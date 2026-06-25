"""Cross-column insight engine — ค้นหาข้อค้นพบเชิงลึกจากการผสมคอลัมน์ (v0.6).

โมดูลนี้เป็น "ตัวค้นหา" (discoverer) ต่างจากโมดูล `insight/` ที่เป็น "ตัวตีความ" (interpreter)
ของผลที่คำนวณไว้แล้ว — ที่นี่เราสร้างมุมมอง (perspective = breakdown × measure × agg) จากคอลัมน์
จริง ๆ แล้วจัดกลุ่ม (group-by) + รวมค่า (aggregate) + ให้คะแนนความน่าสนใจเชิงสถิติ

หลักการสำคัญ:
  * โลจิกล้วน — ไม่มี Jinja/HTML (การเรนเดอร์อยู่ใน report/)
  * ทำงานกับ "ทุก" ชุดข้อมูล — ไม่มีการ hardcode ชื่อคอลัมน์ ไม่ overfit โดเมนใด
    ทุกอย่างขับด้วย ColumnType + cardinality + ช่วงของค่า
  * scipy เป็น optional (thaieda[stats]) — ถ้าไม่มีจะ degrade เป็น effect-size อย่างเดียว + note
  * Benjamini-Hochberg correction บังคับใช้ (มีการทดสอบหลายร้อยครั้ง — กัน false positive)
  * normalize ค่าคีย์ก่อน groupby เสมอ (เลขไทย/อักขระล่องหน/float .0) — กันกลุ่มแตกแบบเงียบ ๆ

4 รูปแบบที่ตรวจต่อ perspective:
  1. outstanding  — กลุ่มเดียวโดดเด่นกว่ากลุ่มอื่นมาก (dominance)
  2. attribution  — กลุ่มเดียวคิดเป็นสัดส่วนใหญ่ของยอดรวม (part-to-whole)
  3. comparison   — กลุ่มเด่นต่างจากกลุ่มที่เหลืออย่างมีนัยสำคัญ (significance-tested)
  4. trend        — แนวโน้มต่อเนื่องตามแกนที่เรียงลำดับได้ (datetime bucketed)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thaieda.analysis import _anova, _scipy_stats
from thaieda.detect import ColumnType, _name_hints_id
from thaieda.schema import _normalize_key_series

# ----------------------------------------------------------------------------
# ค่าคงที่ของ interestingness pipeline (gate → score → penalize → rank)
# ----------------------------------------------------------------------------
_W_PATTERN = 0.5  # น้ำหนักคะแนน pattern
_W_EFFECT = 0.5  # น้ำหนักขนาดผล (effect size)
_MIN_SEGMENT = 30  # แถวขั้นต่ำต่อกลุ่ม (กลุ่มเล็กกว่านี้ไม่น่าเชื่อถือ) — จะปรับตามขนาดข้อมูลใน discover_insights
_SIGNIFICANCE_ALPHA = 0.01  # เข้มกว่า 0.05 เพื่อกัน false positive (multiple comparisons)
_MAX_BREAKDOWNS = 20  # cap จำนวน breakdown candidate (กันระเบิดบนตารางกว้าง)
_MAX_BOOLEAN_BREAKDOWNS = 2  # cap จำนวน boolean breakdown (2 ค่า) — กัน insight ที่ครอบผล
_MAX_MEASURES = 20  # cap จำนวน measure candidate
_DOMINANCE_THRESHOLD = 1.5  # อัตราส่วน top/second ที่ถือว่า "outstanding"
_SHARE_THRESHOLD = 0.5  # สัดส่วนที่ถือว่า "attribution"
# v0.8: เกณฑ์ correlation ที่ถือว่า "strong" (สำหรับ pattern ใหม่)
_CORRELATION_THRESHOLD = 0.7  # |r| >= 0.7 ถือว่า strong correlation
# v0.8: z-score ที่ถือว่าเป็น outlier ในการตรวจ anomaly pattern
_OUTLIER_Z_THRESHOLD = 3.0

# cardinality ของ breakdown ที่ใช้ได้ (น้อยไป=ไม่มีอะไรให้เทียบ, มากไป=ไม่ใช่หมวดหมู่)
_MIN_BREAKDOWN_CARD = 2
_MAX_BREAKDOWN_CARD = 50
# เกณฑ์ Kendall-τ (ขนาด) ที่ถือว่ามีแนวโน้มชัด
_TREND_TAU_MIN = 0.3
# จำนวน bucket ขั้นต่ำสำหรับตรวจ trend (ต้องมีจุดพอจะเห็นทิศทาง)
_TREND_MIN_BUCKETS = 4
# จำนวน segment ขั้นต่ำสำหรับ attribution (>=3 ทำให้สัดส่วน >=50% มีความหมาย)
_ATTRIBUTION_MIN_SEGMENTS = 3
# โทษความซ้ำซ้อน (novelty) สำหรับ card ที่ชี้ไปกลุ่มเดิม/รูปแบบเดิมซ้ำ
_NOVELTY_PENALTY = 0.6
# เมื่อไม่มี scipy: comparison ต้องมี effect ขั้นต่ำเท่านี้จึงจะแสดง (แทนการทดสอบนัยสำคัญ)
_NO_SCIPY_EFFECT_MIN = 0.3
# จำนวน segment ที่เก็บไว้แสดงใน evidence (top-N)
_TOP_SEGMENTS = 3

# ป้ายภาษาไทยของ aggregation
_AGG_TH: dict[str, str] = {"sum": "ผลรวม", "mean": "ค่าเฉลี่ย", "count": "จำนวน"}
# ป้ายของ measure เมื่อเป็นการนับแถว (ไม่มี measure จริง)
_COUNT_MEASURE_TH = "จำนวนข้อมูล"


# ----------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ----------------------------------------------------------------------------
@dataclass
class Perspective:
    """มุมมองการวิเคราะห์หนึ่งคู่ (breakdown × measure × agg)."""

    breakdown: str  # คอลัมน์หมวดหมู่ หรือ datetime ที่ bucket แล้ว
    measure: str | None  # คอลัมน์ตัวเลข; None สำหรับการนับจำนวน (count)
    agg: str  # "sum" | "mean" | "count"

    def to_dict(self) -> dict:
        return {"breakdown": self.breakdown, "measure": self.measure, "agg": self.agg}


@dataclass
class InsightCard:
    """ข้อค้นพบเชิงลึกจากการผสมคอลัมน์ — ภาษาไทย พร้อม evidence."""

    pattern: str  # "outstanding" | "attribution" | "comparison" | "trend"
    perspective: Perspective
    severity: str  # "info" | "warning" (ส่วนใหญ่เป็น info สำหรับข้อค้นพบเชิงวิเคราะห์)
    score: float  # คะแนนความน่าสนใจสุดท้าย 0-1 (ใช้จัดอันดับ)
    title_th: str
    description_th: str
    recommendation_th: str
    evidence: dict  # top segments, shares, lift, p-value, n (JSON-serializable)

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "perspective": self.perspective.to_dict(),
            "severity": self.severity,
            "score": round(self.score, 4),
            "title_th": self.title_th,
            "description_th": self.description_th,
            "recommendation_th": self.recommendation_th,
            "evidence": self.evidence,
        }


@dataclass
class InsightEngineResult:
    """ผลลัพธ์จาก insight engine — รายการ InsightCard เรียงตามความน่าสนใจ."""

    total: int
    cards: list[InsightCard]
    notes: list[str] = field(default_factory=list)  # capping/scipy-missing notes

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "cards": [c.to_dict() for c in self.cards],
            "notes": self.notes,
        }


# ----------------------------------------------------------------------------
# helper: สถิติ (numpy ล้วน — ไม่พึ่ง scipy)
# ----------------------------------------------------------------------------
def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    """แจ้งความคืบหน้าถ้ามี callback."""
    if progress is not None:
        progress(message)


def _norm_cdf(z: float) -> float:
    """ฟังก์ชันการแจกแจงสะสมของ normal มาตรฐาน (ใช้ math.erf — ไม่ต้องมี scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _jsd(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    """Jensen-Shannon divergence ระหว่างการแจกแจงของ a และ b (log ฐาน 2 → อยู่ในช่วง 0-1).

    ใช้วัด "ความต่าง" ของการกระจายค่าระหว่างกลุ่มเด่นกับกลุ่มที่เหลือ
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa.astype("float64")
    pb = pb.astype("float64")
    if pa.sum() == 0 or pb.sum() == 0:
        return 0.0
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    jsd = 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
    # ปัดเข้าช่วง [0,1] (เผื่อ floating error)
    return max(0.0, min(1.0, jsd))


def _mann_kendall(y: np.ndarray) -> tuple[float, float]:
    """Mann-Kendall trend test — คืน (tau, p_value).

    S = ผลรวมของ sign(y_j - y_i) ทุกคู่ i<j; tau = S / C(n,2)
    p-value ใช้ normal approximation พร้อม tie-correction และ continuity correction
    (คำนวณด้วย math.erf จึงไม่ต้องมี scipy)
    """
    n = y.size
    if n < 3:
        return 0.0, 1.0
    s = 0
    for i in range(n - 1):
        s += int(np.sum(np.sign(y[i + 1 :] - y[i])))
    denom = n * (n - 1) / 2.0
    tau = s / denom if denom > 0 else 0.0

    # variance ของ S พร้อมแก้ ties
    _, counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var <= 0:
        return tau, 1.0
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return tau, max(0.0, min(1.0, p))


def _omnibus(values: np.ndarray, groups: list[np.ndarray], st) -> tuple[float, float, str] | None:
    """ทดสอบความต่างของค่าระหว่างกลุ่ม — ANOVA หรือ Kruskal (ถ้าข้อมูลไม่ปกติ).

    คืน (stat, p_value, method) หรือ None ถ้าคำนวณไม่ได้.
    ถ้าไม่มี scipy: ใช้ ANOVA F (p_value = NaN).
    """
    res = _anova(values, groups, st)
    if res is None:
        return None
    f, p = res
    if st is None:
        return f, p, "anova"
    # ข้อมูลมากพอ → ตรวจ normality; ถ้าไม่ปกติใช้ Kruskal-Wallis (ทนต่อการแจกแจงไม่ปกติ)
    try:
        if values.size >= 20:
            _, norm_p = st.normaltest(values)
            if norm_p < 0.05:
                valid = [g for g in groups if g.size > 0]
                if len(valid) >= 2:
                    kr = st.kruskal(*valid)
                    return float(kr[0]), float(kr[1]), "kruskal"
    except Exception:  # noqa: BLE001 — การทดสอบเสริมพังไม่ควรล้มการวิเคราะห์
        pass
    return f, p, "anova"


def _benjamini_hochberg(pvals: list[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg correction — คืน list[bool] ว่าแต่ละ p-value มีนัยสำคัญหรือไม่.

    คุมอัตรา false discovery (FDR) เมื่อทดสอบหลายครั้ง — สำคัญมากเพราะเราอาจทดสอบ
    หลายร้อย perspective (multiple comparisons)
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    threshold_rank = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= (rank / m) * alpha:
            threshold_rank = rank
    significant = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= threshold_rank:
            significant[idx] = True
    return significant


# ----------------------------------------------------------------------------
# helper: เตรียม breakdown key (normalize หมวดหมู่ / bucket datetime)
# ----------------------------------------------------------------------------
@dataclass
class _Breakdown:
    """ข้อมูล breakdown candidate หนึ่งตัว."""

    column: str
    kind: str  # "categorical" | "datetime"
    ordinal: bool  # True ถ้าเรียงลำดับได้ (ใช้ตรวจ trend)
    freq_th: str = ""  # คำอธิบายความถี่ (สำหรับ datetime)


def _normalize_categorical(raw: pd.Series) -> pd.Series:
    """normalize ค่าหมวดหมู่เป็นสตริงมาตรฐานก่อน groupby (เลขไทย/อักขระล่องหน/float .0).

    คืน Series object ที่ค่าว่าง/NaN เป็น np.nan (จะถูก groupby ทิ้งเอง)
    """
    mask = raw.notna()
    out = pd.Series(np.nan, index=raw.index, dtype=object)
    if mask.any():
        out.loc[mask] = _normalize_key_series(raw[mask]).to_numpy()
    return out.where(out != "", np.nan)


# ความถี่ pandas → คำอธิบายภาษาไทย (สำหรับ bucket datetime)
_FREQ_TH: dict[str, str] = {
    "Y": "รายปี",
    "Q": "รายไตรมาส",
    "M": "รายเดือน",
    "W": "รายสัปดาห์",
    "D": "รายวัน",
}


def _bucket_datetime(raw: pd.Series) -> tuple[pd.Series, str] | None:
    """bucket คอลัมน์ datetime เป็นช่วงเวลาที่เรียงลำดับได้ — เลือกความถี่ให้ได้ 4-50 กลุ่ม.

    เลือกความถี่ที่ "ละเอียดสุด" ที่ยังให้จำนวน bucket ไม่เกิน _MAX_BREAKDOWN_CARD
    คืน (Series ของ label ช่วงเวลา, รหัสความถี่) หรือ None ถ้า bucket ไม่ได้
    label เป็นสตริง period ที่เรียงตามตัวอักษร = เรียงตามเวลา (ใช้ตรวจ trend ได้)
    """
    dt = pd.to_datetime(raw, errors="coerce")
    valid = dt.notna()
    if int(valid.sum()) < _TREND_MIN_BUCKETS:
        return None
    for freq in ("D", "W", "M", "Q", "Y"):
        periods = dt.dt.to_period(freq)
        labels = periods.astype(str).where(valid, np.nan)
        nuniq = int(labels[valid].nunique())
        if _MIN_BREAKDOWN_CARD <= nuniq <= _MAX_BREAKDOWN_CARD:
            return labels, freq
    return None


def _build_key(df: pd.DataFrame, bd: _Breakdown) -> pd.Series | None:
    """สร้าง breakdown key series (normalize/bucket แล้ว) สำหรับ breakdown หนึ่งตัว."""
    raw = df[bd.column]
    if bd.kind == "categorical":
        return _normalize_categorical(raw)
    bucketed = _bucket_datetime(raw)
    if bucketed is None:
        return None
    return bucketed[0]


# ----------------------------------------------------------------------------
# helper: คัดเลือก candidate breakdowns / measures
# ----------------------------------------------------------------------------
# ประเภทที่ห้ามเป็น breakdown (ID/ข้อความอิสระ/เบอร์/ว่าง) — กัน insight ที่ไร้สาระ (tautology)
_NON_BREAKDOWN_TYPES = {
    ColumnType.ID,
    ColumnType.PHONE_NUMBER,
    ColumnType.THAI_TEXT,
    ColumnType.ENGLISH_TEXT,
    ColumnType.MIXED_TEXT,
    ColumnType.EMPTY,
    ColumnType.NUMERIC,
}


def _select_breakdowns(
    df: pd.DataFrame, column_types: dict[str, ColumnType], notes: list[str]
) -> list[_Breakdown]:
    """เลือก candidate breakdowns — หมวดหมู่ (2-50 ค่า) + datetime ที่ bucket ได้.

    จัดลำดับความสำคัญ: categorical ที่มีหลายกลุ่มขึ้นก่อน, boolean (2 ค่า) ลด priority
    (boolean breakdown มักให้ insight ที่ถูกต้องแต่ไม่ค่อยน่าสนใจ — กลุ่มที่เยอะกว่าชนะเสมอ)
    """
    out: list[_Breakdown] = []
    for col, ctype in column_types.items():
        if col not in df.columns:
            continue
        if ctype == ColumnType.DATETIME:
            bucketed = _bucket_datetime(df[col])
            if bucketed is not None:
                out.append(
                    _Breakdown(col, "datetime", ordinal=True, freq_th=_FREQ_TH.get(bucketed[1], ""))
                )
            continue
        if ctype in _NON_BREAKDOWN_TYPES:
            continue
        if ctype == ColumnType.CATEGORICAL:
            nunique = int(df[col].dropna().nunique())
            if _MIN_BREAKDOWN_CARD <= nunique <= _MAX_BREAKDOWN_CARD:
                out.append(_Breakdown(col, "categorical", ordinal=False))
    # จัดลำดับ: breakdown ที่มีหลายกลุ่มขึ้นก่อน (boolean = 2 ค่า ลงหลัง) — ให้ categorical จริงได้โอกาสก่อน
    out.sort(key=lambda b: _breakdown_cardinality(df, b), reverse=True)
    # cap จำนวน boolean breakdowns (2 ค่า) ไว้ไม่เกิน 2 ตัว — กันครอบผล
    bool_bds = [b for b in out if _breakdown_cardinality(df, b) <= 2]
    if len(bool_bds) > _MAX_BOOLEAN_BREAKDOWNS:
        kept_bool = set(bool_bds[:_MAX_BOOLEAN_BREAKDOWNS])
        removed = [b.column for b in bool_bds[_MAX_BOOLEAN_BREAKDOWNS:]]
        notes.append(
            f"จำกัด boolean breakdown ไว้ {_MAX_BOOLEAN_BREAKDOWNS} ตัว "
            f"(ข้าม: {', '.join(removed[:5])}) — ลด insight ที่ครอบผล"
        )
        out = [b for b in out if _breakdown_cardinality(df, b) > 2 or b in kept_bool]
    if len(out) > _MAX_BREAKDOWNS:
        notes.append(
            f"จำกัด breakdown ไว้ {_MAX_BREAKDOWNS} จาก {len(out)} คอลัมน์ (กันการคำนวณบานปลาย)"
        )
        out = out[:_MAX_BREAKDOWNS]
    return out


def _breakdown_cardinality(df: pd.DataFrame, bd: _Breakdown) -> int:
    """จำนวนค่าไม่ซ้ำของ breakdown (ใช้เรียงลำดับความสำคัญ)."""
    if bd.kind == "datetime":
        return 99  # datetime มีความสำคัญสูงเสมอ (trend ใช้ได้)
    return int(df[bd.column].dropna().nunique())


def _is_non_additive(values: np.ndarray) -> bool:
    """True ถ้า measure เป็นค่าที่ "บวกรวมแล้วไม่มีความหมาย" (สัดส่วน/เปอร์เซ็นต์) → ข้าม sum.

    เกณฑ์: ค่าอยู่ใน [0,1] (สัดส่วน/ความน่าจะเป็น) หรือ [0,100] แบบ float ที่มีทศนิยม
    (น่าจะเป็นเปอร์เซ็นต์) — แต่จำนวนเต็มใน [0,100] ยังถือว่าบวกได้ (เช่น ยอด/อายุ/คะแนน)
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin >= 0.0 and vmax <= 1.0:
        return True
    has_fraction = bool(np.any(finite != np.floor(finite)))
    return vmin >= 0.0 and vmax <= 100.0 and has_fraction


def _select_measures(
    df: pd.DataFrame, column_types: dict[str, ColumnType], notes: list[str]
) -> dict[str, dict]:
    """เลือก candidate measures — คอลัมน์ตัวเลขที่ความแปรปรวน > 0 (ไม่ใช่ ID/FK).

    กรองคอลัมน์ที่ชื่อบอกใบ้ว่าเป็น ID/FK (เช่น store_id, product_id) ออก —
    แม้ว่า detect จะจัดเป็น NUMERIC (เพราะค่าซ้ำจึงไม่เข้าเกณฑ์ ID) แต่ sum/mean ของ
    ID ไม่มีความหมายทางธุรกิจ จึงไม่ควรเป็น measure

    คืน mapping ชื่อ measure -> {"non_additive": bool} (ใช้ตัดสินว่าจะทำ agg "sum" หรือไม่)
    """
    out: dict[str, dict] = {}
    skipped_fk: list[str] = []
    for col, ctype in column_types.items():
        if ctype != ColumnType.NUMERIC or col not in df.columns:
            continue
        # กรอง FK ที่ชื่อบอกใบ้ว่าเป็น ID (เช่น store_id, customer_id) — ไม่ใช่ measure ที่มีความหมาย
        if _name_hints_id(df[col]):
            skipped_fk.append(col)
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(numeric) < 2 or float(numeric.std()) == 0.0 or numeric.nunique() <= 1:
            continue
        out[col] = {"non_additive": _is_non_additive(numeric.to_numpy(dtype="float64"))}
        if len(out) >= _MAX_MEASURES:
            notes.append(f"จำกัด measure ไว้ {_MAX_MEASURES} คอลัมน์ (กันการคำนวณบานปลาย)")
            break
    if skipped_fk:
        notes.append(f"ข้ามคอลัมน์ ID/FK ที่ไม่ใช่ measure ที่มีความหมาย: {', '.join(skipped_fk[:5])}")
    return out


# ----------------------------------------------------------------------------
# helper: รวมค่า (aggregate) ต่อ breakdown × measure
# ----------------------------------------------------------------------------
def _agg_measure(key: pd.Series, measure: pd.Series) -> dict | None:
    """groupby breakdown แล้วรวมค่า measure — คืน sum/mean/size series + กลุ่มดิบ (สำหรับ comparison).

    คืน None ถ้าไม่มีข้อมูลพอ (น้อยกว่า 2 กลุ่ม)
    """
    num = pd.to_numeric(measure, errors="coerce")
    frame = pd.DataFrame({"_g": key, "_m": num}).dropna()
    if frame.empty:
        return None
    grouped = frame.groupby("_g", observed=True)["_m"]
    sizes = grouped.size()
    if len(sizes) < _MIN_BREAKDOWN_CARD:
        return None
    return {
        "sum": grouped.sum(),
        "mean": grouped.mean(),
        "sizes": sizes,
        "frame": frame,
    }


def _agg_count(key: pd.Series) -> pd.Series | None:
    """นับจำนวนแถวต่อกลุ่ม — คืน size series หรือ None ถ้ากลุ่มน้อยเกินไป."""
    sizes = key.dropna().value_counts()
    if len(sizes) < _MIN_BREAKDOWN_CARD:
        return None
    return sizes


def _top_segments(series: pd.Series, decimals: int = 2) -> list[list]:
    """คืน top-N segment เป็น [[label, value], ...] (JSON-serializable)."""
    top = series.sort_values(ascending=False).head(_TOP_SEGMENTS)
    return [[str(k), round(float(v), decimals)] for k, v in top.items()]


# ----------------------------------------------------------------------------
# ตัวตรวจ 4 รูปแบบ — แต่ละตัวคืน dict candidate หรือ None
# ----------------------------------------------------------------------------
def _detect_outstanding(
    series: pd.Series, sizes: pd.Series, agg: str, min_segment: int
) -> dict | None:
    """รูปแบบ 1: กลุ่มเดียวโดดเด่น (top/second >= threshold) และมีจำนวนแถวมากพอ."""
    if len(series) < 2:
        return None
    ordered = series.sort_values(ascending=False)
    top_seg = str(ordered.index[0])
    top_val = float(ordered.iloc[0])
    second_val = float(ordered.iloc[1])
    if top_val <= 0 or second_val <= 0:
        return None
    ratio = top_val / second_val
    if ratio < _DOMINANCE_THRESHOLD:
        return None
    top_size = int(sizes.get(ordered.index[0], 0))
    if top_size < min_segment:
        return None
    overall_mean = float(series.mean())
    lift = top_val / overall_mean if overall_mean > 0 else 1.0
    return {
        "pattern": "outstanding",
        "pattern_score": max(0.0, min(1.0, 1.0 - second_val / top_val)),
        "effect_norm": min(lift / 5.0, 1.0),
        "p_value": None,
        "top_segment": top_seg,
        "n_segments": int(len(series)),
        "top_size": top_size,
        "evidence": {
            "top_segment": top_seg,
            "top_value": round(top_val, 2),
            "dominance_ratio": round(ratio, 2),
            "top_segments": _top_segments(series),
        },
    }


def _detect_attribution(series: pd.Series, sizes: pd.Series) -> dict | None:
    """รูปแบบ 2: กลุ่มเดียวคิดเป็นสัดส่วนใหญ่ (>= threshold) ของยอดรวม (ต้องมี >= 3 กลุ่ม)."""
    if len(series) < _ATTRIBUTION_MIN_SEGMENTS:
        return None
    total = float(series.sum())
    if total <= 0:
        return None
    ordered = series.sort_values(ascending=False)
    top_seg = str(ordered.index[0])
    top_val = float(ordered.iloc[0])
    share = top_val / total
    if share < _SHARE_THRESHOLD:
        return None
    k = len(series)
    lift = share * k  # เทียบกับสัดส่วนถ้าทุกกลุ่มเท่ากัน (1/k)
    return {
        "pattern": "attribution",
        "pattern_score": max(0.0, min(1.0, share)),
        "effect_norm": min(lift / 5.0, 1.0),
        "p_value": None,
        "top_segment": top_seg,
        "n_segments": int(k),
        "top_size": int(sizes.get(ordered.index[0], 0)),
        "evidence": {
            "top_segment": top_seg,
            "share": round(share * 100.0, 1),
            "total": round(total, 2),
            "top_segments": _top_segments(series),
        },
    }


def _detect_comparison(agg_data: dict, min_segment: int, st) -> dict | None:
    """รูปแบบ 3: กลุ่มเด่นต่างจากกลุ่มที่เหลืออย่างมีนัยสำคัญ (ANOVA/Kruskal + JSD)."""
    means = agg_data["mean"]
    sizes = agg_data["sizes"]
    frame = agg_data["frame"]
    if len(means) < 2:
        return None
    top_label = means.idxmax()
    top_seg = str(top_label)
    if int(sizes.get(top_label, 0)) < min_segment:
        return None

    top_vals = frame.loc[frame["_g"] == top_label, "_m"].to_numpy(dtype="float64")
    rest_vals = frame.loc[frame["_g"] != top_label, "_m"].to_numpy(dtype="float64")
    if top_vals.size < min_segment or rest_vals.size < min_segment:
        return None

    groups = [g["_m"].to_numpy(dtype="float64") for _, g in frame.groupby("_g", observed=True)]
    omni = _omnibus(frame["_m"].to_numpy(dtype="float64"), groups, st)
    if omni is None:
        return None
    _stat, p_value, method = omni

    top_mean = float(top_vals.mean())
    rest_mean = float(rest_vals.mean())
    if rest_mean == 0:
        return None
    lift_pct = (top_mean - rest_mean) / abs(rest_mean) * 100.0
    jsd = _jsd(top_vals, rest_vals)
    return {
        "pattern": "comparison",
        "pattern_score": jsd,
        "effect_norm": min(abs(lift_pct) / 100.0, 1.0),
        "p_value": None if (p_value is None or math.isnan(p_value)) else float(p_value),
        "top_segment": top_seg,
        "n_segments": int(len(means)),
        "top_size": int(top_vals.size),
        "evidence": {
            "top_segment": top_seg,
            "top_mean": round(top_mean, 2),
            "rest_mean": round(rest_mean, 2),
            "lift_pct": round(lift_pct, 1),
            "p_value": (None if (p_value is None or math.isnan(p_value)) else round(p_value, 4)),
            "method": method,
            "n_top": int(top_vals.size),
            "n_rest": int(rest_vals.size),
            "jsd": round(jsd, 3),
        },
    }


def _detect_trend(series: pd.Series, agg: str) -> dict | None:
    """รูปแบบ 4: แนวโน้มต่อเนื่องตามแกนที่เรียงลำดับได้ (Mann-Kendall)."""
    ordered = series.sort_index()  # label ช่วงเวลาเรียงตามตัวอักษร = เรียงตามเวลา
    if len(ordered) < _TREND_MIN_BUCKETS:
        return None
    y = ordered.to_numpy(dtype="float64")
    tau, p_value = _mann_kendall(y)
    if abs(tau) < _TREND_TAU_MIN:
        return None
    up = tau > 0
    first_val = float(y[0])
    last_val = float(y[-1])
    # เก็บ bucket series เต็มสำหรับกราฟ (v0.8 — กัน trend chart แสดงแค่ 2 จุด)
    all_buckets = [[str(idx), round(float(val), 2)] for idx, val in ordered.items()]
    return {
        "pattern": "trend",
        "pattern_score": max(0.0, min(1.0, 1.0 - p_value)),
        "effect_norm": min(abs(tau), 1.0),
        "p_value": float(p_value),
        "top_segment": str(ordered.index[-1]),
        "n_segments": int(len(ordered)),
        "top_size": int(len(ordered)),
        "evidence": {
            "direction": "up" if up else "down",
            "tau": round(tau, 3),
            "p_value": round(p_value, 4),
            "n_buckets": int(len(ordered)),
            "first_bucket": str(ordered.index[0]),
            "last_bucket": str(ordered.index[-1]),
            "first_value": round(first_val, 2),
            "last_value": round(last_val, 2),
            "all_buckets": all_buckets,
        },
    }


# ----------------------------------------------------------------------------
# เก็บ candidate ทั้งหมดจาก breakdown × measure × agg × pattern
# ----------------------------------------------------------------------------
def _collect_candidates(
    df: pd.DataFrame,
    breakdowns: list[_Breakdown],
    measures: dict[str, dict],
    min_segment: int,
    st,
) -> list[dict]:
    """สแกนทุก perspective แล้วเก็บ candidate (ก่อนจัด gate/ranking)."""
    candidates: list[dict] = []
    for bd in breakdowns:
        key = _build_key(df, bd)
        if key is None:
            continue

        # --- count-only perspective (measure=None) ---
        count_series = _agg_count(key)
        if count_series is not None:
            for det in (
                _detect_outstanding(count_series, count_series, "count", min_segment),
                _detect_attribution(count_series, count_series),
                (_detect_trend(count_series, "count") if bd.ordinal else None),
            ):
                if det is not None:
                    _attach(det, bd, None, "count", det["pattern"])
                    candidates.append(det)

        # --- measure-based perspectives ---
        for measure, meta in measures.items():
            if measure == bd.column or measure not in df.columns:
                continue
            agg_data = _agg_measure(key, df[measure])
            if agg_data is None:
                continue

            # sum (เฉพาะ measure ที่บวกได้)
            if not meta["non_additive"]:
                sum_series = agg_data["sum"]
                for det in (
                    _detect_outstanding(sum_series, agg_data["sizes"], "sum", min_segment),
                    _detect_attribution(sum_series, agg_data["sizes"]),
                    (_detect_trend(sum_series, "sum") if bd.ordinal else None),
                ):
                    if det is not None:
                        _attach(det, bd, measure, "sum", det["pattern"])
                        candidates.append(det)

            # mean
            mean_series = agg_data["mean"]
            mean_dets = [
                _detect_outstanding(mean_series, agg_data["sizes"], "mean", min_segment),
                _detect_comparison(agg_data, min_segment, st),
            ]
            if bd.ordinal:
                mean_dets.append(_detect_trend(mean_series, "mean"))
            for det in mean_dets:
                if det is not None:
                    _attach(det, bd, measure, "mean", det["pattern"])
                    candidates.append(det)

    return candidates


def _attach(det: dict, bd: _Breakdown, measure: str | None, agg: str, pattern: str) -> None:
    """ผูกข้อมูล perspective เข้ากับ candidate."""
    det["breakdown"] = bd.column
    det["breakdown_freq_th"] = bd.freq_th
    det["measure"] = measure
    det["agg"] = agg


# ----------------------------------------------------------------------------
# interestingness pipeline: gate → score → penalize → rank
# ----------------------------------------------------------------------------
def _triviality_penalty(pattern: str, n_segments: int, top_size: int, total_n: int) -> float:
    """โทษ insight ที่ "เห็นได้ชัดอยู่แล้ว" (trivial)."""
    penalty = 0.0
    # กลุ่มมีเพียง 2 ระดับ: "outstanding" ก็แค่บอกว่าฝั่งไหนมากกว่า — ไม่ลึก
    if pattern == "outstanding" and n_segments <= 2:
        penalty = max(penalty, 0.2)
    # กลุ่มเด่นครอบคลุมเกือบทั้งชุด (>97% ของแถว) → การที่มันเด่นเป็นเรื่องธรรมดา
    if total_n > 0 and top_size / total_n > 0.97:
        penalty = max(penalty, 0.3)
    return penalty


def _gate_and_score(
    candidates: list[dict], alpha: float, total_n: int, notes: list[str]
) -> list[dict]:
    """ผ่าน significance gate (BH) + คำนวณ base score + novelty + triviality."""
    # --- significance gate ด้วย Benjamini-Hochberg (เฉพาะ comparison/trend ที่มี p-value) ---
    tested = [
        (i, c["p_value"])
        for i, c in enumerate(candidates)
        if c["pattern"] in ("comparison", "trend")
        and c["p_value"] is not None
        and not math.isnan(c["p_value"])
    ]
    sig_map: dict[int, bool] = {}
    if tested:
        flags = _benjamini_hochberg([p for _, p in tested], alpha)
        sig_map = {tested[k][0]: flags[k] for k in range(len(tested))}

    no_scipy_note = False
    kept: list[dict] = []
    for i, c in enumerate(candidates):
        if c["pattern"] in ("comparison", "trend"):
            if i in sig_map:
                if not sig_map[i]:
                    continue  # ไม่ผ่านนัยสำคัญหลังแก้ multiple comparisons → ตัดทิ้ง
            else:
                # ไม่มี p-value (เช่น comparison เมื่อไม่มี scipy) → ใช้ effect size แทน
                if c["effect_norm"] < _NO_SCIPY_EFFECT_MIN:
                    continue
                no_scipy_note = True
        c["base"] = _W_PATTERN * c["pattern_score"] + _W_EFFECT * c["effect_norm"]
        kept.append(c)

    if no_scipy_note:
        notes.append(
            "ไม่มี scipy — ทดสอบนัยสำคัญด้วย effect size อย่างเดียว "
            "(ติดตั้ง pip install thaieda[stats] เพื่อ p-value ที่ถูกต้อง)"
        )

    # --- novelty: ลดคะแนน card ที่ชี้กลุ่มเดิม/รูปแบบเดิมซ้ำ (breakdown × pattern × top_segment) ---
    groups: dict[tuple, list[dict]] = {}
    for c in kept:
        gkey = (c["breakdown"], c["pattern"], c["top_segment"])
        groups.setdefault(gkey, []).append(c)
    for lst in groups.values():
        lst.sort(key=lambda c: -c["base"])
        for rank, c in enumerate(lst):
            c["novelty"] = 1.0 if rank == 0 else _NOVELTY_PENALTY

    # --- v0.8: cross-pattern novelty — ถ้าหลาย pattern ชี้ไปที่ breakdown × measure × top_segment
    #     เดียวกัน (เช่น outlier 1000 ทำให้ outstanding+comparison+attribution พูดเรื่องเดียวกัน)
    #     เก็บเฉพาะอันดับสูงสุดเต็มคะแนน ที่เหลือโดน penalty หนักขึ้น ---
    cross_groups: dict[tuple, list[dict]] = {}
    for c in kept:
        # ใช้ breakdown × measure × top_segment (ไม่รวม pattern) — กันซ้ำข้าม pattern
        measure_key = c.get("measure") or "__count__"
        cgkey = (c["breakdown"], measure_key, c["top_segment"])
        cross_groups.setdefault(cgkey, []).append(c)
    for lst in cross_groups.values():
        if len(lst) <= 1:
            continue
        lst.sort(key=lambda c: -c["base"])
        # อันดับ 0 = เต็มคะแนน, อันดับ 1 = penalty หนัก, อันดับ 2+ = penalty หนักกว่า
        for rank, c in enumerate(lst):
            if rank == 0:
                continue  # ไม่แตะอันดับ 1
            # ทับ novelty เดิมด้วยค่าที่หนักกว่า (ขนาดเล็กกว่า)
            c["novelty"] = min(c.get("novelty", 1.0), _NOVELTY_PENALTY ** (rank + 1))

    # --- final score ---
    for c in kept:
        triv = _triviality_penalty(c["pattern"], c["n_segments"], c["top_size"], total_n)
        c["score"] = 1.0 * c["base"] * c["novelty"] * (1.0 - triv)

    kept.sort(key=lambda c: -c["score"])
    return kept


# ----------------------------------------------------------------------------
# สร้างข้อความภาษาไทย
# ----------------------------------------------------------------------------
def _measure_label(measure: str | None) -> str:
    """ป้ายของ measure (ภาษาไทย) — ใช้ '<measure>' หรือ 'จำนวนข้อมูล' เมื่อเป็นการนับ."""
    return f"'{measure}'" if measure is not None else _COUNT_MEASURE_TH


def _build_text(c: dict) -> tuple[str, str, str]:
    """สร้าง (title_th, description_th, recommendation_th) จาก candidate."""
    pattern = c["pattern"]
    breakdown = c["breakdown"]
    measure = c["measure"]
    agg_th = _AGG_TH.get(c["agg"], c["agg"])
    mlabel = _measure_label(measure)
    ev = c["evidence"]

    if pattern == "outstanding":
        top = ev["top_segment"]
        title = f"กลุ่ม '{top}' โดดเด่นใน {mlabel} (ตาม '{breakdown}')"
        if measure is None:
            desc = (
                f"กลุ่ม '{top}' มีจำนวนข้อมูลสูงสุดที่ {ev['top_value']:,.0f} แถว "
                f"— มากกว่ากลุ่มรอง {ev['dominance_ratio']:.1f} เท่า (จัดกลุ่มตาม '{breakdown}')"
            )
        else:
            desc = (
                f"กลุ่ม '{top}' มี{agg_th} {mlabel} สูงสุดที่ {ev['top_value']:,.1f} "
                f"— สูงกว่ากลุ่มรอง {ev['dominance_ratio']:.1f} เท่า (จัดกลุ่มตาม '{breakdown}')"
            )
        rec = f"ตรวจสอบว่าเหตุใดกลุ่ม '{top}' จึงโดดเด่น และพิจารณาใช้เป็นจุดโฟกัสในการวิเคราะห์/ตัดสินใจ"
        return title, desc, rec

    if pattern == "attribution":
        top = ev["top_segment"]
        title = f"กลุ่ม '{top}' คิดเป็นสัดส่วนใหญ่ของ{agg_th} {mlabel} (ตาม '{breakdown}')"
        desc = (
            f"กลุ่ม '{top}' คิดเป็น {ev['share']:.1f}% ของ{agg_th} {mlabel} ทั้งหมด "
            f"(จัดกลุ่มตาม '{breakdown}')"
        )
        rec = f"กลุ่ม '{top}' มีสัดส่วนสูงต่อภาพรวม — ติดตามเป็นพิเศษ และประเมินความเสี่ยงจากการพึ่งพากลุ่มเดียว"
        return title, desc, rec

    if pattern == "comparison":
        top = ev["top_segment"]
        lift = ev["lift_pct"]
        p_txt = f", p={ev['p_value']:.3f}" if ev["p_value"] is not None else ""
        title = f"กลุ่ม '{top}' ต่างจากกลุ่มอื่นอย่างมีนัยสำคัญใน {mlabel} (ตาม '{breakdown}')"
        desc = (
            f"กลุ่ม '{top}' มี{agg_th} {mlabel} {lift:.0f}% สูงกว่ากลุ่มอื่น"
            f" (เฉลี่ย {ev['top_mean']:,.1f} เทียบกับ {ev['rest_mean']:,.1f}{p_txt})"
        )
        rec = (
            f"ความแตกต่างระหว่างกลุ่มมีนัยสำคัญ — พิจารณาแยกวิเคราะห์ {mlabel} ตาม '{breakdown}' "
            "หรือใช้เป็นฟีเจอร์ในการสร้างโมเดล"
        )
        return title, desc, rec

    # trend
    if pattern == "trend":
        up = ev["direction"] == "up"
        dir_th = "เพิ่มขึ้น" if up else "ลดลง"
        freq = f" ({c['breakdown_freq_th']})" if c.get("breakdown_freq_th") else ""
        title = f"{agg_th} {mlabel} {dir_th}ตาม '{breakdown}'"
        desc = (
            f"{agg_th} {mlabel} {dir_th}อย่างมีนัยสำคัญตาม '{breakdown}'{freq} "
            f"(τ={ev['tau']:.2f}, จาก {ev['first_value']:,.1f} เป็น {ev['last_value']:,.1f})"
        )
        rec = f"พบแนวโน้ม{dir_th}ตาม '{breakdown}' — พิจารณาวางแผน/พยากรณ์โดยคำนึงถึงทิศทางนี้"
        return title, desc, rec

    # v0.8: correlation
    if pattern == "correlation":
        col_a = ev["col_a"]
        col_b = ev["col_b"]
        r = ev["correlation"]
        direction = ev["direction"]
        dir_th = "เป็นบวก" if direction == "positive" else "เป็นลบ"
        title = f"'{col_a}' และ '{col_b}' มีความสัมพันธ์กันสูง (r={r:.2f})"
        desc = (
            f"คอลัมน์ '{col_a}' และ '{col_b}' มีความสัมพันธ์{dir_th}ที่ strong "
            f"(r={r:.3f}, n={ev['n']:,}) — ค่าเคลื่อนไหวไปด้วยกัน"
        )
        rec = (
            "คอลัมน์ทั้งสองสัมพันธ์กันสูง — อาจวัดสิ่งเดียวกัน พิจารณาใช้คอลัมน์ใดคอลัมน์หนึ่ง "
            "หรือรวมเป็น feature เดียวเพื่อกัน multicollinearity"
        )
        return title, desc, rec

    # v0.8: outlier
    if pattern == "outlier":
        col = ev["column"]
        title = (
            f"พบ outlier {ev['outlier_count']} แถวในคอลัมน์ '{col}' "
            f"(z-score ≥ {_OUTLIER_Z_THRESHOLD})"
        )
        desc = (
            f"คอลัมน์ '{col}' มี {ev['outlier_count']} ค่าที่เป็น outlier "
            f"({ev['percentage']:.1f}% ของข้อมูล, max z-score={ev['max_z_score']:.1f}, "
            f"mean={ev['mean']:,.1f}, std={ev['std']:,.1f})"
        )
        rec = (
            f"ตรวจสอบ outlier ในคอลัมน์ '{col}' — อาจเป็นค่าผิดปกติจริงหรือการกรอกผิด "
            "พิจารณา clip/transform ก่อนนำไปวิเคราะห์หรือสร้างโมเดล"
        )
        return title, desc, rec


def _to_card(c: dict) -> InsightCard:
    """แปลง candidate dict เป็น InsightCard พร้อมข้อความภาษาไทย."""
    title, desc, rec = _build_text(c)
    return InsightCard(
        pattern=c["pattern"],
        perspective=Perspective(c["breakdown"], c["measure"], c["agg"]),
        severity="info",
        score=float(c["score"]),
        title_th=title,
        description_th=desc,
        recommendation_th=rec,
        evidence=c["evidence"],
    )


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def discover_insights(
    df: pd.DataFrame,
    column_types: dict[str, ColumnType],
    *,
    top_n: int = 8,
    sample_size: int = 100_000,
    min_segment: int = _MIN_SEGMENT,
    progress: Callable[[str], None] | None = None,
) -> InsightEngineResult:
    """ค้นหาข้อค้นพบจากการผสมคอลัมน์ (group-by + aggregate + statistical scoring).

    ขั้นตอน:
      1. ระบุ candidate breakdowns (categorical 2-50 unique + datetime bucketed)
      2. ระบุ candidate measures (numeric ที่ความแปรปรวน > 0, ไม่ใช่ ID)
      3. สร้าง perspectives (breakdown × measure × agg) แล้วตรวจ 4 รูปแบบ
         + v0.8: correlation (numeric × numeric) + outlier (row-level)
      4. Two-phase: ให้คะแนนบน sample → คำนวณตัวเลขจริงบนข้อมูลเต็มสำหรับ top-N
      5. จัดอันดับด้วย interestingness pipeline (gate → score → penalize → rank)

    Args:
        df: ข้อมูลที่วิเคราะห์.
        column_types: ประเภทคอลัมน์ (จาก detect_all).
        top_n: จำนวน InsightCard สูงสุดที่คืน.
        sample_size: จำนวนแถวสูงสุดในเฟสให้คะแนน (สุ่มถ้าข้อมูลใหญ่กว่านี้).
        min_segment: จำนวนแถวขั้นต่ำต่อกลุ่ม — v0.8: ถ้าเป็น _MIN_SEGMENT (30)
            จะปรับเป็น adaptive ตามขนาดข้อมูล (max(5, total_n // 20)).
        progress: callback(ข้อความ) สำหรับแสดงความคืบหน้า.

    Returns:
        InsightEngineResult พร้อมรายการ InsightCard เรียงตามความน่าสนใจ.
    """
    notes: list[str] = []
    if df is None or len(df) == 0 or len(df.columns) == 0:
        return InsightEngineResult(total=0, cards=[], notes=notes)

    st = _scipy_stats()

    # v0.8: adaptive min_segment — ถ้าผู้ใช้ไม่ระบุ (ใช้ default _MIN_SEGMENT) ให้ปรับตามขนาด
    total_n = len(df)
    if min_segment == _MIN_SEGMENT and total_n < _MIN_SEGMENT * 10:
        # ข้อมูลเล็ก: ลดเกณฑ์เป็น max(5, total_n // 20) กันข้ามทุกกลุ่ม
        adaptive_min = max(5, total_n // 20)
        if adaptive_min < min_segment:
            notes.append(
                f"ปรับ min_segment จาก {min_segment} เป็น {adaptive_min} "
                f"(เหมาะกับขนาดข้อมูล {total_n:,} แถว)"
            )
            min_segment = adaptive_min

    _emit(progress, "เลือกคอลัมน์ผสมสำหรับวิเคราะห์...")
    breakdowns = _select_breakdowns(df, column_types, notes)
    measures = _select_measures(df, column_types, notes)
    if not breakdowns and not measures:
        notes.append("ไม่พบคอลัมน์หมวดหมู่/วันที่/ตัวเลขที่ใช้วิเคราะห์ได้ — ข้ามการค้นหาข้อค้นพบ")
        return InsightEngineResult(total=0, cards=[], notes=notes)

    # --- two-phase: สุ่มตัวอย่างสำหรับเฟสให้คะแนน (ถ้าข้อมูลใหญ่) ---
    sampled = total_n > sample_size
    score_df = df.sample(n=sample_size, random_state=42) if sampled else df
    if sampled:
        notes.append(
            f"ให้คะแนนบนตัวอย่าง {sample_size:,} จาก {total_n:,} แถว "
            "แล้วคำนวณตัวเลขจริงบน top-N ด้วยข้อมูลเต็ม"
        )

    _emit(progress, "ให้คะแนนมุมมองการวิเคราะห์...")
    # บน sample ลดเกณฑ์ขนาดกลุ่มตามสัดส่วน
    if sampled:
        score_min_segment = max(2, int(min_segment * len(score_df) / total_n))
    else:
        score_min_segment = min_segment
    candidates = _collect_candidates(score_df, breakdowns, measures, score_min_segment, st)

    # v0.8: เพิ่ม correlation + outlier candidates
    _emit(progress, "ตรวจสอบความสัมพันธ์และ outlier...")
    corr_candidates = _detect_strong_correlations(score_df, measures)
    candidates.extend(corr_candidates)
    outlier_candidates = _detect_outlier_insights(score_df, measures, column_types)
    candidates.extend(outlier_candidates)

    ranked = _gate_and_score(candidates, _SIGNIFICANCE_ALPHA, len(score_df), notes)

    selected = ranked[:top_n]

    # --- phase 2: คำนวณ evidence จริงบนข้อมูลเต็มสำหรับ top-N ---
    if sampled and selected:
        _emit(progress, "คำนวณตัวเลขจริงบนข้อมูลเต็ม (top-N)...")
        selected = _recompute_full(df, selected, breakdowns, measures, min_segment, st)

    cards = [_to_card(c) for c in selected]
    return InsightEngineResult(total=len(cards), cards=cards, notes=notes)


# ----------------------------------------------------------------------------
# v0.8: pattern ใหม่ — correlation + outlier
# ----------------------------------------------------------------------------
def _detect_strong_correlations(df: pd.DataFrame, measures: dict[str, dict]) -> list[dict]:
    """ตรวจหาความสัมพันธ์ที่ strong (|r| >= 0.7) ระหว่างคู่คอลัมน์ตัวเลข — v0.8.

    คืน list ของ candidate dict ที่มี pattern="correlation"
    จำกัดที่ top 10 คู่เพื่อกันการคำนวณบานปลาย
    """
    measure_cols = [m for m in measures if m in df.columns]
    if len(measure_cols) < 2:
        return []

    numeric = df[measure_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(numeric) < 10:
        return []

    corr_matrix = numeric.corr(numeric_only=True)
    candidates: list[dict] = []

    # ดึงคู่ที่ |r| >= threshold (ข้ามแนวทแยงและคู่ซ้ำ)
    seen: set[tuple[str, str]] = set()
    for i, col_a in enumerate(measure_cols):
        for col_b in measure_cols[i + 1 :]:
            pair = tuple(sorted([col_a, col_b]))
            if pair in seen:
                continue
            seen.add(pair)
            r = float(corr_matrix.get(col_a, {}).get(col_b, 0.0))
            if pd.isna(r):
                continue
            if abs(r) < _CORRELATION_THRESHOLD:
                continue
            direction = "positive" if r > 0 else "negative"
            candidates.append(
                {
                    "pattern": "correlation",
                    "pattern_score": min(1.0, abs(r)),
                    "effect_norm": min(1.0, abs(r)),
                    "p_value": None,
                    "top_segment": col_b,
                    "n_segments": 2,
                    "top_size": len(numeric),
                    "breakdown": col_a,
                    "breakdown_freq_th": "",
                    "measure": col_b,
                    "agg": "correlation",
                    "evidence": {
                        "col_a": col_a,
                        "col_b": col_b,
                        "correlation": round(r, 3),
                        "direction": direction,
                        "n": len(numeric),
                    },
                }
            )
            if len(candidates) >= 10:
                return candidates
    return candidates


def _detect_outlier_insights(
    df: pd.DataFrame,
    measures: dict[str, dict],
    column_types: dict[str, ColumnType],
) -> list[dict]:
    """ตรวจหา outlier ในคอลัมน์ตัวเลข — แถวที่ z-score > 3 — v0.8.

    คืน candidate ที่มี pattern="outlier" สำหรับคอลัมน์ที่มี outlier มากพอ
    """
    candidates: list[dict] = []
    for measure in measures:
        if measure not in df.columns:
            continue
        numeric = pd.to_numeric(df[measure], errors="coerce").dropna()
        if len(numeric) < 20:
            continue
        mean = float(numeric.mean())
        std = float(numeric.std())
        if std == 0:
            continue
        z_scores = (numeric - mean) / std
        outliers = z_scores.abs() >= _OUTLIER_Z_THRESHOLD
        outlier_count = int(outliers.sum())
        if outlier_count == 0:
            continue
        pct = outlier_count / len(numeric) * 100.0
        # มี outlier อย่างน้อย 1% หรือ 5 แถว ถึงจะน่าสนใจ
        if outlier_count < 5 and pct < 1.0:
            continue
        max_z = float(z_scores.abs().max())
        candidates.append(
            {
                "pattern": "outlier",
                "pattern_score": min(1.0, max_z / 6.0),
                "effect_norm": min(1.0, max_z / 6.0),
                "p_value": None,
                "top_segment": measure,
                "n_segments": 1,
                "top_size": outlier_count,
                "breakdown": measure,
                "breakdown_freq_th": "",
                "measure": measure,
                "agg": "outlier",
                "evidence": {
                    "column": measure,
                    "outlier_count": outlier_count,
                    "percentage": round(pct, 1),
                    "max_z_score": round(max_z, 2),
                    "mean": round(mean, 2),
                    "std": round(std, 2),
                },
            }
        )
    return candidates


def _recompute_full(
    df: pd.DataFrame,
    selected: list[dict],
    breakdowns: list[_Breakdown],
    measures: dict[str, dict],
    min_segment: int,
    st,
) -> list[dict]:
    """คำนวณ evidence ใหม่บนข้อมูลเต็มสำหรับ card ที่ถูกเลือก (phase 2 ของ two-phase).

    ใช้ตัวตรวจรูปแบบเดิมแต่บนข้อมูลทั้งหมด เพื่อให้ตัวเลขที่รายงานเป็นค่าจริง (ไม่ใช่ค่าจาก sample)
    ถ้าคำนวณใหม่ไม่ได้ (เช่นรูปแบบหายไปบนข้อมูลเต็ม) จะคงค่าเดิมจาก sample ไว้
    """
    bd_map = {bd.column: bd for bd in breakdowns}
    key_cache: dict[str, pd.Series | None] = {}
    out: list[dict] = []
    for c in selected:
        # v0.8: correlation/outlier ไม่ใช้ breakdown → recompute บนข้อมูลเต็มโดยตรง
        if c["pattern"] == "correlation":
            rec = _detect_strong_correlations(df, measures)
            match = next(
                (
                    r
                    for r in rec
                    if r["evidence"]["col_a"] == c["evidence"]["col_a"]
                    and r["evidence"]["col_b"] == c["evidence"]["col_b"]
                ),
                None,
            )
            if match is not None:
                match["score"] = c["score"]
                out.append(match)
            else:
                out.append(c)
            continue
        if c["pattern"] == "outlier":
            rec = _detect_outlier_insights(df, measures, {})
            match = next(
                (r for r in rec if r["evidence"]["column"] == c["evidence"]["column"]),
                None,
            )
            if match is not None:
                match["score"] = c["score"]
                out.append(match)
            else:
                out.append(c)
            continue

        bd = bd_map.get(c["breakdown"])
        if bd is None:
            out.append(c)
            continue
        if bd.column not in key_cache:
            key_cache[bd.column] = _build_key(df, bd)
        key = key_cache[bd.column]
        if key is None:
            out.append(c)
            continue

        recomputed = _recompute_one(df, key, bd, c, measures, min_segment, st)
        out.append(recomputed if recomputed is not None else c)
    return out


def _recompute_one(
    df: pd.DataFrame,
    key: pd.Series,
    bd: _Breakdown,
    c: dict,
    measures: dict[str, dict],
    min_segment: int,
    st,
) -> dict | None:
    """คำนวณ candidate เดียวใหม่บนข้อมูลเต็ม (คงคะแนน/อันดับเดิม แต่แทนที่ evidence ด้วยค่าจริง)."""
    pattern = c["pattern"]
    measure = c["measure"]
    agg = c["agg"]

    if measure is None:
        series = _agg_count(key)
        sizes = series
        agg_data = None
    else:
        agg_data = _agg_measure(key, df[measure])
        if agg_data is None:
            return None
        sizes = agg_data["sizes"]
        series = agg_data["sum"] if agg == "sum" else agg_data["mean"]

    if pattern == "outstanding":
        det = _detect_outstanding(series, sizes, agg, min_segment)
    elif pattern == "attribution":
        det = _detect_attribution(series, sizes)
    elif pattern == "comparison":
        det = _detect_comparison(agg_data, min_segment, st) if agg_data is not None else None
    else:  # trend
        det = _detect_trend(series, agg) if bd.ordinal else None

    if det is None:
        return None
    # คงคะแนน/อันดับจากเฟสให้คะแนน แต่แทนที่ evidence ด้วยค่าจริงจากข้อมูลเต็ม
    _attach(det, bd, measure, agg, pattern)
    det["score"] = c["score"]
    return det


__all__ = [
    "Perspective",
    "InsightCard",
    "InsightEngineResult",
    "discover_insights",
]
