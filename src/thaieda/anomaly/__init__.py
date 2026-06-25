"""Anomaly detection — ตรวจจับความผิดปกติในข้อมูล (สถิติ/ข้อความ/การเข้ารหัส/หมวดหมู่).

ต่อยอดจาก quality/ (ซึ่งตรวจ "ปัญหาที่รู้จัก") โดย anomaly/ มองหา "ค่าผิดปกติเชิงสถิติ"
เช่น outlier ตัวเลข, ข้อความสั้น/ยาวผิดปกติ, mojibake, หมวดหมู่ที่คล้ายกันจนน่าสงสัยว่าพิมพ์ผิด
ทุกฟังก์ชันคืน AnomalyIssue ที่มีคำอธิบายสองภาษา (ไทย/อังกฤษ) เหมือน QualityIssue
"""

from __future__ import annotations

import difflib
import re
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thaieda.detect import ColumnType, detect_all, detect_column_type

# ----------------------------------------------------------------------------
# ค่าคงที่
# ----------------------------------------------------------------------------
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# จำนวนค่าขั้นต่ำที่ทำให้สถิติมีความหมาย (กันการ flag มั่ว ๆ บนข้อมูลเล็ก)
_MIN_STAT_SAMPLE = 8
# จำนวนค่าขั้นต่ำสำหรับตรวจ outlier ตัวเลข
_MIN_NUMERIC_SAMPLE = 5
# จำนวนแถวขั้นต่ำสำหรับวิธี ML (Isolation Forest / LOF) — น้อยกว่านี้ผลไม่น่าเชื่อถือ
_MIN_ML_SAMPLE = 100
# จำนวนแถวสูงสุดที่ส่งเข้า IF/LOF — เกินนี้สุ่มตัวอย่างลงมา (กัน LOF ช้ามากบนข้อมูลหลักล้านแถว)
# วิธีเชิงสถิติ (z-score/MAD/IQR) ยังรันบนข้อมูลเต็มเสมอ เพราะเป็นเวกเตอร์และเร็วอยู่แล้ว
_MAX_ML_SAMPLE = 10000
# สัดส่วนค่าซ้ำสูงสุดที่ยอมให้ LOF ทำงาน — เกินนี้ผล LOF ไม่น่าเชื่อถือ (ระยะเพื่อนบ้าน = 0)
_MAX_LOF_DUP_RATIO = 0.5
# สัดส่วน outlier สูงสุดที่ยอมรับจากวิธี ML — เกินนี้ถือว่า "ไม่ใช่ค่าหายากแล้ว"
# (contamination='auto' อาจ flag จำนวนมากบนการกระจายแบบ uniform/discrete ซึ่งไม่มี outlier จริง)
# ความผิดปกติควรเป็นของหายาก จึงตัดผลที่กว้างเกินไปทิ้งเพื่อลด noise ในรายงาน
_MAX_ML_OUTLIER_FRAC = 0.20
# จำนวนแถวขั้นต่ำสำหรับกฎ "หมวดหมู่หายาก <1%"
_RARE_MIN_TOTAL = 100
# จำนวน index ตัวอย่างสูงสุดที่เก็บต่อหนึ่ง issue
_MAX_INDICES = 20

# วรรณยุกต์ไทย (ไม้เอก โท ตรี จัตวา)
_THAI_TONE_MARKS = "่้๊๋"
# combining marks ของไทย (สระบน/ล่าง + วรรณยุกต์ + ทัณฑฆาต ฯลฯ)
_THAI_COMBINING = set("ัิีึืุู็่้๊๋์ํฺ")
_THAI_CONSONANTS = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
_THAI_LETTER_RANGE = (0x0E01, 0x0E2E)  # พยัญชนะ ก–ฮ (ไม่รวมเลขไทย/เครื่องหมาย)

# ลายเซ็น mojibake ที่พบบ่อยในข้อมูลไทย:
#   - "à¸"/"à¹" : ไบต์ UTF-8 ของอักษรไทย (E0 B8.. / E0 B9..) ถูกถอดเป็น Latin-1
#   - "Ã."     : อักษรละตินมีเครื่องหมายที่ถูกถอดผิด
#   - "â€"      : เครื่องหมายคำพูด/ขีดที่เพี้ยน (smart punctuation)
#   - "Â."     : NBSP/สัญลักษณ์ที่เพี้ยน
_MOJIBAKE_RE = re.compile("(?:à[¸¹]|Ã[\x80-\xbf]|â€[\x80-\xbf]?|Â[\xa0-\xbf])")

# ลายเซ็น mojibake เฉพาะภาษาไทย (ละเอียดกว่า _MOJIBAKE_RE ทั่วไป):
#   - latin-1/cp1252: อักษรไทย UTF-8 (ไบต์ E0 B8.. / E0 B9..) -> "à¸"/"à¹"
_THAI_MOJIBAKE_LATIN1_RE = re.compile(r"à[¸¹]")
#   - cp874/tis-620: อักษรไทย UTF-8 ทุกตัวกลายเป็น "เ"+อักขระ -> ขึ้นต้นด้วย "เธ"/"เน" ถี่ผิดปกติ
#     (เช่น "สวัสดี" -> "เธชเธงเธฑเธชเธ”เธต") ระวัง false positive จากคำไทยจริงอย่าง "เธอ"/"เนื้อ"
_THAI_MOJIBAKE_CP874_RE = re.compile(r"เธ|เน")

# อักขระเดียวกันซ้ำ 5+ ครั้ง (ผิดปกติเชิงสถิติ — ต่างจาก quality ที่ใช้ 3+)
_EXCESSIVE_REPEAT_RE = re.compile(r"(.)\1{4,}")
# วรรณยุกต์เดียวกันซ้อนติดกัน (เช่น "่่")
_TONE_STACK_RE = re.compile(f"([{_THAI_TONE_MARKS}])\\1+")


# ----------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ----------------------------------------------------------------------------
@dataclass
class AnomalyIssue:
    """ความผิดปกติหนึ่งรายการ (มีคำอธิบายทั้งไทยและอังกฤษ)."""

    check_name: str
    severity: str  # "critical" | "warning" | "info"
    column: str
    anomaly_type: str  # "statistical" | "text" | "encoding" | "pattern" | "categorical"
    count: int
    percentage: float
    description: str
    description_th: str
    examples: list[str] = field(default_factory=list)
    suggestion: str = ""
    suggestion_th: str = ""
    indices: list[int] = field(default_factory=list)  # ตำแหน่งแถว (0-based) สูงสุด 20

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "severity": self.severity,
            "column": self.column,
            "anomaly_type": self.anomaly_type,
            "count": self.count,
            "percentage": round(self.percentage, 2),
            "description": self.description,
            "description_th": self.description_th,
            "examples": self.examples,
            "suggestion": self.suggestion,
            "suggestion_th": self.suggestion_th,
            "indices": self.indices,
        }


# ----------------------------------------------------------------------------
# helper ทั่วไป
# ----------------------------------------------------------------------------
def _col_name(series: pd.Series) -> str:
    return str(series.name) if series.name is not None else ""


def _pct(count: int, total: int) -> float:
    return (count / total * 100.0) if total else 0.0


def _trunc(text: str, limit: int = 60) -> str:
    """ตัดข้อความยาวให้สั้นลงเพื่อใช้เป็นตัวอย่าง."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


# แคชแบ็กเอนด์การวัดความคล้ายสตริง: rapidfuzz (เร็วกว่า) ถ้ามี ไม่งั้น difflib
# 0 = ยังไม่ตรวจ, 1 = ใช้ rapidfuzz, 2 = ใช้ difflib
_SIMILARITY_BACKEND = 0


def _string_similarity(a: str, b: str) -> float:
    """คืนค่าความคล้ายของสองสตริงในช่วง 0–1 — ใช้ rapidfuzz (ติดตั้งผ่าน thaieda[fuzzy]) ถ้ามี.

    ถ้าไม่มี rapidfuzz จะถอยไปใช้ difflib.SequenceMatcher (พฤติกรรมเดิม)
    rapidfuzz.fuzz.ratio คืน 0–100 จึงหารด้วย 100 ให้เทียบเท่ากับ SequenceMatcher (0–1)
    """
    global _SIMILARITY_BACKEND
    if _SIMILARITY_BACKEND == 0:
        try:
            from rapidfuzz import fuzz  # noqa: F401 — lazy probe ว่ามีไหม

            _SIMILARITY_BACKEND = 1
        except ImportError:
            _SIMILARITY_BACKEND = 2

    if _SIMILARITY_BACKEND == 1:
        from rapidfuzz import fuzz

        return fuzz.ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _positional_items(series: pd.Series) -> list[tuple[int, str]]:
    """คืน (ตำแหน่งแถว 0-based, ค่าเป็น str) เฉพาะเซลล์ที่ไม่ว่าง."""
    items: list[tuple[int, str]] = []
    for pos, value in enumerate(series.tolist()):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if pd.isna(value):
            continue
        items.append((pos, str(value)))
    return items


# ----------------------------------------------------------------------------
# (a) Numeric outliers (statistical)
# ----------------------------------------------------------------------------
def _zscore_mask(values: np.ndarray) -> np.ndarray | None:
    """z-score: |z| > 3 (เหมาะกับการกระจายแบบปกติ). คืน None ถ้า std เป็น 0."""
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    if not std or np.isnan(std):
        return None
    z = (values - values.mean()) / std
    return np.abs(z) > 3.0


def _mad_mask(values: np.ndarray) -> np.ndarray | None:
    """Modified z-score: |modified_z| > 3.5 (ทนทานต่อ outlier).

    ถ้า MAD เป็น 0 (ค่าส่วนใหญ่เท่ากัน) ใช้ค่าเฉลี่ยส่วนเบี่ยงเบนสัมบูรณ์ (MeanAD) แทน
    ตามสูตร Iglewicz–Hoaglin. คืน None ก็ต่อเมื่อทั้ง MAD และ MeanAD เป็น 0
    """
    median = float(np.median(values))
    abs_dev = np.abs(values - median)
    mad = float(np.median(abs_dev))
    if mad:
        modified_z = 0.6745 * (values - median) / mad
        return np.abs(modified_z) > 3.5
    mean_ad = float(abs_dev.mean())
    if not mean_ad:
        return None
    modified_z = (values - median) / (1.253314 * mean_ad)
    return np.abs(modified_z) > 3.5


def _iqr_mask(values: np.ndarray) -> np.ndarray | None:
    """IQR: ค่า < Q1-1.5*IQR หรือ > Q3+1.5*IQR. None ถ้า IQR เป็น 0."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    if not iqr:
        return None
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (values < lower) | (values > upper)


def _choose_outlier_method(values: np.ndarray, skew: float) -> tuple[str, np.ndarray | None]:
    """เลือกวิธีตามการกระจาย: เบ้มาก -> MAD (robust), ใกล้ปกติ -> z-score.

    ลองวิธีถัดไปเป็น fallback หากวิธีแรกใช้ไม่ได้ (เช่น std/MAD/IQR เป็น 0)
    """
    if skew > 1.0:
        order = (
            ("modified_z_score (MAD)", _mad_mask),
            ("IQR", _iqr_mask),
            ("z_score", _zscore_mask),
        )
    else:
        order = (
            ("z_score", _zscore_mask),
            ("IQR", _iqr_mask),
            ("modified_z_score (MAD)", _mad_mask),
        )
    for name, fn in order:
        mask = fn(values)
        if mask is not None:
            return name, mask
    return "none", None


def detect_numeric_outliers(series: pd.Series) -> AnomalyIssue | None:
    """ตรวจหา outlier ในคอลัมน์ตัวเลข — เลือกวิธี (z-score/MAD/IQR) ตามการกระจายข้อมูล."""
    col = _col_name(series)
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    valid_mask = ~np.isnan(numeric)
    valid = numeric[valid_mask]
    n = int(valid.size)
    if n < _MIN_NUMERIC_SAMPLE:
        return None

    try:
        skew = abs(float(pd.Series(valid).skew()))
    except (ValueError, TypeError):
        skew = 0.0
    if np.isnan(skew):
        skew = 0.0

    method, out_mask = _choose_outlier_method(valid, skew)
    if out_mask is None or not bool(out_mask.any()):
        return None

    valid_positions = np.flatnonzero(valid_mask)
    outlier_positions = valid_positions[out_mask]
    outlier_values = valid[out_mask]
    count = int(outlier_positions.size)

    examples = [_fmt_number(float(v)) for v in outlier_values[:_MAX_INDICES]]
    indices = [int(p) for p in outlier_positions[:_MAX_INDICES]]

    return AnomalyIssue(
        check_name="numeric_outliers",
        severity="warning",
        column=col,
        anomaly_type="statistical",
        count=count,
        percentage=_pct(count, n),
        description=(
            f"{count} numeric outlier(s) detected using the {method} method "
            f"(distribution skew ≈ {skew:.2f})."
        ),
        description_th=(
            f"พบค่าผิดปกติเชิงตัวเลข {count} ค่า ด้วยวิธี {method} (ความเบ้ของการกระจาย ≈ {skew:.2f})"
        ),
        examples=examples,
        suggestion=(
            "Inspect these values; they may be data-entry errors, "
            "units mismatch, or genuine extremes."
        ),
        suggestion_th="ตรวจสอบค่าเหล่านี้ — อาจเป็นการกรอกผิด หน่วยไม่ตรงกัน หรือค่าสุดขั้วจริง",
        indices=indices,
    )


def _fmt_number(value: float) -> str:
    """แสดงตัวเลขแบบกระชับ (ตัด .0 ของจำนวนเต็มออก)."""
    if value == int(value):
        return str(int(value))
    return str(round(value, 4))


# ----------------------------------------------------------------------------
# (a2) ML-based numeric outliers (optional — ต้องติดตั้ง thaieda[ml] = scikit-learn)
# ----------------------------------------------------------------------------
def sklearn_available() -> bool:
    """คืน True ถ้าติดตั้ง scikit-learn (สำหรับวิธี ML) — ใช้ตัดสินใจระดับบนว่าจะรันวิธี ML ไหม."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def _numeric_valid(series: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """คืน (ค่าตัวเลขที่ไม่ใช่ NaN, ตำแหน่งแถวของค่าเหล่านั้น, อาร์เรย์เต็มหลัง coerce)."""
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    valid_mask = ~np.isnan(numeric)
    return numeric[valid_mask], np.flatnonzero(valid_mask), numeric


def _subsample_for_ml(
    valid: np.ndarray, valid_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """สุ่มตัวอย่างลง _MAX_ML_SAMPLE แถวถ้าข้อมูลใหญ่เกิน — คืน (ค่า, ตำแหน่ง, สุ่มหรือไม่).

    ใช้ seed คงที่ (42) เพื่อให้ผลทำซ้ำได้ และเรียงตำแหน่งที่สุ่มไว้ตามเดิมเพื่อรักษาลำดับแถว
    IF/LOF บนข้อมูลหลักล้านแถวช้ามาก (LOF เป็น O(n²) เชิงระยะ) การสุ่มทำให้เร็วขึ้นมาก
    โดยยังสะท้อนการกระจายของข้อมูลได้
    """
    n = int(valid.size)
    if n <= _MAX_ML_SAMPLE:
        return valid, valid_positions, False
    rng = np.random.default_rng(42)
    sel = rng.choice(n, size=_MAX_ML_SAMPLE, replace=False)
    sel.sort()  # รักษาลำดับตำแหน่งแถวเดิม
    return valid[sel], valid_positions[sel], True


def detect_isolation_forest(series: pd.Series) -> AnomalyIssue | None:
    """ตรวจ outlier ด้วย Isolation Forest (วิธี ML) — เหมาะกับคอลัมน์ตัวเลขที่มี >100 แถว.

    ใช้ contamination='auto', random_state=42 เพื่อให้ผลทำซ้ำได้
    คืน None อย่างสุภาพถ้าไม่ได้ติดตั้ง scikit-learn (thaieda[ml]) หรือข้อมูลน้อยเกินไป
    คะแนน decision_function (ยิ่งต่ำยิ่งผิดปกติ) ถูกแนบไว้ในคำอธิบายและตัวอย่าง
    """
    col = _col_name(series)
    valid, valid_positions, _ = _numeric_valid(series)
    if valid.size <= _MIN_ML_SAMPLE:
        return None
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return None

    # ข้อมูลใหญ่เกิน -> สุ่มตัวอย่างลง _MAX_ML_SAMPLE แถว (เร็วขึ้นมาก ผลยังเชื่อถือได้)
    valid, valid_positions, sampled = _subsample_for_ml(valid, valid_positions)
    sample_size = int(valid.size)

    x = valid.reshape(-1, 1)
    model = IsolationForest(contamination="auto", random_state=42)
    preds = model.fit_predict(x)  # -1 = outlier, 1 = ปกติ
    scores = model.decision_function(x)  # ยิ่งต่ำ (ติดลบ) ยิ่งผิดปกติ
    out_mask = preds == -1
    count = int(out_mask.sum())
    # 0 = ไม่พบ, มากเกินไป = การกระจายไม่มี outlier จริง (ผลไม่น่าเชื่อถือ) -> ข้าม
    if count == 0 or count / sample_size > _MAX_ML_OUTLIER_FRAC:
        return None

    order = np.argsort(scores[out_mask])  # ผิดปกติสุดก่อน
    out_positions = valid_positions[out_mask][order]
    out_values = valid[out_mask][order]
    out_scores = scores[out_mask][order]
    min_score = float(out_scores[0])

    examples = [
        f"{_fmt_number(float(v))} (score={s:.3f})"
        for v, s in zip(out_values[:_MAX_INDICES], out_scores[:_MAX_INDICES], strict=True)
    ]
    sample_en = f" on a {sample_size:,}-row sample" if sampled else ""
    sample_th = f" (สุ่มตัวอย่าง {sample_size:,} แถว)" if sampled else ""
    return AnomalyIssue(
        check_name="isolation_forest",
        severity="warning",
        column=col,
        anomaly_type="statistical",
        count=count,
        percentage=_pct(count, sample_size),
        description=(
            f"Isolation Forest flagged {count} outlier(s){sample_en} (most anomalous score "
            f"{min_score:.3f}; lower = more anomalous)."
        ),
        description_th=(
            f"Isolation Forest พบค่าผิดปกติ {count} ค่า{sample_th} "
            f"(คะแนนผิดปกติสุด {min_score:.3f}; ยิ่งต่ำยิ่งผิดปกติ)"
        ),
        examples=examples,
        suggestion=(
            "ML-based outliers complement statistical methods; cross-check flagged points."
        ),
        suggestion_th="ค่าผิดปกติแบบ ML ใช้เสริมวิธีเชิงสถิติ — ควรตรวจสอบจุดที่ถูก flag ประกอบกัน",
        indices=[int(p) for p in out_positions[:_MAX_INDICES]],
    )


def detect_lof(series: pd.Series) -> AnomalyIssue | None:
    """ตรวจ outlier ด้วย Local Outlier Factor (LOF) — เปรียบเทียบความหนาแน่นกับเพื่อนบ้าน.

    n_neighbors=20 (ปรับลงอัตโนมัติถ้าตัวอย่างน้อยกว่า) เหมาะกับคอลัมน์ตัวเลข >100 แถว
    คืน None อย่างสุภาพถ้าไม่ได้ติดตั้ง scikit-learn (thaieda[ml]) หรือข้อมูลน้อยเกินไป
    """
    col = _col_name(series)
    valid, valid_positions, _ = _numeric_valid(series)
    if valid.size <= _MIN_ML_SAMPLE:
        return None
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        return None

    # ข้อมูลใหญ่เกิน -> สุ่มตัวอย่างลง _MAX_ML_SAMPLE แถว (LOF ช้ามากบนข้อมูลหลักล้านแถว)
    valid, valid_positions, sampled = _subsample_for_ml(valid, valid_positions)
    sample_size = int(valid.size)

    # ค่าซ้ำมาก -> LOF ไม่น่าเชื่อถือ (ระยะถึงเพื่อนบ้าน = 0) และ sklearn จะเตือน -> ข้าม
    n_unique = int(np.unique(valid).size)
    dup_ratio = 1.0 - n_unique / sample_size
    if dup_ratio > _MAX_LOF_DUP_RATIO:
        return None
    # เพิ่ม n_neighbors เมื่อมีค่าซ้ำพอควร เพื่อลด warning เรื่อง duplicate distances
    n_dup = sample_size - n_unique
    n_neighbors = min(max(20, n_dup * 2), sample_size - 1)

    x = valid.reshape(-1, 1)
    model = LocalOutlierFactor(n_neighbors=n_neighbors)
    # กัน warning "Duplicate values..." ที่จัดการแล้วด้านบน ไม่ให้รบกวน output ของผู้ใช้
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        preds = model.fit_predict(x)  # -1 = outlier
    scores = model.negative_outlier_factor_  # ยิ่งต่ำ (ติดลบมาก) ยิ่งผิดปกติ
    out_mask = preds == -1
    count = int(out_mask.sum())
    # 0 = ไม่พบ, มากเกินไป = การกระจายไม่มี outlier จริง (ผลไม่น่าเชื่อถือ) -> ข้าม
    if count == 0 or count / sample_size > _MAX_ML_OUTLIER_FRAC:
        return None

    order = np.argsort(scores[out_mask])  # ผิดปกติสุดก่อน
    out_positions = valid_positions[out_mask][order]
    out_values = valid[out_mask][order]
    out_scores = scores[out_mask][order]
    min_score = float(out_scores[0])

    examples = [
        f"{_fmt_number(float(v))} (LOF={s:.3f})"
        for v, s in zip(out_values[:_MAX_INDICES], out_scores[:_MAX_INDICES], strict=True)
    ]
    sample_en = f" on a {sample_size:,}-row sample" if sampled else ""
    sample_th = f" (สุ่มตัวอย่าง {sample_size:,} แถว)" if sampled else ""
    return AnomalyIssue(
        check_name="local_outlier_factor",
        severity="warning",
        column=col,
        anomaly_type="statistical",
        count=count,
        percentage=_pct(count, sample_size),
        description=(
            f"Local Outlier Factor flagged {count} outlier(s){sample_en} (most anomalous factor "
            f"{min_score:.3f}; more negative = more anomalous)."
        ),
        description_th=(
            f"Local Outlier Factor พบค่าผิดปกติ {count} ค่า{sample_th} (ค่าผิดปกติสุด {min_score:.3f}; "
            "ยิ่งติดลบมากยิ่งผิดปกติ)"
        ),
        examples=examples,
        suggestion=(
            "LOF finds density-based local outliers; useful when global statistics miss them."
        ),
        suggestion_th="LOF จับค่าผิดปกติเชิงความหนาแน่นเฉพาะถิ่น — มีประโยชน์เมื่อสถิติรวมมองไม่เห็น",
        indices=[int(p) for p in out_positions[:_MAX_INDICES]],
    )


# ----------------------------------------------------------------------------
# (b) Text anomalies
# ----------------------------------------------------------------------------
def _length_anomalies(items: list[tuple[int, str]], col: str) -> AnomalyIssue | None:
    """ความยาวข้อความที่เบี่ยงเบนจากค่าเฉลี่ย >3 std (สั้นหรือยาวผิดปกติ)."""
    if len(items) < _MIN_STAT_SAMPLE:
        return None
    lengths = np.array([len(s) for _, s in items], dtype="float64")
    std = float(lengths.std(ddof=1))
    if not std:
        return None
    mean = float(lengths.mean())
    mask = np.abs(lengths - mean) > 3.0 * std
    if not bool(mask.any()):
        return None

    flagged = [items[i] for i in np.flatnonzero(mask)]
    count = len(flagged)
    examples = [f"len={len(s)}: {_trunc(s)}" for _, s in flagged[:_MAX_INDICES]]
    indices = [pos for pos, _ in flagged[:_MAX_INDICES]]
    return AnomalyIssue(
        check_name="text_length_anomaly",
        severity="info",
        column=col,
        anomaly_type="text",
        count=count,
        percentage=_pct(count, len(items)),
        description=(f"{count} text(s) with abnormal length (>3σ from mean length {mean:.1f})."),
        description_th=(f"ข้อความ {count} รายการมีความยาวผิดปกติ (เกิน 3σ จากความยาวเฉลี่ย {mean:.1f})"),
        examples=examples,
        suggestion="Check for truncated, empty, or concatenated records.",
        suggestion_th="ตรวจหาข้อความที่ถูกตัด ว่างเปล่า หรือถูกต่อกันหลายรายการ",
        indices=indices,
    )


def _control_or_replacement_count(text: str) -> int:
    """นับอักขระควบคุม (Cc ยกเว้น \\t\\n\\r) และ replacement char (U+FFFD)."""
    count = 0
    for ch in text:
        if ch == "�" or ch not in "\t\n\r" and unicodedata.category(ch) == "Cc":
            count += 1
    return count


def _mojibake_anomalies(items: list[tuple[int, str]], col: str) -> AnomalyIssue | None:
    """ตรวจ mojibake (UTF-8 ถูกถอดเป็น tis-620/Latin-1) จากลายเซ็น 'à¸', 'Ã', 'â€'."""
    flagged = [(pos, s) for pos, s in items if _MOJIBAKE_RE.search(s)]
    if not flagged:
        return None
    count = len(flagged)
    examples = [_trunc(s) for _, s in flagged[:_MAX_INDICES]]
    indices = [pos for pos, _ in flagged[:_MAX_INDICES]]
    return AnomalyIssue(
        check_name="encoding_mojibake",
        severity="critical",
        column=col,
        anomaly_type="encoding",
        count=count,
        percentage=_pct(count, len(items)),
        description=(
            f"{count} cell(s) look like mojibake (UTF-8 text decoded with the wrong "
            "encoding, e.g. 'à¸ªà¸§à¸±à¸ªà¸”à¸µ')."
        ),
        description_th=(
            f"{count} เซลล์มีลักษณะเป็น mojibake "
            "(ข้อความ UTF-8 ถูกถอดด้วย encoding ผิด เช่น 'à¸ªà¸§à¸±à¸ªà¸”à¸µ')"
        ),
        examples=examples,
        suggestion="Re-decode with the correct encoding or run clean.normalize_encoding (ftfy).",
        suggestion_th="ถอดรหัสใหม่ด้วย encoding ที่ถูกต้อง หรือใช้ clean.normalize_encoding (ftfy)",
        indices=indices,
    )


def _looks_thai_mojibake(text: str) -> str | None:
    """คืนชื่อ encoding ต้นทางที่น่าจะทำให้เกิด mojibake ('latin-1'/'cp874') หรือ None ถ้าไม่ใช่.

    latin-1/cp1252 : พบลายเซ็น "à¸"/"à¹" (อักษรไทย UTF-8 ถูกถอดเป็น latin-1) — เชื่อถือได้สูง
    cp874/tis-620  : พบ "เธ"/"เน" ถี่ผิดปกติ (อักษรไทยทุกตัวกลายเป็น "เ"+x)
                     ต้องเข้มงวด (>=3 ครั้ง และคิดเป็นสัดส่วน >=40%) เพื่อกัน "เธอ"/"เนื้อ" จริง
    """
    if _THAI_MOJIBAKE_LATIN1_RE.search(text):
        return "latin-1"
    hits = len(_THAI_MOJIBAKE_CP874_RE.findall(text))
    if hits >= 3 and (hits * 2) / max(len(text), 1) >= 0.4:
        return "cp874"
    return None


def _thai_mojibake_anomalies(items: list[tuple[int, str]], col: str) -> AnomalyIssue | None:
    """ตรวจ mojibake เฉพาะภาษาไทย — ระบุ encoding ต้นทาง (latin-1/cp874) ที่น่าจะเป็นสาเหตุ."""
    flagged: list[tuple[int, str]] = []
    sources: set[str] = set()
    for pos, s in items:
        src = _looks_thai_mojibake(s)
        if src is not None:
            flagged.append((pos, s))
            sources.add(src)
    if not flagged:
        return None
    count = len(flagged)
    src_label = "/".join(sorted(sources))
    return AnomalyIssue(
        check_name="thai_mojibake",
        severity="critical",
        column=col,
        anomaly_type="encoding",
        count=count,
        percentage=_pct(count, len(items)),
        description=(
            f"{count} cell(s) contain Thai mojibake — UTF-8 Thai text mis-decoded as "
            f"{src_label} (e.g. 'à¸ªà¸§à¸±à¸ªà¸”à¸µ' or 'เธชเธงเธฑเธชเธ”เธต')."
        ),
        description_th=(
            f"{count} เซลล์มี mojibake ภาษาไทย — ข้อความไทย UTF-8 ถูกถอดผิดเป็น {src_label} "
            "(เช่น 'à¸ªà¸§à¸±à¸ªà¸”à¸µ' หรือ 'เธชเธงเธฑเธชเธ”เธต')"
        ),
        examples=[_trunc(s) for _, s in flagged[:_MAX_INDICES]],
        suggestion=(
            "Repair with ftfy (clean.normalize_encoding) or re-decode from the source encoding."
        ),
        suggestion_th="ซ่อมด้วย ftfy (clean.normalize_encoding) หรือถอดรหัสใหม่จาก encoding ต้นทาง",
        indices=[pos for pos, _ in flagged[:_MAX_INDICES]],
    )


def detect_thai_mojibake(series: pd.Series) -> AnomalyIssue | None:
    """ตรวจ mojibake ภาษาไทยในคอลัมน์เดียว — คืน AnomalyIssue หรือ None ถ้าไม่พบ.

    รองรับทั้ง mojibake แบบ latin-1/cp1252 ('à¸ª...') และ cp874/tis-620 ('เธช...')
    """
    items = _positional_items(series)
    if not items:
        return None
    return _thai_mojibake_anomalies(items, _col_name(series))


def _garbled_anomalies(items: list[tuple[int, str]], col: str) -> AnomalyIssue | None:
    """ตรวจข้อความเสียหาย — มี replacement char (U+FFFD) หรืออักขระควบคุม."""
    flagged = [(pos, s) for pos, s in items if _control_or_replacement_count(s) > 0]
    if not flagged:
        return None
    count = len(flagged)
    examples = [repr(_trunc(s)) for _, s in flagged[:_MAX_INDICES]]
    indices = [pos for pos, _ in flagged[:_MAX_INDICES]]
    return AnomalyIssue(
        check_name="garbled_text",
        severity="critical",
        column=col,
        anomaly_type="encoding",
        count=count,
        percentage=_pct(count, len(items)),
        description=(
            f"{count} cell(s) contain replacement (U+FFFD) or control characters — "
            "a sign of corrupted/lossy decoding."
        ),
        description_th=(f"{count} เซลล์มีอักขระแทนที่ (U+FFFD) หรืออักขระควบคุม ซึ่งบ่งชี้การถอดรหัสที่เสียหาย"),
        examples=examples,
        suggestion="Trace the source encoding; data may have been irreversibly corrupted.",
        suggestion_th="ตรวจสอบ encoding ต้นทาง ข้อมูลอาจเสียหายแบบกู้คืนไม่ได้",
        indices=indices,
    )


def _repetition_anomalies(items: list[tuple[int, str]], col: str) -> AnomalyIssue | None:
    """ตรวจการซ้ำอักขระมากเกินไป (อักขระเดียวกัน 5+ ตัวติดกัน) — outlier ของการซ้ำ."""
    flagged = [(pos, s) for pos, s in items if _EXCESSIVE_REPEAT_RE.search(s)]
    if not flagged:
        return None
    count = len(flagged)
    examples = [_trunc(s) for _, s in flagged[:_MAX_INDICES]]
    indices = [pos for pos, _ in flagged[:_MAX_INDICES]]
    return AnomalyIssue(
        check_name="excessive_repetition",
        severity="info",
        column=col,
        anomaly_type="text",
        count=count,
        percentage=_pct(count, len(items)),
        description=(
            f"{count} cell(s) contain a character repeated 5+ times in a row "
            "(e.g. '55555', 'ครับบบบบ')."
        ),
        description_th=(f"{count} เซลล์มีอักขระเดียวกันซ้ำกัน 5 ตัวขึ้นไปติดกัน (เช่น '55555', 'ครับบบบบ')"),
        examples=examples,
        suggestion="Collapse runaway repetition with clean.fix_repeated_chars.",
        suggestion_th="ลดการซ้ำที่มากเกินด้วย clean.fix_repeated_chars",
        indices=indices,
    )


def detect_text_anomalies(series: pd.Series, tokenizer) -> list[AnomalyIssue]:
    """ตรวจความผิดปกติเชิงข้อความ: ความยาว, mojibake, ข้อความเสียหาย, การซ้ำมากเกิน.

    รับ tokenizer ไว้เพื่อความสม่ำเสมอของ API (ปัจจุบันการตรวจเหล่านี้ไม่ต้องตัดคำ)
    """
    col = _col_name(series)
    items = _positional_items(series)
    if not items:
        return []

    issues: list[AnomalyIssue] = []
    for check in (
        _length_anomalies,
        _mojibake_anomalies,
        _thai_mojibake_anomalies,
        _garbled_anomalies,
        _repetition_anomalies,
    ):
        issue = check(items, col)
        if issue is not None:
            issues.append(issue)
    return issues


# ----------------------------------------------------------------------------
# (c) Thai-specific text anomalies
# ----------------------------------------------------------------------------
def _has_orphan_combining(text: str) -> bool:
    """True ถ้ามี combining mark ของไทยที่ขึ้นต้น หรือตามหลังอักขระที่ไม่ใช่ฐาน."""
    prev_is_base = False
    for ch in text:
        if ch in _THAI_COMBINING and not prev_is_base:
            return True
        prev_is_base = (ch in _THAI_CONSONANTS) or (ch in _THAI_COMBINING)
    return False


def _thai_latin_letter_counts(text: str) -> tuple[int, int]:
    """นับ (อักษรไทย, อักษรละติน) ในข้อความ — ไม่รวมเลข/ช่องว่าง/เครื่องหมาย."""
    thai = latin = 0
    for ch in text:
        cp = ord(ch)
        if _THAI_LETTER_RANGE[0] <= cp <= _THAI_LETTER_RANGE[1]:
            thai += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            latin += 1
    return thai, latin


def _orphan_combining_anomalies(items: list[tuple[int, str]], col: str) -> AnomalyIssue | None:
    flagged = [(pos, s) for pos, s in items if _has_orphan_combining(s)]
    if not flagged:
        return None
    count = len(flagged)
    return AnomalyIssue(
        check_name="invalid_thai_sequence",
        severity="warning",
        column=col,
        anomaly_type="text",
        count=count,
        percentage=_pct(count, len(items)),
        description=(
            f"{count} cell(s) have Thai tone marks/vowels without a preceding base "
            "consonant (invalid character sequence)."
        ),
        description_th=(f"{count} เซลล์มีวรรณยุกต์/สระไทยที่ไม่มีพยัญชนะฐานนำหน้า (ลำดับอักขระไม่ถูกต้อง)"),
        examples=[repr(_trunc(s)) for _, s in flagged[:_MAX_INDICES]],
        suggestion=(
            "Validate text input; orphan diacritics often indicate corrupted or sliced text."
        ),
        suggestion_th="ตรวจสอบการกรอกข้อความ — สระ/วรรณยุกต์ลอยมักเกิดจากข้อความที่เสียหายหรือถูกตัด",
        indices=[pos for pos, _ in flagged[:_MAX_INDICES]],
    )


def _tone_stacking_anomalies(items: list[tuple[int, str]], col: str) -> AnomalyIssue | None:
    flagged = [(pos, s) for pos, s in items if _TONE_STACK_RE.search(s)]
    if not flagged:
        return None
    count = len(flagged)
    return AnomalyIssue(
        check_name="tone_mark_stacking",
        severity="warning",
        column=col,
        anomaly_type="text",
        count=count,
        percentage=_pct(count, len(items)),
        description=(
            f"{count} cell(s) have consecutive identical tone marks (e.g. '่่') — "
            "likely an input error."
        ),
        description_th=(f"{count} เซลล์มีวรรณยุกต์เดียวกันซ้อนติดกัน (เช่น '่่') ซึ่งน่าจะเป็นการพิมพ์ผิด"),
        examples=[repr(_trunc(s)) for _, s in flagged[:_MAX_INDICES]],
        suggestion="Remove duplicate tone marks with clean.fix_tone_mark_stacking.",
        suggestion_th="ลบวรรณยุกต์ซ้ำด้วย clean.fix_tone_mark_stacking",
        indices=[pos for pos, _ in flagged[:_MAX_INDICES]],
    )


def _diacritic_order_anomalies(items: list[tuple[int, str]], col: str) -> AnomalyIssue | None:
    flagged: list[tuple[int, str]] = []
    for pos, s in items:
        if unicodedata.normalize("NFC", s) != s and any(c in _THAI_COMBINING for c in s):
            flagged.append((pos, s))
    if not flagged:
        return None
    count = len(flagged)
    return AnomalyIssue(
        check_name="diacritic_order",
        severity="info",
        column=col,
        anomaly_type="text",
        count=count,
        percentage=_pct(count, len(items)),
        description=(
            f"{count} cell(s) have Thai combining marks in non-canonical order "
            "(NFC normalization would change them)."
        ),
        description_th=(
            f"{count} เซลล์มีลำดับ combining mark ไทยไม่เป็นมาตรฐาน (การ normalize แบบ NFC จะเปลี่ยนค่า)"
        ),
        examples=[repr(_trunc(s)) for _, s in flagged[:_MAX_INDICES]],
        suggestion="Apply Unicode NFC normalization with clean.normalize_unicode.",
        suggestion_th="ใช้การ normalize แบบ NFC ด้วย clean.normalize_unicode",
        indices=[pos for pos, _ in flagged[:_MAX_INDICES]],
    )


def _script_mixing_anomalies(items: list[tuple[int, str]], col: str) -> AnomalyIssue | None:
    """เซลล์ที่สัดส่วนไทย/ละตินผิดปกติเทียบกับค่าเฉลี่ยคอลัมน์ (เช่น คอลัมน์ไทยแต่เซลล์เป็นละตินล้วน)."""
    ratios: list[tuple[int, str, float]] = []  # (pos, text, thai_ratio)
    for pos, s in items:
        thai, latin = _thai_latin_letter_counts(s)
        if thai + latin >= 3:
            ratios.append((pos, s, thai / (thai + latin)))
    if len(ratios) < _MIN_STAT_SAMPLE:
        return None

    col_mean = float(np.mean([r for _, _, r in ratios]))
    # ตรวจเฉพาะคอลัมน์ที่ "ส่วนใหญ่เป็นไทย" แล้วมีเซลล์ที่เป็นละตินเด่นผิดปกติ
    if col_mean < 0.6:
        return None
    flagged = [(pos, s) for pos, s, r in ratios if r <= 0.2]
    if not flagged:
        return None
    count = len(flagged)
    return AnomalyIssue(
        check_name="abnormal_script_mixing",
        severity="info",
        column=col,
        anomaly_type="text",
        count=count,
        percentage=_pct(count, len(items)),
        description=(
            f"{count} cell(s) are mostly Latin while the column is mostly Thai "
            f"(column Thai ratio ≈ {col_mean:.2f})."
        ),
        description_th=(
            f"{count} เซลล์เป็นอักษรละตินเป็นหลัก ทั้งที่คอลัมน์ส่วนใหญ่เป็นไทย "
            f"(สัดส่วนไทยของคอลัมน์ ≈ {col_mean:.2f})"
        ),
        examples=[_trunc(s) for _, s in flagged[:_MAX_INDICES]],
        suggestion="Verify these rows are in the right column/language.",
        suggestion_th="ตรวจสอบว่าแถวเหล่านี้อยู่ในคอลัมน์/ภาษาที่ถูกต้อง",
        indices=[pos for pos, _ in flagged[:_MAX_INDICES]],
    )


def detect_thai_text_anomalies(series: pd.Series) -> list[AnomalyIssue]:
    """ตรวจความผิดปกติเฉพาะภาษาไทย: สระ/วรรณยุกต์ลอย, วรรณยุกต์ซ้อน, ลำดับ NFC, การปนสคริปต์."""
    col = _col_name(series)
    items = _positional_items(series)
    if not items:
        return []

    issues: list[AnomalyIssue] = []
    for check in (
        _orphan_combining_anomalies,
        _tone_stacking_anomalies,
        _diacritic_order_anomalies,
        _script_mixing_anomalies,
    ):
        issue = check(items, col)
        if issue is not None:
            issues.append(issue)
    return issues


# ----------------------------------------------------------------------------
# (d) Categorical anomalies
# ----------------------------------------------------------------------------
def _rare_category_anomaly(
    counts: Counter[str], items: list[tuple[int, str]], col: str
) -> AnomalyIssue | None:
    total = len(items)
    if total < _RARE_MIN_TOTAL or len(counts) < 3:
        return None
    rare = {cat for cat, c in counts.items() if (c / total) < 0.01}
    if not rare:
        return None
    indices = [pos for pos, v in items if v in rare][:_MAX_INDICES]
    count = sum(counts[c] for c in rare)
    rarest = sorted(rare, key=lambda x: counts[x])[:_MAX_INDICES]
    examples = [f"{cat} (×{counts[cat]})" for cat in rarest]
    return AnomalyIssue(
        check_name="rare_categories",
        severity="info",
        column=col,
        anomaly_type="categorical",
        count=count,
        percentage=_pct(count, total),
        description=(
            f"{len(rare)} category value(s) occur in <1% of rows — possible typos or rare cases."
        ),
        description_th=(
            f"พบหมวดหมู่ {len(rare)} ค่า ที่ปรากฏน้อยกว่า 1% ของแถว — อาจเป็นการพิมพ์ผิดหรือค่าที่หายาก"
        ),
        examples=examples,
        suggestion="Review rare categories; consolidate typos or group into an 'other' bucket.",
        suggestion_th="ตรวจสอบหมวดหมู่หายาก รวมค่าที่พิมพ์ผิดหรือจัดเข้ากลุ่ม 'อื่น ๆ'",
        indices=indices,
    )


def _fuzzy_duplicate_anomaly(
    counts: Counter[str], items: list[tuple[int, str]], col: str
) -> AnomalyIssue | None:
    """หมวดหมู่ที่คล้ายกันมาก (SequenceMatcher ratio > 0.8) แต่ไม่เหมือนกัน และไม่ใช่แค่ตัวพิมพ์ใหญ่/เล็ก."""
    cats = [c for c, _ in counts.most_common(300)]
    pairs: list[tuple[str, str]] = []
    involved: set[str] = set()
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            a, b = cats[i], cats[j]
            if a.casefold() == b.casefold():
                continue  # ต่างแค่ตัวพิมพ์ -> จัดเป็น case inconsistency แทน
            if _string_similarity(a, b) > 0.8:
                pairs.append((a, b))
                involved.update((a, b))
                if len(pairs) >= _MAX_INDICES:
                    break
        if len(pairs) >= _MAX_INDICES:
            break
    if not pairs:
        return None
    indices = [pos for pos, v in items if v in involved][:_MAX_INDICES]
    count = sum(counts[c] for c in involved)
    examples = [f"{a} ↔ {b}" for a, b in pairs[:_MAX_INDICES]]
    return AnomalyIssue(
        check_name="fuzzy_duplicates",
        severity="warning",
        column=col,
        anomaly_type="categorical",
        count=count,
        percentage=_pct(count, len(items)),
        description=f"{len(pairs)} near-duplicate category pair(s) detected (similarity > 0.8).",
        description_th=f"พบคู่หมวดหมู่ที่คล้ายกันเกือบเหมือน {len(pairs)} คู่ (ความคล้าย > 0.8)",
        examples=examples,
        suggestion=(
            "These likely refer to the same value (e.g. 'กรุงเทพ' vs 'กรุงเทพฯ'); standardize them."
        ),
        suggestion_th=(
            "ค่าเหล่านี้น่าจะหมายถึงสิ่งเดียวกัน (เช่น 'กรุงเทพ' กับ 'กรุงเทพฯ') ควรทำให้เป็นค่ามาตรฐานเดียว"
        ),
        indices=indices,
    )


def _case_inconsistency_anomaly(
    counts: Counter[str], items: list[tuple[int, str]], col: str
) -> AnomalyIssue | None:
    groups: dict[str, set[str]] = {}
    for cat in counts:
        groups.setdefault(cat.casefold(), set()).add(cat)
    inconsistent = {k: v for k, v in groups.items() if len(v) > 1}
    if not inconsistent:
        return None
    involved = {c for variants in inconsistent.values() for c in variants}
    indices = [pos for pos, v in items if v in involved][:_MAX_INDICES]
    count = sum(counts[c] for c in involved)
    examples = [" / ".join(sorted(v)) for v in list(inconsistent.values())[:_MAX_INDICES]]
    return AnomalyIssue(
        check_name="case_inconsistency",
        severity="warning",
        column=col,
        anomaly_type="categorical",
        count=count,
        percentage=_pct(count, len(items)),
        description=(
            f"{len(inconsistent)} value(s) appear with inconsistent casing "
            "(e.g. 'Bangkok' vs 'bangkok')."
        ),
        description_th=(
            f"พบ {len(inconsistent)} ค่าที่ใช้ตัวพิมพ์ใหญ่/เล็กไม่สม่ำเสมอ (เช่น 'Bangkok' กับ 'bangkok')"
        ),
        examples=examples,
        suggestion="Normalize casing (e.g. lowercase) before grouping or joining.",
        suggestion_th="ทำให้ตัวพิมพ์ใหญ่/เล็กสม่ำเสมอ (เช่น แปลงเป็นตัวพิมพ์เล็ก) ก่อนจัดกลุ่มหรือ join",
        indices=indices,
    )


def detect_categorical_anomalies(series: pd.Series) -> list[AnomalyIssue]:
    """ตรวจความผิดปกติของหมวดหมู่: หมวดหมู่หายาก, ค่าที่คล้ายกันจนน่าสงสัย, ตัวพิมพ์ไม่สม่ำเสมอ."""
    col = _col_name(series)
    items = _positional_items(series)
    if not items:
        return []
    counts: Counter[str] = Counter(v for _, v in items)

    issues: list[AnomalyIssue] = []
    for check in (
        _rare_category_anomaly,
        _fuzzy_duplicate_anomaly,
        _case_inconsistency_anomaly,
    ):
        issue = check(counts, items, col)
        if issue is not None:
            issues.append(issue)
    return issues


# ----------------------------------------------------------------------------
# (e) Column-level / cross-column anomalies
# ----------------------------------------------------------------------------
_DATE_LIKE_RE = re.compile(r"\d{1,4}[-/.]\d{1,2}([-/.]\d{1,4})?")


def _date_signature(text: str) -> str:
    """แทนหลักตัวเลขด้วย '9' เพื่อให้ได้ 'ลายเซ็นรูปแบบ' ของวันที่ (เช่น 9999-99-99)."""
    return re.sub(r"\d", "9", text.strip())


def _type_mixing_anomaly(series: pd.Series, col: str) -> AnomalyIssue | None:
    items = _positional_items(series)
    total = len(items)
    if total < _MIN_STAT_SAMPLE:
        return None
    numeric = pd.to_numeric(pd.Series([v for _, v in items]), errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    non_numeric = [(pos, v) for (pos, v), ok in zip(items, numeric.notna(), strict=True) if not ok]
    non_numeric_ratio = len(non_numeric) / total
    # ส่วนใหญ่เป็นเลข แต่มีบางส่วน (>5%) ที่ไม่ใช่เลข -> ปนชนิด
    if numeric_ratio < 0.5 or non_numeric_ratio <= 0.05:
        return None
    count = len(non_numeric)
    return AnomalyIssue(
        check_name="type_mixing",
        severity="warning",
        column=col,
        anomaly_type="pattern",
        count=count,
        percentage=_pct(count, total),
        description=(
            f"Column is {numeric_ratio * 100:.0f}% numeric but {count} value(s) "
            f"({non_numeric_ratio * 100:.1f}%) are non-numeric strings."
        ),
        description_th=(
            f"คอลัมน์เป็นตัวเลข {numeric_ratio * 100:.0f}% แต่มี {count} ค่า "
            f"({non_numeric_ratio * 100:.1f}%) ที่เป็นสตริงไม่ใช่ตัวเลข"
        ),
        examples=[_trunc(v) for _, v in non_numeric[:_MAX_INDICES]],
        suggestion=(
            "Coerce to numeric and inspect non-parseable values (placeholders like 'N/A', '-')."
        ),
        suggestion_th="แปลงเป็นตัวเลขแล้วตรวจค่าที่แปลงไม่ได้ (ค่าแทน เช่น 'N/A', '-')",
        indices=[pos for pos, _ in non_numeric[:_MAX_INDICES]],
    )


def _mixed_date_format_anomaly(series: pd.Series, col: str) -> AnomalyIssue | None:
    items = _positional_items(series)
    total = len(items)
    if total < _MIN_STAT_SAMPLE:
        return None
    date_items = [(pos, v) for pos, v in items if _DATE_LIKE_RE.search(v)]
    if len(date_items) / total < 0.6:
        return None
    sig_counts: Counter[str] = Counter(_date_signature(v) for _, v in date_items)
    # ต้องมีอย่างน้อย 2 รูปแบบที่ต่างกัน
    if len(sig_counts) < 2:
        return None
    formats = sorted(sig_counts)
    return AnomalyIssue(
        check_name="mixed_date_formats",
        severity="warning",
        column=col,
        anomaly_type="pattern",
        count=len(date_items),
        percentage=_pct(len(date_items), total),
        description=(
            f"Date column mixes {len(formats)} distinct format patterns ({', '.join(formats[:5])})."
        ),
        description_th=(f"คอลัมน์วันที่มีรูปแบบที่ต่างกัน {len(formats)} รูปแบบ ({', '.join(formats[:5])})"),
        examples=[_trunc(v) for _, v in date_items[:_MAX_INDICES]],
        suggestion="Parse with a single canonical format; mixed formats break sorting and parsing.",
        suggestion_th="แปลงเป็นรูปแบบมาตรฐานเดียว — รูปแบบที่ปนกันทำให้การเรียงลำดับและ parse ผิดพลาด",
        indices=[pos for pos, _ in date_items[:_MAX_INDICES]],
    )


def _is_textual_dtype(series: pd.Series) -> bool:
    """True ถ้าคอลัมน์อาจเก็บสตริง (object/StringDtype/category) — ไม่ใช่ตัวเลข/วันที่/bool.

    จำเป็นเพราะ pandas 3.0 ใช้ StringDtype กับคอลัมน์ข้อความโดยปริยาย (ไม่ใช่ object)
    """
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )


def _high_null_anomaly(series: pd.Series, col: str) -> AnomalyIssue | None:
    total = len(series)
    if total == 0:
        return None
    null_count = int(series.isna().sum())
    null_ratio = null_count / total
    if null_ratio <= 0.5:
        return None
    return AnomalyIssue(
        check_name="high_null_spike",
        severity="warning",
        column=col,
        anomaly_type="pattern",
        count=null_count,
        percentage=null_ratio * 100.0,
        description=f"Column is {null_ratio * 100:.1f}% null — most values are missing.",
        description_th=f"คอลัมน์มีค่าว่าง {null_ratio * 100:.1f}% — ข้อมูลส่วนใหญ่หายไป",
        examples=[],
        suggestion="Consider dropping the column or investigating why data is missing.",
        suggestion_th="พิจารณาตัดคอลัมน์ทิ้ง หรือตรวจสอบสาเหตุที่ข้อมูลหายไป",
        indices=[],
    )


def _constant_anomaly(series: pd.Series, col: str) -> AnomalyIssue | None:
    non_null = series.dropna()
    if len(non_null) < 2:
        return None
    if non_null.nunique() != 1:
        return None
    value = str(non_null.iloc[0])
    return AnomalyIssue(
        check_name="constant_column",
        severity="info",
        column=col,
        anomaly_type="pattern",
        count=int(len(non_null)),
        percentage=100.0,
        description=f"Column has a single unique value ('{_trunc(value)}') — no information.",
        description_th=f"คอลัมน์มีค่าไม่ซ้ำเพียงค่าเดียว ('{_trunc(value)}') — ไม่มีสารสนเทศ",
        examples=[_trunc(value)],
        suggestion="Constant columns can usually be dropped.",
        suggestion_th="คอลัมน์ที่มีค่าเดียวมักตัดทิ้งได้",
        indices=[],
    )


def detect_column_anomalies(
    df: pd.DataFrame, column_types: dict[str, ColumnType]
) -> list[AnomalyIssue]:
    """ตรวจความผิดปกติระดับคอลัมน์: ปนชนิด, รูปแบบวันที่ปนกัน, ค่าว่างสูง, คอลัมน์ค่าคงที่."""
    issues: list[AnomalyIssue] = []
    for col in df.columns:
        col_name = str(col)
        series = df[col]
        ctype = column_types.get(col_name, ColumnType.EMPTY)
        if ctype == ColumnType.EMPTY:
            issue = _high_null_anomaly(series, col_name)
            if issue is not None:
                issues.append(issue)
            continue

        if (issue := _high_null_anomaly(series, col_name)) is not None:
            issues.append(issue)
        if (issue := _constant_anomaly(series, col_name)) is not None:
            issues.append(issue)

        # ปนชนิด — เฉพาะคอลัมน์ที่เก็บสตริง (คอลัมน์ตัวเลขจริงไม่มีสตริงปน)
        if _is_textual_dtype(series):
            if (issue := _type_mixing_anomaly(series, col_name)) is not None:
                issues.append(issue)

            # รูปแบบวันที่ปนกัน — คอลัมน์สตริงที่ดูเหมือนวันที่
            if (issue := _mixed_date_format_anomaly(series, col_name)) is not None:
                issues.append(issue)

    return issues


# ----------------------------------------------------------------------------
# (f) Main entry point
# ----------------------------------------------------------------------------
_TEXT_TYPES = {ColumnType.THAI_TEXT, ColumnType.MIXED_TEXT, ColumnType.ENGLISH_TEXT}
_THAI_TYPES = {ColumnType.THAI_TEXT, ColumnType.MIXED_TEXT}
_CATEGORICAL_TYPES = {ColumnType.CATEGORICAL, ColumnType.ID}


def _detect_anomalies_frame(
    df: pd.DataFrame,
    column_types: dict[str, ColumnType] | None = None,
    tokenizer=None,
) -> list[AnomalyIssue]:
    """ตรวจความผิดปกติทั้งหมดใน DataFrame คืนรายการ AnomalyIssue เรียงตามความรุนแรง.

    Args:
        df: ข้อมูลที่ต้องการตรวจ.
        column_types: ผลจำแนกประเภทคอลัมน์ (ถ้าไม่ให้ จะเรียก detect_all ให้เอง).
        tokenizer: ตัวตัดคำ — ถ้าเป็น None จะข้ามการตรวจเชิงข้อความทั่วไป
            (แต่ยังตรวจตัวเลข/หมวดหมู่/เฉพาะภาษาไทยที่ไม่ต้องตัดคำ).

    Returns:
        list[AnomalyIssue] เรียงจากวิกฤต -> เตือน -> ข้อมูล.
    """
    if column_types is None:
        column_types = detect_all(df)

    issues: list[AnomalyIssue] = list(detect_column_anomalies(df, column_types))

    # รันวิธี ML เฉพาะเมื่อมี scikit-learn (ตรวจครั้งเดียว ไม่ import ซ้ำต่อคอลัมน์)
    run_ml = sklearn_available()

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        ctype = column_types.get(col_name, ColumnType.EMPTY)

        if ctype == ColumnType.NUMERIC:
            if (issue := detect_numeric_outliers(series)) is not None:
                issues.append(issue)
            # วิธี ML — เสริมวิธีเชิงสถิติเมื่อข้อมูลพอ (>100 แถว) และมี sklearn
            # ไม่ deduplicate: การที่หลายวิธี flag จุดเดียวกันช่วยเพิ่มความมั่นใจ
            if run_ml:
                for ml_detect in (detect_isolation_forest, detect_lof):
                    if (issue := ml_detect(series)) is not None:
                        issues.append(issue)

        if ctype in _TEXT_TYPES:
            if tokenizer is not None:
                issues.extend(detect_text_anomalies(series, tokenizer))
            if ctype in _THAI_TYPES:
                # การตรวจเฉพาะภาษาไทยเป็น unicode-level ไม่ต้องใช้ tokenizer
                issues.extend(detect_thai_text_anomalies(series))

        if ctype in _CATEGORICAL_TYPES:
            issues.extend(detect_categorical_anomalies(series))

    issues.sort(key=lambda i: (SEVERITY_ORDER.get(i.severity, 99), -i.percentage))
    return issues


# ----------------------------------------------------------------------------
# (g) Unified single-column API (แรงบันดาลใจจาก PyCaret) — ฟังก์ชันเดียว เลือกวิธีด้วย method
# ----------------------------------------------------------------------------
@dataclass
class AnomalySummary:
    """สรุปการตรวจ outlier ของคอลัมน์เดียวแบบรวม (จากหลายวิธีถ้า method='auto')."""

    column: str
    method: str
    total_anomalies: int
    anomaly_rate: float  # ร้อยละ
    issues: list[AnomalyIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "method": self.method,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": round(self.anomaly_rate, 2),
            "issues": [i.to_dict() for i in self.issues],
        }


# วิธีเชิงสถิติที่ระบุชื่อได้โดยตรง: ชื่อ method -> (ป้ายแสดงผล, ฟังก์ชันสร้าง mask)
_STAT_METHODS = {
    "zscore": ("z_score", _zscore_mask),
    "mad": ("modified_z_score (MAD)", _mad_mask),
    "iqr": ("IQR", _iqr_mask),
}
_VALID_METHODS = ("auto", "zscore", "mad", "iqr", "iforest", "lof")


def _stat_issue(series: pd.Series, method_key: str) -> AnomalyIssue | None:
    """รัน outlier เชิงสถิติด้วยวิธีที่ระบุ (zscore/mad/iqr) บนคอลัมน์ตัวเลข."""
    label, mask_fn = _STAT_METHODS[method_key]
    valid, valid_positions, _ = _numeric_valid(series)
    if valid.size < _MIN_NUMERIC_SAMPLE:
        return None
    out_mask = mask_fn(valid)
    if out_mask is None or not bool(out_mask.any()):
        return None
    outlier_positions = valid_positions[out_mask]
    outlier_values = valid[out_mask]
    count = int(outlier_positions.size)
    return AnomalyIssue(
        check_name=f"numeric_outliers_{method_key}",
        severity="warning",
        column=_col_name(series),
        anomaly_type="statistical",
        count=count,
        percentage=_pct(count, int(valid.size)),
        description=f"{count} numeric outlier(s) detected using the {label} method.",
        description_th=f"พบค่าผิดปกติเชิงตัวเลข {count} ค่า ด้วยวิธี {label}",
        examples=[_fmt_number(float(v)) for v in outlier_values[:_MAX_INDICES]],
        suggestion="Inspect these values; they may be data-entry errors or genuine extremes.",
        suggestion_th="ตรวจสอบค่าเหล่านี้ — อาจเป็นการกรอกผิดหรือค่าสุดขั้วจริง",
        indices=[int(p) for p in outlier_positions[:_MAX_INDICES]],
    )


def _auto_series_issues(series: pd.Series) -> tuple[str, list[AnomalyIssue]]:
    """เลือกวิธีตรวจ outlier ของคอลัมน์อัตโนมัติตามประเภท/ขนาดข้อมูล — คืน (ป้ายวิธี, รายการ issue)."""
    ctype = detect_column_type(series)

    if ctype == ColumnType.NUMERIC:
        issues = [i for i in (detect_numeric_outliers(series),) if i is not None]
        # ข้อมูลใหญ่พอ (>100 แถว) + มี scikit-learn -> เสริมวิธี ML
        if sklearn_available():
            for fn in (detect_isolation_forest, detect_lof):
                issue = fn(series)
                if issue is not None:
                    issues.append(issue)
            return "auto:statistical+ml", issues
        return "auto:statistical", issues

    if ctype in _TEXT_TYPES:
        # การตรวจข้อความ (ความยาว/mojibake/ซ้ำ) ไม่ต้องใช้ tokenizer
        issues = list(detect_text_anomalies(series, None))
        if ctype in _THAI_TYPES:
            issues.extend(detect_thai_text_anomalies(series))
        return "auto:text", issues

    if ctype in _CATEGORICAL_TYPES:
        return "auto:categorical", list(detect_categorical_anomalies(series))

    return "auto:none", []


def _detect_anomalies_series(
    series: pd.Series, method: str = "auto", **kwargs
) -> AnomalySummary | None:
    """ตรวจ outlier ของคอลัมน์เดียวแบบรวม — คืน AnomalySummary หรือ None ถ้าไม่พบ/ใช้ไม่ได้."""
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown anomaly method {method!r}. Expected one of: {', '.join(_VALID_METHODS)}."
        )
    col = _col_name(series)

    if method == "auto":
        used, issues = _auto_series_issues(series)
    elif method in _STAT_METHODS:
        used, issues = method, [i for i in (_stat_issue(series, method),) if i is not None]
    elif method == "iforest":
        used, issues = method, [i for i in (detect_isolation_forest(series),) if i is not None]
    else:  # "lof"
        used, issues = method, [i for i in (detect_lof(series),) if i is not None]

    if not issues:
        return None

    # total_anomalies/rate ใช้ค่าสูงสุดในบรรดา issue (ไม่บวกรวมเพื่อเลี่ยงการนับซ้ำข้ามวิธี)
    total = max(i.count for i in issues)
    rate = max(i.percentage for i in issues)
    return AnomalySummary(
        column=col, method=used, total_anomalies=total, anomaly_rate=rate, issues=issues
    )


def detect_anomalies(data, *args, **kwargs):
    """ตรวจความผิดปกติ — รับได้ทั้ง Series (API รวมแบบคอลัมน์เดียว) และ DataFrame (ทั้งตาราง).

    * ``detect_anomalies(series, method="auto")`` -> ``AnomalySummary | None``
        method = "auto" (เลือกตามชนิด/ขนาด), "zscore", "mad", "iqr",
        "iforest" (Isolation Forest), "lof" (Local Outlier Factor).
    * ``detect_anomalies(df, column_types=None, tokenizer=None)`` -> ``list[AnomalyIssue]``
        ตรวจทุกคอลัมน์ คืนรายการ AnomalyIssue เรียงตามความรุนแรง (พฤติกรรมเดิม).
    """
    if isinstance(data, pd.Series):
        return _detect_anomalies_series(data, *args, **kwargs)
    return _detect_anomalies_frame(data, *args, **kwargs)


def detect_anomalies_all(
    df: pd.DataFrame, method: str = "auto"
) -> dict[str, AnomalySummary | None]:
    """ตรวจ outlier ทุกคอลัมน์ด้วย API รวม — คืน dict {ชื่อคอลัมน์: AnomalySummary | None}."""
    return {str(col): _detect_anomalies_series(df[col], method=method) for col in df.columns}


__all__ = [
    "AnomalyIssue",
    "AnomalySummary",
    "detect_numeric_outliers",
    "detect_isolation_forest",
    "detect_lof",
    "detect_thai_mojibake",
    "detect_text_anomalies",
    "detect_thai_text_anomalies",
    "detect_categorical_anomalies",
    "detect_column_anomalies",
    "detect_anomalies",
    "detect_anomalies_all",
    "sklearn_available",
]
