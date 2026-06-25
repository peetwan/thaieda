"""Target variable analysis — วิเคราะห์ความสัมพันธ์ของทุกคอลัมน์กับ target column.

แรงบันดาลใจจาก Sweetviz: เมื่อระบุคอลัมน์เป้าหมาย (target) จะคำนวณว่าคอลัมน์อื่น ๆ
สัมพันธ์กับเป้าหมายมากแค่ไหน โดยเลือกสถิติให้เหมาะกับชนิดข้อมูล:
  * ตัวเลข × ตัวเลข   -> Pearson correlation
  * ตัวเลข × หมวดหมู่ -> ANOVA F-statistic
  * หมวดหมู่ × หมวดหมู่ -> Chi-square test

scipy เป็น dependency เสริม (thaieda[stats]) — ใช้คำนวณ p-value ที่ถูกต้อง
ถ้าไม่มี scipy จะยังคำนวณค่าสถิติ (score) ได้ แต่ p-value เป็น NaN
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from thaieda.detect import ColumnType, detect_column_type

# ----------------------------------------------------------------------------
# ค่าคงที่
# ----------------------------------------------------------------------------
# จำนวนหมวดหมู่สูงสุดที่ยอมรับให้คอลัมน์เป็น "categorical" (มากกว่านี้ chi-square ไม่มีความหมาย)
_MAX_CATEGORIES = 50
# จำนวนแถวขั้นต่ำที่ทำให้สถิติมีความหมาย
_MIN_SAMPLE = 3


# ----------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ----------------------------------------------------------------------------
@dataclass
class TargetAssociation:
    """ความสัมพันธ์ของคอลัมน์หนึ่งกับ target column."""

    column: str
    target: str
    association_type: str  # "correlation" | "chi_square" | "anova"
    score: float  # correlation coefficient, chi2 หรือ F-statistic
    p_value: float  # NaN ถ้าไม่มี scipy
    description_th: str

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "target": self.target,
            "association_type": self.association_type,
            "score": round(self.score, 4) if not math.isnan(self.score) else None,
            "p_value": round(self.p_value, 4) if not math.isnan(self.p_value) else None,
            "description_th": self.description_th,
        }


# ----------------------------------------------------------------------------
# helper
# ----------------------------------------------------------------------------
def _scipy_stats():
    """คืนโมดูล scipy.stats ถ้าติดตั้ง (thaieda[stats]) ไม่งั้นคืน None."""
    try:
        import scipy.stats as st
    except ImportError:
        return None
    return st


def _classify(series: pd.Series) -> str | None:
    """จัดคอลัมน์เป็น 'numeric'/'categorical' สำหรับการวิเคราะห์ target หรือ None ถ้าใช้ไม่ได้.

    หมวดหมู่ต้องมีจำนวนค่าไม่ซ้ำพอเหมาะ (2–50) — ID/ข้อความ cardinality สูงจะถูกข้าม
    """
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    ctype = detect_column_type(series)
    if ctype == ColumnType.NUMERIC:
        return "numeric"
    if ctype in (ColumnType.CATEGORICAL, ColumnType.ID):
        nunique = int(series.dropna().nunique())
        if 2 <= nunique <= _MAX_CATEGORIES:
            return "categorical"
    return None


def _sig_th(p_value: float, alpha: float) -> str:
    """ข้อความไทยบอกนัยสำคัญทางสถิติ."""
    if math.isnan(p_value):
        return "ไม่มี scipy จึงไม่ได้คำนวณ p-value (ติดตั้ง thaieda[stats] เพื่อทดสอบนัยสำคัญ)"
    if p_value < alpha:
        return f"มีนัยสำคัญทางสถิติ (p={p_value:.4f} < {alpha})"
    return f"ไม่มีนัยสำคัญทางสถิติ (p={p_value:.4f} ≥ {alpha})"


# ----------------------------------------------------------------------------
# การคำนวณแต่ละชนิด
# ----------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray, st) -> tuple[float, float] | None:
    """Pearson correlation — คืน (r, p) หรือ None ถ้าคำนวณไม่ได้ (ความแปรปรวนเป็น 0)."""
    if x.size < _MIN_SAMPLE or x.std() == 0 or y.std() == 0:
        return None
    if st is not None:
        result = st.pearsonr(x, y)
        return float(result[0]), float(result[1])
    r = float(np.corrcoef(x, y)[0, 1])
    return r, float("nan")


def _anova(num: np.ndarray, groups: list[np.ndarray], st) -> tuple[float, float] | None:
    """ANOVA F-statistic (one-way) — num คือค่าตัวเลข, groups คือค่าตัวเลขแยกตามหมวดหมู่."""
    groups = [g for g in groups if g.size > 0]
    k = len(groups)
    n = sum(g.size for g in groups)
    if k < 2 or n - k <= 0:
        return None
    if st is not None:
        result = st.f_oneway(*groups)
        f, p = float(result[0]), float(result[1])
        if math.isnan(f):
            return None
        return f, p
    grand = num.mean()
    ss_between = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    if ss_within == 0:
        return None
    f = (ss_between / (k - 1)) / (ss_within / (n - k))
    return float(f), float("nan")


def _chi_square(observed: np.ndarray, st) -> tuple[float, float] | None:
    """Chi-square test of independence จากตาราง contingency — คืน (chi2, p)."""
    if observed.size == 0 or observed.sum() == 0 or min(observed.shape) < 2:
        return None
    if st is not None:
        chi2, p, _, _ = st.chi2_contingency(observed)
        return float(chi2), float(p)
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row @ col / total
    nonzero = expected > 0
    chi2 = float((((observed - expected) ** 2)[nonzero] / expected[nonzero]).sum())
    return chi2, float("nan")


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def analyze_target(
    df: pd.DataFrame, target_column: str, alpha: float = 0.05
) -> list[TargetAssociation]:
    """วิเคราะห์ความสัมพันธ์ของทุกคอลัมน์กับ target column.

    Args:
        df: ข้อมูล.
        target_column: ชื่อคอลัมน์เป้าหมาย (ต้องเป็นตัวเลขหรือหมวดหมู่ cardinality ต่ำ).
        alpha: ระดับนัยสำคัญสำหรับทดสอบสมมติฐาน (ค่าเริ่มต้น 0.05).

    Returns:
        list[TargetAssociation] เรียงจากสัมพันธ์ชัด/มีนัยสำคัญที่สุดก่อน.

    Raises:
        KeyError: ถ้าไม่พบ target_column ใน df.
        ValueError: ถ้า target_column ไม่ใช่ตัวเลขหรือหมวดหมู่ที่วิเคราะห์ได้.
    """
    if target_column not in df.columns:
        raise KeyError(f"target_column {target_column!r} not found in DataFrame.")

    target_kind = _classify(df[target_column])
    if target_kind is None:
        raise ValueError(
            f"target_column {target_column!r} must be numeric or a low-cardinality "
            "categorical column (2–50 distinct values)."
        )

    st = _scipy_stats()
    target = df[target_column]
    results: list[TargetAssociation] = []

    for col in df.columns:
        name = str(col)
        if name == str(target_column):
            continue
        col_kind = _classify(df[col])
        if col_kind is None:
            continue

        assoc = _associate(df, name, target, target_kind, col_kind, st, alpha)
        if assoc is not None:
            results.append(assoc)

    # เรียง: มี p-value (มีนัยสำคัญ) ก่อน แล้วตามความแรงของความสัมพันธ์
    def sort_key(a: TargetAssociation) -> tuple:
        p = a.p_value if not math.isnan(a.p_value) else 2.0
        return (p, -abs(a.score))

    results.sort(key=sort_key)
    return results


def _associate(
    df: pd.DataFrame,
    col: str,
    target: pd.Series,
    target_kind: str,
    col_kind: str,
    st,
    alpha: float,
) -> TargetAssociation | None:
    """คำนวณความสัมพันธ์หนึ่งคู่ (คอลัมน์ vs target) ตามชนิดข้อมูล."""
    target_name = str(target.name)
    series = df[col]

    # numeric × numeric -> Pearson correlation
    if target_kind == "numeric" and col_kind == "numeric":
        x = pd.to_numeric(series, errors="coerce")
        y = pd.to_numeric(target, errors="coerce")
        mask = x.notna() & y.notna()
        if int(mask.sum()) < _MIN_SAMPLE:
            return None
        out = _pearson(x[mask].to_numpy("float64"), y[mask].to_numpy("float64"), st)
        if out is None:
            return None
        r, p = out
        desc = f"สหสัมพันธ์ Pearson ระหว่าง '{col}' กับ '{target_name}' = {r:.3f} — {_sig_th(p, alpha)}"
        return TargetAssociation(col, target_name, "correlation", r, p, desc)

    # numeric × categorical (ทิศใดก็ได้) -> ANOVA F
    if "numeric" in (target_kind, col_kind) and "categorical" in (target_kind, col_kind):
        if target_kind == "numeric":
            num_series, cat_series = target, series
        else:
            num_series, cat_series = series, target
        num = pd.to_numeric(num_series, errors="coerce")
        frame = pd.DataFrame({"num": num, "cat": cat_series.astype("object")}).dropna()
        if len(frame) < _MIN_SAMPLE or frame["cat"].nunique() < 2:
            return None
        groups = [grp["num"].to_numpy("float64") for _, grp in frame.groupby("cat")]
        out = _anova(frame["num"].to_numpy("float64"), groups, st)
        if out is None:
            return None
        f, p = out
        desc = f"ANOVA F ของ '{col}' เทียบกับ '{target_name}' = {f:.3f} — {_sig_th(p, alpha)}"
        return TargetAssociation(col, target_name, "anova", f, p, desc)

    # categorical × categorical -> Chi-square
    if target_kind == "categorical" and col_kind == "categorical":
        frame = pd.DataFrame({"a": series.astype("object"), "b": target.astype("object")}).dropna()
        if len(frame) < _MIN_SAMPLE:
            return None
        observed = pd.crosstab(frame["a"], frame["b"]).to_numpy(dtype="float64")
        out = _chi_square(observed, st)
        if out is None:
            return None
        chi2, p = out
        desc = f"Chi-square ระหว่าง '{col}' กับ '{target_name}' = {chi2:.3f} — {_sig_th(p, alpha)}"
        return TargetAssociation(col, target_name, "chi_square", chi2, p, desc)

    return None


__all__ = [
    "TargetAssociation",
    "analyze_target",
]
