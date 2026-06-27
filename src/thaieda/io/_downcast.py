"""dtype downcasting — ลด dtype เพื่อประหยัด memory บนเครื่อง low-resource — v2.0.

แปลงชนิดข้อมูลให้เล็กลงโดยไม่เสียค่า:
  * int64   → int32/int16/int8 (เลือกชนิดเล็กสุดที่ครอบคลุมช่วงค่าจริง)
  * float64 → float32
  * object  → category (ถ้า cardinality < 50% ของจำนวนแถว — ประหยัดเมื่อค่าซ้ำเยอะ)

หลักการ:
  * ไม่เปลี่ยนค่า — เฉพาะชนิดข้อมูล (downcast แบบ lossless สำหรับ int; float32 ยอมรับ precision ที่ลดลง)
  * คืนรายงาน before/after memory เพื่อให้ตรวจสอบได้ว่าประหยัดไปเท่าไร
  * ไม่แตะคอลัมน์ที่ downcast ไม่ได้ (เช่น datetime, bool, category อยู่แล้ว)
"""

from __future__ import annotations

import pandas as pd

# cardinality สูงสุด (สัดส่วนต่อจำนวนแถว) ที่ยังคุ้มจะแปลง object → category
_CATEGORY_MAX_CARDINALITY_RATIO = 0.50


def downcast_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """ลด dtype เพื่อประหยัด memory — v2.0.

    int64 → int32 (หรือ int16/int8 ถ้าช่วงค่าพอดี)
    float64 → float32
    object → category (ถ้า cardinality < 50% ของจำนวนแถว)

    Args:
        df: DataFrame ต้นฉบับ.

    Returns:
        (downcasted_df, report_dict) — report มี memory ก่อน/หลัง (MB), % ที่ลด,
        และรายละเอียดการเปลี่ยน dtype ต่อคอลัมน์.
    """
    out = df.copy()
    n_rows = len(out)

    mem_before = int(out.memory_usage(deep=True).sum())
    columns_changed: dict[str, dict[str, str]] = {}

    for col in out.columns:
        series = out[col]
        before_dtype = str(series.dtype)
        new_series = _downcast_series(series, n_rows)
        if new_series is not None and str(new_series.dtype) != before_dtype:
            out[col] = new_series
            columns_changed[str(col)] = {
                "before": before_dtype,
                "after": str(new_series.dtype),
            }

    mem_after = int(out.memory_usage(deep=True).sum())
    reduction = mem_before - mem_after
    reduction_pct = (reduction / mem_before * 100.0) if mem_before else 0.0

    report = {
        "memory_before_bytes": mem_before,
        "memory_after_bytes": mem_after,
        "memory_before_mb": round(mem_before / 1e6, 3),
        "memory_after_mb": round(mem_after / 1e6, 3),
        "reduction_bytes": reduction,
        "reduction_pct": round(reduction_pct, 1),
        "columns_changed": columns_changed,
        "n_columns_changed": len(columns_changed),
    }
    return out, report


def _downcast_series(series: pd.Series, n_rows: int) -> pd.Series | None:
    """ลด dtype ของ Series หนึ่งคอลัมน์ — คืน None ถ้าไม่มีอะไรเปลี่ยน."""
    dtype = series.dtype

    # bool / category / datetime → ไม่แตะ (เล็กอยู่แล้ว หรือไม่เหมาะ downcast)
    if pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return None
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return None

    # integer → เลือกชนิดเล็กสุด
    if pd.api.types.is_integer_dtype(dtype):
        return pd.to_numeric(series, downcast="integer")

    # float → float32 (downcast="float" เลือก float32 ถ้าค่าพอดี)
    if pd.api.types.is_float_dtype(dtype):
        return pd.to_numeric(series, downcast="float")

    # object/string → category ถ้าค่าซ้ำเยอะ (cardinality < 50% ของแถว)
    if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
        if n_rows == 0:
            return None
        nunique = series.nunique(dropna=True)
        if nunique / n_rows < _CATEGORY_MAX_CARDINALITY_RATIO:
            return series.astype("category")
        return None

    return None


__all__ = ["downcast_dtypes"]
