"""DataFrame-level cleaning pipeline — ``thaieda.clean(df)`` ในครั้งเดียว — v2.0.

รวมการทำความสะอาดทั้ง DataFrame เข้าด้วยกัน: encoding, whitespace, เลขไทย, วันที่/พ.ศ.,
สกุลเงิน, แปลง numeric, จัดการค่าว่าง, ลบแถวซ้ำ และ dtype downcasting เพื่อประหยัด memory.

หลักการ:
  * ทำงานบน copy — ไม่แก้ DataFrame ต้นฉบับ
  * คืน CleaningReport ที่ตรวจสอบได้ว่าทำอะไรไปบ้าง (operation, แถวที่กระทบ, ก่อน/หลัง)
  * วันที่/พ.ศ. แปลงเฉพาะคอลัมน์ข้อความที่ "ดูเหมือนวันที่" — กันการเผลอแปลงเลขปีในคอลัมน์ราคา
  * ไม่มี silent fallback — operation ที่ระบุผิดจะ raise (ผ่าน clean_thai_text)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from thaieda._validation import ensure_unique_column_names
from thaieda.clean import (
    _THAI_MONTH_MAP,
    CleaningResult,
    clean_thai_text,
    coerce_numeric_column,
    handle_missing_values,
    normalize_currency,
    normalize_dates,
    remove_duplicate_rows,
)

# รูปแบบตรวจจับ "ดูเหมือนวันที่" — ไม่มี capture group (เลี่ยง UserWarning ของ pandas .str)
_DATELIKE_DETECT_RE = re.compile(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}")
_THAI_MONTH_DETECT_RE = re.compile("(?:" + "|".join(re.escape(k) for k in _THAI_MONTH_MAP) + ")")

# operation ที่ขับด้วย flag fix_encoding / fix_numerals (text-level)
_ENCODING_OPS = ("encoding", "zwspace", "whitespace", "unicode")
_NUMERAL_OPS = ("numerals",)

# คำใบ้ชื่อคอลัมน์ที่บ่งว่าเป็นวันที่ (ใช้คัดคอลัมน์สำหรับ normalize_dates)
_DATE_NAME_HINTS = (
    "date",
    "time",
    "วันที่",
    "ปี",
    "เดือน",
    "dob",
    "timestamp",
    "created",
    "updated",
)


@dataclass
class CleaningReport:
    """สรุปผลการทำความสะอาดทั้ง DataFrame — v2.0.

    Attributes:
        operations_run: รายการ CleaningResult ของ operation ที่กระทบข้อมูลจริง.
        rows_before: จำนวนแถวก่อนทำความสะอาด.
        rows_after: จำนวนแถวหลังทำความสะอาด (ต่างจาก before เมื่อลบแถวซ้ำ/ค่าว่าง).
        columns_affected: รายชื่อคอลัมน์ที่มีการเปลี่ยนแปลง.
        total_changes: จำนวนการเปลี่ยนแปลงรวม (ระดับเซลล์/แถว ไม่นับ downcast).
        warnings: คำเตือน (เช่น คอลัมน์ขาดข้อมูลสูง, กลไกค่าว่าง MNAR).
    """

    operations_run: list[CleaningResult]
    rows_before: int
    rows_after: int
    columns_affected: list[str]
    total_changes: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "operations_run": [r.to_dict() for r in self.operations_run],
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "columns_affected": self.columns_affected,
            "total_changes": self.total_changes,
            "warnings": self.warnings,
        }

    def to_json(self, path: str | None = None) -> str:
        """ส่งออกเป็น JSON — ถ้าระบุ path จะเขียนไฟล์ด้วย."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text

    def summary_th(self) -> str:
        """สรุปผลการทำความสะอาดเป็นภาษาไทย — v2.0."""
        lines = ["📋 สรุปการทำความสะอาดข้อมูล (ThaiEDA v2.0)"]
        lines.append(f"  แถว: {self.rows_before:,} → {self.rows_after:,}")
        lines.append(
            f"  คอลัมน์ที่เปลี่ยน: {len(self.columns_affected)} "
            f"({', '.join(self.columns_affected) if self.columns_affected else 'ไม่มี'})"
        )
        lines.append(f"  การเปลี่ยนแปลงรวม: {self.total_changes:,} จุด")
        if self.operations_run:
            lines.append("  รายการที่ทำ:")
            for r in self.operations_run:
                detail = r.description_th or r.operation
                lines.append(f"    • [{r.column}] {detail}")
        else:
            lines.append("  (ข้อมูลสะอาดอยู่แล้ว — ไม่มีการเปลี่ยนแปลง)")
        if self.warnings:
            lines.append("  ⚠️ คำเตือน:")
            for w in self.warnings:
                lines.append(f"    • {w}")
        return "\n".join(lines)


def clean(
    df: pd.DataFrame,
    *,
    operations: list[str] | None = None,
    handle_missing: str = "flag",
    remove_duplicates: bool = True,
    fix_dates: bool = True,
    fix_numerals: bool = True,
    fix_encoding: bool = True,
    downcast: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """ทำความสะอาด DataFrame ทั้งหมดในครั้งเดียว — v2.0.

    รวมการดำเนินการทั้งหมด: encoding, whitespace, numerals, dates,
    Buddhist era, currency, duplicates, missing values, dtype downcasting.

    Args:
        df: DataFrame ต้นฉบับ.
        operations: รายการ text operations เฉพาะ (None = ใช้ flag fix_encoding/fix_numerals).
        handle_missing: strategy สำหรับ missing values (flag/median/mode/drop/unknown/ml).
        remove_duplicates: ลบแถวซ้ำ.
        fix_dates: แปลงวันที่/พ.ศ. (เฉพาะคอลัมน์ข้อความที่ดูเหมือนวันที่).
        fix_numerals: แปลงเลขไทย ๐-๙ → 0-9.
        fix_encoding: แก้ encoding/zero-width/whitespace/unicode.
        downcast: ลด dtype เพื่อประหยัด memory (int64→int32, float64→float32, object→category).

    Returns:
        (cleaned_df, CleaningReport) — ตรวจสอบได้ว่าทำอะไรไปบ้าง.

    Raises:
        TypeError: ถ้า df ไม่ใช่ pandas DataFrame.
        ValueError: ถ้าระบุ operation ที่ไม่รู้จัก หรือ handle_missing ไม่รองรับ.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("thaieda.clean() ต้องรับ pandas DataFrame.")
    ensure_unique_column_names(df, context="thaieda.clean()")

    _valid_missing = {"flag", "median", "mode", "drop", "unknown", "ml"}
    if handle_missing not in _valid_missing:
        raise ValueError(
            f"handle_missing ไม่รองรับ: {handle_missing!r} — รองรับ {sorted(_valid_missing)}"
        )

    rows_before = len(df)
    out = df.copy()
    results: list[CleaningResult] = []
    warnings: list[str] = []

    # ----- 1. ทำความสะอาดคอลัมน์ข้อความ (text-level) -----
    text_ops = _build_text_ops(operations, fix_encoding=fix_encoding, fix_numerals=fix_numerals)
    object_cols = [c for c in out.columns if _is_text_column(out[c])]

    for col in object_cols:
        if text_ops:
            cleaned, op_results = clean_thai_text(out[col], text_ops)
            out[col] = cleaned
            results.extend(r for r in op_results if r.rows_affected > 0)

        # currency normalization (auto-detect >10% สัญลักษณ์สกุลเงิน)
        out[col], cur_res = normalize_currency(out[col])
        if cur_res.rows_affected > 0:
            results.append(cur_res)

        # dates — เฉพาะคอลัมน์ข้อความที่ยังเป็น object และ "ดูเหมือนวันที่"
        if fix_dates and _is_text_column(out[col]) and _is_datelike_column(out[col]):
            out[col], date_res = normalize_dates(out[col])
            if date_res.rows_affected > 0:
                results.append(date_res)

        # coerce numeric — แปลงคอลัมน์ข้อความที่ควรเป็นตัวเลข (เช่น เลขไทยที่ normalize แล้ว)
        if _is_text_column(out[col]):
            out[col], num_res = coerce_numeric_column(out[col])
            if num_res.rows_affected > 0:
                results.append(num_res)

    # ----- 2. ลบแถวซ้ำ (หลังทำความสะอาดค่า — เผยให้เห็น dup ที่ซ่อนด้วย encoding/ช่องว่าง) -----
    if remove_duplicates:
        out, dup_res = remove_duplicate_rows(out)
        if dup_res.rows_affected > 0:
            results.append(dup_res)

    # ----- 3. จัดการค่าว่าง -----
    if handle_missing == "drop":
        before = len(out)
        out = out.dropna().reset_index(drop=True)
        removed = before - len(out)
        if removed > 0:
            results.append(
                CleaningResult(
                    operation="handle_missing_values",
                    rows_affected=removed,
                    column="(entire df)",
                    description_th=f"ลบ {removed:,} แถวที่มีค่าว่าง (strategy=drop)",
                )
            )
    elif handle_missing == "ml":
        from thaieda.clean._impute import ml_impute

        out, ml_results, ml_warnings = ml_impute(out)
        results.extend(r for r in ml_results if r.rows_affected > 0)
        warnings.extend(ml_warnings)
    else:
        for col in out.columns:
            if not out[col].isna().any():
                continue
            out[col], miss_res = handle_missing_values(out[col], handle_missing)
            if miss_res.rows_affected > 0:
                results.append(miss_res)
            if "เตือน" in miss_res.description_th or "mostly_missing" in miss_res.description_th:
                warnings.append(miss_res.description_th)

    # ----- 4. dtype downcasting (ประหยัด memory) -----
    if downcast:
        from thaieda.io._downcast import downcast_dtypes

        out, dc_report = downcast_dtypes(out)
        if dc_report["n_columns_changed"] > 0:
            results.append(
                CleaningResult(
                    operation="downcast_dtypes",
                    rows_affected=dc_report["n_columns_changed"],
                    column="(entire df)",
                    description_th=(
                        f"ลด memory {dc_report['memory_before_mb']}"
                        f"→{dc_report['memory_after_mb']} MB "
                        f"(-{dc_report['reduction_pct']}%), "
                        f"ปรับ {dc_report['n_columns_changed']} คอลัมน์"
                    ),
                )
            )

    # ----- 5. สร้าง CleaningReport -----
    columns_affected = sorted({r.column for r in results if r.column not in ("(entire df)", "")})
    total_changes = sum(r.rows_affected for r in results if r.operation != "downcast_dtypes")

    report = CleaningReport(
        operations_run=results,
        rows_before=rows_before,
        rows_after=len(out),
        columns_affected=columns_affected,
        total_changes=total_changes,
        warnings=warnings,
    )
    return out, report


# ----------------------------------------------------------------------------
# helper
# ----------------------------------------------------------------------------
def _build_text_ops(
    operations: list[str] | None, *, fix_encoding: bool, fix_numerals: bool
) -> list[str]:
    """สร้างรายการ text operation — ถ้าผู้ใช้ระบุ operations ใช้ตามนั้น มิฉะนั้นสร้างจาก flag."""
    if operations is not None:
        return list(operations)
    ops: list[str] = []
    if fix_encoding:
        ops.extend(_ENCODING_OPS)
    if fix_numerals:
        ops.extend(_NUMERAL_OPS)
    return ops


def _is_text_column(series: pd.Series) -> bool:
    """คอลัมน์เป็นข้อความ (object/string) ที่ควรทำ text cleaning หรือไม่."""
    return pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)


def _is_datelike_column(series: pd.Series, sample: int = 200) -> bool:
    """คอลัมน์ข้อความ "ดูเหมือนวันที่" หรือไม่ — กันการแปลงเลขปีในคอลัมน์ราคา/จำนวน.

    ตัดสินจาก: ชื่อคอลัมน์มีคำใบ้วันที่ หรือ >50% ของค่าตรงรูปแบบวันที่/ชื่อเดือนไทย.
    """
    name = str(series.name or "").lower()
    if any(hint in name for hint in _DATE_NAME_HINTS):
        return True
    vals = series.dropna().astype(str)
    if vals.empty:
        return False
    if len(vals) > sample:
        vals = vals.head(sample)
    matches = vals.str.contains(_DATELIKE_DETECT_RE, na=False) | vals.str.contains(
        _THAI_MONTH_DETECT_RE, na=False
    )
    return float(matches.mean()) > 0.5


__all__ = ["CleaningReport", "clean"]
