"""Smart cleaning — ตัดสินใจอัตโนมัติว่าควรทำความสะอาดอะไรบ้าง (v1.1).

แทนที่การเรียก clean ทุกฟังก์ชันแบบเดิม โมดูลนี้ตรวจข้อมูลก่อนแล้วเลือกเฉพาะ
การทำความสะอาดที่จำเป็นจริง ๆ — ลดเวลาประมวลผลและกัน side-effect ที่ไม่ต้องการ

หลักการ:
  * ตรวจข้อมูลก่อน — ไม่ทำอะไรเลยถ้าไม่จำเป็น
  * แต่ละ check คืน bool + จำนวนที่เจอ (เพื่อ report)
  * ผู้ใช้สามารถ override ได้ด้วย force=True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd


# ----------------------------------------------------------------------------
# ผลลัพธ์การตรวจสอบ smart cleaning
# ----------------------------------------------------------------------------
@dataclass
class CleaningPlan:
    """แผนการทำความสะอาดที่ smart cleaning ตัดสินใจได้.

    Attributes:
        actions: รายการการทำความสะอาดที่แนะนำ (เช่น "encoding", "zwspace", "numerals")
        skipped: รายการที่ข้ามเพราะไม่จำเป็น
        details: รายละเอียดแต่ละ action (เช่น {"zwspace": 15} = เจอ 15 ตัว)
    """

    actions: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    details: dict[str, int] = field(default_factory=dict)

    @property
    def has_actions(self) -> bool:
        """มีการทำความสะอาดที่ต้องทำหรือไม่."""
        return len(self.actions) > 0


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def plan_cleaning(df: pd.DataFrame) -> CleaningPlan:
    """ตรวจข้อมูลแล้วตัดสินใจว่าควรทำความสะอาดอะไรบ้าง.

    ตรวจ:
      * encoding — มี mojibake หรือไม่ (TIS-620 ผิด, replacement char)
      * zwspace — มี zero-width space หรือไม่
      * numerals — มีเลขไทย (๐-๙) ในคอลัมน์ object หรือไม่
      * whitespace — มี whitespace ซ้ำหรือระหว่างข้อความหรือไม่
      * buddhist_era — มีปี พ.ศ. (> 2400) ในคอลัมน์วันที่หรือไม่
      * duplicates — มีแถวซ้ำหรือไม่
      * missing — มีค่าว่างที่ placeholder (เช่น '-', 'N/A') หรือไม่

    Args:
        df: DataFrame ที่จะตรวจ.

    Returns:
        CleaningPlan — แผนการทำความสะอาด.
    """
    plan = CleaningPlan()

    # 1. encoding — ตรวจ mojibake (replacement char, TIS-620 artifacts)
    encoding_count = _count_mojibake(df)
    if encoding_count > 0:
        plan.actions.append("encoding")
        plan.details["encoding"] = encoding_count
    else:
        plan.skipped.append("encoding")

    # 2. zwspace — ตรวจ zero-width space (\u200b, \u200c, \u200d, \ufeff)
    zw_count = _count_zwspace(df)
    if zw_count > 0:
        plan.actions.append("zwspace")
        plan.details["zwspace"] = zw_count
    else:
        plan.skipped.append("zwspace")

    # 3. numerals — ตรวจเลขไทย ๐-๙ ในคอลัมน์ object
    numeral_count = _count_thai_numerals(df)
    if numeral_count > 0:
        plan.actions.append("numerals")
        plan.details["numerals"] = numeral_count
    else:
        plan.skipped.append("numerals")

    # 4. whitespace — ตรวจ whitespace ซ้ำหรือหน้าหลัง
    ws_count = _count_extra_whitespace(df)
    if ws_count > 0:
        plan.actions.append("whitespace")
        plan.details["whitespace"] = ws_count
    else:
        plan.skipped.append("whitespace")

    # 5. buddhist_era — ตรวจปี พ.ศ. ในคอลัมน์ที่ดูเหมือนวันที่
    be_count = _count_buddhist_era(df)
    if be_count > 0:
        plan.actions.append("buddhist_era")
        plan.details["buddhist_era"] = be_count
    else:
        plan.skipped.append("buddhist_era")

    # 6. duplicates — ตรวจแถวซ้ำ
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        plan.actions.append("duplicates")
        plan.details["duplicates"] = dup_count
    else:
        plan.skipped.append("duplicates")

    # 7. missing placeholders — ตรวจ placeholder values
    placeholder_count = _count_placeholders(df)
    if placeholder_count > 0:
        plan.actions.append("missing")
        plan.details["missing"] = placeholder_count
    else:
        plan.skipped.append("missing")

    return plan


# ----------------------------------------------------------------------------
# helper — แต่ละการตรวจสอบ (vectorized)
# ----------------------------------------------------------------------------
_ZWSPACES = {"\u200b", "\u200c", "\u200d", "\ufeff"}
_ZWSPACE_PATTERN = "[" + "".join(_ZWSPACES) + "]"
_THAI_NUMERALS = "๐๑๒๓๔๕๖๗๘๙"
_PLACEHOLDERS = {"-", "N/A", "n/a", "NA", "ไม่มี", "ไม่มีข้อมูล", "—", "?"}
_MOJIBAKE_PATTERNS = ["Ã", "Â¸", "Ã©", "Ã§", "â€", "\ufffd"]
_MOJIBAKE_PATTERN = "|".join(re.escape(p) for p in _MOJIBAKE_PATTERNS)


def _count_zwspace(df: pd.DataFrame) -> int:
    """นับจำนวน zero-width space ในคอลัมน์ object (vectorized)."""
    text_cols = df.select_dtypes(include=["object", "string"])
    if text_cols.empty:
        return 0
    count = 0
    for col in text_cols.columns:
        s = text_cols[col].dropna().astype(str)
        count += s.str.contains(_ZWSPACE_PATTERN, regex=True, na=False).sum()
    return int(count)


def _count_thai_numerals(df: pd.DataFrame) -> int:
    """นับจำนวนเซลล์ที่มีเลขไทย ๐-๙ (vectorized)."""
    text_cols = df.select_dtypes(include=["object", "string"])
    if text_cols.empty:
        return 0
    count = 0
    for col in text_cols.columns:
        s = text_cols[col].dropna().astype(str)
        count += s.str.contains(r"[๐-๙]", regex=True, na=False).sum()
    return int(count)


def _count_extra_whitespace(df: pd.DataFrame) -> int:
    """นับเซลล์ที่มี whitespace ซ้ำหรือหน้าหลัง (vectorized)."""
    text_cols = df.select_dtypes(include=["object", "string"])
    if text_cols.empty:
        return 0
    count = 0
    for col in text_cols.columns:
        s = text_cols[col].dropna().astype(str)
        # whitespace ซ้ำ (2 ขึ้นไป)
        count += s.str.contains(r"  +", regex=True, na=False).sum()
        # หน้าหรือหลัง whitespace
        count += s.str.contains(r"^\s|\s$", regex=True, na=False).sum()
    return int(count)


def _count_mojibake(df: pd.DataFrame) -> int:
    """นับเซลล์ที่มี mojibake (replacement char, Ã, Â¸, เป็นต้น) (vectorized)."""
    text_cols = df.select_dtypes(include=["object", "string"])
    if text_cols.empty:
        return 0
    count = 0
    for col in text_cols.columns:
        s = text_cols[col].dropna().astype(str)
        count += s.str.contains(_MOJIBAKE_PATTERN, regex=True, na=False).sum()
    return int(count)


def _count_buddhist_era(df: pd.DataFrame) -> int:
    """นับเซลล์ที่มีปี พ.ศ. (> 2400) ในคอลัมน์ object (vectorized)."""
    text_cols = df.select_dtypes(include=["object", "string"])
    if text_cols.empty:
        return 0
    count = 0
    for col in text_cols.columns:
        s = text_cols[col].dropna().astype(str)
        # ปี พ.ศ. มักอยู่ในช่วง 2400-2699
        count += s.str.contains(r"\b2[4-6]\d{2}\b", regex=True, na=False).sum()
    return int(count)


def _count_placeholders(df: pd.DataFrame) -> int:
    """นับเซลล์ที่มี placeholder values (vectorized)."""
    text_cols = df.select_dtypes(include=["object", "string"])
    if text_cols.empty:
        return 0
    count = 0
    for col in text_cols.columns:
        s = text_cols[col].dropna().astype(str)
        count += s.isin(_PLACEHOLDERS).sum()
    return int(count)


__all__ = ["CleaningPlan", "plan_cleaning"]
