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

    # แปลงคอลัมน์ข้อความเป็น str เพียงครั้งเดียว แล้วใช้ซ้ำในทุก check ด้านล่าง —
    # เดิมแต่ละ _count_* เรียก select_dtypes + .dropna().astype(str) ของตัวเอง ทำให้
    # คอลัมน์ข้อความเดียวกันถูกแปลงซ้ำ 6 รอบบน DataFrame ที่มีคอลัมน์ข้อความเยอะ
    text_series = _text_str_series(df)

    # (action, จำนวนที่ตรวจพบ) — เรียงลำดับเดิมไว้เพื่อความเสถียรของรายงาน
    checks: list[tuple[str, int]] = [
        ("encoding", _count_mojibake(text_series)),
        ("zwspace", _count_zwspace(text_series)),
        ("numerals", _count_thai_numerals(text_series)),
        ("whitespace", _count_extra_whitespace(text_series)),
        ("buddhist_era", _count_buddhist_era(text_series)),
        ("duplicates", int(df.duplicated().sum())),
        ("missing", _count_placeholders(text_series)),
    ]
    for action, count in checks:
        if count > 0:
            plan.actions.append(action)
            plan.details[action] = count
        else:
            plan.skipped.append(action)

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


def _text_str_series(df: pd.DataFrame) -> list[pd.Series]:
    """คืนรายการ Series ของคอลัมน์ข้อความ (object/string) ที่ dropna + astype(str) แล้ว.

    แปลงครั้งเดียวเพื่อให้ทุก _count_* ใช้ซ้ำได้ — เลี่ยงการ astype(str) คอลัมน์เดิมหลายรอบ.
    """
    text_cols = df.select_dtypes(include=["object", "string"])
    return [text_cols[col].dropna().astype(str) for col in text_cols.columns]


def _count_zwspace(series_list: list[pd.Series]) -> int:
    """นับจำนวน zero-width space ในคอลัมน์ object (vectorized)."""
    return int(
        sum(s.str.contains(_ZWSPACE_PATTERN, regex=True, na=False).sum() for s in series_list)
    )


def _count_thai_numerals(series_list: list[pd.Series]) -> int:
    """นับจำนวนเซลล์ที่มีเลขไทย ๐-๙ (vectorized)."""
    return int(sum(s.str.contains(r"[๐-๙]", regex=True, na=False).sum() for s in series_list))


def _count_extra_whitespace(series_list: list[pd.Series]) -> int:
    """นับเซลล์ที่มี whitespace ซ้ำหรือหน้าหลัง (vectorized)."""
    count = 0
    for s in series_list:
        # ใช้ logical OR ระหว่างสองเงื่อนไขเพื่อป้องกันการนับซ้ำในเซลล์เดียวกัน
        mask = s.str.contains(r"  +", regex=True, na=False) | s.str.contains(
            r"^\s|\s$", regex=True, na=False
        )
        count += mask.sum()
    return int(count)


def _count_mojibake(series_list: list[pd.Series]) -> int:
    """นับเซลล์ที่มี mojibake (replacement char, Ã, Â¸, เป็นต้น) (vectorized)."""
    return int(
        sum(s.str.contains(_MOJIBAKE_PATTERN, regex=True, na=False).sum() for s in series_list)
    )


def _count_buddhist_era(series_list: list[pd.Series]) -> int:
    """นับเซลล์ที่มีปี พ.ศ. (> 2400) ในคอลัมน์ object (vectorized)."""
    # ปี พ.ศ. มักอยู่ในช่วง 2400-2699
    return int(
        sum(s.str.contains(r"\b2[4-6]\d{2}\b", regex=True, na=False).sum() for s in series_list)
    )


def _count_placeholders(series_list: list[pd.Series]) -> int:
    """นับเซลล์ที่มี placeholder values (vectorized)."""
    return int(sum(s.isin(_PLACEHOLDERS).sum() for s in series_list))


__all__ = ["CleaningPlan", "plan_cleaning"]
