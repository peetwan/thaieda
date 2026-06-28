"""Thai data quality checks — หัวใจของ ThaiEDA (the moat).

ตรวจจับปัญหาคุณภาพข้อมูลที่เครื่องมือ EDA แบบ English-centric มองข้าม เช่น
ปี พ.ศ. ปนกับ ค.ศ., เลขไทย, อักขระล่องหน, การ normalize ผิด ฯลฯ
"""

from __future__ import annotations

import contextlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from thaieda.detect import (
    _THAI_PHONE_RE,
    ColumnType,
    _clean_phone_str,
    _detect_language,
    _name_hints_id,
    script_ratio,
)
from thaieda.detect import (
    _looks_like_phone_column as _detect_looks_like_phone_column,
)
from thaieda.quality._thai_id import check_thai_id, validate_thai_id, validate_thai_id_column

# ----------------------------------------------------------------------------
# ค่าคงที่
# ----------------------------------------------------------------------------
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# ช่วงปี พ.ศ. ที่เป็นไปได้ (≈ ค.ศ. 1900–2056)
_BE_MIN, _BE_MAX = 2440, 2599
_DATE_YEAR_COLUMN_HINTS = ("date", "year", "yr", "ปี")

# อักขระล่องหน/ความกว้างศูนย์ที่พบบ่อยในข้อความไทยบนเว็บ
_ZERO_WIDTH_CHARS = {
    "​": "ZERO WIDTH SPACE (U+200B)",
    "‌": "ZERO WIDTH NON-JOINER (U+200C)",
    "‍": "ZERO WIDTH JOINER (U+200D)",
    "﻿": "BYTE ORDER MARK (U+FEFF)",
    "⁠": "WORD JOINER (U+2060)",
}

# เลขไทย
_THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
_THAI_DIGIT_SET = set(_THAI_DIGITS)
_THAI_TO_ARABIC = str.maketrans(_THAI_DIGITS, "0123456789")

# วรรณยุกต์และสระไทยสำหรับตรวจ normalization
_THAI_TONE_MARKS = "่้๊๋"  # ไม้เอก โท ตรี จัตวา
_THAI_UPPER_VOWELS = "ัิีึืุู็ํ"  # สระบน/ล่างที่ซ้ำกันไม่ได้

# สระ/วรรณยุกต์ที่ต้องตามหลังพยัญชนะ (combining marks)
_THAI_COMBINING = set("ัิีึืุู็่้๊๋์ํฺ")
_THAI_CONSONANTS = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")

# ----------------------------------------------------------------------------
# regex สำหรับตรวจแบบ vectorized (.str.contains / .str.count)
# ----------------------------------------------------------------------------
# หมายเหตุ: เรียก .str accessor บน Series ที่ cast เป็น object dtype (ใช้ engine ของ
# Python re) เพราะ regex บางตัวมี backreference/lookbehind ที่ Arrow (RE2) ไม่รองรับ
# — ให้ผลตรงกับ re.search() เดิมทุกประการ และยังเร็วกว่าการวน Python ทีละอักขระมาก
_THAI_DIGIT_RE = re.compile(f"[{_THAI_DIGITS}]")
_ARABIC_DIGIT_RE = re.compile(r"[0-9]")
_ZW_CLASS_RE = re.compile("[" + "".join(_ZERO_WIDTH_CHARS) + "]")
_FULLWIDTH_RE = re.compile("[！-～]")  # อักขระ full-width U+FF01–U+FF5E
_THAI_BLOCK_RE = re.compile("[฀-๿]")  # บล็อกอักษรไทย (รวมเลขไทย)
# combining mark ที่ขึ้นต้น หรือตามหลังอักขระที่ไม่ใช่พยัญชนะ/combining (orphan)
# lookbehind ที่ start-of-string ก็เป็นจริง → จับ combining ที่ขึ้นต้นด้วย
_THAI_BASE_CHARS = "".join(sorted(_THAI_CONSONANTS | _THAI_COMBINING))
_THAI_COMBINING_CHARS = "".join(sorted(_THAI_COMBINING))
_COMBINING_ORDER_RE = re.compile(f"(?<![{_THAI_BASE_CHARS}])[{_THAI_COMBINING_CHARS}]")

# non-breaking space
_NBSP = " "


# ----------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ----------------------------------------------------------------------------
@dataclass
class QualityIssue:
    """ปัญหาคุณภาพข้อมูลหนึ่งรายการ (มีคำอธิบายทั้งไทยและอังกฤษ)."""

    check_name: str
    severity: str  # "critical" | "warning" | "info"
    column: str
    count: int
    percentage: float
    description: str
    description_th: str
    examples: list[str] = field(default_factory=list)
    suggestion: str = ""
    suggestion_th: str = ""

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "severity": self.severity,
            "column": self.column,
            "count": self.count,
            "percentage": round(self.percentage, 2),
            "description": self.description,
            "description_th": self.description_th,
            "examples": self.examples,
            "suggestion": self.suggestion,
            "suggestion_th": self.suggestion_th,
        }


# ----------------------------------------------------------------------------
# helper
# ----------------------------------------------------------------------------
def _non_null_str(series: pd.Series) -> pd.Series:
    """คืน Series ของค่าที่ไม่ว่าง แปลงเป็น str."""
    return series.dropna().astype(str)


def _pct(count: int, total: int) -> float:
    return (count / total * 100.0) if total else 0.0


def _visible_repr(text: str) -> str:
    """แสดงอักขระล่องหนให้มองเห็นได้ผ่าน repr (เช่น '\\u200b')."""
    return repr(text)


def _is_date_year_column(column: str) -> bool:
    """True ถ้าชื่อคอลัมน์สื่อว่าเป็นวันที่/ปี เพื่อกัน false positive จาก ID."""
    normalized = str(column).strip().lower()
    return any(hint in normalized for hint in _DATE_YEAR_COLUMN_HINTS)


# ----------------------------------------------------------------------------
# (a) Buddhist Era detection
# ----------------------------------------------------------------------------
def check_buddhist_era(
    series: pd.Series, column: str, *, allow_non_date_name: bool = False
) -> QualityIssue | None:
    """ตรวจหาเลขปี พ.ศ. (2440–2599) ที่อาจปนกับ ค.ศ. ในคอลัมน์ตัวเลข/วันที่ — v0.8 vectorized.

    v1.1 fix: กรองค่าที่ไม่ใช่รูปแบบวันที่ออก — กัน false positive จากตัวเลขทั่วไป
    เช่น ticket number 21171, room count 2401 ที่ตรงช่วง พ.ศ. แต่ไม่ใช่ปี
    """
    values = _non_null_str(series)
    if len(values) == 0:
        return None
    if not allow_non_date_name and not _is_date_year_column(column):
        return None

    # v1.1: กรองเฉพาะค่าที่มีรูปแบบวันที่ (มี - หรือ / คั่น และมี 4 หลัก)
    # กัน false positive จากตัวเลขทั่วไป เช่น 21171, 7099
    # แต่ถ้าค่าเป็นตัวเลขล้วน (int/float → string) ให้ตรวจเฉพาะช่วง 2400-2700
    date_like_re = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b")
    year_only_re = re.compile(r"^\d{4}$")  # ค่าที่เป็นเลข 4 หลักล้วน (เช่น 2567)

    date_like_values = []
    year_only_values = []
    ce_year_values = []  # CE years (1900-2100) ที่ไม่ใช่ พ.ศ.
    for v in values:
        if date_like_re.search(v):
            date_like_values.append(v)
        elif year_only_re.match(v.strip()):
            year_int = int(v.strip())
            if 2400 <= year_int <= 2700:
                year_only_values.append(v)
            elif 1900 <= year_int <= 2100:
                ce_year_values.append(v)

    # ถ้ามี date-like values ให้ใช้เฉพาะ那些
    # ถ้าไม่มี date-like แต่มี year-only ให้ใช้ year-only (รองรับคอลัมน์ปีแบบ int)
    check_values = date_like_values if date_like_values else year_only_values
    if len(check_values) == 0:
        return None

    year_re = re.compile(r"\b(\d{4})\b")
    # ตรวจ CE จากทุกค่า (รวม date strings เช่น 2024-02-20) เพื่อตัดสิน mixed → critical
    ce_seen = len(ce_year_values) > 0
    if not ce_seen:
        for v in values:
            for m in year_re.finditer(v):
                year = int(m.group(1))
                if 1900 <= year <= 2100:
                    ce_seen = True
                    break
            if ce_seen:
                break

    be_examples: list[str] = []

    # v0.8: vectorize — ใช้ .str.extractall แทนการวนลูป
    try:
        # แปลงเป็น string Series แล้ว extractall (ใช้ check_values เท่านั้น)
        s = pd.Series(check_values)
        extracted = s.str.extractall(year_re)
        if extracted.empty:
            return None
        years = pd.to_numeric(extracted[0], errors="coerce").dropna()
        if years.between(1900, 2100).any():
            ce_seen = True
        be_match_mask = (years >= _BE_MIN) & (years <= _BE_MAX)
        if not be_match_mask.any():
            return None
        # นับ "rows ที่มี พ.ศ. อย่างน้อย 1 ค่า" ไม่ใช่ "จำนวน match" — กัน percentage >100%
        be_year_vals = years[be_match_mask].unique()
        be_rows_mask = s.apply(lambda v: any(str(int(y)) in v for y in be_year_vals))
        be_count = int(be_rows_mask.sum())
        if be_count == 0:
            return None
        # หา examples — ดึง rows ที่มี พ.ศ.
        for v in check_values:
            if any(str(int(y)) in v for y in be_year_vals[:5]) and len(be_examples) < 5:
                be_examples.append(v)
    except Exception:  # noqa: BLE001 — fallback ถ้า extractall พัง
        # วิธีเดิม (row-by-row) เป็น fallback
        be_count = 0
        for v in values:
            found_be = False
            for m in year_re.finditer(v):
                year = int(m.group(1))
                if _BE_MIN <= year <= _BE_MAX:
                    found_be = True
                elif 1900 <= year <= 2100:
                    ce_seen = True
            if found_be:
                be_count += 1
                if len(be_examples) < 5:
                    be_examples.append(v)
        if be_count == 0:
            return None

    all_year_values = date_like_values + year_only_values
    total = len(all_year_values) + len(ce_year_values)
    # critical ถ้าปนกัน (ทั้ง พ.ศ. และ ค.ศ.), warning ถ้าเป็น พ.ศ. ล้วน
    mixed = ce_seen and be_count > 0 and ce_seen
    severity = "critical" if mixed else "warning"

    return QualityIssue(
        check_name="buddhist_era",
        severity=severity,
        column=column,
        count=be_count,
        percentage=_pct(be_count, total),
        description=(
            "Column may contain Buddhist Era (พ.ศ.) years"
            + (" mixed with CE years" if mixed else "")
            + ". Values in 2440–2599 look like B.E. years."
        ),
        description_th=(
            "คอลัมน์อาจมีปีพุทธศักราช (พ.ศ.)"
            + ("ปนกับคริสต์ศักราช (ค.ศ.)" if mixed else "")
            + " ค่าในช่วง 2440–2599 มีลักษณะเป็นปี พ.ศ."
        ),
        examples=be_examples,
        suggestion="Standardize all years to a single era (subtract 543 to convert B.E. to C.E.).",
        suggestion_th="ทำให้ปีเป็นศักราชเดียวกันทั้งหมด (ลบ 543 เพื่อแปลง พ.ศ. เป็น ค.ศ.)",
    )


# ----------------------------------------------------------------------------
# (b) Thai numeral detection
# ----------------------------------------------------------------------------
def check_thai_numerals(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจหาเลขไทย ๐–๙ โดยเฉพาะเมื่อปนกับเลขอารบิกในคอลัมน์เดียวกัน."""
    values = _non_null_str(series)
    if len(values) == 0:
        return None

    # vectorized: ตรวจเลขไทยต่อเซลล์ผ่าน .str.contains (object dtype = Python re engine)
    obj = values.astype(object)
    has_thai = obj.str.contains(_THAI_DIGIT_RE, regex=True, na=False)
    thai_num_count = int(has_thai.sum())
    if thai_num_count == 0:
        return None

    # มีเลขอารบิก (0–9) ปนในคอลัมน์ไหม → ใช้กำหนดความรุนแรง (warning เมื่อปน)
    arabic_seen = bool(obj.str.contains(_ARABIC_DIGIT_RE, regex=True, na=False).any())
    examples = list(obj[has_thai].head(5))

    total = len(values)
    severity = "warning" if arabic_seen else "info"

    return QualityIssue(
        check_name="thai_numerals",
        severity=severity,
        column=column,
        count=thai_num_count,
        percentage=_pct(thai_num_count, total),
        description=(
            "Thai numerals (๐–๙) found"
            + (" mixed with Arabic numerals" if arabic_seen else "")
            + ". This breaks numeric parsing and sorting."
        ),
        description_th=(
            "พบเลขไทย (๐–๙)"
            + ("ปนกับเลขอารบิก" if arabic_seen else "")
            + " ซึ่งทำให้การแปลงเป็นตัวเลขและการเรียงลำดับผิดพลาด"
        ),
        examples=examples,
        suggestion="Normalize Thai digits to Arabic digits for consistent numeric handling.",
        suggestion_th="แปลงเลขไทยเป็นเลขอารบิกเพื่อให้จัดการตัวเลขได้สม่ำเสมอ",
    )


# ----------------------------------------------------------------------------
# (c) Zero-width / invisible character detection
# ----------------------------------------------------------------------------
def check_zero_width(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจหาอักขระความกว้างศูนย์/ล่องหน ที่ทำให้ groupby/join พังแบบเงียบ ๆ."""
    values = _non_null_str(series)
    if len(values) == 0:
        return None

    # vectorized: ตรวจอักขระล่องหนต่อเซลล์ผ่าน .str.contains (object dtype = Python re)
    obj = values.astype(object)
    mask = obj.str.contains(_ZW_CLASS_RE, regex=True, na=False)
    count = int(mask.sum())
    if count == 0:
        return None

    # หาว่ามีอักขระล่องหนตัวใดบ้าง (เพื่อแสดงชื่อ) — เช็คทีละตัวแบบ vectorized
    found_chars = {c for c in _ZERO_WIDTH_CHARS if obj.str.contains(c, regex=False, na=False).any()}
    examples = [_visible_repr(v) for v in obj[mask].head(5)]

    total = len(values)
    char_names = ", ".join(sorted(_ZERO_WIDTH_CHARS[c] for c in found_chars))

    return QualityIssue(
        check_name="zero_width_chars",
        severity="critical",
        column=column,
        count=count,
        percentage=_pct(count, total),
        description=(
            f"Invisible/zero-width characters found ({char_names}). "
            "These silently break groupby, joins, and deduplication."
        ),
        description_th=(
            f"พบอักขระล่องหน/ความกว้างศูนย์ ({char_names}) "
            "ซึ่งทำให้ groupby, join และการตัดข้อมูลซ้ำผิดพลาดแบบเงียบ ๆ"
        ),
        examples=examples,
        suggestion=(
            "Strip these characters (e.g. re.sub('[\\u200b-\\u200d\\ufeff\\u2060]', '', s))."
        ),
        suggestion_th="ลบอักขระเหล่านี้ออกก่อนประมวลผล (เช่น ใช้ regex แทนที่ด้วยค่าว่าง)",
    )


# ----------------------------------------------------------------------------
# (d) Script composition per column
# ----------------------------------------------------------------------------
def _aggregate_script_ratio(values: pd.Series) -> dict[str, float]:
    """หาสัดส่วนสคริปต์รวมของทั้งคอลัมน์ (ถ่วงน้ำหนักตามจำนวนอักขระ)."""
    totals = {
        "thai": 0.0,
        "latin": 0.0,
        "digit": 0.0,
        "thai_digit": 0.0,
        "whitespace": 0.0,
        "emoji": 0.0,
        "other": 0.0,
    }
    total_chars = 0
    for v in values:
        s = str(v)
        n = len(s)
        if n == 0:
            continue
        r = script_ratio(s)
        for k in totals:
            totals[k] += r[k] * n
        total_chars += n
    if total_chars == 0:
        return totals
    return {k: totals[k] / total_chars for k in totals}


def check_script_composition(
    series: pd.Series, column: str, expected_thai: bool
) -> QualityIssue | None:
    """รายงานสัดส่วนสคริปต์รวม และเตือนถ้าคอลัมน์ที่คาดว่าเป็นไทยมีอักษรไทย <30%."""
    values = _non_null_str(series)
    if len(values) == 0:
        return None

    # vectorized: thai_total = สัดส่วนอักขระในบล็อกไทย (รวมเลขไทย) ต่ออักขระทั้งหมด
    # เทียบเท่า _aggregate_script_ratio()["thai"] + ["thai_digit"] แต่ไม่วนทีละอักขระ
    # (อักขระไทยทุกตัวอยู่ในช่วง U+0E00–U+0E7F และไม่มีตัวใดถูกนับเป็น emoji)
    obj = values.astype(object)
    thai_chars = int(obj.str.count(_THAI_BLOCK_RE).sum())
    total_chars = int(obj.str.len().sum())
    thai_total = (thai_chars / total_chars) if total_chars else 0.0
    total = len(values)

    # เตือนเฉพาะกรณีที่ "ควรเป็นไทย" แต่อักษรไทยน้อย (mislabeled)
    if expected_thai and thai_total < 0.30:
        examples = list(values.head(5))
        return QualityIssue(
            check_name="mislabeled_thai_column",
            severity="warning",
            column=column,
            count=total,
            percentage=thai_total * 100.0,
            description=(
                f"Column expected to be Thai but only {thai_total * 100:.1f}% Thai script. "
                "It may be mislabeled or contain mostly non-Thai content."
            ),
            description_th=(
                f"คอลัมน์ที่คาดว่าเป็นภาษาไทย แต่มีอักษรไทยเพียง {thai_total * 100:.1f}% "
                "อาจติดป้ายผิดหรือมีเนื้อหาที่ไม่ใช่ไทยเป็นส่วนใหญ่"
            ),
            examples=examples,
            suggestion="Verify the column language; re-run detection or relabel it.",
            suggestion_th="ตรวจสอบภาษาของคอลัมน์ และจำแนกประเภทใหม่หรือเปลี่ยนป้ายกำกับ",
        )
    return None


# ----------------------------------------------------------------------------
# (e) Normalization issues
# ----------------------------------------------------------------------------
_DUP_TONE_RE = re.compile(f"[{_THAI_TONE_MARKS}]{{2,}}")
_DUP_VOWEL_RE = re.compile(f"[{_THAI_UPPER_VOWELS}]{{2,}}")
# AC-6 (grapheme validation): พยัญชนะ + วรรณยุกต์ซ้อนกัน 2 ตัวขึ้นไป (เช่น ก่้ = ไม้เอก+ไม้โท)
# ภาษาไทยอนุญาตวรรณยุกต์ได้เพียงตัวเดียวต่อพยางค์ — การมีหลายวรรณยุกต์บนพยัญชนะตัวเดียว
# เป็น grapheme ที่ผิดหลักภาษา (มักเกิดจากการพิมพ์ผิด/วาง combining mark ผิด) — REPORT-ONLY
_MULTI_TONE_ON_BASE_RE = re.compile(r"[ก-ฮ][่้๊๋]{2,}")
# อักขระเดียวกันซ้ำ 3+ ครั้ง (เช่น 5555, ๆๆๆ, อืมมม)
_REPEAT_SPAM_RE = re.compile(r"(.)\1{2,}")
_THAI_LAUGHTER_RE = re.compile(r"^5{3,}$")
# v1.x (C2): รหัส/ID แบบ alphanumeric คั่นด้วย hyphen เช่น "FUR-BO-10001798",
# "CA-2017-152156" — เป็น code ปกติ ไม่ใช่ repeated-char spam (ไม่มีช่องว่าง = ไม่ใช่ข้อความธรรมชาติ)
_ID_CODE_RE = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+$")


def _has_repeated_char_spam(text: str) -> bool:
    """True ถ้ามี run อักขระซ้ำที่บ่งชี้สแปม/การยืดเสียง (elongation) จริง.

    แยกเกณฑ์ตามชนิดอักขระเพื่อลด false positive บนข้อมูลสะอาด:
    - ตัวเลขซ้ำ = ส่วนหนึ่งของจำนวน/ปี/รหัส/ไปรษณีย์/timestamp (เช่น 2000, 10000,
      ".000" ในเวลา ISO) ไม่ใช่สแปม — ยกเว้นเลข '5' ซ้ำ (หัวเราะแบบไทย 555).
    - ตัวอักษรในคำ (ไทย/ละติน) ซ้ำ 3 ตัวเกิดได้ตามขอบเขตคำ/พยางค์ (เช่น ถนน+นคร→"นนน",
      เลขโรมัน "iii") จึง flag เฉพาะ 4 ตัวขึ้นไป — สอดคล้องกับ fix_repeated_chars
      ที่ยอมให้ซ้ำได้ถึง max_repeat=3.
    - อักขระอื่น (ไม้ยมก ๆ, เครื่องหมายวรรคตอน ฯลฯ) ซ้ำ 3+ = สแปมเสมอ.
    """
    for match in _REPEAT_SPAM_RE.finditer(text):
        char = match.group(1)
        run_len = len(match.group(0))
        category = unicodedata.category(char)
        if category == "Nd":  # ตัวเลข — เป็นส่วนของจำนวน ไม่ใช่สแปม (ยกเว้น 555 = หัวเราะ)
            if char == "5":
                return True
            continue
        if category in ("Lo", "Ll", "Lu", "Lt"):  # ตัวอักษรในคำ — flag เฉพาะ 4+ ตัว
            if run_len >= 4:
                return True
            continue
        return True  # ไม้ยมก/เครื่องหมาย/สัญลักษณ์ ซ้ำ 3+ = สแปม
    return False


def _skip_repeated_spam_check(text: str) -> bool:
    """ข้าม repeated-char spam ใน code/category/ID ที่ไม่ใช่ข้อความธรรมชาติ.

    ข้ามเมื่อ:
    - เป็น code/category สั้น (<15 ตัว) ที่มีตัวเลขปน เช่น Ticket/Cabin, หรือ
    - เป็นรหัส ID แบบ alphanumeric คั่นด้วย hyphen ที่มีตัวเลข >=3 ตัว
      เช่น "FUR-BO-10001798", "CA-2017-152156" (เป็นรหัส ไม่ใช่สแปม)
    ไม่ข้าม Thai laughter (5555) เพื่อให้ยัง flag เป็นสแปมได้.
    """
    stripped = text.strip()
    if _THAI_LAUGHTER_RE.match(stripped):
        return False
    # code/category สั้นที่มีเลขปน
    if len(stripped) < 15 and any(ch.isdigit() for ch in stripped):
        return True
    # รหัส ID: alphanumeric + hyphen + ตัวเลข >=3 ตัว
    return bool(_ID_CODE_RE.match(stripped)) and sum(ch.isdigit() for ch in stripped) >= 3


def _has_combining_order_issue(text: str) -> bool:
    """True ถ้ามี combining mark ของไทยที่ขึ้นต้น หรือมาหลังอักขระที่ไม่ใช่พยัญชนะ/สระ.

    ใช้ regex lookbehind (_COMBINING_ORDER_RE) แทนการวนทีละอักขระ — ให้ผลเท่ากันทุกกรณี:
    lookbehind ที่ start-of-string เป็นจริง = combining ขึ้นต้น; ตามหลัง non-base = orphan
    """
    return bool(_COMBINING_ORDER_RE.search(text))


def check_normalization(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจปัญหา normalization: วรรณยุกต์/สระซ้ำ, ลำดับ combining ผิด, อักขระสแปม, full-width."""
    values = _non_null_str(series)
    if len(values) == 0:
        return None

    # vectorized: คำนวณ mask ของแต่ละปัญหาแบบขนาน (object dtype = Python re engine
    # รองรับ backreference ของ _REPEAT_SPAM_RE และ lookbehind ของ _COMBINING_ORDER_RE)
    obj = values.astype(object)
    dup_tone = obj.str.contains(_DUP_TONE_RE, regex=True, na=False)
    dup_vowel = obj.str.contains(_DUP_VOWEL_RE, regex=True, na=False)
    comb_order = obj.str.contains(_COMBINING_ORDER_RE, regex=True, na=False)
    # AC-6: วรรณยุกต์หลายตัวบนพยัญชนะตัวเดียว (grapheme ผิดหลัก เช่น ก่้) — REPORT-ONLY
    multi_tone = obj.str.contains(_MULTI_TONE_ON_BASE_RE, regex=True, na=False)
    # repeated-char spam: ตรวจ run อักขระซ้ำแบบแยกเกณฑ์ตามชนิดอักขระ (ดู _has_repeated_char_spam)
    # เพื่อไม่ flag ตัวเลขในจำนวน/timestamp หรือพยัญชนะซ้ำ 3 ตัวตามขอบเขตคำที่ถูกต้อง
    # แล้วยังข้าม code/ID ตามกฎ _skip_repeated_spam_check เดิม
    repeat_raw = obj.map(_has_repeated_char_spam)
    skip_spam = obj.map(_skip_repeated_spam_check).astype(bool)
    repeat_spam = repeat_raw & ~skip_spam
    # full-width: NFKC เปลี่ยนค่า "และ" มีอักขระ full-width จริง
    # คำนวณ NFKC เฉพาะแถวที่มีอักขระ full-width (รายอื่นเป็น False อยู่แล้ว) เพื่อความเร็ว
    has_fw = obj.str.contains(_FULLWIDTH_RE, regex=True, na=False)
    full_width = pd.Series(False, index=obj.index)
    if has_fw.any():
        fw = obj[has_fw]
        full_width.loc[fw.index] = fw.map(lambda v: v != unicodedata.normalize("NFKC", v)).astype(
            bool
        )

    any_problem = dup_tone | dup_vowel | comb_order | repeat_spam | full_width | multi_tone
    count = int(any_problem.sum())
    if count == 0:
        return None

    # reasons: ปัญหาชนิดใดที่พบใน "แถวที่ถูกนับ" บ้าง — mask True ใด ๆ = แถวนั้นถูกนับด้วย
    reasons: set[str] = set()
    if dup_tone.any():
        reasons.add("duplicate tone marks")
    if dup_vowel.any():
        reasons.add("duplicate vowels")
    if comb_order.any():
        reasons.add("combining order")
    if repeat_spam.any():
        reasons.add("repeated-char spam")
    if full_width.any():
        reasons.add("full-width characters")
    if multi_tone.any():
        reasons.add("multiple tone marks on one base")

    examples = [v if len(v) <= 60 else v[:57] + "..." for v in obj[any_problem].head(5)]

    total = len(values)
    reason_str = ", ".join(sorted(reasons))

    return QualityIssue(
        check_name="normalization",
        severity="warning",
        column=column,
        count=count,
        percentage=_pct(count, total),
        description=(
            f"Text normalization issues found ({reason_str}). "
            "These cause duplicate-looking but unequal strings."
        ),
        description_th=(f"พบปัญหาการ normalize ข้อความ ({reason_str}) ทำให้สตริงที่ดูเหมือนกันแต่ไม่เท่ากันจริง"),
        examples=examples,
        suggestion="Apply Unicode normalization (e.g. pythainlp.util.normalize) before analysis.",
        suggestion_th="ใช้การ normalize ข้อความ (เช่น pythainlp.util.normalize) ก่อนวิเคราะห์",
    )


def _has_fullwidth(text: str) -> bool:
    """True ถ้ามีอักขระ full-width (ช่วง U+FF01–U+FF5E)."""
    return any(0xFF01 <= ord(c) <= 0xFF5E for c in text)


# ----------------------------------------------------------------------------
# (f) Whitespace issues
# ----------------------------------------------------------------------------
_MULTI_SPACE_RE = re.compile(r"  +")


def check_whitespace(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจปัญหาช่องว่าง: เว้นหน้า/หลัง, ช่องว่างซ้อน, non-breaking space."""
    values = _non_null_str(series)
    if len(values) == 0:
        return None

    # vectorized: object dtype → .str.strip() ใช้ str.strip ของ Python (ตัด NBSP/unicode ws
    # เหมือน v.strip() เดิม), ส่วน _MULTI_SPACE_RE/_NBSP ตรวจด้วย .str.contains
    obj = values.astype(object)
    lead_trail = obj != obj.str.strip()
    multi = obj.str.contains(_MULTI_SPACE_RE, regex=True, na=False)
    nbsp = obj.str.contains(_NBSP, regex=False, na=False)

    any_problem = lead_trail | multi | nbsp
    count = int(any_problem.sum())
    if count == 0:
        return None

    reasons: set[str] = set()
    if lead_trail.any():
        reasons.add("leading/trailing space")
    if multi.any():
        reasons.add("multiple consecutive spaces")
    if nbsp.any():
        reasons.add("non-breaking space (U+00A0)")

    examples = [_visible_repr(v) for v in obj[any_problem].head(5)]

    total = len(values)
    reason_str = ", ".join(sorted(reasons))

    return QualityIssue(
        check_name="whitespace",
        severity="info",
        column=column,
        count=count,
        percentage=_pct(count, total),
        description=(
            f"Whitespace issues found ({reason_str}). "
            "These cause subtle mismatches in grouping and joins."
        ),
        description_th=(f"พบปัญหาช่องว่าง ({reason_str}) ทำให้การจับกลุ่มและ join ไม่ตรงกันแบบสังเกตยาก"),
        examples=examples,
        suggestion="Trim and collapse whitespace; replace U+00A0 with a regular space.",
        suggestion_th="ตัดและยุบช่องว่าง และแทน U+00A0 ด้วยช่องว่างปกติ",
    )


# ----------------------------------------------------------------------------
# (g) Keyboard layout anomaly — เซลล์ที่น่าจะพิมพ์ผิดแป้น (ลืมสลับเป็นไทย) — v1.6
# ----------------------------------------------------------------------------
# เกณฑ์ของ check_keyboard_layout_suspect
_KB_COLUMN_THAI_RATIO = 0.50  # คอลัมน์ต้องมีเซลล์ที่มีอักษรไทย > 50% จึงถือว่าเป็นคอลัมน์ไทย
_KB_CELL_LATIN_RATIO = 0.50  # เซลล์ที่ Latin > 50% ของตัวอักษร (ไทย+ละติน) = ต้องสงสัย
_KB_CELL_MIN_LATIN = 3  # ต้องมี Latin อย่างน้อย 3 ตัวจึงพิจารณา (กัน noise สั้น ๆ)


def _count_latin_alpha(text: str) -> int:
    """นับจำนวนตัวอักษรละติน (a–z, A–Z) ในข้อความ."""
    return sum(1 for c in text if ("a" <= c <= "z") or ("A" <= c <= "Z"))


def _count_thai_letters(text: str) -> int:
    """นับจำนวน "ตัวอักษร" ไทย (ช่วง U+0E00–U+0E7F แต่ไม่รวมเลขไทย U+0E50–U+0E59)."""
    return sum(1 for c in text if "฀" <= c <= "๿" and not ("๐" <= c <= "๙"))


def _is_keyboard_layout_suspect(text: str) -> bool:
    """True ถ้าเซลล์น่าจะพิมพ์ผิด keyboard layout (Latin ครอบงำทั้งที่อยู่ในคอลัมน์ไทย).

    เกณฑ์: มี Latin >= 3 ตัว และสัดส่วน Latin ต่อ "ตัวอักษรทั้งหมด" (ไทย+ละติน) > 50%
    เช่น 'l;ylfu' (พิมพ์ 'สวัสดี' โดยลืมสลับแป้น) — Latin ล้วน ในคอลัมน์ที่ปกติเป็นไทย
    """
    latin = _count_latin_alpha(text)
    if latin < _KB_CELL_MIN_LATIN:
        return False
    thai = _count_thai_letters(text)
    alpha = latin + thai
    if alpha == 0:
        return False
    return (latin / alpha) > _KB_CELL_LATIN_RATIO


def check_keyboard_layout_suspect(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจหาเซลล์ที่สงสัยว่าพิมพ์ผิด keyboard layout (ละตินแต่น่าจะเป็นไทย).

    ตรวจเฉพาะคอลัมน์ที่มี Thai text เป็นหลัก — หาเซลล์ที่มี Latin chars ผสมอยู่
    แต่ไม่ใช่การผสมภาษาปกติ (เช่น สวัสดี l;ylfu) คือเซลล์ที่ Latin ครอบงำ (>50%)
    ทั้งที่คอลัมน์โดยรวมเป็นภาษาไทย — มักเกิดจากการลืมสลับแป้นพิมพ์เป็นไทย

    เป็น REPORT-ONLY (เป็น "ข้อสงสัย" เชิงฮิวริสติก — ไม่แก้ไขข้อมูล) เพื่อให้ผู้ใช้ตรวจเอง
    ว่าเป็นการพิมพ์ผิดจริงหรือเป็นคำอังกฤษ/ทับศัพท์ที่ตั้งใจ. ใช้ clean.fix_keyboard_layout
    (ซึ่งตรวจกับพจนานุกรมไทย) เพื่อแก้แบบระมัดระวัง
    """
    values = _non_null_str(series)
    if len(values) == 0:
        return None

    obj = values.astype(object)
    total = len(values)

    # คอลัมน์ต้องเป็นภาษาไทยเป็นหลัก — กันการรันบนคอลัมน์อังกฤษ/รหัสที่ Latin เป็นปกติ
    has_thai = obj.map(lambda v: _count_thai_letters(v) > 0)
    if (int(has_thai.sum()) / total) <= _KB_COLUMN_THAI_RATIO:
        return None

    suspect = obj.map(_is_keyboard_layout_suspect).astype(bool)
    count = int(suspect.sum())
    if count == 0:
        return None

    examples = list(obj[suspect].head(5))
    return QualityIssue(
        check_name="keyboard_layout_suspect",
        severity="info",
        column=column,
        count=count,
        percentage=_pct(count, total),
        description=(
            "Cells dominated by Latin letters were found in a mostly-Thai column. "
            "They may be mistyped with the wrong keyboard layout (e.g. 'l;ylfu' for 'สวัสดี')."
        ),
        description_th=(
            "พบเซลล์ที่เป็นอักษรละตินเป็นส่วนใหญ่ในคอลัมน์ที่ส่วนใหญ่เป็นภาษาไทย "
            "อาจเกิดจากการพิมพ์ผิดแป้นพิมพ์ (ลืมสลับเป็นไทย เช่น 'l;ylfu' แทน 'สวัสดี')"
        ),
        examples=examples,
        suggestion=(
            "Verify these cells; if mistyped, fix with clean.fix_keyboard_layout "
            "(it converts only when the result is a real Thai word)."
        ),
        suggestion_th=(
            "ตรวจสอบเซลล์เหล่านี้ หากพิมพ์ผิดจริงให้แก้ด้วย clean.fix_keyboard_layout "
            "(แปลงเฉพาะเมื่อผลลัพธ์เป็นคำไทยจริง)"
        ),
    )


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------
# ประเภทคอลัมน์ที่ถือว่าเป็นข้อความ
_TEXT_TYPES = {
    ColumnType.THAI_TEXT,
    ColumnType.ENGLISH_TEXT,
    ColumnType.MIXED_TEXT,
    ColumnType.CATEGORICAL,
}
# คอลัมน์ที่ควรรัน string-based checks (รวม phone + pandas 3.x str dtype)
_STRINGISH_TYPES = _TEXT_TYPES | {ColumnType.PHONE_NUMBER}


def _is_stringish_column(series: pd.Series, ctype: ColumnType) -> bool:
    """True ถ้าคอลัมน์เก็บค่าแบบข้อความ (object/str/phone) — รองรับ pandas 3.x str dtype."""
    return (
        ctype in _STRINGISH_TYPES or pd.api.types.is_string_dtype(series) or series.dtype == object
    )


# ประเภทคอลัมน์ที่อาจมีปี
_YEAR_TYPES = {ColumnType.NUMERIC, ColumnType.DATETIME}


def check_missing_values(series: pd.Series, column: str) -> QualityIssue | None:
    """รายงานค่าว่างแยกตามคอลัมน์ตามขนาดปัญหา: info >1%, warning >5%, critical ≥50%."""
    total = len(series)
    if total == 0:
        return None
    count = int(series.isna().sum())
    percentage = _pct(count, total)
    if percentage < 1.0:
        return None
    # คอลัมน์ว่างทั้งหมดให้ check_schema_hints (empty_column) รายงานแทน — กันนับซ้ำ
    if count >= total:
        return None
    if percentage >= 50.0:
        severity = "critical"
    elif percentage > 5.0:
        severity = "warning"
    else:
        severity = "info"
    return QualityIssue(
        check_name="missing_values",
        severity=severity,
        column=column,
        count=count,
        percentage=percentage,
        description=f"Column has {count} missing values ({percentage:.1f}%).",
        description_th=f"คอลัมน์มีค่าว่าง {count} ค่า ({percentage:.1f}%)",
        examples=["<NA>"],
        suggestion="Handle missing values before analysis (impute, flag, or drop as appropriate).",
        suggestion_th="จัดการค่าว่างก่อนวิเคราะห์ (เติมค่า, flag, หรือลบตามความเหมาะสม)",
    )


def check_infinite_values(series: pd.Series, column: str) -> QualityIssue | None:
    """รายงานค่า +/-inf ในคอลัมน์ตัวเลข เพราะทำให้สถิติและกราฟเพี้ยนได้."""
    total = len(series)
    if total == 0:
        return None
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    inf_mask = np.isinf(numeric)
    count = int(inf_mask.sum())
    if count == 0:
        return None
    examples = [str(v) for v in series.loc[pd.Series(inf_mask, index=series.index)].head(5)]
    return QualityIssue(
        check_name="infinite_values",
        severity="warning",
        column=column,
        count=count,
        percentage=_pct(count, total),
        description=(
            f"Column contains {count} infinite value(s) (+/-inf), which can distort "
            "statistics, charts, and anomaly detection."
        ),
        description_th=(
            f"คอลัมน์มีค่าอนันต์ (+/-inf) {count} ค่า ซึ่งอาจทำให้สถิติ กราฟ และการตรวจ outlier เพี้ยน"
        ),
        examples=examples,
        suggestion=(
            "Replace +/-inf with NaN, cap them to a domain limit, or fix the upstream source."
        ),
        suggestion_th="แทน +/-inf ด้วย NaN, จำกัดค่าตาม domain, หรือแก้จากต้นทางข้อมูล",
    )


# ค่าที่มีตัวเลขพอจะน่าจะเป็นเบอร์ (หลังลบสัญลักษณ์)
_PHONE_DIGIT_RE = re.compile(r"\d{9,12}")
_UNIQUE_KEY_RATIO = 0.95


def _looks_like_phone_column(series: pd.Series) -> bool:
    """True ถ้าชื่อหรือค่าบ่งว่าเป็นคอลัมน์เบอร์โทร (ใช้ logic เดียวกับ detect)."""
    values = _non_null_str(series)
    if values.empty:
        return False
    return _detect_looks_like_phone_column(series, values.head(200).tolist())


def check_phone_format(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจเบอร์โทรที่ดูเป็นเบอร์แต่รูปแบบไม่ถูกต้องหลัง normalize."""
    if not _looks_like_phone_column(series):
        return None
    values = _non_null_str(series)
    if values.empty:
        return None
    invalid = 0
    examples: list[str] = []
    for val in values:
        cleaned = _clean_phone_str(val)
        if _PHONE_DIGIT_RE.search(cleaned) and not _THAI_PHONE_RE.match(cleaned):
            invalid += 1
            if len(examples) < 3:
                examples.append(val)
    if invalid == 0:
        return None
    total = len(values)
    return QualityIssue(
        check_name="phone_format",
        severity="warning",
        column=column,
        count=invalid,
        percentage=_pct(invalid, total),
        description=f"{invalid} phone-like value(s) do not match Thai phone format (0XXXXXXXXX)",
        description_th=f"พบ {invalid} ค่าที่ดูเป็นเบอร์โทรแต่รูปแบบไม่ถูก (ควรเป็น 0XXXXXXXXX)",
        examples=examples,
        suggestion="Normalize phones with normalize_phone_numbers() or thaieda.clean()",
        suggestion_th="ทำความสะอาดเบอร์ด้วย normalize_phone_numbers() หรือ thaieda.clean()",
    )


def check_schema_hints(
    series: pd.Series,
    column: str,
    *,
    ctype: ColumnType,
) -> QualityIssue | None:
    """แนะนำ schema hints: near-unique key, dtype mismatch, empty column."""
    non_null = series.dropna()
    total = len(series)
    if total == 0:
        return None

    if series.isna().all():
        return QualityIssue(
            check_name="empty_column",
            severity="critical",
            column=column,
            count=total,
            percentage=100.0,
            description="Column is entirely empty (100% missing)",
            description_th="คอลัมน์ว่างทั้งหมด (100% missing)",
            examples=[],
            suggestion="Drop this column or investigate why it has no data",
            suggestion_th="พิจารณาลบคอลัมน์หรือตรวจสอบว่าทำไมไม่มีข้อมูล",
        )

    n = len(non_null)
    if n >= 5 and _name_hints_id(series):
        unique_ratio = non_null.nunique() / n
        if _UNIQUE_KEY_RATIO <= unique_ratio < 1.0:
            dup_count = n - int(non_null.nunique())
            return QualityIssue(
                check_name="near_unique_key",
                severity="warning",
                column=column,
                count=dup_count,
                percentage=_pct(dup_count, n),
                description=(
                    f"Column name suggests an ID/key but {dup_count} duplicate value(s) exist "
                    f"({unique_ratio * 100:.1f}% unique)"
                ),
                description_th=(
                    f"ชื่อคอลัมน์บ่งว่าเป็น ID/คีย์ แต่มีค่าซ้ำ {dup_count} รายการ "
                    f"(unique {unique_ratio * 100:.1f}%)"
                ),
                examples=non_null.astype(str).value_counts().head(3).index.tolist(),
                suggestion="Verify primary key integrity or rename if not an identifier",
                suggestion_th="ตรวจสอบความ unique ของคีย์หรือเปลี่ยนชื่อถ้าไม่ใช่ identifier",
            )

    if (
        ctype != ColumnType.EMPTY
        and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
        and n > 0
    ):
        coerced = pd.to_numeric(
            non_null.astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        numeric_ratio = coerced.notna().sum() / n
        if numeric_ratio > 0.80:
            return QualityIssue(
                check_name="dtype_mismatch",
                severity="info",
                column=column,
                count=int(coerced.notna().sum()),
                percentage=numeric_ratio * 100.0,
                description=(
                    f"{numeric_ratio * 100:.0f}% of values parse as numeric "
                    "but column is stored as text"
                ),
                description_th=(
                    f"ค่า {numeric_ratio * 100:.0f}% แปลงเป็นตัวเลขได้ แต่คอลัมน์เก็บเป็นข้อความ"
                ),
                examples=non_null.astype(str).head(3).tolist(),
                suggestion="Use coerce_numeric_column() or thaieda.clean() to convert dtype",
                suggestion_th="ใช้ coerce_numeric_column() หรือ thaieda.clean() เพื่อแปลง dtype",
            )

    return None


def check_duplicate_rows(df: pd.DataFrame) -> QualityIssue | None:
    """ตรวจแถวซ้ำทั้ง dataset."""
    n = len(df)
    if n == 0:
        return None
    dup_count = int(df.duplicated(keep=False).sum())
    if dup_count == 0:
        return None
    pct = _pct(dup_count, n)
    severity = "warning" if pct >= 1.0 else "info"
    return QualityIssue(
        check_name="duplicate_rows",
        severity=severity,
        column="_dataset_",
        count=dup_count,
        percentage=pct,
        description=f"{dup_count} row(s) involved in duplicate groups ({pct:.1f}%)",
        description_th=f"พบแถวที่เกี่ยวข้องกับการซ้ำ {dup_count} แถว ({pct:.1f}%)",
        examples=[],
        suggestion="Use remove_duplicate_rows() or thaieda.clean(remove_duplicates=True)",
        suggestion_th="ใช้ remove_duplicate_rows() หรือ thaieda.clean(remove_duplicates=True)",
    )


def run_quality_checks(
    df: pd.DataFrame,
    column_types: dict[str, ColumnType],
    language_info: dict[str, Any] | None = None,
) -> list[QualityIssue]:
    """รันการตรวจคุณภาพทั้งหมด คืนรายการ QualityIssue เรียงตามความรุนแรง (วิกฤตก่อน)."""
    issues: list[QualityIssue] = []
    language_info = language_info or _detect_language(df)
    detected_language = str(language_info.get("language", "numeric"))
    language_columns = language_info.get("columns", {})
    run_thai_specific = detected_language in {"thai", "mixed"}

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        ctype = column_types.get(col_name, ColumnType.EMPTY)

        if (issue := check_missing_values(series, col_name)) is not None:
            issues.append(issue)
        if (
            ctype == ColumnType.NUMERIC
            and (issue := check_infinite_values(series, col_name)) is not None
        ):
            issues.append(issue)

        if ctype == ColumnType.EMPTY:
            if (issue := check_schema_hints(series, col_name, ctype=ctype)) is not None:
                issues.append(issue)
            continue

        column_language = str(language_columns.get(col_name, "numeric"))
        column_has_thai = column_language in {"thai", "mixed"}
        should_run_thai = run_thai_specific or column_has_thai

        # Buddhist Era — เฉพาะคอลัมน์วันที่/ปีจริง ไม่รันบน ID เพื่อกัน false positive
        be_candidate = ctype == ColumnType.DATETIME or (
            ctype != ColumnType.ID
            and _is_date_year_column(col_name)
            and (ctype in _YEAR_TYPES or ctype in _TEXT_TYPES)
        )
        if (
            should_run_thai
            and be_candidate
            and (
                issue := check_buddhist_era(
                    series,
                    col_name,
                    allow_non_date_name=ctype == ColumnType.DATETIME,
                )
            )
            is not None
        ):
            issues.append(issue)

        # เลขไทย — ทุกคอลัมน์ที่เป็น string-ish (ข้อความ/เบอร์ + pandas 3.x str dtype)
        if (
            should_run_thai
            and _is_stringish_column(series, ctype)
            and (issue := check_thai_numerals(series, col_name)) is not None
        ):
            issues.append(issue)

        # เช็คที่เกี่ยวกับข้อความล้วน ๆ
        if ctype in _TEXT_TYPES:
            for check in (check_zero_width, check_normalization, check_whitespace):
                if (issue := check(series, col_name)) is not None:
                    issues.append(issue)

            expected_thai = ctype == ColumnType.THAI_TEXT
            if (issue := check_script_composition(series, col_name, expected_thai)) is not None:
                issues.append(issue)

            # v1.6 (AC-5): keyboard layout suspect — เฉพาะคอลัมน์ไทย/ผสม (รายงานอย่างเดียว)
            if (
                ctype in {ColumnType.THAI_TEXT, ColumnType.MIXED_TEXT}
                and (issue := check_keyboard_layout_suspect(series, col_name)) is not None
            ):
                issues.append(issue)

        # v0.8: placeholder/dash แทน NaN
        if (
            _is_stringish_column(series, ctype)
            and (issue := check_placeholder_values(series, col_name)) is not None
        ):
            issues.append(issue)

        # v0.8: constant column (zero variance) — flag เพื่อให้ user รู้ว่าไม่มีประโยชน์
        if (issue := check_constant_column(series, col_name)) is not None:
            issues.append(issue)

        # Thai national ID validation
        if (
            issue := check_thai_id(series, col_name, is_id_type=(ctype == ColumnType.ID))
        ) is not None:
            issues.append(issue)

        # Phone format validation
        if (issue := check_phone_format(series, col_name)) is not None:
            issues.append(issue)

        # Schema hints (near-unique key, dtype mismatch, empty column)
        if (issue := check_schema_hints(series, col_name, ctype=ctype)) is not None:
            issues.append(issue)

    if (issue := check_duplicate_rows(df)) is not None:
        issues.append(issue)

    issues.sort(key=lambda i: (SEVERITY_ORDER.get(i.severity, 99), -i.percentage))
    return issues


# ------------------------------------------------------------------------------
# v0.8: placeholder/dash แทน NaN + constant column
# ------------------------------------------------------------------------------
_PLACEHOLDER_SET = frozenset(
    {
        "-",
        "--",
        "---",
        "na",
        "n/a",
        "N/A",
        "NA",
        "null",
        "NULL",
        "None",
        "none",
        "nan",
        "NaN",
        "ไม่ระบุ",
        "ไม่มี",
        "?",
        "ไม่ทราบ",
    }
)


def check_placeholder_values(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจหาค่าที่ใช้แทน NaN ('-', 'N/A', 'ไม่มี') ในคอลัมน์ — v0.8."""
    values = _non_null_str(series)
    if len(values) == 0:
        return None
    # vectorized: ใช้ .isin()
    s = pd.Series(values)
    stripped = s.str.strip()
    placeholder_mask = stripped.isin(_PLACEHOLDER_SET)
    count = int(placeholder_mask.sum())
    if count == 0:
        return None
    total = len(values)
    # v1.x (C1): ลด false positive — "-" เดี่ยวที่พบน้อย (<1%) มักเป็น ad-hoc missing
    # ในข้อมูลอังกฤษ ไม่ใช่ placeholder จริงที่ต้องเตือน. ข้ามเฉพาะกรณีที่เจอ placeholder
    # ชนิดเดียวคือ "-" และคิดเป็น <1% ของคอลัมน์ — ถ้าหลายชนิดหรือ >=1% ให้ flag ตามปกติ
    found_types = set(stripped[placeholder_mask].unique())
    if found_types == {"-"} and count / total < 0.01:
        return None
    examples = s[placeholder_mask].unique()[:5].tolist()
    return QualityIssue(
        column=column,
        check_name="placeholder_values",
        severity="warning",
        count=count,
        percentage=round(count / total * 100.0, 1) if total else 0.0,
        description=f"Found {count} placeholder values ('-', 'N/A', 'ไม่มี') that should be NaN",
        description_th=f"พบ {count} ค่าที่ใช้แทน NaN ('-', 'N/A', 'ไม่มี') — ควรแปลงเป็น NaN",
        examples=examples,
        suggestion="Replace placeholder values with NaN before analysis",
        suggestion_th=(
            "แทนที่ค่า placeholder ด้วย NaN ก่อนวิเคราะห์ (ใช้ coerce_numeric_column หรือ replace)"
        ),
    )


def check_constant_column(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจหาคอลัมน์ที่ค่าเดียวทั้งหมด (zero variance) — v0.8."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    nunique = int(non_null.nunique())
    if nunique > 1:
        return None
    val = non_null.iloc[0]
    return QualityIssue(
        column=column,
        check_name="constant_column",
        severity="info",
        count=len(non_null),
        percentage=100.0,
        description=f"Column has only one unique value: {val!r} — no information for analysis",
        description_th=f"คอลัมน์มีค่าเดียวคือ {val!r} — ไม่มีประโยชน์สำหรับการวิเคราะห์",
        examples=[str(val)],
        suggestion="Consider dropping this column — it has no variance",
        suggestion_th="พิจารณาลบคอลัมน์นี้ — ไม่มีความแปรปรวน",
    )


def normalize_thai_digits(text: str) -> str:
    """ช่วยเหลือ: แปลงเลขไทยเป็นเลขอารบิกในสตริง."""
    return text.translate(_THAI_TO_ARABIC)


# ------------------------------------------------------------------------------
# v1.8: Missing data mechanism detection (MCAR / MAR / MNAR)
# ------------------------------------------------------------------------------
_MISSING_MECHANISM_MIN_ROWS = 50
_MISSING_MECHANISM_MIN_COLS = 3


@dataclass
class MissingMechanismResult:
    """ผลการวิเคราะห์รูปแบบค่าว่าง — v1.8."""

    mechanism: str  # "MCAR", "MAR_likely", "MNAR_likely", "insufficient_data"
    missing_pct: float
    missing_cols: int
    total_cols: int
    description: str
    description_th: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "mechanism": self.mechanism,
            "missing_pct": round(self.missing_pct, 2),
            "missing_cols": self.missing_cols,
            "total_cols": self.total_cols,
            "description": self.description,
            "description_th": self.description_th,
            "evidence": self.evidence,
        }


def detect_missing_mechanism(df: pd.DataFrame) -> MissingMechanismResult | None:
    """วิเคราะห์รูปแบบค่าว่าง (missing data mechanism) — v1.8.

    จำแนก pattern ของ missing data ออกเป็น:
      - MCAR (Missing Completely at Random): ค่าว่างกระจายสุ่ม ไม่ขึ้นกคอลัมน์อื่น
      - MAR_likely (Missing at Random): ค่าว่างมีความสัมพันธ์กับคอลัมน์ที่สังเกตได้
      - MNAR_likely (Missing Not at Random): ค่าว่างน่าจะขึ้นกับค่าตัวเอง

    วิธี:
      1. สร้าง missing indicator matrix (1 = missing, 0 = present)
      2. ตรวจ Little's MCAR test approximation: เปรียบเทียบ covariance ของ
         observed data กับ covariance ของ data ที่ไม่มี missing
      3. ถ้า missing pattern ไม่ correlated กับคอลัมน์อื่น → MCAR
      4. ถ้า missing pattern ในคอลัมน์ A correlated กับค่าในคอลัมน์ B → MAR_likely
      5. ถ้าไม่มีคอลัมน์อื่นอธิบายได้ แต่ missing rate สูง → MNAR_likely

    Args:
        df: ข้อมูลที่วิเคราะห์.

    Returns:
        MissingMechanismResult หรือ None ถ้าข้อมูลไม่พอหรือไม่มี missing.
    """
    n_rows, n_cols = df.shape
    if n_rows < _MISSING_MECHANISM_MIN_ROWS or n_cols < _MISSING_MECHANISM_MIN_COLS:
        return None

    # คำนวณ missing rate รวม
    missing_matrix = df.isna()
    total_cells = n_rows * n_cols
    total_missing = int(missing_matrix.sum().sum())
    if total_missing == 0:
        return None

    missing_pct = (total_missing / total_cells) * 100.0

    # หาคอลัมน์ที่มี missing
    col_missing = missing_matrix.sum()
    missing_cols = col_missing[col_missing > 0]
    n_missing_cols = len(missing_cols)

    if n_missing_cols == 0:
        return None

    evidence: dict[str, Any] = {
        "total_missing": total_missing,
        "missing_by_column": {str(k): int(v) for k, v in missing_cols.items()},
    }

    # ตรวจ correlation ของ missing indicators ระหว่างคอลัมน์
    # ถ้า missing ในคอลัมน์ A correlated กับ missing ในคอลัมน์ B → co-missingness
    missing_indicators = missing_matrix.astype(float)
    # เลือกเฉพาะคอลัมน์ที่มี missing
    cols_with_missing = list(missing_cols.index)
    sub_matrix = missing_indicators[cols_with_missing]

    if len(cols_with_missing) >= 2:
        corr = sub_matrix.corr()
        # หาคู่ที่ correlated สูง (|r| > 0.3)
        high_corr_pairs: list[dict] = []
        for i, col_a in enumerate(cols_with_missing):
            for col_b in cols_with_missing[i + 1 :]:
                r = corr.loc[col_a, col_b]
                if pd.notna(r) and abs(r) > 0.3:
                    high_corr_pairs.append(
                        {
                            "col_a": str(col_a),
                            "col_b": str(col_b),
                            "correlation": round(float(r), 3),
                        }
                    )
        evidence["co_missing_correlations"] = high_corr_pairs
    else:
        high_corr_pairs = []

    # ตรวจ: missing indicator correlated กับค่าจริงของคอลัมน์อื่นหรือไม่ (MAR detection)
    # สำหรับแต่ละคอลัมน์ที่มี missing: เช็คว่า missing pattern ใน col A
    # correlated กับค่าใน col B (numeric) หรือไม่
    mar_evidence: list[dict] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for miss_col in cols_with_missing:
        miss_indicator = missing_matrix[miss_col].astype(int)
        for val_col in numeric_cols:
            if val_col == miss_col:
                continue
            val_series = pd.to_numeric(df[val_col], errors="coerce")
            # ต้องมีค่าที่ไม่ missing พอ
            valid = val_series.notna() & miss_indicator.notna()
            if valid.sum() < 30:
                continue
            # แปลงเป็น indicator เพื่อคำนวณ correlation
            miss_vals = miss_indicator[valid].astype(float)
            obs_vals = val_series[valid].astype(float)
            if miss_vals.std() == 0 or obs_vals.std() == 0:
                continue
            r = miss_vals.corr(obs_vals)
            if pd.notna(r) and abs(r) > 0.2:
                mar_evidence.append(
                    {
                        "missing_col": str(miss_col),
                        "predictor_col": str(val_col),
                        "correlation": round(float(r), 3),
                    }
                )
    evidence["mar_indicators"] = mar_evidence

    # ตัดสินใจ mechanism
    if n_missing_cols == df.shape[1] and missing_pct > 50:
        # ทุกคอลัมน์มี missing สูง → MNAR_likely
        mechanism = "MNAR_likely"
        desc = (
            "High missing rate across all columns suggests data may be "
            "MNAR (Missing Not at Random)."
        )
        desc_th = "อัตราค่าว่างสูงทุกคอลัมน์ ข้อมูลอาจเป็น MNAR (ค่าว่างขึ้นกับค่าตัวเอง)"
    elif mar_evidence:
        mechanism = "MAR_likely"
        desc = (
            f"Missing patterns correlated with observed values "
            f"({len(mar_evidence)} pairs) — likely MAR (Missing at Random)."
        )
        desc_th = f"รูปแบบค่าว่างมีความสัมพันธ์กับค่าที่สังเกตได้ ({len(mar_evidence)} คู่) — น่าจะเป็น MAR"
    elif high_corr_pairs:
        mechanism = "MAR_likely"
        desc = (
            f"Co-missingness detected ({len(high_corr_pairs)} column pairs) — "
            "missing values cluster together."
        )
        desc_th = f"พบค่าว่างเกิดพร้อมกัน ({len(high_corr_pairs)} คู่คอลัมน์) — ค่าว่างกลุ่มติดกัน"
    else:
        mechanism = "MCAR"
        desc = (
            "No correlation between missing patterns and observed data — "
            "consistent with MCAR (Missing Completely at Random)."
        )
        desc_th = "ไม่พบความสัมพันธ์ระหว่างรูปแบบค่าว่างกับข้อมูล — สอดคล้องกับ MCAR (สุ่มสมบูรณ์)"

    return MissingMechanismResult(
        mechanism=mechanism,
        missing_pct=missing_pct,
        missing_cols=n_missing_cols,
        total_cols=n_cols,
        description=desc,
        description_th=desc_th,
        evidence=evidence,
    )


# ------------------------------------------------------------------------------
# v1.8: Distribution fitting + Kolmogorov-Smirnov goodness-of-fit test
# ------------------------------------------------------------------------------
_KS_MIN_SAMPLE = 30
_KS_ALPHA = 0.05
_DISTRIBUTION_CANDIDATES = ("normal", "lognormal", "exponential", "uniform")


@dataclass
class DistributionFitResult:
    """ผลการ fitting distribution — v1.8."""

    column: str
    best_fit: str  # ชื่อ distribution ที่ fit ที่สุด
    ks_statistic: float
    p_value: float
    parameters: dict[str, float]
    description: str
    description_th: str
    all_fits: list[dict[str, Any]]

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "best_fit": self.best_fit,
            "ks_statistic": round(self.ks_statistic, 4),
            "p_value": round(self.p_value, 4),
            "parameters": {k: round(v, 4) for k, v in self.parameters.items()},
            "description": self.description,
            "description_th": self.description_th,
            "all_fits": self.all_fits,
        }


def fit_distributions(series: pd.Series, column: str) -> DistributionFitResult | None:
    """ทดสอบ fitting distribution และ KS goodness-of-fit — v1.8.

    ลอง fit 4 distributions (normal, lognormal, exponential, uniform) แล้ว
    เลือกที่ p-value สูงสุด (fit ที่ดีที่สุด) โดยใช้ Kolmogorov-Smirnov test

    Args:
        series: คอลัมน์ตัวเลข.
        column: ชื่อคอลัมน์.

    Returns:
        DistributionFitResult หรือ None ถ้าข้อมูลไม่พอ.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    n = len(numeric)
    if n < _KS_MIN_SAMPLE:
        return None

    values = numeric.to_numpy(dtype="float64")
    if values.std() == 0:
        return None  # constant column — ไม่มี distribution ที่ fit ได้

    # พยายาม import scipy (optional)
    try:
        from scipy import stats as st
    except ImportError:
        return None  # ไม่มี scipy — skip

    all_fits: list[dict[str, Any]] = []

    # Normal distribution
    with contextlib.suppress(Exception):
        mu, sigma = st.norm.fit(values)
        ks_stat, p_val = st.kstest(values, st.norm.cdf, args=(mu, sigma))
        all_fits.append(
            {
                "distribution": "normal",
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "parameters": {"mean": float(mu), "std": float(sigma)},
            }
        )

    # Lognormal distribution (ต้องมีค่าบวก)
    if (values > 0).all():
        with contextlib.suppress(Exception):
            shape, loc, scale = st.lognorm.fit(values, floc=0)
            ks_stat, p_val = st.kstest(values, st.lognorm.cdf, args=(shape, loc, scale))
            all_fits.append(
                {
                    "distribution": "lognormal",
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_val),
                    "parameters": {"shape": float(shape), "loc": float(loc), "scale": float(scale)},
                }
            )

    # Exponential distribution (ต้องมีค่าไม่ติดลบ)
    if (values >= 0).all():
        with contextlib.suppress(Exception):
            loc, scale = st.expon.fit(values)
            ks_stat, p_val = st.kstest(values, st.expon.cdf, args=(loc, scale))
            all_fits.append(
                {
                    "distribution": "exponential",
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_val),
                    "parameters": {"loc": float(loc), "scale": float(scale)},
                }
            )

    # Uniform distribution
    with contextlib.suppress(Exception):
        loc, scale = st.uniform.fit(values)
        ks_stat, p_val = st.kstest(values, st.uniform.cdf, args=(loc, scale))
        all_fits.append(
            {
                "distribution": "uniform",
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "parameters": {"loc": float(loc), "scale": float(scale)},
            }
        )

    if not all_fits:
        return None

    # เลือก distribution ที่ p-value สูงสุด (fit ดีสุด)
    best = max(all_fits, key=lambda x: x["p_value"])
    best_name = best["distribution"]

    # คำอธิบาย
    _DIST_DESC_TH = {
        "normal": "การกระจายปกติ",
        "lognormal": "การกระจาย log-normal",
        "exponential": "การกระจายเอกซ์โพเนนเชียล",
        "uniform": "การกระจายสม่ำเสมอ",
    }
    fit_th = _DIST_DESC_TH.get(best_name, best_name)
    is_good_fit = best["p_value"] >= _KS_ALPHA
    if is_good_fit:
        desc = f"'{column}' follows {best_name} distribution (KS p={best['p_value']:.4f})."
        desc_th = f"'{column}' ตาม{fit_th} (KS p={best['p_value']:.4f})"
    else:
        desc = (
            f"'{column}' does not follow any tested distribution well "
            f"(best: {best_name}, p={best['p_value']:.4f})."
        )
        desc_th = f"'{column}' ไม่ตรงกับการกระจายใดที่ทดสอบ (ดีที่สุด: {fit_th}, p={best['p_value']:.4f})"

    return DistributionFitResult(
        column=column,
        best_fit=best_name,
        ks_statistic=best["ks_statistic"],
        p_value=best["p_value"],
        parameters=best["parameters"],
        description=desc,
        description_th=desc_th,
        all_fits=all_fits,
    )


from thaieda.quality._score import (  # noqa: E402 — import หลัง QualityIssue เพื่อเลี่ยง circular import
    QualityBreakdown,
    QualityComparisonResult,
    QualityScoreResult,
    compute_quality_comparison,
    compute_quality_score,
)

__all__ = [
    "QualityIssue",
    "QualityBreakdown",
    "QualityComparisonResult",
    "QualityScoreResult",
    "MissingMechanismResult",
    "DistributionFitResult",
    "check_buddhist_era",
    "check_thai_numerals",
    "check_zero_width",
    "check_script_composition",
    "check_normalization",
    "check_whitespace",
    "check_keyboard_layout_suspect",
    "check_missing_values",
    "check_infinite_values",
    "check_placeholder_values",
    "check_constant_column",
    "check_thai_id",
    "check_phone_format",
    "check_schema_hints",
    "check_duplicate_rows",
    "run_quality_checks",
    "normalize_thai_digits",
    "compute_quality_score",
    "compute_quality_comparison",
    "validate_thai_id",
    "validate_thai_id_column",
    "detect_missing_mechanism",
    "fit_distributions",
]
