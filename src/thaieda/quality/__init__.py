"""Thai data quality checks — หัวใจของ ThaiEDA (the moat).

ตรวจจับปัญหาคุณภาพข้อมูลที่เครื่องมือ EDA แบบ English-centric มองข้าม เช่น
ปี พ.ศ. ปนกับ ค.ศ., เลขไทย, อักขระล่องหน, การ normalize ผิด ฯลฯ
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from thaieda.detect import ColumnType, _detect_language, script_ratio
from thaieda.quality._thai_id import validate_thai_id, validate_thai_id_column

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

    # ตรวจ CE จากค่าทั้งหมด (ไม่ใช่แค่ check_values) เพื่อตัดสินใจ mixed
    all_year_values = date_like_values + year_only_values
    ce_seen = len(ce_year_values) > 0

    be_examples: list[str] = []
    year_re = re.compile(r"\b(\d{4})\b")

    # v0.8: vectorize — ใช้ .str.extractall แทนการวนลูป
    try:
        # แปลงเป็น string Series แล้ว extractall (ใช้ check_values เท่านั้น)
        s = pd.Series(check_values)
        extracted = s.str.extractall(year_re)
        if extracted.empty:
            return None
        years = pd.to_numeric(extracted[0], errors="coerce").dropna()
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

    total = len(all_year_values) + len(ce_year_values)
    # critical ถ้าปนกัน (ทั้ง พ.ศ. และ ค.ศ.), warning ถ้าเป็น พ.ศ. ล้วน
    mixed = ce_seen and be_count < total
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
    # repeated-char spam: (.)\1{2,} แต่ข้าม code/ID/หัวเราะ ตามกฎ _skip_repeated_spam_check เดิม
    # ใช้ .map กับ regex ที่มี capturing group (เลี่ยง UserWarning ของ .str.contains)
    repeat_raw = obj.map(lambda v: _REPEAT_SPAM_RE.search(v) is not None)
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
# ประเภทคอลัมน์ที่อาจมีปี
_YEAR_TYPES = {ColumnType.NUMERIC, ColumnType.DATETIME}


def check_missing_values(series: pd.Series, column: str) -> QualityIssue | None:
    """รายงานค่าว่างแยกตามคอลัมน์เมื่อเกิน threshold: info >1%, warning >5%."""
    total = len(series)
    if total == 0:
        return None
    count = int(series.isna().sum())
    percentage = _pct(count, total)
    if percentage < 1.0:
        return None
    severity = "warning" if percentage > 5.0 else "info"
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

        if ctype == ColumnType.EMPTY:
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

        # เลขไทย — ทุกคอลัมน์ที่เป็น string-ish (ข้อความ + อาจเป็นเลขไทยใน object)
        if (
            should_run_thai
            and (ctype in _TEXT_TYPES or series.dtype == object)
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
        if (ctype in _TEXT_TYPES or series.dtype == object) and (
            issue := check_placeholder_values(series, col_name)
        ) is not None:
            issues.append(issue)

        # v0.8: constant column (zero variance) — flag เพื่อให้ user รู้ว่าไม่มีประโยชน์
        if (issue := check_constant_column(series, col_name)) is not None:
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


from thaieda.quality._score import (  # noqa: E402 — import หลัง QualityIssue เพื่อเลี่ยง circular import
    QualityBreakdown,
    QualityScoreResult,
    compute_quality_score,
)

__all__ = [
    "QualityIssue",
    "QualityBreakdown",
    "QualityScoreResult",
    "check_buddhist_era",
    "check_thai_numerals",
    "check_zero_width",
    "check_script_composition",
    "check_normalization",
    "check_whitespace",
    "check_keyboard_layout_suspect",
    "check_missing_values",
    "check_placeholder_values",
    "check_constant_column",
    "run_quality_checks",
    "normalize_thai_digits",
    "compute_quality_score",
    "validate_thai_id",
    "validate_thai_id_column",
]
