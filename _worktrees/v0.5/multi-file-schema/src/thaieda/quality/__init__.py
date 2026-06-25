"""Thai data quality checks — หัวใจของ ThaiEDA (the moat).

ตรวจจับปัญหาคุณภาพข้อมูลที่เครื่องมือ EDA แบบ English-centric มองข้าม เช่น
ปี พ.ศ. ปนกับ ค.ศ., เลขไทย, อักขระล่องหน, การ normalize ผิด ฯลฯ
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

from thaieda.detect import ColumnType, script_ratio

# ----------------------------------------------------------------------------
# ค่าคงที่
# ----------------------------------------------------------------------------
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# ช่วงปี พ.ศ. ที่เป็นไปได้ (≈ ค.ศ. 1900–2056)
_BE_MIN, _BE_MAX = 2440, 2599

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


# ----------------------------------------------------------------------------
# (a) Buddhist Era detection
# ----------------------------------------------------------------------------
def check_buddhist_era(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจหาเลขปี พ.ศ. (2440–2599) ที่อาจปนกับ ค.ศ. ในคอลัมน์ตัวเลข/วันที่."""
    values = _non_null_str(series)
    if len(values) == 0:
        return None

    be_examples: list[str] = []
    ce_seen = False
    be_count = 0
    year_re = re.compile(r"\b(\d{4})\b")

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

    total = len(values)
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

    thai_num_count = 0
    arabic_seen = False
    examples: list[str] = []

    for v in values:
        has_thai = any(c in _THAI_DIGIT_SET for c in v)
        has_arabic = any(c.isdigit() and c not in _THAI_DIGIT_SET for c in v)
        if has_arabic:
            arabic_seen = True
        if has_thai:
            thai_num_count += 1
            if len(examples) < 5:
                examples.append(v)

    if thai_num_count == 0:
        return None

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

    count = 0
    examples: list[str] = []
    found_chars: set[str] = set()

    for v in values:
        present = [c for c in _ZERO_WIDTH_CHARS if c in v]
        if present:
            count += 1
            found_chars.update(present)
            if len(examples) < 5:
                examples.append(_visible_repr(v))

    if count == 0:
        return None

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

    agg = _aggregate_script_ratio(values)
    thai_total = agg["thai"] + agg["thai_digit"]
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
# อักขระเดียวกันซ้ำ 3+ ครั้ง (เช่น 5555, ๆๆๆ, อืมมม)
_REPEAT_SPAM_RE = re.compile(r"(.)\1{2,}")


def _has_combining_order_issue(text: str) -> bool:
    """True ถ้ามี combining mark ของไทยที่ขึ้นต้น หรือมาหลังอักขระที่ไม่ใช่พยัญชนะ/สระ."""
    prev_is_base = False
    for ch in text:
        # combining mark ต้องตามหลัง base (พยัญชนะ) หรือ combining ตัวอื่น (สระ+วรรณยุกต์)
        if ch in _THAI_COMBINING and not prev_is_base:
            return True
        prev_is_base = (ch in _THAI_CONSONANTS) or (ch in _THAI_COMBINING)
    return False


def check_normalization(series: pd.Series, column: str) -> QualityIssue | None:
    """ตรวจปัญหา normalization: วรรณยุกต์/สระซ้ำ, ลำดับ combining ผิด, อักขระสแปม, full-width."""
    values = _non_null_str(series)
    if len(values) == 0:
        return None

    count = 0
    examples: list[str] = []
    reasons: set[str] = set()

    for v in values:
        problems = []
        if _DUP_TONE_RE.search(v):
            problems.append("duplicate tone marks")
        if _DUP_VOWEL_RE.search(v):
            problems.append("duplicate vowels")
        if _has_combining_order_issue(v):
            problems.append("combining order")
        if _REPEAT_SPAM_RE.search(v):
            problems.append("repeated-char spam")
        # full-width vs half-width — ตรวจผ่าน NFKC ว่าเปลี่ยนไหม
        if v != unicodedata.normalize("NFKC", v) and _has_fullwidth(v):
            problems.append("full-width characters")

        if problems:
            count += 1
            reasons.update(problems)
            if len(examples) < 5:
                examples.append(v if len(v) <= 60 else v[:57] + "...")

    if count == 0:
        return None

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

    count = 0
    examples: list[str] = []
    reasons: set[str] = set()

    for v in values:
        problems = []
        if v != v.strip():
            problems.append("leading/trailing space")
        if _MULTI_SPACE_RE.search(v):
            problems.append("multiple consecutive spaces")
        if _NBSP in v:
            problems.append("non-breaking space (U+00A0)")

        if problems:
            count += 1
            reasons.update(problems)
            if len(examples) < 5:
                examples.append(_visible_repr(v))

    if count == 0:
        return None

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
_YEAR_TYPES = {ColumnType.NUMERIC, ColumnType.DATETIME, ColumnType.ID}


def run_quality_checks(df: pd.DataFrame, column_types: dict[str, ColumnType]) -> list[QualityIssue]:
    """รันการตรวจคุณภาพทั้งหมด คืนรายการ QualityIssue เรียงตามความรุนแรง (วิกฤตก่อน)."""
    issues: list[QualityIssue] = []

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        ctype = column_types.get(col_name, ColumnType.EMPTY)

        if ctype == ColumnType.EMPTY:
            continue

        # Buddhist Era — คอลัมน์เลข/วันที่ และข้อความ (date strings)
        if (ctype in _YEAR_TYPES or ctype in _TEXT_TYPES) and (
            issue := check_buddhist_era(series, col_name)
        ) is not None:
            issues.append(issue)

        # เลขไทย — ทุกคอลัมน์ที่เป็น string-ish (ข้อความ + อาจเป็นเลขไทยใน object)
        if (ctype in _TEXT_TYPES or series.dtype == object) and (
            issue := check_thai_numerals(series, col_name)
        ) is not None:
            issues.append(issue)

        # เช็คที่เกี่ยวกับข้อความล้วน ๆ
        if ctype in _TEXT_TYPES:
            for check in (check_zero_width, check_normalization, check_whitespace):
                if (issue := check(series, col_name)) is not None:
                    issues.append(issue)

            expected_thai = ctype == ColumnType.THAI_TEXT
            if (issue := check_script_composition(series, col_name, expected_thai)) is not None:
                issues.append(issue)

    issues.sort(key=lambda i: (SEVERITY_ORDER.get(i.severity, 99), -i.percentage))
    return issues


def normalize_thai_digits(text: str) -> str:
    """ช่วยเหลือ: แปลงเลขไทยเป็นเลขอารบิกในสตริง."""
    return text.translate(_THAI_TO_ARABIC)


__all__ = [
    "QualityIssue",
    "check_buddhist_era",
    "check_thai_numerals",
    "check_zero_width",
    "check_script_composition",
    "check_normalization",
    "check_whitespace",
    "run_quality_checks",
    "normalize_thai_digits",
]
