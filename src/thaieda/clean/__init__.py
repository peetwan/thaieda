"""Data cleaning — ฟังก์ชันทำความสะอาดข้อความไทย ที่แก้ปัญหาที่ quality/anomaly ตรวจพบ.

แต่ละฟังก์ชันคืน (Series ที่สะอาดแล้ว, CleaningResult) เพื่อให้ตรวจสอบได้ว่าทำอะไรไปบ้าง
(กี่แถวที่เปลี่ยน, ตัวอย่างก่อน/หลัง) — เหมาะกับการแสดงเป็น "คำแนะนำการทำความสะอาด" ในรายงาน
หลักการ: ไม่มี fallback แบบเงียบ ๆ, import ของเสริม (ftfy) แบบ lazy
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

# ----------------------------------------------------------------------------
# ค่าคงที่
# ----------------------------------------------------------------------------
# อักขระความกว้างศูนย์/ล่องหน
_ZERO_WIDTH_RE = re.compile("[​‌‍﻿⁠]")

# เลขไทย -> เลขอารบิก
_THAI_DIGITS = "๐๑๒๓๔๕๖๗๘๙"
_THAI_TO_ARABIC = str.maketrans(_THAI_DIGITS, "0123456789")

# non-breaking space และช่องว่างซ้อน
_NBSP = " "
_MULTI_SPACE_RE = re.compile(r" {2,}")

# วรรณยุกต์ไทย (สำหรับลบวรรณยุกต์ซ้อน)
_THAI_TONE_MARKS = "่้๊๋"
_TONE_STACK_RE = re.compile(f"([{_THAI_TONE_MARKS}])\\1+")

# ไม้ยมก (ๆ) ที่ซ้ำกัน -> ตัวเดียว
_YAMOK_REPEAT_RE = re.compile("ๆ{2,}")

# จำนวนตัวอย่างก่อน/หลังที่เก็บต่อหนึ่งการทำความสะอาด
_MAX_EXAMPLES = 5


# ----------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ----------------------------------------------------------------------------
@dataclass
class CleaningResult:
    """สรุปผลการทำความสะอาดหนึ่งการดำเนินการ."""

    operation: str
    rows_affected: int
    column: str
    before_examples: list[str] = field(default_factory=list)
    after_examples: list[str] = field(default_factory=list)
    description_th: str = ""
    # คำอธิบายว่า "แก้อะไรไป" (เช่น แผนการซ่อมของ ftfy) — โปร่งใสขึ้น มีค่าเฉพาะบางการดำเนินการ
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "rows_affected": self.rows_affected,
            "column": self.column,
            "before_examples": self.before_examples,
            "after_examples": self.after_examples,
            "description_th": self.description_th,
            "explanation": self.explanation,
        }


# ----------------------------------------------------------------------------
# helper หลัก — ใช้ฟังก์ชันแปลงระดับสตริงกับทั้งคอลัมน์
# ----------------------------------------------------------------------------
def _apply_str_transform(
    series: pd.Series,
    func: Callable[[str], str],
    operation: str,
    description_th: str,
    *,
    visible: bool = False,
) -> tuple[pd.Series, CleaningResult]:
    """ใช้ func กับทุกเซลล์ที่ไม่ว่าง นับแถวที่เปลี่ยน และเก็บตัวอย่างก่อน/หลัง.

    visible=True จะแสดงตัวอย่างผ่าน repr() เพื่อให้เห็นอักขระล่องหน/ช่องว่างที่เปลี่ยนไป
    """
    column = str(series.name) if series.name is not None else ""
    out = series.copy()
    affected = 0
    before_examples: list[str] = []
    after_examples: list[str] = []

    def show(text: str) -> str:
        return repr(text) if visible else text

    for idx, value in series.items():
        if pd.isna(value):
            continue
        original = str(value)
        cleaned = func(original)
        if cleaned != original:
            out.at[idx] = cleaned
            affected += 1
            if len(before_examples) < _MAX_EXAMPLES:
                before_examples.append(show(original))
                after_examples.append(show(cleaned))

    result = CleaningResult(
        operation=operation,
        rows_affected=affected,
        column=column,
        before_examples=before_examples,
        after_examples=after_examples,
        description_th=description_th,
    )
    return out, result


# ----------------------------------------------------------------------------
# การทำความสะอาดแต่ละแบบ
# ----------------------------------------------------------------------------
def remove_zero_width_chars(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """ลบอักขระความกว้างศูนย์ (U+200B/C/D, U+FEFF, U+2060) ที่ทำให้ join/groupby พังเงียบ ๆ."""
    return _apply_str_transform(
        series,
        lambda s: _ZERO_WIDTH_RE.sub("", s),
        operation="remove_zero_width_chars",
        description_th="ลบอักขระความกว้างศูนย์/ล่องหน",
        visible=True,
    )


def normalize_thai_numerals(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """แปลงเลขไทย (๐๑๒๓) เป็นเลขอารบิก (0123) เพื่อให้แปลงเป็นตัวเลข/เรียงลำดับได้."""
    return _apply_str_transform(
        series,
        lambda s: s.translate(_THAI_TO_ARABIC),
        operation="normalize_thai_numerals",
        description_th="แปลงเลขไทยเป็นเลขอารบิก (๐๑๒๓ → 0123)",
    )


def _manual_demojibake(text: str) -> str:
    """แก้ mojibake เบื้องต้นแบบไม่พึ่ง ftfy: ถอด Latin-1 กลับเป็น UTF-8 ถ้าได้ผลเป็นไทย."""
    # mojibake ของไทยมักมีลายเซ็น "à¸"/"à¹" (ไบต์ E0 B8.. ถูกถอดเป็น Latin-1)
    if "à¸" not in text and "à¹" not in text:
        return text
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    # ยอมรับการแก้เฉพาะเมื่อผลลัพธ์มีอักษรไทยจริง (กันการเปลี่ยนที่ผิดพลาด)
    if any("฀" <= ch <= "๿" for ch in repaired):
        return repaired
    return text


def _format_ftfy_plan(explanation) -> str:
    """แปลงแผนการซ่อมของ ftfy ให้เป็นข้อความอ่านง่าย (เช่น 'encode latin-1 → decode utf-8').

    ftfy.fix_and_explain คืน explanation เป็น list ของ tuple (operation, ...) แต่ละขั้นตอน
    """
    steps: list[str] = []
    for step in explanation or []:
        if isinstance(step, (tuple, list)):
            steps.append(" ".join(str(part) for part in step))
        else:
            steps.append(str(step))
    return " → ".join(steps)


def normalize_encoding(
    series: pd.Series, target: str = "utf-8"
) -> tuple[pd.Series, CleaningResult]:
    """แก้ mojibake ที่พบบ่อย — ใช้ ftfy ถ้ามี (ติดตั้งผ่าน thaieda[fix]) ไม่งั้นใช้วิธี manual.

    เมื่อมี ftfy จะใช้ fix_and_explain เพื่อเก็บ "คำอธิบายว่าซ่อมอะไรไป" ไว้ใน CleaningResult.explanation
    (โปร่งใสกว่า fix_text เพราะบอกว่าข้อความถูกถอด/เข้ารหัสผิดแบบไหน)
    target เก็บไว้เพื่อความเข้ากันได้ของ API (ปัจจุบันคืนผลเป็นสตริง UTF-8 เสมอ)
    """
    try:
        from ftfy import fix_and_explain  # lazy import — optional dependency [fix]
    except ImportError:
        # ไม่มี ftfy — ถอยไปใช้วิธี manual (ไม่มีคำอธิบายการซ่อม)
        return _apply_str_transform(
            series,
            _manual_demojibake,
            operation="normalize_encoding",
            description_th="แก้ข้อความที่เข้ารหัสผิด (mojibake)",
        )

    # เก็บแผนการซ่อมที่ "ไม่ว่าง" ทั้งหมด (เซลล์ที่ถูกแก้จริง) แบบไม่ซ้ำ เพื่อรายงานรวม
    plans: dict[str, None] = {}  # dict รักษาลำดับที่พบ

    def fixer(text: str) -> str:
        result = fix_and_explain(text)
        if result.explanation:
            plan = _format_ftfy_plan(result.explanation)
            if plan:
                plans.setdefault(plan, None)
        return result.text

    out, cleaning = _apply_str_transform(
        series,
        fixer,
        operation="normalize_encoding",
        description_th="แก้ข้อความที่เข้ารหัสผิด (mojibake)",
    )
    if plans:
        cleaning.explanation = "ftfy: " + "; ".join(plans)
    return out, cleaning


def _strip_whitespace_str(text: str) -> str:
    text = text.replace(_NBSP, " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def strip_whitespace(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """ตัดช่องว่างหน้า/หลัง, ยุบช่องว่างซ้อนให้เหลือช่องเดียว, แปลง non-breaking space เป็นช่องว่างปกติ."""
    return _apply_str_transform(
        series,
        _strip_whitespace_str,
        operation="strip_whitespace",
        description_th="ตัด/ยุบช่องว่าง และแปลง non-breaking space เป็นช่องว่างปกติ",
        visible=True,
    )


def normalize_unicode(series: pd.Series, form: str = "NFC") -> tuple[pd.Series, CleaningResult]:
    """ทำ Unicode normalization (ค่าเริ่มต้น NFC) เพื่อจัดลำดับ combining mark ให้เป็นมาตรฐาน."""
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(f"Unsupported normalization form {form!r}; expected NFC/NFD/NFKC/NFKD.")
    return _apply_str_transform(
        series,
        lambda s: unicodedata.normalize(form, s),
        operation="normalize_unicode",
        description_th=f"ทำ Unicode normalization แบบ {form}",
    )


def _fix_repeated_str(text: str, max_repeat: int) -> str:
    # ยุบไม้ยมกที่ซ้ำ (ๆๆๆ) ให้เหลือตัวเดียว — การเขียน ๆ ซ้ำไม่มีความหมาย
    text = _YAMOK_REPEAT_RE.sub("ๆ", text)
    # ยุบอักขระอื่นที่ซ้ำเกิน max_repeat ให้เหลือ max_repeat ตัว (เช่น 55555 -> 555)
    pattern = re.compile(r"(.)\1{" + str(max_repeat) + r",}")
    return pattern.sub(lambda m: m.group(1) * max_repeat, text)


def fix_repeated_chars(series: pd.Series, max_repeat: int = 3) -> tuple[pd.Series, CleaningResult]:
    """ลดการซ้ำอักขระที่มากเกินไป (55555 → 555, ๆๆๆ → ๆ)."""
    if max_repeat < 1:
        raise ValueError("max_repeat must be >= 1")
    return _apply_str_transform(
        series,
        lambda s: _fix_repeated_str(s, max_repeat),
        operation="fix_repeated_chars",
        description_th=f"ลดการซ้ำอักขระที่เกิน {max_repeat} ตัว (เช่น 55555 → 555, ๆๆๆ → ๆ)",
    )


def fix_tone_mark_stacking(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """ลบวรรณยุกต์ไทยที่ซ้อนติดกัน (เช่น '่่' → '่') ซึ่งมักเกิดจากการพิมพ์ผิด."""
    return _apply_str_transform(
        series,
        lambda s: _TONE_STACK_RE.sub(r"\1", s),
        operation="fix_tone_mark_stacking",
        description_th="ลบวรรณยุกต์ที่ซ้อนติดกัน (เช่น '่่' → '่')",
    )


# ----------------------------------------------------------------------------
# ตัวประกอบหลายการดำเนินการ
# ----------------------------------------------------------------------------
# ทะเบียนชื่อ operation -> ฟังก์ชัน (เรียงตามลำดับการทำงานที่ปลอดภัย)
_OPERATIONS: dict[str, Callable[[pd.Series], tuple[pd.Series, CleaningResult]]] = {
    "encoding": normalize_encoding,
    "zwspace": remove_zero_width_chars,
    "whitespace": strip_whitespace,
    "unicode": normalize_unicode,
    "tonemarks": fix_tone_mark_stacking,
    "repeat": fix_repeated_chars,
    "numerals": normalize_thai_numerals,
}

# ลำดับการทำความสะอาดเริ่มต้น (ทั้งหมด): แก้ encoding ก่อน แล้วค่อยลบ/ยุบ/normalize
DEFAULT_OPERATIONS: tuple[str, ...] = (
    "encoding",
    "zwspace",
    "whitespace",
    "unicode",
    "tonemarks",
    "repeat",
    "numerals",
)


def clean_thai_text(
    series: pd.Series, operations: list[str] | None = None
) -> tuple[pd.Series, list[CleaningResult]]:
    """ทำความสะอาดข้อความไทยหลายขั้นตอนตามลำดับ — คืน (Series สะอาด, รายการ CleaningResult).

    Args:
        series: คอลัมน์ข้อความ.
        operations: รายชื่อการดำเนินการ (จาก _OPERATIONS) ตามลำดับที่ต้องการ
            ถ้าเป็น None ใช้ DEFAULT_OPERATIONS (การทำความสะอาดที่ปลอดภัยทั้งหมด).
            ใช้ "all" เป็นชื่อย่อแทนทุกการดำเนินการได้.

    Raises:
        ValueError: เมื่อระบุชื่อ operation ที่ไม่รู้จัก.
    """
    if operations is None:
        op_names: list[str] = list(DEFAULT_OPERATIONS)
    else:
        op_names = []
        for name in operations:
            if name == "all":
                op_names.extend(DEFAULT_OPERATIONS)
            else:
                op_names.append(name)

    unknown = [n for n in op_names if n not in _OPERATIONS]
    if unknown:
        raise ValueError(
            f"Unknown cleaning operation(s): {', '.join(unknown)}. "
            f"Available: {', '.join(_OPERATIONS)}, all."
        )

    current = series
    results: list[CleaningResult] = []
    for name in op_names:
        current, result = _OPERATIONS[name](current)
        results.append(result)
    return current, results


def available_operations() -> list[str]:
    """คืนรายชื่อการดำเนินการทำความสะอาดที่รองรับ."""
    return list(_OPERATIONS)


__all__ = [
    "CleaningResult",
    "remove_zero_width_chars",
    "normalize_thai_numerals",
    "normalize_encoding",
    "strip_whitespace",
    "normalize_unicode",
    "fix_repeated_chars",
    "fix_tone_mark_stacking",
    "clean_thai_text",
    "available_operations",
    "DEFAULT_OPERATIONS",
]
