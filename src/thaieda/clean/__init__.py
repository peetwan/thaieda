"""Data cleaning — ฟังก์ชันทำความสะอาดข้อความไทย ที่แก้ปัญหาที่ quality/anomaly ตรวจพบ.

แต่ละฟังก์ชันคืน (Series ที่สะอาดแล้ว, CleaningResult) เพื่อให้ตรวจสอบได้ว่าทำอะไรไปบ้าง
(กี่แถวที่เปลี่ยน, ตัวอย่างก่อน/หลัง) — เหมาะกับการแสดงเป็น "คำแนะนำการทำความสะอาด" ในรายงาน
หลักการ: ไม่มี fallback แบบเงียบ ๆ, import ของเสริม (ftfy) แบบ lazy
"""

from __future__ import annotations

import contextlib
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

# ลายเซ็นที่ "อาจเป็น mojibake / ข้อความที่ ftfy ซ่อมได้" — ใช้กรองแถวก่อนเรียก ftfy
# (เร็วขึ้นมากบนข้อมูลใหญ่: ข้อความไทย UTF-8 ปกติไม่มีอักขระเหล่านี้ จึงข้ามได้ทันที)
#   * U+0080–U+00FF : Latin-1 supplement (ไบต์ของ mojibake ไทยตกในช่วงนี้ เช่น 'à¸')
#   * U+FFFD        : replacement char (ถอดรหัสเสีย)
#   * &...;          : HTML entity ที่ ftfy ถอดกลับได้
_MOJIBAKE_HINT_RE = re.compile(r"[-ÿ�]|&[#a-zA-Z0-9]{1,8};")

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
# helper หลัก — ใช้ฟังก์ชันแปลงระดับสตริงกับทั้งคอลัมน์ (เวกเตอร์ไรซ์)
# ----------------------------------------------------------------------------
def _apply_str_transform(
    series: pd.Series,
    func: Callable[[str], str],
    operation: str,
    description_th: str,
    *,
    visible: bool = False,
    vectorized: Callable[[pd.Series], pd.Series] | None = None,
) -> tuple[pd.Series, CleaningResult]:
    """ใช้การแปลงระดับสตริงกับทุกเซลล์ที่ไม่ว่าง นับแถวที่เปลี่ยน และเก็บตัวอย่างก่อน/หลัง.

    เวอร์ชันนี้ทำงานแบบเวกเตอร์ (vectorized) เพื่อให้เร็วบนข้อมูลขนาดใหญ่ (>1M แถว):
      * คัดเฉพาะเซลล์ที่ไม่ว่างครั้งเดียว (notna mask) แทนการเช็ค pd.isna ทีละแถว
      * ถ้ามี ``vectorized`` (เช่น .str.translate / .str.replace) จะใช้ตัวนั้น —
        เป็น C-level ไม่วน Python ทีละเซลล์; ไม่งั้นถอยไปใช้ ``series.map(func)``
        ซึ่งยังเร็วกว่าการวน .items() + กำหนดค่าด้วย .at ทีละแถวมาก
      * กำหนดค่ากลับเฉพาะเซลล์ที่ "เปลี่ยนจริง" แบบกลุ่มเดียว (ไม่ใช่ทีละ .at)

    visible=True จะแสดงตัวอย่างผ่าน repr() เพื่อให้เห็นอักขระล่องหน/ช่องว่างที่เปลี่ยนไป
    """
    column = str(series.name) if series.name is not None else ""

    notna = series.notna()
    originals = series[notna].astype(str)
    cleaned = vectorized(originals) if vectorized is not None else originals.map(func)
    # เทียบก่อน/หลังแบบเวกเตอร์ — ได้ mask ของเซลล์ที่เปลี่ยนจริง
    changed = cleaned.to_numpy() != originals.to_numpy()
    affected = int(changed.sum())

    out = series
    before_examples: list[str] = []
    after_examples: list[str] = []
    if affected:
        # แปลงเป็น object ก่อนกำหนดค่ากลับ — กันข้อผิดพลาดเมื่อคอลัมน์เป็นชนิดตัวเลข/บูลีน
        # (เช่น customer_id เป็น int64 แล้วการทำความสะอาดคืนเป็นสตริง)
        out = series.astype(object)
        changed_originals = originals[changed]
        changed_cleaned = cleaned[changed]
        out.loc[changed_originals.index] = changed_cleaned

        show = (lambda t: repr(t)) if visible else (lambda t: t)
        for before, after in zip(
            changed_originals.to_numpy()[:_MAX_EXAMPLES],
            changed_cleaned.to_numpy()[:_MAX_EXAMPLES],
            strict=True,
        ):
            before_examples.append(show(str(before)))
            after_examples.append(show(str(after)))

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
        vectorized=lambda s: s.str.replace(_ZERO_WIDTH_RE.pattern, "", regex=True),
    )


def normalize_thai_numerals(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """แปลงเลขไทย (๐๑๒๓) เป็นเลขอารบิก (0123) เพื่อให้แปลงเป็นตัวเลข/เรียงลำดับได้."""
    return _apply_str_transform(
        series,
        lambda s: s.translate(_THAI_TO_ARABIC),
        operation="normalize_thai_numerals",
        description_th="แปลงเลขไทยเป็นเลขอารบิก (๐๑๒๓ → 0123)",
        vectorized=lambda s: s.str.translate(_THAI_TO_ARABIC),
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
        # ข้ามแถวที่ไม่มีลายเซ็น mojibake เลย — ftfy จะไม่เปลี่ยนอยู่แล้ว (เร็วขึ้นมากบนข้อมูลใหญ่)
        if not _MOJIBAKE_HINT_RE.search(text):
            return text
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


def _vec_strip_whitespace(s: pd.Series) -> pd.Series:
    """รุ่นเวกเตอร์ของ _strip_whitespace_str — ใช้ .str accessor (เทียบเท่ากันทุกขั้น)."""
    s = s.str.replace(_NBSP, " ", regex=False)
    s = s.str.replace(_MULTI_SPACE_RE.pattern, " ", regex=True)
    return s.str.strip()


def strip_whitespace(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """ตัดช่องว่างหน้า/หลัง, ยุบช่องว่างซ้อนให้เหลือช่องเดียว, แปลง non-breaking space เป็นช่องว่างปกติ."""
    return _apply_str_transform(
        series,
        _strip_whitespace_str,
        operation="strip_whitespace",
        description_th="ตัด/ยุบช่องว่าง และแปลง non-breaking space เป็นช่องว่างปกติ",
        visible=True,
        vectorized=_vec_strip_whitespace,
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


def _vec_fix_repeated(s: pd.Series, max_repeat: int) -> pd.Series:
    """รุ่นเวกเตอร์ของ _fix_repeated_str — ยุบไม้ยมกซ้ำ แล้วยุบอักขระซ้ำเกิน max_repeat."""
    s = s.str.replace(_YAMOK_REPEAT_RE.pattern, "ๆ", regex=True)
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    return s.str.replace(pattern, lambda m: m.group(1) * max_repeat, regex=True)


def fix_repeated_chars(series: pd.Series, max_repeat: int = 3) -> tuple[pd.Series, CleaningResult]:
    """ลดการซ้ำอักขระที่มากเกินไป (55555 → 555, ๆๆๆ → ๆ)."""
    if max_repeat < 1:
        raise ValueError("max_repeat must be >= 1")
    return _apply_str_transform(
        series,
        lambda s: _fix_repeated_str(s, max_repeat),
        operation="fix_repeated_chars",
        description_th=f"ลดการซ้ำอักขระที่เกิน {max_repeat} ตัว (เช่น 55555 → 555, ๆๆๆ → ๆ)",
        vectorized=lambda s: _vec_fix_repeated(s, max_repeat),
    )


def fix_tone_mark_stacking(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """ลบวรรณยุกต์ไทยที่ซ้อนติดกัน (เช่น '่่' → '่') ซึ่งมักเกิดจากการพิมพ์ผิด."""
    return _apply_str_transform(
        series,
        lambda s: _TONE_STACK_RE.sub(r"\1", s),
        operation="fix_tone_mark_stacking",
        description_th="ลบวรรณยุกต์ที่ซ้อนติดกัน (เช่น '่่' → '่')",
        vectorized=lambda s: s.str.replace(_TONE_STACK_RE.pattern, r"\1", regex=True),
    )


# ----------------------------------------------------------------------------
# การทำความสะอาดที่ใช้ pythainlp (optional dependency [thai])
# ----------------------------------------------------------------------------
_PYTHAINLP_INSTALL_HINT = "ต้องติดตั้ง pip install thaieda[thai] (แพ็กเกจ 'pythainlp') ก่อนใช้งานฟังก์ชันนี้"


def _pythainlp_available() -> bool:
    """ตรวจว่าติดตั้ง pythainlp ไหม โดยไม่ import เนื้อหนัก ๆ (ใช้ตัดสินใจ skip ใน default pipeline)."""
    import importlib.util

    return importlib.util.find_spec("pythainlp") is not None


def pythainlp_normalize(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """Normalize ข้อความไทยด้วย pythainlp.util.normalize().

    normalize() ของ pythainlp รวมหลายกฎการจัดระเบียบไว้ในขั้นตอนเดียว:
      * remove_zw — ลบ zero-width spaces
      * remove_dup_spaces — ยุบช่องว่างซ้ำ
      * remove_spaces_before_marks — ลบช่องว่างหน้าวรรณยุกต์/สระ
      * remove_repeat_vowels — ลบสระ/เครื่องหมายที่ซ้ำ (เช่น 'นานาาา' → 'นานา')
      * remove_dangling — ลบอักขระลอยที่ขึ้นต้นผิด (เช่น 'เเปลก' → 'แปลก')

    เหมาะกับการจัดข้อความไทยให้เป็นรูปสะกดมาตรฐานในขั้นตอนเดียว (เสริม/แทนการเรียกทีละกฎ)

    Raises:
        ImportError: ถ้าไม่ได้ติดตั้ง pythainlp (ติดตั้งผ่าน thaieda[thai]).
    """
    try:
        from pythainlp.util import normalize  # lazy import — optional dependency [thai]
    except ImportError as exc:
        raise ImportError(f"pythainlp_normalize {_PYTHAINLP_INSTALL_HINT}.") from exc

    return _apply_str_transform(
        series,
        normalize,
        operation="pythainlp_normalize",
        description_th="จัดระเบียบข้อความไทยด้วย pythainlp.normalize (ลบ zw/ช่องว่างซ้ำ/สระซ้ำ ฯลฯ)",
    )


# จำนวน token ที่เป็นคำไทยจริงขั้นต่ำ และสัดส่วนขั้นต่ำ เพื่อยอมรับการแก้ keyboard layout
# (กัน false positive: ข้อความอังกฤษจริงที่แปลงแล้วได้คำไทยมั่ว ๆ)
_KEYBOARD_MIN_KNOWN = 1
_KEYBOARD_MIN_RATIO = 0.67
_KEYBOARD_MIN_THAI_CHARS = 3

# แคช dictionary คำไทย (โหลดครั้งเดียว) — None = ยังไม่โหลด
_THAI_WORDS_CACHE: frozenset[str] | None = None


def _thai_words() -> frozenset[str]:
    """คืนชุดคำไทยจาก pythainlp (แคชไว้) — ใช้ตรวจว่าผลการแก้ layout เป็นคำจริงไหม."""
    global _THAI_WORDS_CACHE
    if _THAI_WORDS_CACHE is None:
        from pythainlp.corpus.common import thai_words

        _THAI_WORDS_CACHE = frozenset(thai_words())
    return _THAI_WORDS_CACHE


def _count_thai_chars(text: str) -> int:
    """นับจำนวนอักขระไทย (ช่วง U+0E00–U+0E7F) ในข้อความ."""
    return sum(1 for ch in text if "฀" <= ch <= "๿")


def _make_keyboard_fixer() -> Callable[[str], str]:
    """สร้างฟังก์ชันแก้ keyboard layout (eng_to_thai) แบบระมัดระวัง — โหลด pythainlp ครั้งเดียว.

    หลักการตัดสิน: แปลง Latin → Thai ด้วย eng_to_thai แล้ว "ยอมรับ" ก็ต่อเมื่อ
    ผลลัพธ์ตัดคำแล้วเป็นคำไทยจริง (อยู่ใน dictionary) เป็นสัดส่วนสูงพอ — เพื่อกัน
    การแปลงข้อความอังกฤษจริง (เช่น 'hello') เป็นภาษาไทยมั่ว ๆ
    """
    from pythainlp.tokenize import word_tokenize
    from pythainlp.util import eng_to_thai

    thai_words = _thai_words()

    def fix(text: str) -> str:
        letters = [c for c in text if c.isalpha()]
        # ต้องมีตัวอักษร และต้องเป็น ASCII ล้วน (ผู้พิมพ์ลืมสลับเป็นแป้นไทย)
        if not letters or any(ord(c) > 127 for c in letters):
            return text
        converted = eng_to_thai(text)
        if _count_thai_chars(converted) < _KEYBOARD_MIN_THAI_CHARS:
            return text
        tokens = [t for t in word_tokenize(converted) if t.strip()]
        if not tokens:
            return text
        known = sum(1 for t in tokens if t in thai_words)
        if known >= _KEYBOARD_MIN_KNOWN and (known / len(tokens)) >= _KEYBOARD_MIN_RATIO:
            return converted
        return text

    return fix


def fix_keyboard_layout(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """ตรวจและแก้การพิมพ์ผิด keyboard layout — พิมพ์อังกฤษทั้งที่ตั้งใจพิมพ์ไทย.

    เช่น พิมพ์ 'l;ylfu' (ลืมสลับแป้น) แทน 'สวัสดี' — ฟังก์ชันนี้จะแปลงกลับด้วย
    pythainlp.util.eng_to_thai เฉพาะเซลล์ที่แปลงแล้วได้ "คำไทยจริง" เท่านั้น
    (ข้อความอังกฤษจริงจะไม่ถูกแตะต้อง)

    Raises:
        ImportError: ถ้าไม่ได้ติดตั้ง pythainlp (ติดตั้งผ่าน thaieda[thai]).
    """
    try:
        fixer = _make_keyboard_fixer()
    except ImportError as exc:
        raise ImportError(f"fix_keyboard_layout {_PYTHAINLP_INSTALL_HINT}.") from exc

    return _apply_str_transform(
        series,
        fixer,
        operation="fix_keyboard_layout",
        description_th="แก้การพิมพ์ผิด keyboard layout (พิมพ์อังกฤษทั้งที่ตั้งใจพิมพ์ไทย เช่น 'l;ylfu' → 'สวัสดี')",
    )


def normalize_phone_numbers(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """ทำความสะอาดเบอร์โทรศัพท์ไทยในคอลัมน์ — แปลงเลขไทย, ลบ dash/space, +66 → 0.

    เบอร์โทรเป็นข้อมูลที่ sensitive ต่อการเปลี่ยน type: ถ้าเขียนเป็น int แล้วอ่านใหม่
    leading zero จะหายไป (0812345678 → 812345678) ฟังก์ชันนี้เก็บเป็น string
    และทำความสะอาดรูปแบบให้เป็นมาตรฐาน 10 หลักขึ้นต้น 0
    """
    from thaieda.detect import _THAI_PHONE_RE, _clean_phone_str

    def _phone_fixer(text: str) -> str:
        cleaned = _clean_phone_str(text)
        if _THAI_PHONE_RE.match(cleaned):
            return cleaned
        return text

    return _apply_str_transform(
        series,
        _phone_fixer,
        operation="normalize_phone_numbers",
        description_th="ทำความสะอาดเบอร์โทรศัพท์ (แปลงเลขไทย, ลบ dash/space, +66 → 0)",
    )


# ----------------------------------------------------------------------------
# v0.8: การแปลงคอลัมน์ string ที่ควรเป็นตัวเลข → numeric dtype
# ----------------------------------------------------------------------------
# ค่าที่ใช้แทน NaN ที่พบบ่อยในข้อมูลไทย (นอกจาก empty string)
_PLACEHOLDER_VALUES = frozenset(
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
    }
)

# ค่าที่ใช้แทน NaN ในข้อมูลไทย (ขยายจาก placeholder)
_THAI_MISSING_VALUES = frozenset(
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
        "ไม่ต้องมี",
    }
)


def coerce_numeric_column(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """แปลงคอลัมน์ string ที่ควรเป็นตัวเลข → numeric dtype (v0.8).

    แก้ปัญหา: เลขไทย '๑๐๐' หลัง normalize_thai_numerals กลายเป็น '100' แต่ยังเป็น string
    — ทำให้ pd.to_numeric เจอ '100' แปลงเป็น 100.0 ได้ แต่ค่าที่แปลงไม่ได้กลายเป็น NaN

    ขั้นตอน:
      1. แปลง placeholder values ('-', 'N/A', 'ไม่มี') → NaN ก่อน (กันเป็น numeric ผิด)
      2. ลอง pd.to_numeric — ถ้าแปลงได้ >50% ของค่าที่ไม่ใช่ placeholder ถือว่าคอลัมน์เป็น numeric
      3. คืน Series ที่เป็น numeric แล้ว + CleaningResult

    ถ้าแปลงไม่ได้ (เป็นข้อความจริง ๆ) คืน series เดิม ไม่แตะ
    """
    notna = series.notna()
    if not notna.any():
        return series, CleaningResult(
            operation="coerce_numeric", rows_affected=0, column=str(series.name or "")
        )

    # 1. แปลง placeholder → NaN
    str_vals = series[notna].astype(str).str.strip()
    placeholder_mask = str_vals.isin(_PLACEHOLDER_VALUES)
    placeholders_found = int(placeholder_mask.sum())

    # 2. ลองแปลงเป็น numeric (หลังจากลบ placeholder แล้ว)
    non_placeholder = str_vals[~placeholder_mask]
    if non_placeholder.empty:
        return series, CleaningResult(
            operation="coerce_numeric", rows_affected=0, column=str(series.name or "")
        )

    coerced = pd.to_numeric(non_placeholder, errors="coerce")
    converted_ok = coerced.notna()
    ok_count = int(converted_ok.sum())
    total_count = int(non_placeholder.shape[0])

    # ถ้าแปลงได้ >50% ถือว่าคอลัมน์นี้ควรเป็น numeric
    if total_count == 0 or ok_count / total_count < 0.5:
        return series, CleaningResult(
            operation="coerce_numeric", rows_affected=0, column=str(series.name or "")
        )

    # 3. สร้าง Series ผลลัพธ์
    out = series.astype(object)
    # placeholder → NaN
    placeholder_idx = str_vals[placeholder_mask].index
    out.loc[placeholder_idx] = pd.NA
    # numeric values
    numeric_idx = non_placeholder[converted_ok].index
    out.loc[numeric_idx] = coerced[converted_ok].to_numpy()
    # ค่าที่แปลงไม่ได้ (แปลงได้ <100%) → คงค่าเดิมไว้
    failed_idx = non_placeholder[~converted_ok].index
    out.loc[failed_idx] = non_placeholder[~converted_ok].to_numpy()

    # พยายามแปลงเป็น numeric dtype ถ้าทุกค่าที่ไม่ใช่ placeholder/failed แปลงได้
    if failed_idx.empty:
        with contextlib.suppress(Exception):
            out = pd.to_numeric(out, errors="coerce")

    affected = ok_count + placeholders_found
    return out, CleaningResult(
        operation="coerce_numeric",
        rows_affected=affected,
        column=str(series.name or ""),
        description_th=(
            f"แปลงคอลัมน์เป็นตัวเลข ({ok_count} ค่า) + แทนที่ placeholder {placeholders_found} ค่าด้วย NaN"
        ),
    )


def convert_buddhist_era(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """แปลงปีพุทธศักราช (พ.ศ.) → คริสต์ศักราช (ค.ศ.) ในคอลัมน์ตัวเลขหรือวันที่ (v0.8).

    ตรวจหาค่าในช่วง 2440–2599 (พ.ศ. ที่พบบ่อย) แล้วลบ 543 เพื่อแปลงเป็น ค.ศ.
    ค่าที่อยู่นอกช่วงนี้จะไม่ถูกแตะ (เป็น ค.ศ. อยู่แล้ว)

    รองรับทั้ง:
      - คอลัมน์ numeric (เช่น ปีเกิด 2530 → 1987)
      - คอลัมน์ string ที่มีปี (เช่น '2567-01-15' → '2024-01-15')
    """
    notna = series.notna()
    if not notna.any():
        return series, CleaningResult(
            operation="convert_buddhist_era", rows_affected=0, column=str(series.name or "")
        )

    _BE_MIN = 2440
    _BE_MAX = 2599
    _BE_OFFSET = 543

    # กรณี 1: numeric column — แปลงเฉพาะค่าในช่วง พ.ศ.
    numeric = pd.to_numeric(series, errors="coerce")
    be_mask = numeric.notna() & (numeric >= _BE_MIN) & (numeric <= _BE_MAX)
    be_count = int(be_mask.sum())

    if be_count > 0:
        out = series.astype(object)
        converted = numeric - _BE_OFFSET
        out.loc[be_mask] = converted[be_mask].to_numpy()
        # พยายามแปลงกลับเป็น numeric
        with contextlib.suppress(Exception):
            out = pd.to_numeric(out, errors="coerce")
        return out, CleaningResult(
            operation="convert_buddhist_era",
            rows_affected=be_count,
            column=str(series.name or ""),
            description_th=f"แปลงปี พ.ศ. → ค.ศ. (ลบ 543) จำนวน {be_count} ค่า",
        )

    # กรณี 2: string column ที่มีปี พ.ศ. ฝังอยู่ (เช่น '2567-01-15')
    str_vals = series[notna].astype(str)
    # หา ปี 4 หลัก ในช่วง พ.ศ.
    year_pattern = re.compile(r"\b(24\d{2}|25\d{2})\b")

    def _replace_year(text: str) -> str:
        def _sub(m):
            return str(int(m.group(1)) - _BE_OFFSET)

        return year_pattern.sub(_sub, text)

    converted = str_vals.map(_replace_year)
    changed = converted.to_numpy() != str_vals.to_numpy()
    affected = int(changed.sum())

    if affected == 0:
        return series, CleaningResult(
            operation="convert_buddhist_era", rows_affected=0, column=str(series.name or "")
        )

    out = series.astype(object)
    changed_idx = str_vals[changed].index
    out.loc[changed_idx] = converted[changed].to_numpy()

    return out, CleaningResult(
        operation="convert_buddhist_era",
        rows_affected=affected,
        column=str(series.name or ""),
        description_th=f"แปลงปี พ.ศ. → ค.ศ. ในข้อความ จำนวน {affected} ค่า",
    )


# ชื่อเดือนไทย → เลขเดือน (สำหรับ normalize_dates)
_THAI_MONTH_MAP = {
    "มกราคม": "01",
    "ม.ค.": "01",
    "มกรา": "01",
    "กุมภาพันธ์": "02",
    "ก.พ.": "02",
    "กุมภา": "02",
    "มีนาคม": "03",
    "มี.ค.": "03",
    "มีนา": "03",
    "เมษายน": "04",
    "เม.ย.": "04",
    "เมษา": "04",
    "พฤษภาคม": "05",
    "พ.ค.": "05",
    "พฤษภ": "05",
    "มิถุนายน": "06",
    "มิ.ย.": "06",
    "มิถุน": "06",
    "กรกฎาคม": "07",
    "ก.ค.": "07",
    "กรกฎา": "07",
    "สิงหาคม": "08",
    "ส.ค.": "08",
    "สิงหา": "08",
    "กันยายน": "09",
    "ก.ย.": "09",
    "กันยา": "09",
    "ตุลาคม": "10",
    "ต.ค.": "10",
    "ตุลา": "10",
    "พฤศจิกายน": "11",
    "พ.ย.": "11",
    "พฤศจิ": "11",
    "ธันวาคม": "12",
    "ธ.ค.": "12",
    "ธันวา": "12",
}
_THAI_MONTH_RE = re.compile(
    r"(\d{1,2})\s+(" + "|".join(_THAI_MONTH_MAP.keys()) + r")\s+(\d{2,4})",
    re.IGNORECASE,
)


def normalize_dates(series: pd.Series) -> tuple[pd.Series, CleaningResult]:
    """แปลงวันที่ที่มีหลายรูปแบบ (รวม Thai month names + พ.ศ.) → ISO มาตรฐาน — v0.8.

    ขั้นตอน:
      1. แปลง Thai month names → เลขเดือน ("15 มกราคม 2567" → "15/01/2567")
      2. แปลง พ.ศ. → ค.ศ. ใน string วันที่ ("2567" → "2024")
      3. ลอง parse เป็น datetime
    คืน Series ที่เป็น datetime แล้ว (ถ้า parse ได้) หรือ string ที่แปลงแล้ว
    """
    notna = series.notna()
    if not notna.any():
        return series, CleaningResult(
            operation="normalize_dates", rows_affected=0, column=str(series.name or "")
        )

    str_vals = series[notna].astype(str)

    # 1. แปลง Thai month names → เลขเดือน
    def _replace_thai_month(text: str) -> str:
        def _sub(m):
            day = m.group(1).zfill(2)
            month = _THAI_MONTH_MAP.get(m.group(2), m.group(2))
            year = m.group(3)
            return f"{day}/{month}/{year}"

        return _THAI_MONTH_RE.sub(_sub, text)

    converted = str_vals.map(_replace_thai_month)

    # 2. แปลง พ.ศ. → ค.ศ. ในปี (4 หลัก ในช่วง 2440-2599)
    year_re = re.compile(r"\b(24\d{2}|25\d{2})\b")

    def _replace_be_year(text: str) -> str:
        def _sub(m):
            return str(int(m.group(1)) - 543)

        return year_re.sub(_sub, text)

    converted = converted.map(_replace_be_year)

    # 3. เปรียบเทียบก่อน/หลัง
    changed = converted.to_numpy() != str_vals.to_numpy()
    affected = int(changed.sum())

    if affected == 0:
        return series, CleaningResult(
            operation="normalize_dates", rows_affected=0, column=str(series.name or "")
        )

    out = series.astype(object)
    changed_idx = str_vals[changed].index
    out.loc[changed_idx] = converted[changed].to_numpy()

    return out, CleaningResult(
        operation="normalize_dates",
        rows_affected=affected,
        column=str(series.name or ""),
        description_th=f"แปลงรูปแบบวันที่ {affected} ค่า (Thai month → ISO, พ.ศ. → ค.ศ.)",
    )


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningResult]:
    """ตรวจหาและลบแถวที่ซ้ำกันทั้งหมด (v0.8).

    คืน DataFrame ที่ลบ duplicate แล้ว + CleaningResult บอกจำนวนที่ลบ
    """
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return df, CleaningResult(
            operation="remove_duplicate_rows",
            rows_affected=0,
            column="(entire df)",
            description_th="ไม่พบแถวซ้ำ",
        )
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, CleaningResult(
        operation="remove_duplicate_rows",
        rows_affected=dup_count,
        column="(entire df)",
        description_th=f"ลบแถวซ้ำ {dup_count} แถว",
    )


def handle_missing_values(
    series: pd.Series, strategy: str = "flag"
) -> tuple[pd.Series, CleaningResult]:
    """จัดการค่าว่างในคอลัมน์ — แทนที่หรือ flag ตาม strategy (v0.8).

    strategy:
      - 'flag'    : แทน NaN ด้วย 'ไม่ระบุ' (text) หรือ 0 (numeric) — ทุกค่ายังใช้งานได้
      - 'drop'    : ลบแถวที่ NaN (เฉพาะคอลัมน์นี้)
      - 'median'  : แทนด้วย median (numeric only)
      - 'mode'    : แทนด้วย mode (ทุกประเภท)
      - 'unknown' : แทนด้วย 'unknown' (text/categorical)
    """
    missing = int(series.isna().sum())
    if missing == 0:
        return series, CleaningResult(
            operation="handle_missing_values",
            rows_affected=0,
            column=str(series.name or ""),
            description_th="ไม่มีค่าว่าง",
        )

    col_name = str(series.name or "")
    out = series

    if strategy == "drop":
        out = series.dropna()
        desc = f"ลบ {missing} แถวที่ว่างในคอลัมน์ '{col_name}'"
    elif strategy == "median" and pd.api.types.is_numeric_dtype(series):
        med = series.median()
        out = series.fillna(med)
        desc = f"แทน {missing} ค่าว่างด้วย median ({med})"
    elif strategy == "mode":
        mode_val = series.mode().iloc[0] if not series.mode().empty else None
        if mode_val is not None:
            out = series.fillna(mode_val)
            desc = f"แทน {missing} ค่าว่างด้วย mode ({mode_val})"
        else:
            desc = f"ไม่สามารถหา mode ของคอลัมน์ '{col_name}' ได้"
    elif strategy == "unknown":
        out = series.fillna("unknown")
        desc = f"แทน {missing} ค่าว่างด้วย 'unknown'"
    else:  # 'flag' (default)
        if pd.api.types.is_numeric_dtype(series):
            out = series.fillna(0)
            desc = f"แทน {missing} ค่าว่างด้วย 0 (numeric)"
        else:
            out = series.fillna("ไม่ระบุ")
            desc = f"แทน {missing} ค่าว่างด้วย 'ไม่ระบุ'"

    return out, CleaningResult(
        operation="handle_missing_values",
        rows_affected=missing,
        column=col_name,
        description_th=desc,
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
    "phone": normalize_phone_numbers,
    "pythainlp_normalize": pythainlp_normalize,
    "keyboard_layout": fix_keyboard_layout,
    "coerce_numeric": coerce_numeric_column,
    "buddhist_era": convert_buddhist_era,
    "normalize_dates": normalize_dates,
}

# การดำเนินการที่ต้องใช้ pythainlp (ติดตั้งผ่าน thaieda[thai]) — ใช้ตัดสินใจ skip ใน default pipeline
_PYTHAINLP_OPERATIONS = frozenset({"pythainlp_normalize", "keyboard_layout"})

# ลำดับการทำความสะอาดเริ่มต้น (ทั้งหมด): แก้ encoding ก่อน แล้วค่อยลบ/ยุบ/normalize
# จบด้วยขั้นตอนที่ใช้ pythainlp (normalize + แก้ keyboard layout) ถ้าติดตั้งไว้
DEFAULT_OPERATIONS: tuple[str, ...] = (
    "encoding",
    "zwspace",
    "whitespace",
    "unicode",
    "tonemarks",
    "repeat",
    "numerals",
    "phone",
    "pythainlp_normalize",
    "keyboard_layout",
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
    # ติดตามว่า op ใด "ถูกเพิ่มอัตโนมัติ" (จาก default/all) เทียบกับ "ผู้ใช้ระบุเอง"
    # op ที่ใช้ pythainlp และถูกเพิ่มอัตโนมัติจะถูก skip เงียบ ๆ ถ้าไม่ได้ติดตั้ง pythainlp
    # แต่ถ้าผู้ใช้ระบุเอง จะปล่อยให้ฟังก์ชัน fail loudly (ตามหลักการ no silent fallback)
    op_names: list[str] = []
    auto_added: set[str] = set()
    if operations is None:
        op_names = list(DEFAULT_OPERATIONS)
        auto_added = set(DEFAULT_OPERATIONS)
    else:
        for name in operations:
            if name == "all":
                op_names.extend(DEFAULT_OPERATIONS)
                auto_added.update(DEFAULT_OPERATIONS)
            else:
                op_names.append(name)

    unknown = [n for n in op_names if n not in _OPERATIONS]
    if unknown:
        raise ValueError(
            f"Unknown cleaning operation(s): {', '.join(unknown)}. "
            f"Available: {', '.join(_OPERATIONS)}, all."
        )

    # default/all pipeline: ถ้าไม่มี pythainlp ให้ข้ามขั้นตอนที่ต้องใช้มัน (degrade อย่างสุภาพ)
    if auto_added & _PYTHAINLP_OPERATIONS and not _pythainlp_available():
        op_names = [n for n in op_names if not (n in _PYTHAINLP_OPERATIONS and n in auto_added)]

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
    "normalize_phone_numbers",
    "pythainlp_normalize",
    "fix_keyboard_layout",
    "coerce_numeric_column",
    "convert_buddhist_era",
    "normalize_dates",
    "remove_duplicate_rows",
    "handle_missing_values",
    "clean_thai_text",
    "available_operations",
    "DEFAULT_OPERATIONS",
]
