"""Column type detection — จำแนกประเภทคอลัมน์ โดยเน้นการแยกข้อความไทย/อังกฤษ/ผสม."""

from __future__ import annotations

import re
import warnings
from enum import Enum
from typing import Any

import pandas as pd

from thaieda.detect._thai_address import parse_thai_address, parse_thai_address_column

# ----------------------------------------------------------------------------
# ช่วง Unicode ที่ใช้คำนวณสัดส่วนสคริปต์
# ----------------------------------------------------------------------------
_THAI_RANGE = (0x0E00, 0x0E7F)  # บล็อกอักษรไทยทั้งหมด
_THAI_DIGIT_RANGE = (0x0E50, 0x0E59)  # ๐-๙

# emoji แบบคร่าว ๆ ครอบคลุมบล็อกหลัก ๆ ที่พบบ่อย
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"  # สัญลักษณ์ & รูปภาพ, emoji เสริม
    "\U00002600-\U000027bf"  # สัญลักษณ์เบ็ดเตล็ด & dingbats
    "\U0001f000-\U0001f0ff"  # ไพ่นกกระจอก/โดมิโน/ไพ่
    "\U00002b00-\U00002bff"  # ลูกศร/รูปทรงเบ็ดเตล็ด
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0001f1e6-\U0001f1ff"  # ธงประเทศ (regional indicators)
    "]",
    flags=re.UNICODE,
)


class ColumnType(str, Enum):
    """ประเภทคอลัมน์ที่ ThaiEDA จำแนกได้."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    THAI_TEXT = "thai_text"
    ENGLISH_TEXT = "english_text"
    MIXED_TEXT = "mixed_text"
    DATETIME = "datetime"
    ID = "id"
    PHONE_NUMBER = "phone_number"
    EMPTY = "empty"

    def __str__(self) -> str:  # ให้แสดงเป็นค่าตรง ๆ เวลา print/format
        return self.value


def script_ratio(text: str) -> dict[str, float]:
    """คืนสัดส่วน (0–1) ของสคริปต์ต่าง ๆ ในข้อความหนึ่งสตริง.

    คีย์ที่คืน: thai, latin, digit, thai_digit, whitespace, emoji, other.
    ผลรวมของทุกคีย์ ≈ 1.0 (สำหรับสตริงที่ไม่ว่าง). สตริงว่างคืนค่า 0 ทั้งหมด.
    """
    keys = ("thai", "latin", "digit", "thai_digit", "whitespace", "emoji", "other")
    counts = dict.fromkeys(keys, 0)

    if not isinstance(text, str) or len(text) == 0:
        return {k: 0.0 for k in keys}

    # หา emoji ก่อน เพื่อไม่ให้ถูกนับเป็น "other"
    emoji_chars: set[int] = set()
    for match in _EMOJI_PATTERN.finditer(text):
        for i in range(match.start(), match.end()):
            emoji_chars.add(i)

    total = len(text)
    for idx, ch in enumerate(text):
        if idx in emoji_chars:
            counts["emoji"] += 1
            continue
        cp = ord(ch)
        if _THAI_DIGIT_RANGE[0] <= cp <= _THAI_DIGIT_RANGE[1]:
            counts["thai_digit"] += 1
        elif _THAI_RANGE[0] <= cp <= _THAI_RANGE[1]:
            counts["thai"] += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            counts["latin"] += 1
        elif "0" <= ch <= "9":
            counts["digit"] += 1
        elif ch.isspace():
            counts["whitespace"] += 1
        else:
            counts["other"] += 1

    return {k: counts[k] / total for k in keys}


def _thai_content_ratio(text: str) -> float:
    """สัดส่วน "อักษร" ไทย เทียบกับอักขระที่ไม่ใช่ช่องว่าง.

    นับเฉพาะตัวอักษรไทย ไม่รวมเลขไทย — เพราะคอลัมน์ที่เป็นเลขไทยล้วน (เช่น ราคา)
    คือ "ตัวเลข" ไม่ใช่ "ข้อความไทย" จึงไม่ควรถูกจัดเป็น thai_text
    ใช้ตัวหารเป็นอักขระที่ไม่ใช่ช่องว่าง เพื่อให้ "อาหาร ดี" ยังถือว่าเป็นไทยเต็ม ๆ
    """
    if not isinstance(text, str) or not text:
        return 0.0
    ratios = script_ratio(text)
    non_ws = 1.0 - ratios["whitespace"]
    if non_ws <= 0:
        return 0.0
    return ratios["thai"] / non_ws


def _latin_content_ratio(text: str) -> float:
    """สัดส่วนอักษรละติน เทียบกับอักขระที่ไม่ใช่ช่องว่าง."""
    if not isinstance(text, str) or not text:
        return 0.0
    ratios = script_ratio(text)
    non_ws = 1.0 - ratios["whitespace"]
    if non_ws <= 0:
        return 0.0
    return ratios["latin"] / non_ws


def is_thai_text(series: pd.Series, threshold: float = 0.15) -> bool:
    """True ถ้าสัดส่วนเซลล์ที่มีอักษรไทย >30% มากกว่า threshold ของเซลล์ที่ไม่ว่าง."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    sample = non_null.head(1000)
    thai_cells = sum(1 for v in sample if _thai_content_ratio(str(v)) > 0.30)
    return (thai_cells / len(sample)) > threshold


_LANGUAGE_SAMPLE_ROWS = 500
_LOW_CARDINALITY_MAX_UNIQUE = 20
_SHORT_LABEL_AVG_LEN = 15
_SEMICOLON_DELIMITER_MIN_COUNT = 3

# ชื่อคอลัมน์ที่บอกว่าเป็นข้อความ/รีวิว — ไม่ควรถูกบังคับเป็น categorical แม้ cardinality ต่ำ
_TEXT_NAME_HINTS_RE = re.compile(
    r"(review|feedback|comment|description|text|note|notes|summary|message|"
    r"รีวิว|ความคิดเห็น|ข้อความ|บันทึก|คำอธิบาย|รายละเอียด)",
    re.IGNORECASE,
)
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")
_THAI_BLOCK_RE = re.compile(r"[\u0e00-\u0e7f]+")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")
_CODE_LIKE_RE = re.compile(r"^[A-Za-z0-9_.:/#\-]+$")
_COMMON_THAI_WORDS = frozenset({"ครับ", "ค่ะ", "ไทย", "อร่อย", "ดี", "ไม่", "มี", "และ"})

_WORD_TOKENIZER: Any | None = None
_WORD_TOKENIZER_LOADED = False


def _is_thai_text_char(ch: str) -> bool:
    """True ถ้าเป็นตัวอักษร/สระ/วรรณยุกต์ไทย ไม่รวมเลขไทยและ punctuation ล้วน."""
    cp = ord(ch)
    return (
        0x0E01 <= cp <= 0x0E2E  # consonants
        or 0x0E30 <= cp <= 0x0E4D  # vowels + tone marks + leading vowels
        or cp in {0x0E45, 0x0E46}  # ๅ, ๆ
    )


def _is_thai_vowel_or_tone(ch: str) -> bool:
    """True ถ้าอยู่ในช่วง U+0E30-U+0E4D ตาม Thai vowel/tone marks."""
    return 0x0E30 <= ord(ch) <= 0x0E4D


def _get_thai_word_tokenizer() -> Any | None:
    """Lazy import pythainlp tokenizer; ถ้าไม่มี/โหลดไม่ได้ให้ fallback เป็น regex."""
    global _WORD_TOKENIZER, _WORD_TOKENIZER_LOADED
    if not _WORD_TOKENIZER_LOADED:
        try:
            from pythainlp.tokenize import word_tokenize
        except Exception:
            _WORD_TOKENIZER = None
        else:
            _WORD_TOKENIZER = word_tokenize
        _WORD_TOKENIZER_LOADED = True
    return _WORD_TOKENIZER


def _thai_word_tokens(text: str) -> list[str]:
    """ตัดคำไทยแบบ optional: ใช้ pythainlp ถ้าพร้อม ไม่พร้อมใช้ Unicode block fallback."""
    tokenizer = _get_thai_word_tokenizer()
    if tokenizer is not None:
        try:
            tokens = tokenizer(text, keep_whitespace=False)
        except Exception:
            tokens = []
        else:
            thai_tokens = [t for t in tokens if any(_is_thai_text_char(ch) for ch in t)]
            if thai_tokens:
                return thai_tokens
    return _THAI_BLOCK_RE.findall(text)


def _empty_language_stats() -> dict[str, Any]:
    """โครง stats กลางสำหรับรวม evidence ระดับ cell/column/dataset."""
    return {
        "sample_size": 0,
        "non_empty": 0,
        "thai_chars": 0,
        "latin_chars": 0,
        "thai_digits": 0,
        "digits": 0,
        "char_count": 0,
        "thai_vowel_tone_chars": 0,
        "thai_punctuation_chars": 0,
        "zero_width_chars": 0,
        "thai_word_tokens": 0,
        "english_word_tokens": 0,
        "common_thai_word_hits": 0,
        "thai_cells": 0,
        "english_cells": 0,
        "mixed_cells": 0,
        "text_cells": 0,
        "code_like_cells": 0,
        "examples": [],
    }


def _merge_language_stats(total: dict[str, Any], part: dict[str, Any]) -> None:
    """รวม stats แบบ in-place โดยเว้น examples ไว้จำกัดจำนวน."""
    for key, value in part.items():
        if key == "examples":
            continue
        if isinstance(value, int):
            total[key] = int(total.get(key, 0)) + value
    examples = total.setdefault("examples", [])
    for example in part.get("examples", []):
        if len(examples) >= 3:
            break
        examples.append(example)


def _analyze_language_text(value: str) -> dict[str, Any]:
    """วิเคราะห์หนึ่ง cell ด้วย Unicode block + token/common-word heuristics."""
    stats = _empty_language_stats()
    text = str(value)
    stats["sample_size"] = 1
    stats["zero_width_chars"] = len(_ZERO_WIDTH_RE.findall(text))
    cleaned = _ZERO_WIDTH_RE.sub("", text).strip()
    if not cleaned:
        return stats

    stats["non_empty"] = 1
    stats["char_count"] = len(cleaned)
    for ch in cleaned:
        cp = ord(ch)
        if _THAI_DIGIT_RANGE[0] <= cp <= _THAI_DIGIT_RANGE[1]:
            stats["thai_digits"] += 1
        elif _is_thai_text_char(ch):
            stats["thai_chars"] += 1
            if _is_thai_vowel_or_tone(ch):
                stats["thai_vowel_tone_chars"] += 1
        elif _THAI_RANGE[0] <= cp <= _THAI_RANGE[1]:
            stats["thai_punctuation_chars"] += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            stats["latin_chars"] += 1
        elif "0" <= ch <= "9":
            stats["digits"] += 1

    has_thai = stats["thai_chars"] > 0
    has_english = stats["latin_chars"] > 0
    stats["common_thai_word_hits"] = sum(cleaned.count(w) for w in _COMMON_THAI_WORDS)
    stats["thai_word_tokens"] = len(_thai_word_tokens(cleaned)) if has_thai else 0
    stats["english_word_tokens"] = len(_LATIN_WORD_RE.findall(cleaned))

    if has_thai:
        stats["thai_cells"] = 1
    if has_english:
        stats["english_cells"] = 1
    if has_thai and has_english:
        stats["mixed_cells"] = 1
    if has_thai or has_english:
        stats["text_cells"] = 1
        stats["examples"] = [cleaned[:40]]

    if _CODE_LIKE_RE.fullmatch(cleaned) and any(ch.isdigit() for ch in cleaned):
        stats["code_like_cells"] = 1
    return stats


def _ratio(numerator: int | float, denominator: int | float) -> float:
    """Safe division for language heuristics."""
    return float(numerator) / float(denominator) if denominator else 0.0


def _classify_language_stats(stats: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    """จำแนกภาษาและ confidence จาก stats ที่รวมแล้ว."""
    non_empty = int(stats.get("non_empty", 0))
    thai_chars = int(stats.get("thai_chars", 0))
    latin_chars = int(stats.get("latin_chars", 0))
    text_chars = thai_chars + latin_chars
    thai_tokens = int(stats.get("thai_word_tokens", 0))
    english_tokens = int(stats.get("english_word_tokens", 0))
    word_tokens = thai_tokens + english_tokens

    ratios = {
        "thai_ratio": _ratio(thai_chars, text_chars),
        "english_ratio": _ratio(latin_chars, text_chars),
        "thai_cell_ratio": _ratio(int(stats.get("thai_cells", 0)), non_empty),
        "english_cell_ratio": _ratio(int(stats.get("english_cells", 0)), non_empty),
        "mixed_cell_ratio": _ratio(int(stats.get("mixed_cells", 0)), non_empty),
        "thai_word_ratio": _ratio(thai_tokens, word_tokens),
        "english_word_ratio": _ratio(english_tokens, word_tokens),
        "code_like_ratio": _ratio(int(stats.get("code_like_cells", 0)), non_empty),
    }

    if text_chars == 0 and word_tokens == 0:
        # เลขไทย/สัญลักษณ์ไทยใน object column เป็นสัญญาณว่าควรเปิด Thai-specific checks
        if int(stats.get("thai_digits", 0)) > 0 or int(stats.get("thai_punctuation_chars", 0)) > 0:
            return "thai", 0.65, ratios
        return "numeric", 1.0, ratios

    # รหัสเช่น ORD001/SKU-123 ไม่ใช่ภาษาอังกฤษเชิงเนื้อหา แม้มี A-Z ปนตัวเลข
    if thai_chars == 0 and ratios["code_like_ratio"] >= 0.80:
        return "numeric", 0.9, ratios

    common_bonus = min(0.18, int(stats.get("common_thai_word_hits", 0)) * 0.04)
    mark_bonus = 0.08 if int(stats.get("thai_vowel_tone_chars", 0)) > 0 else 0.0
    thai_strength = min(
        1.0,
        max(ratios["thai_ratio"], ratios["thai_cell_ratio"], ratios["thai_word_ratio"])
        + common_bonus
        + mark_bonus,
    )
    english_strength = max(
        ratios["english_ratio"], ratios["english_cell_ratio"], ratios["english_word_ratio"]
    )

    has_thai = thai_chars > 0 or int(stats.get("common_thai_word_hits", 0)) > 0
    has_english = latin_chars > 0 or english_tokens > 0
    if has_thai and has_english:
        balance = 1.0 - abs(ratios["thai_ratio"] - ratios["english_ratio"])
        coverage = max(
            ratios["mixed_cell_ratio"],
            min(ratios["thai_cell_ratio"], ratios["english_cell_ratio"]),
            min(ratios["thai_word_ratio"], ratios["english_word_ratio"]),
        )
        confidence = 0.45 + 0.30 * balance + 0.20 * coverage + common_bonus + mark_bonus
        return "mixed", round(min(1.0, confidence), 3), ratios

    if has_thai:
        confidence = 0.45 + 0.45 * thai_strength + 0.10 * ratios["thai_cell_ratio"]
        return "thai", round(min(1.0, confidence), 3), ratios

    confidence = 0.45 + 0.45 * english_strength + 0.10 * ratios["english_cell_ratio"]
    return "english", round(min(1.0, confidence), 3), ratios


def _summarize_language_column(series: pd.Series, sample_rows: int) -> dict[str, Any]:
    """สรุปภาษาในคอลัมน์เดียวจาก sample แรก เพื่อ performance บนไฟล์ใหญ่."""
    stats = _empty_language_stats()
    values = series.dropna().astype(str).head(sample_rows)
    for value in values:
        _merge_language_stats(stats, _analyze_language_text(value))

    language, confidence, ratios = _classify_language_stats(stats)
    return {
        "language": language,
        "confidence": confidence,
        "sample_size": int(stats["sample_size"]),
        "non_empty": int(stats["non_empty"]),
        "thai_ratio": round(ratios["thai_ratio"], 4),
        "english_ratio": round(ratios["english_ratio"], 4),
        "thai_cell_ratio": round(ratios["thai_cell_ratio"], 4),
        "english_cell_ratio": round(ratios["english_cell_ratio"], 4),
        "mixed_cell_ratio": round(ratios["mixed_cell_ratio"], 4),
        "thai_word_ratio": round(ratios["thai_word_ratio"], 4),
        "common_thai_word_hits": int(stats["common_thai_word_hits"]),
        "thai_vowel_tone_chars": int(stats["thai_vowel_tone_chars"]),
        "zero_width_chars": int(stats["zero_width_chars"]),
        "char_count": int(stats["char_count"]),
        "examples": list(stats["examples"]),
        "_stats": stats,
    }


def _dataset_language_from_columns(column_details: dict[str, dict[str, Any]]) -> tuple[str, float]:
    """ตัดสินภาษาทั้ง dataset จากผลรายคอลัมน์ แทนการนับ char รวมอย่างเดียว."""
    text_details = [d for d in column_details.values() if d.get("language") != "numeric"]
    if not text_details:
        return "numeric", 1.0

    thai_cols = [d for d in text_details if d.get("language") == "thai"]
    english_cols = [d for d in text_details if d.get("language") == "english"]
    mixed_cols = [d for d in text_details if d.get("language") == "mixed"]
    avg_conf = sum(float(d.get("confidence", 0.0)) for d in text_details) / len(text_details)

    if mixed_cols:
        return "mixed", round(max(0.65, avg_conf), 3)
    if thai_cols and english_cols:
        # ถ้ามีคอลัมน์ไทยชัดเจนและคอลัมน์อังกฤษเป็น minority สั้น ๆ (ชื่อ, code, label)
        # ให้จัด dataset เป็นไทย เพื่อไม่ปิด Thai-specific checks จาก metadata อังกฤษ
        english_cell_count = sum(int(d.get("non_empty", 0)) for d in english_cols)
        thai_cell_count = sum(int(d.get("non_empty", 0)) for d in thai_cols)
        english_chars = sum(int(d.get("char_count", 0)) for d in english_cols)
        if thai_cell_count >= english_cell_count and english_chars <= 24:
            return "thai", round(max(0.65, avg_conf), 3)
        return "mixed", round(max(0.65, avg_conf), 3)
    if thai_cols:
        return "thai", round(avg_conf, 3)
    return "english", round(avg_conf, 3)


def _detect_language(df: pd.DataFrame) -> dict[str, Any]:
    """ตรวจภาษาแบบ column-aware ด้วย Unicode block, token, common-word และ sample heuristics.

    คืนค่า backward-compatible keys เดิม (`language`, `columns`, `thai_ratio`, `evidence`)
    และเพิ่ม `confidence`, `english_ratio`, `column_details`, `sample_rows` สำหรับ v2.
    ใช้ตัวอย่างแรกไม่เกิน 500 แถว/คอลัมน์เพื่อให้เร็วบน DataFrame ใหญ่ และ lazy-import
    pythainlp เฉพาะตอนต้องตัดคำไทยเท่านั้น
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("_detect_language expects a pandas DataFrame.")

    sample_rows = min(_LANGUAGE_SAMPLE_ROWS, len(df)) if len(df) else _LANGUAGE_SAMPLE_ROWS
    column_languages: dict[str, str] = {}
    column_details: dict[str, dict[str, Any]] = {}
    evidence: list[str] = []
    dataset_stats = _empty_language_stats()

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ):
            detail = {
                "language": "numeric",
                "confidence": 1.0,
                "sample_size": int(min(sample_rows, int(series.notna().sum()))),
                "non_empty": int(series.notna().head(sample_rows).sum()),
                "thai_ratio": 0.0,
                "english_ratio": 0.0,
                "thai_cell_ratio": 0.0,
                "english_cell_ratio": 0.0,
                "mixed_cell_ratio": 0.0,
                "thai_word_ratio": 0.0,
                "common_thai_word_hits": 0,
                "thai_vowel_tone_chars": 0,
                "zero_width_chars": 0,
                "char_count": 0,
                "examples": [],
            }
            column_languages[col_name] = "numeric"
            column_details[col_name] = detail
            continue

        detail = _summarize_language_column(series, sample_rows)
        stats = detail.pop("_stats")
        _merge_language_stats(dataset_stats, stats)
        language = str(detail["language"])
        column_languages[col_name] = language
        column_details[col_name] = detail

        if detail["non_empty"]:
            extras: list[str] = []
            if detail["common_thai_word_hits"]:
                extras.append(f"common words {detail['common_thai_word_hits']}")
            if detail["thai_vowel_tone_chars"]:
                extras.append(f"vowel/tone {detail['thai_vowel_tone_chars']}")
            if detail["zero_width_chars"]:
                extras.append(f"ZWSP {detail['zero_width_chars']}")
            extra_txt = f", {', '.join(extras)}" if extras else ""
            evidence.append(
                f"{col_name}: {language} (conf {detail['confidence']:.0%}, "
                f"ไทย {detail['thai_ratio']:.1%}, อังกฤษ {detail['english_ratio']:.1%}, "
                f"sample {detail['non_empty']}{extra_txt})"
            )

    language, confidence = _dataset_language_from_columns(column_details)
    _dataset_lang, _dataset_conf, ratios = _classify_language_stats(dataset_stats)
    thai_ratio = ratios["thai_ratio"]
    english_ratio = ratios["english_ratio"]
    if language == "numeric":
        confidence = _dataset_conf

    return {
        "language": language,
        "confidence": round(float(confidence), 3),
        "columns": column_languages,
        "column_details": column_details,
        "thai_ratio": round(thai_ratio, 4),
        "english_ratio": round(english_ratio, 4),
        "sample_rows": sample_rows,
        "evidence": evidence[:8],
    }


def _name_is_id_or_fk(series: pd.Series) -> bool:
    """ชื่อคอลัมน์เป็น primary id หรือ foreign key แบบ *_id หรือไม่."""
    name = str(series.name).lower() if series.name is not None else ""
    return name == "id" or name.endswith("_id")


_ID_LIKE_KEYWORDS = ("postal", "zipcode", "zip_code", "zip")


def _name_hints_id(series: pd.Series) -> bool:
    """ชื่อคอลัมน์บอกใบ้ว่าเป็น ID หรือไม่ (เช่น 'id', 'user_id', 'postal code')."""
    name = str(series.name).lower() if series.name is not None else ""
    if _name_is_id_or_fk(series) or name.endswith("id"):
        return True
    # รหัสไปรษณีย์ (postal code, zip) เป็น identifier ไม่ใช่ measure ทางสถิติ
    return any(kw in name for kw in _ID_LIKE_KEYWORDS)


def _looks_like_id(series: pd.Series, non_null: pd.Series) -> bool:
    """เดาว่าเป็นคอลัมน์ตัวระบุ (ID/FK) เชิงตัวเลข.

    ถ้าชื่อเป็น ``id`` หรือ ``*_id`` ให้ถือเป็น ID/FK แม้ค่าซ้ำเยอะ (foreign key)
    ส่วนชื่อที่แค่ลงท้าย ``id`` แบบหลวม ๆ ยังต้องไม่ซ้ำเกือบทั้งหมดเพื่อกัน false positive
    จากค่าวัดตัวเลขทั่วไปที่บังเอิญไม่ซ้ำ.
    """
    if _name_is_id_or_fk(series):
        return True
    n = len(non_null)
    if n < 5:
        return False
    if non_null.nunique() / n < 0.95:
        return False
    return _name_hints_id(series)


def _looks_like_string_id(series: pd.Series, non_null: pd.Series, str_sample: list[str]) -> bool:
    """เดาว่าเป็น ID แบบสตริง (รหัส/UUID) — ไม่ซ้ำ, token เดี่ยว, สั้น, ไม่ใช่ข้อความไทยยาว."""
    if _name_is_id_or_fk(series):
        return True
    n = len(non_null)
    if n < 5:
        return False
    if non_null.nunique() / n < 0.95:
        return False
    single_token = all(" " not in s for s in str_sample)
    avg_len = sum(len(s) for s in str_sample) / len(str_sample)
    not_thai_text = (
        sum(1 for s in str_sample if _thai_content_ratio(s) > 0.30) / len(str_sample) < 0.5
    )
    # ต้องเป็น token เดี่ยว สั้น และไม่ใช่ข้อความไทย หรืออย่างน้อยชื่อบอกใบ้
    return (
        single_token
        and avg_len <= 40
        and not_thai_text
        and (_name_hints_id(series) or avg_len <= 24)
    )


# ----------------------------------------------------------------------------
# ตรวจจับเบอร์โทรศัพท์ไทย
# ----------------------------------------------------------------------------
# รูปแบบเบอร์ไทย: 0812345678, 08-1234-5678, 081-234-5678, +668****5678
# ลักษณะเด่น: 10 หลัก (หลังตัด prefix/symbol), ขึ้นต้นด้วย 0 หรือ +66
_PHONE_NAME_HINTS = ("phone", "tel", "mobile", "เบอร์", "โทร", "contact")

# pattern เบอร์ไทย (หลัง strip สัญลักษณ์แล้ว): 10 หลัก ขึ้นต้น 0 หรือ +66 + 9 หลัก
_THAI_PHONE_RE = re.compile(r"^(?:\+66|0)\d{9}$")

# เลขไทย → อารบิก (สำหรับแปลงเบอร์ที่พิมพ์ด้วยเลขไทย)
_THAI_DIGIT_MAP = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def _clean_phone_str(value: str) -> str:
    """ลบสัญลักษณ์ออกจากเบอร์โทร — แปลงเลขไทย, ลบ dash/space/parenthesis, +66 → 0."""
    s = value.strip()
    s = s.translate(_THAI_DIGIT_MAP)  # เลขไทย → อารบิก
    s = re.sub(r"[-\s()\.]", "", s)
    if s.startswith("+66"):
        s = "0" + s[3:]
    return s


def _looks_like_phone_column(series: pd.Series, str_sample: list[str]) -> bool:
    """เดาว่าคอลัมน์เป็นเบอร์โทรศัพท์ไทยหรือไม่.

    เงื่อนไข: ชื่อบอกใบ้ (phone/tel/mobile/เบอร์/โทร) + ≥50% ตรงรูปแบบ
    หรือไม่มีชื่อบอกใบ้แต่ ≥70% ตรงรูปแบบเบอร์ไทย
    """
    if not str_sample:
        return False
    name = str(series.name).lower() if series.name is not None else ""
    name_hints = any(h in name for h in _PHONE_NAME_HINTS)

    matches = sum(1 for s in str_sample if _THAI_PHONE_RE.match(_clean_phone_str(s)))
    ratio = matches / len(str_sample)
    if name_hints and ratio >= 0.50:
        return True
    return ratio >= 0.70


def normalize_phone_number(value: object) -> str:
    """ทำความสะอาดเบอร์โทรศัพท์ไทย — ละลักษณะเป็น 10 หลักขึ้นต้น 0.

    แปลงเลขไทยเป็นอารบิก, ลบ dash/space/parenthesis, เปลี่ยน +66 → 0.
    คืนค่าเดิม (เป็น str) ถ้าไม่ใช่เบอร์ไทย
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return str(value) if value is not None else ""
    cleaned = _clean_phone_str(str(value))
    if _THAI_PHONE_RE.match(cleaned):
        return cleaned
    return str(value)


def is_phone_number(value: object) -> bool:
    """ตรวจว่าค่าเป็นเบอร์โทรศัพท์ไทยหรือไม่ (หลังทำความสะอาด)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(_THAI_PHONE_RE.match(_clean_phone_str(str(value))))


def clean_phone_string(value: str) -> str:
    """ลบสัญลักษณ์จากเบอร์โทร — alias ของ _clean_phone_str สำหรับ public API."""
    return _clean_phone_str(str(value))


def detect_column_type(series: pd.Series) -> ColumnType:
    """จำแนกประเภทของ pandas Series หนึ่งคอลัมน์.

    ลำดับการตัดสิน: empty -> datetime -> id/fk -> numeric -> low-cardinality labels
    -> text(thai/eng/mixed) -> categorical.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return ColumnType.EMPTY

    # --- datetime ---
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    # ID/FK จากชื่อคอลัมน์ต้องมาก่อน numeric เพื่อไม่ให้ order_id/store_id
    # ที่ซ้ำเยอะ (foreign key) ถูกตีเป็นตัวเลขเชิงสถิติ
    if _name_is_id_or_fk(series):
        return ColumnType.ID

    # --- numeric (รวมกรณี dtype เป็นเลขอยู่แล้ว) ---
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if _looks_like_id(series, non_null):
            return ColumnType.ID
        return ColumnType.NUMERIC

    if pd.api.types.is_bool_dtype(series):
        return ColumnType.CATEGORICAL

    # --- สุ่มตัวอย่างไม่เกิน 1000 ค่า เพื่อตรวจ object column ---
    sample = non_null.head(1000)
    str_sample = [str(v) for v in sample]
    n = len(str_sample)

    # ตรวจเบอร์โทรศัพท์ — ต้องเช็คก่อน numeric เพราะเบอร์โทรเป็นเลขล้วน
    # แต่ไม่ควรถือเป็น numeric สำหรับสถิติ (เบอร์ไม่มีความหมายทางสถิติ)
    if _looks_like_phone_column(series, str_sample):
        return ColumnType.PHONE_NUMBER

    # ลองตีความเป็นตัวเลข (object ที่จริง ๆ เป็นเลข)
    coerced_num = pd.to_numeric(pd.Series(str_sample), errors="coerce")
    num_valid = coerced_num.notna().mean()
    if num_valid >= 0.95:
        as_series = pd.Series(coerced_num.dropna().values)
        if _looks_like_id(series, non_null):
            return ColumnType.ID
        # ถ้าค่าน้อยและไม่ซ้ำมาก ถือเป็น numeric
        return ColumnType.NUMERIC if as_series.nunique() > 1 else ColumnType.CATEGORICAL

    # ลองตีความเป็นวันที่
    if _looks_like_datetime(str_sample):
        return ColumnType.DATETIME

    # ตัวระบุแบบสตริง (รหัส/UUID) ก่อนวิเคราะห์เป็นข้อความ
    if _looks_like_string_id(series, non_null, str_sample):
        return ColumnType.ID

    nunique = non_null.nunique()
    unique_ratio = nunique / len(non_null)
    avg_len = sum(len(s.strip()) for s in str_sample) / n

    # หมวดหมู่คำสั้นซ้ำ ๆ ต้องชนะการตรวจภาษาไทย/ผสม เช่น serve_type
    # (ร้อน/เย็น/ปั่น) หรือ payment_method (QR/PromptPay) ไม่ใช่ free text
    # แต่ถ้าชื่อคอลัมน์บอกว่าเป็นข้อความ/review/feedback ให้ข้าม เพราะประโยคสั้นซ้ำ ๆ ก็ยังเป็น text
    col_name = str(series.name).lower() if series.name is not None else ""
    if (
        nunique <= _LOW_CARDINALITY_MAX_UNIQUE
        and avg_len < _SHORT_LABEL_AVG_LEN
        and not _TEXT_NAME_HINTS_RE.search(col_name)
    ):
        return ColumnType.CATEGORICAL

    # --- วิเคราะห์สคริปต์ของข้อความด้วยสัดส่วนรวม (mean ต่อเซลล์) ---
    mean_thai = sum(_thai_content_ratio(s) for s in str_sample) / n
    mean_latin = sum(_latin_content_ratio(s) for s in str_sample) / n

    # ไทยเด่นชัด และละตินน้อย -> ข้อความไทย
    if mean_thai >= 0.50 and mean_latin < 0.20:
        return ColumnType.THAI_TEXT

    # มีไทยพอสมควร (15%+) แต่ปนกับละตินอย่างมีนัยสำคัญ หรือไทยอยู่ช่วง 15–50% -> ผสม
    if mean_thai >= 0.15 and (mean_latin >= 0.20 or mean_thai < 0.50):
        return ColumnType.MIXED_TEXT

    # ไทยล้วนแต่สั้นมาก (เช่น คำเดียว) ที่หลุดเงื่อนไขข้างบน
    if mean_thai >= 0.50:
        return ColumnType.THAI_TEXT

    # ข้อความที่ไม่ใช่ไทย: ตัดสินว่าเป็น categorical หรือ english_text
    # cardinality check ก่อน
    avg_tokens = sum(len(s.split()) for s in str_sample) / n

    if (nunique < 50 or unique_ratio < 0.05) and avg_tokens < 4:
        return ColumnType.CATEGORICAL

    # มีอักษรละตินพอสมควร และเป็นข้อความยาว ๆ
    has_latin = sum(1 for s in str_sample if script_ratio(s)["latin"] > 0.30) / n
    if has_latin > 0.30 or avg_tokens >= 2:
        return ColumnType.ENGLISH_TEXT

    # default — ถือเป็นหมวดหมู่
    return ColumnType.CATEGORICAL


def _looks_like_datetime(values: list[str]) -> bool:
    """เดาว่ารายการสตริงเป็นวันที่หรือไม่ (parse ผ่าน >90%) — v0.8: รองรับ Thai month names."""
    if not values:
        return False
    # v1.x (D1): ค่าที่ขึ้นต้นด้วยตัวอักษร 2+ ตัวตามด้วยตัวคั่น (เช่น "CA-2017-152156",
    # "US-2016-108966", "OFF-EN-10001492") เป็นรหัส/ID ไม่ใช่วันที่ — กัน false positive
    # จาก segment ที่มีรูป 9999-9999 ภายในรหัส
    _id_prefix_re = re.compile(r"^[A-Za-z]{2,}[-_/.]")
    id_prefix_count = sum(1 for v in values if _id_prefix_re.match(v.strip()))
    if id_prefix_count / len(values) >= 0.6:
        return False
    # ต้องมีตัวคั่นแบบวันที่ ป้องกัน false positive จากเลขล้วน
    date_like = re.compile(r"\d{1,4}[-/.]\d{1,2}([-/.]\d{1,4})?|\d{4}[-/]\d{2}")
    # v0.8: เพิ่ม pattern สำหรับ Thai month names (เช่น "15 มกราคม 2567", "1 ก.พ. 67")
    _THAI_MONTH_NAMES = (
        "มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|"
        "ตุลาคม|พฤศจิกายน|ธันวาคม|ม.ค.|ก.พ.|มี.ค.|เม.ย.|พ.ค.|มิ.ย.|ก.ค.|ส.ค.|ก.ย.|ต.ค.|พ.ย.|ธ.ค."
    )
    thai_date_re = re.compile(rf"\d{{1,2}}\s+({_THAI_MONTH_NAMES})\s+\d{{2,4}}", re.IGNORECASE)

    date_count = sum(1 for v in values if date_like.search(v) or thai_date_re.search(v))
    if date_count / len(values) < 0.6:
        return False
    # ลอง parse — แปลง Thai month และ พ.ศ. ก่อน parse เสมอ
    thai_month_map = {
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
    thai_month_re = re.compile(
        r"(\d{1,2})\s+(" + "|".join(thai_month_map.keys()) + r")\s+(\d{2,4})",
        re.IGNORECASE,
    )
    year_re = re.compile(
        r"\b(24\d{2}|25\d{2})\b|"
        r"(\b\d{1,2}/\d{1,2}/)(\d{2})\b|"
        r"(\b\d{1,2}-\d{1,2}-)(\d{2})\b"
    )

    def _replace_be_year(m):
        if m.group(1):
            return str(int(m.group(1)) - 543)
        elif m.group(3):
            prefix = m.group(2)
            y_val = int(m.group(3))
            be_year = (2500 + y_val) if y_val <= 75 else (2400 + y_val)
            return f"{prefix}{be_year - 543}"
        elif m.group(5):
            prefix = m.group(4)
            y_val = int(m.group(5))
            be_year = (2500 + y_val) if y_val <= 75 else (2400 + y_val)
            return f"{prefix}{be_year - 543}"
        return m.group(0)

    def _sub_month(m):
        day = m.group(1).zfill(2)
        month = thai_month_map.get(m.group(2), m.group(2))
        year = m.group(3)
        return f"{day}/{month}/{year}"

    converted_values = []
    for v in values:
        v_conv = thai_month_re.sub(_sub_month, v)
        v_conv = year_re.sub(_replace_be_year, v_conv)
        converted_values.append(v_conv)

    parsed = pd.to_datetime(pd.Series(converted_values), errors="coerce", format="mixed")
    return bool(parsed.notna().mean() > 0.90)


def detect_all(df: pd.DataFrame) -> dict[str, ColumnType]:
    """จำแนกทุกคอลัมน์ใน DataFrame คืน mapping ชื่อคอลัมน์ -> ColumnType."""
    _warn_if_likely_wrong_semicolon_delimiter(df)
    return {str(col): detect_column_type(df[col]) for col in df.columns}


def _warn_if_likely_wrong_semicolon_delimiter(df: pd.DataFrame) -> None:
    """เตือนกรณี CSV คั่นด้วย ';' แต่ถูกอ่านเป็นคอลัมน์เดียว."""
    if len(df.columns) != 1 or df.empty:
        return

    sample = df.iloc[:, 0].dropna().astype(str).head(100)
    if sample.empty:
        return

    semicolon_counts = sample.str.count(";")
    rows_with_many = int((semicolon_counts >= _SEMICOLON_DELIMITER_MIN_COUNT).sum())
    min_rows = max(3, int(len(sample) * 0.5))
    if rows_with_many < min_rows:
        return

    warnings.warn(
        "DataFrame has one column but many values contain repeated ';'. "
        "The CSV may have been read with the wrong delimiter; try pd.read_csv(..., sep=';').",
        UserWarning,
        stacklevel=2,
    )


__all__ = [
    "ColumnType",
    "script_ratio",
    "is_thai_text",
    "_detect_language",
    "detect_column_type",
    "detect_all",
    "normalize_phone_number",
    "is_phone_number",
    "clean_phone_string",
    "parse_thai_address",
    "parse_thai_address_column",
]
