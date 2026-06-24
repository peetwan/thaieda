"""ทดสอบ thaieda.clean — ฟังก์ชันทำความสะอาดข้อความไทย."""

from __future__ import annotations

import unicodedata

import pandas as pd
import pytest

from thaieda.clean import (
    CleaningResult,
    clean_thai_text,
    fix_repeated_chars,
    fix_tone_mark_stacking,
    normalize_encoding,
    normalize_thai_numerals,
    normalize_unicode,
    remove_zero_width_chars,
    strip_whitespace,
)

MOJIBAKE = "สวัสดี".encode().decode("latin-1")


# ------------------------------------------------------------- zero-width
def test_remove_zero_width_chars():
    s = pd.Series(["ก​ข", "ปกติ"])  # มี U+200B คั่นระหว่าง ก ข
    cleaned, result = remove_zero_width_chars(s)
    assert cleaned.iloc[0] == "กข"
    assert "​" not in cleaned.iloc[0]
    assert result.rows_affected == 1
    assert result.operation == "remove_zero_width_chars"


# ------------------------------------------------------------- thai numerals
def test_normalize_thai_numerals():
    s = pd.Series(["๑๒๓", "456", "๗๘๙"])
    cleaned, result = normalize_thai_numerals(s)
    assert cleaned.iloc[0] == "123"
    assert cleaned.iloc[2] == "789"
    assert result.rows_affected == 2


# ------------------------------------------------------------- whitespace
def test_strip_whitespace():
    s = pd.Series(["  hello   world  ", "ok"])
    cleaned, result = strip_whitespace(s)
    assert cleaned.iloc[0] == "hello world"
    assert result.rows_affected == 1


def test_strip_whitespace_nbsp():
    s = pd.Series(["a b"])
    cleaned, result = strip_whitespace(s)
    assert cleaned.iloc[0] == "a b"
    assert result.rows_affected == 1


# ------------------------------------------------------------- unicode
def test_normalize_unicode():
    decomposed = unicodedata.normalize("NFD", "é")  # e + combining acute
    s = pd.Series([decomposed])
    cleaned, result = normalize_unicode(s, "NFC")
    assert cleaned.iloc[0] == unicodedata.normalize("NFC", decomposed)
    assert result.rows_affected == 1


def test_normalize_unicode_bad_form_raises():
    s = pd.Series(["x"])
    with pytest.raises(ValueError):
        normalize_unicode(s, "BOGUS")


# ------------------------------------------------------------- repeated chars
def test_fix_repeated_chars():
    s = pd.Series(["55555", "ๆๆๆ", "ปกติ"])
    cleaned, result = fix_repeated_chars(s, max_repeat=3)
    assert cleaned.iloc[0] == "555"
    assert cleaned.iloc[1] == "ๆ"
    assert result.rows_affected == 2


# ------------------------------------------------------------- tone marks
def test_fix_tone_mark_stacking():
    s = pd.Series(["น้้ำ", "ปกติ"])
    cleaned, result = fix_tone_mark_stacking(s)
    assert "้้" not in cleaned.iloc[0]
    assert result.rows_affected == 1


# ------------------------------------------------------------- encoding
def test_normalize_encoding_fixes_mojibake():
    s = pd.Series([MOJIBAKE, "ปกติ"])
    cleaned, result = normalize_encoding(s)
    assert cleaned.iloc[0] == "สวัสดี"
    assert result.rows_affected == 1


# ------------------------------------------------------------- composite
def test_clean_thai_text_composite():
    s = pd.Series(["  ๑๒๓  ", "55555", "ก​ข"])
    cleaned, results = clean_thai_text(s)
    assert isinstance(results, list)
    assert all(isinstance(r, CleaningResult) for r in results)
    # row0: strip -> "๑๒๓" -> numerals -> "123"
    assert cleaned.iloc[0] == "123"
    # row2: zero-width removed
    assert cleaned.iloc[2] == "กข"


def test_clean_thai_text_custom_operations():
    s = pd.Series(["๑๒๓"])
    cleaned, results = clean_thai_text(s, operations=["numerals"])
    assert cleaned.iloc[0] == "123"
    assert len(results) == 1
    assert results[0].operation == "normalize_thai_numerals"


def test_clean_thai_text_all_keyword():
    s = pd.Series(["๑๒๓"])
    cleaned, results = clean_thai_text(s, operations=["all"])
    assert cleaned.iloc[0] == "123"
    assert len(results) >= 1


def test_clean_thai_text_unknown_operation_raises():
    s = pd.Series(["x"])
    with pytest.raises(ValueError):
        clean_thai_text(s, operations=["bogus"])


# ------------------------------------------------------------- dataclass
def test_cleaning_result_fields():
    s = pd.Series(["๑"])
    _, result = normalize_thai_numerals(s)
    assert result.operation == "normalize_thai_numerals"
    assert result.column == ""  # series ไม่มีชื่อ
    assert result.rows_affected == 1
    assert isinstance(result.before_examples, list)
    assert isinstance(result.after_examples, list)
    assert isinstance(result.description_th, str)
    d = result.to_dict()
    expected_keys = {
        "operation",
        "rows_affected",
        "column",
        "before_examples",
        "after_examples",
        "description_th",
    }
    assert expected_keys <= set(d)


def test_cleaning_preserves_column_name():
    s = pd.Series(["๑๒๓"], name="price")
    _, result = normalize_thai_numerals(s)
    assert result.column == "price"


def test_clean_then_anomaly_free():
    # ทำความสะอาดแล้วความผิดปกติเชิงข้อความควรลดลง/หายไป
    from thaieda.anomaly import detect_text_anomalies

    s = pd.Series(["55555", MOJIBAKE, "ปกติ", "ดีมาก"])
    before = detect_text_anomalies(s, tokenizer=None)
    assert before  # มีความผิดปกติก่อนทำความสะอาด

    cleaned, _ = clean_thai_text(s)
    after = detect_text_anomalies(cleaned, tokenizer=None)
    after_names = {i.check_name for i in after}
    assert "encoding_mojibake" not in after_names
    assert "excessive_repetition" not in after_names
