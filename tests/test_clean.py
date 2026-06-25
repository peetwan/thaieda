"""ทดสอบ thaieda.clean — ฟังก์ชันทำความสะอาดข้อความไทย."""

from __future__ import annotations

import sys
import unicodedata

import pandas as pd
import pytest

from thaieda.clean import (
    CleaningResult,
    clean_thai_text,
    fix_keyboard_layout,
    fix_repeated_chars,
    fix_tone_mark_stacking,
    normalize_encoding,
    normalize_thai_numerals,
    normalize_unicode,
    pythainlp_normalize,
    remove_zero_width_chars,
    strip_whitespace,
)

MOJIBAKE = "สวัสดี".encode().decode("latin-1")


def _ftfy_installed() -> bool:
    try:
        import ftfy  # noqa: F401
    except ImportError:
        return False
    return True


def _pythainlp_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("pythainlp") is not None


requires_pythainlp = pytest.mark.skipif(
    not _pythainlp_installed(), reason="pythainlp not installed"
)


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


def test_cleaning_result_has_explanation_field():
    # ฟิลด์ explanation ต้องมีเสมอ (ค่าเริ่มต้นเป็นสตริงว่าง) และอยู่ใน to_dict
    s = pd.Series(["๑"])
    _, result = normalize_thai_numerals(s)
    assert hasattr(result, "explanation")
    assert isinstance(result.explanation, str)
    assert "explanation" in result.to_dict()


@pytest.mark.skipif(not _ftfy_installed(), reason="ftfy not installed")
def test_normalize_encoding_explanation_with_ftfy():
    # เมื่อมี ftfy: ใช้ fix_and_explain -> explanation ต้องอธิบายว่าซ่อมอะไร
    s = pd.Series([MOJIBAKE, "ปกติ"])
    cleaned, result = normalize_encoding(s)
    assert cleaned.iloc[0] == "สวัสดี"
    assert result.explanation  # ไม่ว่าง
    assert "ftfy" in result.explanation


def test_normalize_encoding_without_ftfy_falls_back(monkeypatch):
    # จำลองว่าไม่มี ftfy -> ใช้วิธี manual, ยังแก้ mojibake ได้ แต่ explanation ว่าง
    monkeypatch.setitem(sys.modules, "ftfy", None)
    s = pd.Series([MOJIBAKE, "ปกติ"])
    cleaned, result = normalize_encoding(s)
    assert cleaned.iloc[0] == "สวัสดี"
    assert result.rows_affected == 1
    assert result.explanation == ""


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


# ------------------------------------------------------ pythainlp_normalize
@requires_pythainlp
def test_pythainlp_normalize_fixes_spelling():
    # 'เเปลก' (สระเอสองตัว) -> 'แปลก', 'นานาาา' (สระซ้ำ) -> 'นานา'
    s = pd.Series(["เเปลก", "นานาาา", "ปกติ"])
    cleaned, result = pythainlp_normalize(s)
    assert cleaned.iloc[0] == "แปลก"
    assert cleaned.iloc[1] == "นานา"
    assert result.operation == "pythainlp_normalize"
    assert result.rows_affected == 2


@requires_pythainlp
def test_pythainlp_normalize_no_change_returns_zero():
    s = pd.Series(["สวัสดีครับ", "ขอบคุณ"])
    cleaned, result = pythainlp_normalize(s)
    assert cleaned.iloc[0] == "สวัสดีครับ"
    assert result.rows_affected == 0


def test_pythainlp_normalize_without_pythainlp_raises(monkeypatch):
    # ไม่มี pythainlp -> ต้อง fail loudly (ไม่ silent fallback)
    monkeypatch.setitem(sys.modules, "pythainlp", None)
    monkeypatch.setitem(sys.modules, "pythainlp.util", None)
    s = pd.Series(["x"])
    with pytest.raises(ImportError):
        pythainlp_normalize(s)


# ------------------------------------------------------ fix_keyboard_layout
@requires_pythainlp
def test_fix_keyboard_layout_fixes_mistyped():
    from pythainlp.util import thai_to_eng

    mistyped = thai_to_eng("ขอบคุณมาก")  # ลืมสลับแป้น -> ได้ตัวอักษรอังกฤษ
    s = pd.Series([mistyped, "hello", "ปกติ"])
    cleaned, result = fix_keyboard_layout(s)
    assert cleaned.iloc[0] == "ขอบคุณมาก"
    assert cleaned.iloc[1] == "hello"  # ภาษาอังกฤษจริงต้องไม่ถูกแตะ
    assert result.rows_affected == 1
    assert result.operation == "fix_keyboard_layout"


@requires_pythainlp
def test_fix_keyboard_layout_leaves_english_alone():
    s = pd.Series(["python", "data", "the quick brown"])
    cleaned, result = fix_keyboard_layout(s)
    assert result.rows_affected == 0
    assert list(cleaned) == ["python", "data", "the quick brown"]


def test_fix_keyboard_layout_without_pythainlp_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "pythainlp", None)
    monkeypatch.setitem(sys.modules, "pythainlp.tokenize", None)
    monkeypatch.setitem(sys.modules, "pythainlp.util", None)
    s = pd.Series(["x"])
    with pytest.raises(ImportError):
        fix_keyboard_layout(s)


# ------------------------------------------------------ default pipeline integration
@requires_pythainlp
def test_default_operations_include_pythainlp_ops():
    from thaieda.clean import DEFAULT_OPERATIONS

    assert "pythainlp_normalize" in DEFAULT_OPERATIONS
    assert "keyboard_layout" in DEFAULT_OPERATIONS


def test_clean_thai_text_default_skips_pythainlp_when_absent(monkeypatch):
    # default pipeline ต้องข้าม pythainlp ops อย่างสุภาพเมื่อไม่ได้ติดตั้ง (ไม่ crash)
    monkeypatch.setitem(sys.modules, "pythainlp", None)
    s = pd.Series(["๑๒๓"])
    cleaned, results = clean_thai_text(s)
    ops = {r.operation for r in results}
    assert "pythainlp_normalize" not in ops
    assert "fix_keyboard_layout" not in ops
    assert cleaned.iloc[0] == "123"  # ขั้นตอนอื่นยังทำงานปกติ


def test_clean_thai_text_explicit_pythainlp_op_fails_loudly(monkeypatch):
    # ผู้ใช้ระบุ op ที่ต้องใช้ pythainlp เอง -> ต้อง fail loudly ไม่ skip เงียบ ๆ
    monkeypatch.setitem(sys.modules, "pythainlp", None)
    monkeypatch.setitem(sys.modules, "pythainlp.util", None)
    s = pd.Series(["x"])
    with pytest.raises(ImportError):
        clean_thai_text(s, operations=["pythainlp_normalize"])
