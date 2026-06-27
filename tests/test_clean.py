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
    normalize_nfkc,
    normalize_phone_numbers,
    normalize_thai_numerals,
    normalize_unicode,
    pythainlp_normalize,
    remove_zero_width_chars,
    spell_correct,
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


def _khamyo_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("khamyo") is not None


requires_pythainlp = pytest.mark.skipif(
    not _pythainlp_installed(), reason="pythainlp not installed"
)

requires_khamyo = pytest.mark.skipif(
    not (_pythainlp_installed() and _khamyo_installed()),
    reason="khamyo (pythainlp[abbreviation]) not installed",
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


def test_fix_repeated_chars_skips_product_codes():
    s = pd.Series([
        "SKU-AAA111",
        "PROD-0001",
        "CA-2017-152156",
        "AAB-222",
        "AAA111",
        "ปกติ",
        "55555"
    ])
    cleaned, result = fix_repeated_chars(s, max_repeat=3)
    assert cleaned.iloc[0] == "SKU-AAA111"
    assert cleaned.iloc[1] == "PROD-0001"
    assert cleaned.iloc[2] == "CA-2017-152156"
    assert cleaned.iloc[3] == "AAB-222"
    assert cleaned.iloc[4] == "AAA111"
    assert cleaned.iloc[6] == "555"
    assert result.rows_affected == 1

    # Column name awareness test
    s2 = pd.Series(["AAAAA", "BBBBB"], name="product_sku")
    cleaned2, result2 = fix_repeated_chars(s2, max_repeat=3)
    assert cleaned2.iloc[0] == "AAAAA"
    assert cleaned2.iloc[1] == "BBBBB"
    assert result2.rows_affected == 0


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


# ------------------------------------------------------------------ phone numbers
def test_normalize_phone_numbers_thai_numerals():
    """เบอร์โทรเลขไทย → อารบิก 10 หลัก"""
    s = pd.Series(["๐๘๑๒๓๔๕๖๗๘", "๐๘๙๘๗๖๕๔๓๒"], name="phone")
    cleaned, result = normalize_phone_numbers(s)
    assert cleaned.iloc[0] == "0812345678"
    assert cleaned.iloc[1] == "0898765432"
    assert result.rows_affected == 2


def test_normalize_phone_numbers_with_dashes():
    """เบอร์ที่มี dash/space → 10 หลัก"""
    s = pd.Series(["08-1234-5678", "08 9876 5432"], name="tel")
    cleaned, result = normalize_phone_numbers(s)
    assert cleaned.iloc[0] == "0812345678"
    assert cleaned.iloc[1] == "0898765432"


def test_normalize_phone_numbers_plus66():
    """+66 → 0"""
    s = pd.Series(["+66812345678"], name="mobile")
    cleaned, result = normalize_phone_numbers(s)
    assert cleaned.iloc[0] == "0812345678"


def test_normalize_phone_numbers_not_phone_passthrough():
    """ค่าที่ไม่ใช่เบอร์ → ไม่เปลี่ยน"""
    s = pd.Series(["hello", "12345"], name="text")
    cleaned, result = normalize_phone_numbers(s)
    assert cleaned.iloc[0] == "hello"
    assert result.rows_affected == 0


# ------------------------------------------------------------ AC-3: NFKC
def test_normalize_nfkc_fullwidth():
    # full-width Ａ(U+FF21)/９(U+FF19) → half-width A/9
    s = pd.Series(["Ａ９", "２０２４", "ปกติ"])
    cleaned, result = normalize_nfkc(s)
    assert cleaned.iloc[0] == "A9"
    assert cleaned.iloc[1] == "2024"
    assert cleaned.iloc[2] == "ปกติ"
    assert result.rows_affected == 2
    assert result.operation == "normalize_nfkc"


def test_normalize_nfkc_no_change_when_already_halfwidth():
    s = pd.Series(["hello", "123", "ปกติ"])
    cleaned, result = normalize_nfkc(s)
    assert result.rows_affected == 0
    assert list(cleaned) == ["hello", "123", "ปกติ"]


def test_normalize_nfkc_empty_series():
    s = pd.Series([], dtype=object)
    cleaned, result = normalize_nfkc(s)
    assert result.rows_affected == 0


def test_normalize_nfkc_not_in_default_operations():
    from thaieda.clean import DEFAULT_OPERATIONS, available_operations

    assert "nfkc" not in DEFAULT_OPERATIONS
    assert "nfkc" in available_operations()


def test_normalize_nfkc_via_clean_thai_text():
    # เรียกผ่าน pipeline แบบระบุ op เองได้
    s = pd.Series(["Ａ９"])
    cleaned, results = clean_thai_text(s, operations=["nfkc"])
    assert cleaned.iloc[0] == "A9"
    assert results[0].operation == "normalize_nfkc"


# ------------------------------------------------------ AC-2: spell_correct
@requires_pythainlp
def test_spell_correct_leaves_correct_text_unchanged():
    s = pd.Series(["สวัสดีครับ", "ขอบคุณค่ะ"])
    cleaned, result = spell_correct(s)
    assert cleaned.iloc[0] == "สวัสดีครับ"
    assert cleaned.iloc[1] == "ขอบคุณค่ะ"
    assert result.rows_affected == 0
    assert result.operation == "spell_correct"


@requires_pythainlp
def test_spell_correct_changes_misspelling():
    # คำที่สะกดผิดต้องถูกแก้/เปลี่ยน (โมเดล spell ของ pythainlp ไม่สมบูรณ์ —
    # ทดสอบว่ามีการเปลี่ยนแปลงเกิดขึ้น ไม่ใช่ผลลัพธ์ที่ถูกต้องเป๊ะ)
    s = pd.Series(["เรอม", "ปกติดี"])
    cleaned, result = spell_correct(s)
    assert cleaned.iloc[0] != "เรอม"
    assert result.rows_affected >= 1


@requires_pythainlp
def test_spell_correct_empty_and_none():
    s = pd.Series(["", None])
    cleaned, result = spell_correct(s)
    assert result.rows_affected == 0


def test_spell_correct_without_pythainlp_raises(monkeypatch):
    # ไม่มี pythainlp -> ต้อง fail loudly (ไม่ silent fallback)
    monkeypatch.setitem(sys.modules, "pythainlp", None)
    monkeypatch.setitem(sys.modules, "pythainlp.spell", None)
    monkeypatch.setitem(sys.modules, "pythainlp.tokenize", None)
    s = pd.Series(["x"])
    with pytest.raises(ImportError):
        spell_correct(s)


def test_spell_correct_not_in_default_operations():
    from thaieda.clean import DEFAULT_OPERATIONS, available_operations

    assert "spell_correct" not in DEFAULT_OPERATIONS
    assert "spell_correct" in available_operations()


# ------------------------------------------------ AC-1: expand_abbreviations
@requires_khamyo
def test_expand_abbreviations():
    from thaieda.clean import expand_abbreviations

    s = pd.Series(["กทม.", "บจ.", "ทดสอบ"])
    result, info = expand_abbreviations(s)
    assert "กรุงเทพมหานคร" in result.iloc[0]
    assert info.rows_affected > 0
    assert info.operation == "expand_abbreviations"


@requires_pythainlp
def test_expand_abbreviations_fails_loudly_without_khamyo():
    # ถ้าไม่มี khamyo การเรียกต้อง fail loudly (ImportError) — ไม่ silent fallback
    from thaieda.clean import expand_abbreviations

    if _khamyo_installed():
        pytest.skip("khamyo installed; ไม่สามารถทดสอบ path ที่ไม่มี khamyo ได้")
    with pytest.raises(ImportError):
        expand_abbreviations(pd.Series(["กทม."]))


def test_expand_abbreviations_without_pythainlp_raises(monkeypatch):
    from thaieda.clean import expand_abbreviations

    monkeypatch.setitem(sys.modules, "pythainlp", None)
    monkeypatch.setitem(sys.modules, "pythainlp.util", None)
    s = pd.Series(["x"])
    with pytest.raises(ImportError):
        expand_abbreviations(s)


def test_expand_abbreviations_not_in_default_operations():
    from thaieda.clean import DEFAULT_OPERATIONS, available_operations

    assert "expand_abbreviations" not in DEFAULT_OPERATIONS
    assert "expand_abbreviations" in available_operations()
