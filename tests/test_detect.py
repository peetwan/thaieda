"""ทดสอบ thaieda.detect — script_ratio, detect_column_type, is_thai_text."""

from __future__ import annotations

import pandas as pd
import pytest

from thaieda.detect import (
    ColumnType,
    detect_column_type,
    is_phone_number,
    is_thai_text,
    normalize_phone_number,
    script_ratio,
)


# ---------------------------------------------------------------- script_ratio
def test_script_ratio_pure_thai():
    r = script_ratio("อาหารอร่อย")
    assert r["thai"] == pytest.approx(1.0)
    assert r["latin"] == 0.0
    assert r["digit"] == 0.0


def test_script_ratio_pure_english():
    r = script_ratio("hello")
    assert r["latin"] == pytest.approx(1.0)
    assert r["thai"] == 0.0


def test_script_ratio_mixed():
    r = script_ratio("ดีmak")  # 2 thai + 3 latin
    assert r["thai"] == pytest.approx(2 / 5)
    assert r["latin"] == pytest.approx(3 / 5)


def test_script_ratio_empty_string():
    r = script_ratio("")
    assert all(v == 0.0 for v in r.values())


def test_script_ratio_numbers():
    r = script_ratio("12345")
    assert r["digit"] == pytest.approx(1.0)


def test_script_ratio_thai_digits():
    r = script_ratio("๑๒๓")
    assert r["thai_digit"] == pytest.approx(1.0)
    # เลขไทยไม่ควรนับเป็น thai (ตัวอักษร)
    assert r["thai"] == 0.0


def test_script_ratio_ratios_sum_to_one():
    r = script_ratio("สวัสดี hello 123 ๑๒๓ 😀")
    assert sum(r.values()) == pytest.approx(1.0)


def test_script_ratio_emoji():
    r = script_ratio("😀😀")
    assert r["emoji"] > 0.0


# ----------------------------------------------------------- detect_column_type
def test_detect_numeric():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    assert detect_column_type(s) == ColumnType.NUMERIC


def test_detect_categorical():
    s = pd.Series(["red", "green", "blue", "red", "green", "blue"] * 5)
    assert detect_column_type(s) == ColumnType.CATEGORICAL


def test_detect_thai_text():
    s = pd.Series(
        [
            "อาหารร้านนี้อร่อยมากแนะนำเลยครับ",
            "บริการดีพนักงานน่ารัก",
            "รสชาติใช้ได้แต่ราคาแพงไป",
            "บรรยากาศดีเหมาะกับครอบครัว",
        ]
    )
    assert detect_column_type(s) == ColumnType.THAI_TEXT


def test_detect_mixed_text():
    s = pd.Series(
        [
            "สินค้า quality ดีมาก recommend",
            "ราคา good value for money คุ้ม",
            "delivery เร็ว packaging ดี",
            "service ใช้ได้ แต่ shipping ช้า",
        ]
    )
    assert detect_column_type(s) == ColumnType.MIXED_TEXT


def test_detect_empty():
    s = pd.Series([None, None, None], dtype="object")
    assert detect_column_type(s) == ColumnType.EMPTY


def test_detect_english_text():
    s = pd.Series(
        [
            "this product is absolutely wonderful and highly recommended",
            "the service was slow but the quality made up for it",
            "great value for money would definitely buy again here",
            "shipping was fast and the packaging was very secure indeed",
        ]
    )
    assert detect_column_type(s) == ColumnType.ENGLISH_TEXT


def test_detect_datetime():
    s = pd.Series(["2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10"])
    assert detect_column_type(s) == ColumnType.DATETIME


def test_detect_id():
    s = pd.Series(range(100), name="user_id")
    assert detect_column_type(s) == ColumnType.ID


# ------------------------------------------------------------------ is_thai_text
def test_is_thai_text_true():
    s = pd.Series(["สวัสดีครับ", "ขอบคุณมาก", "ยินดีที่ได้รู้จัก"])
    assert is_thai_text(s) is True


def test_is_thai_text_false_english():
    s = pd.Series(["hello", "world", "foo", "bar"])
    assert is_thai_text(s) is False


def test_is_thai_text_threshold_behavior():
    # 1 จาก 4 เซลล์เป็นไทย = 25%
    s = pd.Series(["สวัสดีครับ", "hello", "world", "foo"])
    assert is_thai_text(s, threshold=0.20) is True
    assert is_thai_text(s, threshold=0.50) is False


def test_is_thai_text_empty_series():
    s = pd.Series([], dtype="object")
    assert is_thai_text(s) is False


# ------------------------------------------------------------------ phone number
def test_detect_phone_column_by_name():
    """คอลัมน์ชื่อ phone ที่เป็นเบอร์ไทย → PHONE_NUMBER"""
    s = pd.Series(["0812345678", "0898765432", "0213456789"], name="phone")
    assert detect_column_type(s) == ColumnType.PHONE_NUMBER


def test_detect_phone_column_thai_numerals():
    """เบอร์ที่พิมพ์ด้วยเลขไทย → PHONE_NUMBER"""
    s = pd.Series(["๐๘๑๒๓๔๕๖๗๘", "๐๘๙๘๗๖๕๔๓๒", "๐๒๑๓๔๕๖๗๘๙"], name="tel")
    assert detect_column_type(s) == ColumnType.PHONE_NUMBER


def test_detect_phone_column_with_dashes():
    """เบอร์ที่มี dash → PHONE_NUMBER"""
    s = pd.Series(["08-1234-5678", "08-9876-5432", "02-1345-6789"], name="mobile")
    assert detect_column_type(s) == ColumnType.PHONE_NUMBER


def test_detect_phone_column_plus66():
    """เบอร์ +66 → PHONE_NUMBER"""
    s = pd.Series(["+66812345678", "+66896543219", "+66213456789"], name="contact")
    assert detect_column_type(s) == ColumnType.PHONE_NUMBER


def test_detect_phone_not_numeric():
    """เบอร์โทรไม่ควรถูกจัดเป็น NUMERIC"""
    s = pd.Series(["0812345678", "0898765432", "0213456789", "0551234567"], name="phone")
    result = detect_column_type(s)
    assert result == ColumnType.PHONE_NUMBER
    assert result != ColumnType.NUMERIC


def test_normalize_phone_number_basic():
    """แปลงเบอร์ไทยเป็นมาตรฐาน 10 หลัก"""
    assert normalize_phone_number("08-1234-5678") == "0812345678"
    assert normalize_phone_number("08 1234 5678") == "0812345678"
    assert normalize_phone_number("+66812345678") == "0812345678"


def test_normalize_phone_number_thai_numerals():
    """เบอร์เลขไทย → อารบิก"""
    assert normalize_phone_number("๐๘๑๒๓๔๕๖๗๘") == "0812345678"


def test_normalize_phone_number_int_missing_leading_zero():
    """CSV อ่าน 0801234567 เป็น int 801234567 — เติม 0 นำหน้า"""
    assert normalize_phone_number(801234567) == "0801234567"
    assert normalize_phone_number("801234567") == "0801234567"


def test_normalize_phone_number_not_phone():
    """ค่าที่ไม่ใช่เบอร์ → คืนค่าเดิม"""
    assert normalize_phone_number("hello") == "hello"
    assert normalize_phone_number("12345") == "12345"  # สั้นเกิน


def test_is_phone_number_true():
    assert is_phone_number("0812345678") is True
    assert is_phone_number("+66812345678") is True
    assert is_phone_number("๐๘๑๒๓๔๕๖๗๘") is True


def test_is_phone_number_false():
    assert is_phone_number("12345") is False
    assert is_phone_number("hello") is False
