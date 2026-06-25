"""ทดสอบการตรวจสอบเลขบัตรประชาชนไทย — thaieda.quality._thai_id."""

from __future__ import annotations

import pandas as pd
import pytest

from thaieda.quality import validate_thai_id, validate_thai_id_column

# ----------------------------------------------------------- เลขบัตรที่ถูกต้อง
# เลขบัตรที่ถูกต้อง (คำนวณ checksum แล้วผ่าน)
_VALID_IDS = [
    "1100700000001",
    "1234567890121",
    "1111111111119",
    "3100200000008",
    "5901000000004",
]


@pytest.mark.parametrize("id_str", _VALID_IDS)
def test_valid_thai_id(id_str: str):
    """เลขบัตรที่ถูกต้องตาม checksum ต้องคืนค่า True."""
    assert validate_thai_id(id_str) is True


def test_valid_ids_multiple():
    """ตรวจหลายเลขบัตรพร้อมกัน — ทุกตัวต้องถูกต้อง."""
    results = [validate_thai_id(tid) for tid in _VALID_IDS]
    assert all(results)


# ----------------------------------------------------------- checksum ผิด
def test_invalid_checksum():
    """เลขบัตรที่ checksum ผิดต้องคืนค่า False."""
    # เปลี่ยนหลักสุดท้ายของเลขที่ถูกต้อง
    assert validate_thai_id("1100700000002") is False
    assert validate_thai_id("1234567890122") is False
    assert validate_thai_id("1111111111110") is False


def test_invalid_checksum_all_digits_same_wrong():
    """เลขบัตร 1111111111111 (checksum ผิด) ต้องคืนค่า False."""
    # checksum ที่ถูกต้องสำหรับ 111111111111 คือ 9
    assert validate_thai_id("1111111111111") is False


# ----------------------------------------------------------- ความยาวผิด
def test_wrong_length_too_short():
    """เลขบัตรสั้นเกินไปต้องคืนค่า False."""
    assert validate_thai_id("123456789012") is False  # 12 หลัก
    assert validate_thai_id("12345") is False


def test_wrong_length_too_long():
    """เลขบัตรยาวเกินไปต้องคืนค่า False."""
    assert validate_thai_id("12345678901234") is False  # 14 หลัก
    assert validate_thai_id("12345678901234567890") is False


def test_empty_string():
    """สตริงว่างต้องคืนค่า False."""
    assert validate_thai_id("") is False


# ----------------------------------------------------------- ตัวอักษรที่ไม่ใช่ตัวเลข
def test_non_numeric_characters():
    """สตริงที่มีตัวอักษรที่ไม่ใช่ตัวเลขต้องคืนค่า False."""
    assert validate_thai_id("11007000000A1") is False
    assert validate_thai_id("abcdefghijabc") is False
    assert validate_thai_id("11007000000 1") is False  # มีช่องว่างกลาง


def test_dashes_in_id():
    """เลขบัตรที่มีขีด (รูปแบบ X-XXXX-XXXXX-XX-X) ต้องคืนค่า False เพราะไม่ใช่ตัวเลข 13 หลักล้วน."""
    # ฟังก์ชัน validate_thai_id รับเฉพาะตัวเลข 13 หลัก ไม่รองรับรูปแบบที่มีขีด
    assert validate_thai_id("1-1007-00000-00-1") is False
    assert validate_thai_id("1-1007-00000-00-0") is False


def test_thai_numerals_not_accepted():
    """เลขไทย ๐–๙ ไม่ได้รับการสนับสนุนใน validate_thai_id — ต้องคืนค่า False."""
    assert validate_thai_id("๑๑๐๐๗๐๐๐๐๐๐๐๑") is False


# ----------------------------------------------------------- การตรวจสอบทั้งคอลัมน์
def test_column_all_valid():
    """คอลัมน์ที่เลขบัตรถูกต้องทั้งหมดต้องมี valid_count เท่ากับจำนวนแถว."""
    series = pd.Series(_VALID_IDS)
    result = validate_thai_id_column(series)
    assert result["valid_count"] == len(_VALID_IDS)
    assert result["invalid_count"] == 0
    assert result["invalid_indices"] == []


def test_column_all_invalid():
    """คอลัมน์ที่เลขบัตรผิดทั้งหมดต้องมี invalid_count เท่ากับจำนวนแถว."""
    series = pd.Series(["1100700000002", "1234567890122", "abcdefghijabc"])
    result = validate_thai_id_column(series)
    assert result["valid_count"] == 0
    assert result["invalid_count"] == 3
    assert result["invalid_indices"] == [0, 1, 2]


def test_column_mixed_valid_invalid():
    """คอลัมน์ที่มีทั้งเลขบัตรถูกและผิดต้องนับแยกกันถูกต้อง."""
    series = pd.Series(
        [
            "1100700000001",  # ถูกต้อง
            "1100700000002",  # checksum ผิด
            "1234567890121",  # ถูกต้อง
            "12345",  # สั้นเกินไป
            "1111111111119",  # ถูกต้อง
            "abcdefghijabc",  # ไม่ใช่ตัวเลข
        ]
    )
    result = validate_thai_id_column(series)
    assert result["valid_count"] == 3
    assert result["invalid_count"] == 3
    assert result["invalid_indices"] == [1, 3, 5]


def test_column_with_none_values():
    """คอลัมน์ที่มี None ต้องนับว่าไม่ถูกต้อง."""
    series = pd.Series(["1100700000001", None, "1234567890121"])
    result = validate_thai_id_column(series)
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 1
    assert 1 in result["invalid_indices"]


def test_column_with_nan_values():
    """คอลัมน์ที่มี NaN ต้องนับว่าไม่ถูกต้อง."""
    series = pd.Series(["1100700000001", float("nan"), "1234567890121"])
    result = validate_thai_id_column(series)
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 1
    assert 1 in result["invalid_indices"]


def test_column_empty_series():
    """คอลัมน์ว่างต้องคืนค่าทั้งหมดเป็น 0."""
    series = pd.Series([], dtype=str)
    result = validate_thai_id_column(series)
    assert result["valid_count"] == 0
    assert result["invalid_count"] == 0
    assert result["invalid_indices"] == []


def test_column_integer_values():
    """คอลัมน์ที่เก็บเลขบัตรเป็น int ต้องแปลงแล้วตรวจได้ถูกต้อง."""
    # pandas อาจเก็บเลข 13 หลักเป็น int ได้
    series = pd.Series([1100700000001, 1100700000002])
    result = validate_thai_id_column(series)
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1
    assert result["invalid_indices"] == [1]


def test_column_custom_index():
    """คอลัมน์ที่มี index กำหนดเอง— invalid_indices ต้องใช้ index ของ Series."""
    series = pd.Series(
        ["1100700000001", "WRONG", "1234567890121"],
        index=[100, 200, 300],
    )
    result = validate_thai_id_column(series)
    assert result["valid_count"] == 2
    assert result["invalid_count"] == 1
    assert result["invalid_indices"] == [200]


# ----------------------------------------------------------- edge cases เพิ่มเติม
def test_whitespace_around_id():
    """เลขบัตรที่มีช่องว่างหน้าหลังต้องถูกตัดก่อนตรวจ—ต้องคืนค่า True."""
    assert validate_thai_id("  1100700000001  ") is True
    assert validate_thai_id("\t1100700000001\n") is True


def test_leading_zeros_valid():
    """เลขบัตรที่เริ่มด้วย 0 ต้องตรวจได้ถูกต้อง."""
    # 0000000000000 — ตรวจสอบว่า checksum คำนวณถูก
    # weighted_sum = 0*13+0*12+...+0*2 = 0, checksum = (11-0)%10 = 1
    assert validate_thai_id("0000000000001") is True
    assert validate_thai_id("0000000000000") is False


def test_none_raises_type_error():
    """ส่ง None เข้า validate_thai_id ต้อง raise TypeError."""
    with pytest.raises(TypeError):
        validate_thai_id(None)  # type: ignore[arg-type]


def test_int_raises_type_error():
    """ส่ง int เข้า validate_thai_id ต้อง raise TypeError."""
    with pytest.raises(TypeError):
        validate_thai_id(1100700000001)  # type: ignore[arg-type]


def test_non_series_raises_type_error():
    """ส่ง object ที่ไม่ใช่ pd.Series เข้า validate_thai_id_column ต้อง raise TypeError."""
    with pytest.raises(TypeError):
        validate_thai_id_column(["1100700000001", "1234567890121"])  # type: ignore[arg-type]
