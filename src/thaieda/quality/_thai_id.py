"""การตรวจสอบเลขบัตรประจำตัวประชาชนไทย (Thai national ID card checksum validation).

เลขบัตรประชาชนไทยมี 13 หลัก โดยหลักที่ 13 เป็น checksum:
- หลักที่ 1–12 เป็นเลขหมายประจำตัว
- หลักที่ 13 เป็น checksum คำนวณจากสูตร:
  sum = (digit[0]*13 + digit[1]*12 + ... + digit[11]*2) % 11
  checksum = (11 - sum) % 10
- ถ้า checksum ที่คำนวณได้ไม่ตรงกับหลักที่ 13 ถือว่าเลขบัตรไม่ถูกต้อง
"""

from __future__ import annotations

import re

import pandas as pd

# นิพจน์ปกติสำหรับตรวจว่าเป็นตัวเลขอารบิก 13 หลักเท่านั้น (ไม่มีขีด ไม่มีช่องว่าง ไม่ใช่เลขไทย ๐-๙)
_ID_RE = re.compile(r"^[0-9]{13}$")

# น้ำหนักสำหรับการคำนวณ checksum: หลักแรกคูณ 13, หลักที่สองคูณ 12, ... หลักที่ 12 คูณ 2
_WEIGHTS = (13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2)


def validate_thai_id(id_str: str) -> bool:
    """ตรวจสอบความถูกต้องของเลขบัตรประชาชนไทย 13 หลัก.

    คืนค่า True ถ้าเลขบัตรถูกต้องตามหลัก checksum, False ถ้าไม่ถูกต้อง.
    รับเฉพาะสตริงที่เป็นตัวเลขอารบิก 13 หลักเท่านั้น (ไม่มีขีด ไม่มีช่องว่าง ไม่ใช่เลขไทย).

    Args:
        id_str: สตริงเลขบัตรประชาชน 13 หลัก

    Returns:
        True ถ้าถูกต้อง, False ถ่าไม่ถูกต้อง

    Raises:
        TypeError: ถ้า id_str ไม่ใช่สตริง
    """
    if not isinstance(id_str, str):
        raise TypeError(f"id_str ต้องเป็น str เท่านั้น ได้รับ {type(id_str).__name__}")

    # ตัดช่องว่างหน้าหลัง แล้วตรวจรูปแบบ: ต้องเป็นตัวเลขอารบิก 13 หลัก
    cleaned = id_str.strip()
    if not _ID_RE.match(cleaned):
        return False

    digits = [int(ch) for ch in cleaned]

    # คำนวณ checksum
    weighted_sum = sum(d * w for d, w in zip(digits[:12], _WEIGHTS, strict=True))
    expected_checksum = (11 - (weighted_sum % 11)) % 10

    return expected_checksum == digits[12]


def validate_thai_id_column(series: pd.Series) -> dict:
    """ตรวจสอบเลขบัตรประชาชนไทยทั้งคอลัมน์.

    วนตรวจทุกค่าใน Series ว่าเป็นเลขบัตรประชาชนไทยที่ถูกต้องหรือไม่.
    ค่าที่เป็น None/NaN ถือว่าไม่ถูกต้อง.

    Args:
        series: pd.Series ที่มีเลขบัตรประชาชนเป็นสตริง

    Returns:
        dict ที่มี keys:
        - valid_count: จำนวนเลขบัตรที่ถูกต้อง
        - invalid_count: จำนวนเลขบัตรที่ไม่ถูกต้อง
        - invalid_indices: ลิสต์ index ของแถวที่ไม่ถูกต้อง

    Raises:
        TypeError: ถ้า series ไม่ใช่ pd.Series
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"series ต้องเป็น pd.Series เท่านั้น ได้รับ {type(series).__name__}")

    valid_count = 0
    invalid_count = 0
    invalid_indices: list[int] = []

    for idx, value in series.items():
        # จัดการค่า None/NaN และค่าที่ไม่ใช่สตริง
        if value is None or (isinstance(value, float) and pd.isna(value)):
            invalid_count += 1
            invalid_indices.append(idx)
            continue

        # แปลงเป็นสตริงเพื่อตรวจสอบ (รองรับกรณีที่ค่าเป็น int)
        str_value = str(value).strip() if not isinstance(value, str) else value.strip()

        if validate_thai_id(str_value):
            valid_count += 1
        else:
            invalid_count += 1
            invalid_indices.append(idx)

    return {
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "invalid_indices": invalid_indices,
    }
