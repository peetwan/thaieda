"""ทดสอบ thaieda.detect._thai_address — แยกวิเคราะห์ที่อยู่ไทย."""

from __future__ import annotations

import pandas as pd

from thaieda.detect._thai_address import (
    parse_thai_address,
    parse_thai_address_column,
)


# --------------------------------------------------------------- รูปแบบเต็ม
def test_full_format_address():
    """ที่อยู่แบบเต็ม: เลขที่ หมู่ ตำบล อำเภอ จังหวัด รหัสไปรษณีย์."""
    addr = "123 หมู่ 4 ตำบลบางบัว อำเภอบางบัว จังหวัดกรุงเทพมหานคร 10230"
    result = parse_thai_address(addr)
    assert result["house_number"] == "123"
    assert result["moo"] == "4"
    assert result["subdistrict"] == "บางบัว"
    assert result["district"] == "บางบัว"
    assert result["province"] == "กรุงเทพมหานคร"
    assert result["postal_code"] == "10230"


def test_full_format_with_slash_house_number():
    """เลขที่แบบ x/y เช่น 123/45."""
    addr = "123/45 หมู่ 4 ตำบลบางบัว อำเภอบางบัว จังหวัดกรุงเทพมหานคร 10230"
    result = parse_thai_address(addr)
    assert result["house_number"] == "123/45"
    assert result["moo"] == "4"


# --------------------------------------------------------------- รูปแบบย่อ
def test_short_form_address():
    """ใช้คำย่อ จ. อ. ต. แทนชื่อเต็ม."""
    addr = "123/45 ม.5 ต.บางบัว อ.บางบัว จ.กรุงเทพฯ 10310"
    result = parse_thai_address(addr)
    assert result["house_number"] == "123/45"
    assert result["moo"] == "5"
    assert result["subdistrict"] == "บางบัว"
    assert result["district"] == "บางบัว"
    assert result["province"] == "กรุงเทพฯ"
    assert result["postal_code"] == "10310"


def test_short_form_moo_with_dot():
    """หมู่แบบย่อ 'ม.' มีจุด."""
    addr = "88 ม.12 ต.หลักสี่ อ.ดอนเมือง จ.กรุงเทพฯ 10210"
    result = parse_thai_address(addr)
    assert result["moo"] == "12"
    assert result["subdistrict"] == "หลักสี่"
    assert result["district"] == "ดอนเมือง"
    assert result["province"] == "กรุงเทพฯ"


# --------------------------------------------------------------- กรุงเทพฯ vs ต่างจังหวัด
def test_bangkok_format_แขวง_เขต():
    """รูปแบบกรุงเทพฯ: แขวง + เขต (ไม่ใช้ ตำบล/อำเภอ)."""
    addr = "123/45 ม.5 แขวงบางบัว เขตห้วยขวาง กรุงเทพฯ 10310"
    result = parse_thai_address(addr)
    assert result["house_number"] == "123/45"
    assert result["moo"] == "5"
    assert result["subdistrict"] == "บางบัว"
    assert result["district"] == "ห้วยขวาง"
    assert result["province"] == "กรุงเทพฯ"
    assert result["postal_code"] == "10310"


def test_bangkok_with_full_province_no_prefix():
    """รูปแบบ 'กรุงเทพมหานคร' โดยไม่มี prefix 'จังหวัด'."""
    addr = "99 หมู่ 3 แขวงดินแดง เขตดินแดง กรุงเทพมหานคร 10400"
    result = parse_thai_address(addr)
    assert result["subdistrict"] == "ดินแดง"
    assert result["district"] == "ดินแดง"
    assert result["province"] == "กรุงเทพมหานคร"
    assert result["postal_code"] == "10400"


def test_province_format_outside_bangkok():
    """ที่อยู่ต่างจังหวัด: ตำบล + อำเภอ + จังหวัด."""
    addr = "45 หมู่ 7 ตำบลเชิงทะเล อำเภอถลาง จังหวัดภูเก็ต 83110"
    result = parse_thai_address(addr)
    assert result["house_number"] == "45"
    assert result["moo"] == "7"
    assert result["subdistrict"] == "เชิงทะเล"
    assert result["district"] == "ถลาง"
    assert result["province"] == "ภูเก็ต"
    assert result["postal_code"] == "83110"


def test_bangkok_short_no_prefix():
    """รูปแบบ 'กรุงเทพฯ' โดยไม่มี prefix 'จ.'."""
    addr = "10 หมู่ 1 ต.ลาดพร้าว อ.จตุจักร กรุงเทพฯ 10900"
    result = parse_thai_address(addr)
    assert result["province"] == "กรุงเทพฯ"


# --------------------------------------------------------------- ส่วนที่ขาด
def test_missing_moo():
    """ไม่มีหมู่."""
    addr = "123 ตำบลบางบัว อำเภอบางบัว จังหวัดกรุงเทพมหานคร 10230"
    result = parse_thai_address(addr)
    assert result["house_number"] == "123"
    assert result["moo"] == ""
    assert result["subdistrict"] == "บางบัว"


def test_missing_postal_code():
    """ไม่มีรหัสไปรษณีย์."""
    addr = "123 หมู่ 4 ตำบลบางบัว อำเภอบางบัว จังหวัดกรุงเทพมหานคร"
    result = parse_thai_address(addr)
    assert result["house_number"] == "123"
    assert result["moo"] == "4"
    assert result["postal_code"] == ""


def test_missing_province():
    """ไม่มีจังหวัด."""
    addr = "123 หมู่ 4 ตำบลบางบัว อำเภอบางบัว 10230"
    result = parse_thai_address(addr)
    assert result["house_number"] == "123"
    assert result["subdistrict"] == "บางบัว"
    assert result["district"] == "บางบัว"
    assert result["province"] == ""
    assert result["postal_code"] == "10230"


def test_only_house_number():
    """มีแค่เลขที่."""
    result = parse_thai_address("123/45")
    assert result["house_number"] == "123/45"
    assert result["moo"] == ""
    assert result["subdistrict"] == ""
    assert result["district"] == ""
    assert result["province"] == ""
    assert result["postal_code"] == ""


def test_only_postal_code():
    """มีแค่รหัสไปรษณีย์."""
    result = parse_thai_address("10230")
    assert result["house_number"] == ""
    assert result["postal_code"] == "10230"


# --------------------------------------------------------------- empty/None
def test_empty_string():
    """สตริงว่าง — คืนค่าว่างทั้งหมด."""
    result = parse_thai_address("")
    assert all(v == "" for v in result.values())


def test_none_input():
    """None — คืนค่าว่างทั้งหมด."""
    result = parse_thai_address(None)  # type: ignore[arg-type]
    assert all(v == "" for v in result.values())


def test_whitespace_only():
    """สตริงที่มีแต่ช่องว่าง — คืนค่าว่างทั้งหมด."""
    result = parse_thai_address("   ")
    assert all(v == "" for v in result.values())


def test_non_string_input():
    """ค่าที่ไม่ใช่สตริง — คืนค่าว่างทั้งหมด."""
    result = parse_thai_address(12345)  # type: ignore[arg-type]
    assert all(v == "" for v in result.values())


# --------------------------------------------------------------- dict keys
def test_result_has_all_keys():
    """ผลลัพธ์ต้องมีคีย์ครบทั้ง 6 ตัว."""
    result = parse_thai_address("123 10230")
    expected_keys = {
        "house_number",
        "moo",
        "subdistrict",
        "district",
        "province",
        "postal_code",
    }
    assert set(result.keys()) == expected_keys


# --------------------------------------------------------------- column parsing
def test_parse_column_basic():
    """แยกวิเคราะห์ทั้งคอลัมน์ — คืน DataFrame ที่มีคอลัมน์ครบ."""
    s = pd.Series(
        [
            "123 หมู่ 4 ตำบลบางบัว อำเภอบางบัว จังหวัดกรุงเทพมหานคร 10230",
            "123/45 ม.5 แขวงบางบัว เขตห้วยขวาง กรุงเทพฯ 10310",
            "45 หมู่ 7 ตำบลเชิงทะเล อำเภอถลาง จังหวัดภูเก็ต 83110",
        ]
    )
    df = parse_thai_address_column(s)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == [
        "house_number",
        "moo",
        "subdistrict",
        "district",
        "province",
        "postal_code",
    ]
    assert df.iloc[0]["house_number"] == "123"
    assert df.iloc[0]["province"] == "กรุงเทพมหานคร"
    assert df.iloc[1]["subdistrict"] == "บางบัว"
    assert df.iloc[1]["district"] == "ห้วยขวาง"
    assert df.iloc[2]["province"] == "ภูเก็ต"
    assert df.iloc[2]["postal_code"] == "83110"


def test_parse_column_with_empty_values():
    """คอลัมน์ที่มีค่าว่าง/None ปน — คืนสตริงว่างสำหรับแถวนั้น."""
    s = pd.Series(
        [
            "123 หมู่ 4 ตำบลบางบัว อำเภอบางบัว จังหวัดกรุงเทพมหานคร 10230",
            None,
            "",
        ]
    )
    df = parse_thai_address_column(s)
    assert len(df) == 3
    assert df.iloc[0]["house_number"] == "123"
    assert all(df.iloc[1][c] == "" for c in df.columns)
    assert all(df.iloc[2][c] == "" for c in df.columns)


def test_parse_column_empty_series():
    """Series ว่าง — คืน DataFrame ที่มี 0 แถว แต่มีคอลัมน์ครบ."""
    s = pd.Series([], dtype="object")
    df = parse_thai_address_column(s)
    assert len(df) == 0
    assert list(df.columns) == [
        "house_number",
        "moo",
        "subdistrict",
        "district",
        "province",
        "postal_code",
    ]


def test_parse_column_short_forms():
    """คอลัมน์ที่ใช้รูปแบบย่อทั้งหมด."""
    s = pd.Series(
        [
            "123/45 ม.5 ต.บางบัว อ.บางบัว จ.กรุงเทพฯ 10310",
            "88 ม.12 ต.หลักสี่ อ.ดอนเมือง จ.กรุงเทพฯ 10210",
        ]
    )
    df = parse_thai_address_column(s)
    assert df.iloc[0]["moo"] == "5"
    assert df.iloc[0]["subdistrict"] == "บางบัว"
    assert df.iloc[0]["province"] == "กรุงเทพฯ"
    assert df.iloc[1]["moo"] == "12"
    assert df.iloc[1]["subdistrict"] == "หลักสี่"
