"""แยกวิเคราะห์ที่อยู่ไทย — สกัดเลขที่ หมู่ ตำบล/แขวง อำเภอ/เขต จังหวัด และรหัสไปรษณีย์.

ที่อยู่ไทยโดยทั่วไปประกอบด้วย:
- เลขที่ (house number) เช่น 123 หรือ 123/45
- หมู่ (moo) เช่น หมู่ 4, ม.5
- ตำบล/แขวง (subdistrict) เช่น ตำบลบางบัว, ต.บางบัว, แขวงบางบัว
- อำเภอ/เขต (district) เช่น อำเภอบางบัว, อ.บางบัว, เขตห้วยขวาง
- จังหวัด (province) เช่น จังหวัดกรุงเทพมหานคร, จ.กรุงเทพฯ
- รหัสไปรษณีย์ (postal code) เช่น 10230

รองรับทั้งแบบเต็ม (จังหวัด, อำเภอ, ตำบล) และแบบย่อ (จ., อ., ต.)
รองรับทั้งรูปแบบกรุงเทพฯ (แขวง/เขต) และต่างจังหวัด (ตำบล/อำเภอ)
"""

from __future__ import annotations

import re

import pandas as pd

__all__ = [
    "parse_thai_address",
    "parse_thai_address_column",
]

# ----------------------------------------------------------------------------
# รูปแบบ regex สำหรับสกัดแต่ละส่วนของที่อยู่ไทย
# ----------------------------------------------------------------------------

# เลขที่: เลขอารบิก อาจมี "/" เช่น 123, 123/45, 12/3
# มักอยู่ต้นสตริง อาจมี prefix "เลขที่" หรือ "เลขที่ "
_HOUSE_NUMBER_RE = re.compile(
    r"(?:เลขที่\s*)?(?P<num>\d+(?:/\d+)?(?:-\d+)?)",
)

# หมู่: "หมู่ 4", "หมู่ที่ 4", "ม.5", "ม 5", "หมู่บ้าน ..."
# สกัดเฉพาะเลขหมู่ ไม่รวมหมู่บ้าน (ที่เป็นชื่อ)
_MOO_RE = re.compile(
    r"(?:หมู่ที่|หมู่|ม\.?)(?:\s*)(?P<moo>\d+)",
)

# ตำบล/แขวง: "ตำบลบางบัว", "ต.บางบัว", "แขวงบางบัว"
# สกัดชื่อตำบล/แขวงหลังคำนำหน้า
_SUBDISTRICT_RE = re.compile(
    r"(?:ตำบล|ต\.|แขวง)(?:\s*)(?P<name>[ก-๛A-Za-z0-9.]+)",
)

# อำเภอ/เขต: "อำเภอบางบัว", "อ.บางบัว", "เขตห้วยขวาง"
# สกัดชื่ออำเภอ/เขตหลังคำนำหน้า
_DISTRICT_RE = re.compile(
    r"(?:อำเภอ|อ\.|เขต)(?:\s*)(?P<name>[ก-๛A-Za-z0-9.]+)",
)

# จังหวัด: "จังหวัดกรุงเทพมหานคร", "จ.กรุงเทพฯ", "กรุงเทพฯ"
# สกัดชื่อจังหวัดหลังคำนำหน้า หรือกรณีไม่มี prefix แต่รู้จักคำพิเศษ
_PROVINCE_RE = re.compile(
    r"(?:จังหวัด|จ\.)(?:\s*)(?P<name>[ก-๛A-Za-z0-9.]+)",
)

# รหัสไปรษณีย์: เลข 5 หลัก ท้ายสตริง หรือหลังชื่อจังหวัด
_POSTAL_CODE_RE = re.compile(
    r"\b(?P<code>\d{5})\b",
)

# รูปแบบกรุงเทพฯ แบบย่อที่ไม่มี prefix "จังหวัด" เช่น "กรุงเทพฯ", "กรุงเทพมหานคร"
_BANGKOK_SHORT_RE = re.compile(
    r"(?P<name>กรุงเทพ(?:มหานคร|ฯ)?)",
)

# จังหวัดไทยที่พบบ่อย — ใช้ตรวจเมื่อไม่มี prefix "จ." หรือ "จังหวัด"
# เรียงตามความยาว (ยาวก่อน) เพื่อกัน match สั้นกว่าก่อน เช่น "กรุงเทพ" ก่อน "กรุง"
_THAI_PROVINCES = [
    "กรุงเทพมหานคร",
    "กรุงเทพฯ",
    "กรุงเทพ",
    "เชียงใหม่",
    "เชียงราย",
    "ขอนแก่น",
    "ภูเก็ต",
    "พัทยา",
    "นนทบุรี",
    "ปทุมธานี",
    "สมุทรปราการ",
    "สมุทรสาคร",
    "สมุทรสงคราม",
    "นครปฐม",
    "ราชบุรี",
    "กาญจนบุรี",
    "เพชรบุรี",
    "ประจวบคีรีขันธ์",
    "ชลบุรี",
    "ระยอง",
    "จันทบุรี",
    "ตราด",
    "ฉะเชิงเทรา",
    "นครราชสีมา",
    "บุรีรัมย์",
    "สุรินทร์",
    "ศรีสะเกษ",
    "อุบลราชธานี",
    "อุดรธานี",
    "หนองคาย",
    "เลย",
    "สกลนคร",
    "นครพนม",
    "มุกดาหาร",
    "พิษณุโลก",
    "สุโขทัย",
    "พิจิตร",
    "เพชรบูรณ์",
    "นครสวรรค์",
    "ลำปาง",
    "ลำพูน",
    "พะเยา",
    "น่าน",
    "แพร่",
    "ตาก",
    "สุราษฎร์ธานี",
    "นครศรีธรรมราช",
    "ระนอง",
    "ชุมพร",
    "สงขลา",
    "พัทลุง",
    "ตรัง",
    "พังงา",
    "กระบี่",
    "สตูล",
    "ยะลา",
    "ปัตตานี",
    "นราธิวาส",
]

# สร้าง regex สำหรับจังหวัดที่ไม่มี prefix — match คำยาวก่อน
_PROVINCE_NO_PREFIX_RE = re.compile(
    r"(?P<name>" + "|".join(re.escape(p) for p in _THAI_PROVINCES) + ")",
)


def parse_thai_address(text: str) -> dict[str, str]:
    """แยกวิเคราะห์ที่อยู่ไทยจากสตริง คืน dict ของแต่ละส่วน.

    คืน dict ที่มีคีย์: house_number, moo, subdistrict, district, province, postal_code
    ส่วนที่หาไม่ได้คืนค่าเป็นสตริงว่าง ""

    Parameters
    ----------
    text : str
        ที่อยู่ภาษาไทยที่ต้องการแยกวิเคราะห์

    Returns
    -------
    dict[str, str]
        {house_number, moo, subdistrict, district, province, postal_code}

    Examples
    --------
    >>> parse_thai_address("123 หมู่ 4 ตำบลบางบัว อำเภอบางบัว จังหวัดกรุงเทพมหานคร 10230")
    {'house_number': '123', 'moo': '4', 'subdistrict': 'บางบัว',
     'district': 'บางบัว', 'province': 'กรุงเทพมหานคร', 'postal_code': '10230'}
    """
    result: dict[str, str] = {
        "house_number": "",
        "moo": "",
        "subdistrict": "",
        "district": "",
        "province": "",
        "postal_code": "",
    }

    if not isinstance(text, str) or not text.strip():
        return result

    text = text.strip()

    # --- สกัดรหัสไปรษณีย์ (5 หลัก) — ทำก่อนเพื่อไม่ให้บังส่วนอื่น ---
    postal_match = _POSTAL_CODE_RE.search(text)
    if postal_match:
        result["postal_code"] = postal_match.group("code")
        # ลบรหัสไปรษณีย์ออกจาก text เพื่อไม่ให้กระทบการสกัดส่วนอื่น
        text_no_postal = text[: postal_match.start()] + text[postal_match.end() :]
    else:
        text_no_postal = text

    # --- หมู่ — สกัดก่อนเพื่อไม่ให้เลขหมู่ไปติดกับเลขที่ ---
    moo_match = _MOO_RE.search(text_no_postal)
    if moo_match:
        result["moo"] = moo_match.group("moo")

    # --- เลขที่ — มักอยู่ต้นสตริง สกัดตัวเลขแรกที่พบ ---
    house_match = _HOUSE_NUMBER_RE.search(text_no_postal)
    if house_match:
        result["house_number"] = house_match.group("num")

    # --- ตำบล/แขวง ---
    subdist_match = _SUBDISTRICT_RE.search(text_no_postal)
    if subdist_match:
        result["subdistrict"] = subdist_match.group("name")

    # --- อำเภอ/เขต ---
    dist_match = _DISTRICT_RE.search(text_no_postal)
    if dist_match:
        result["district"] = dist_match.group("name")

    # --- จังหวัด — ลองหาด้วย prefix ก่อน ---
    prov_match = _PROVINCE_RE.search(text_no_postal)
    if prov_match:
        result["province"] = prov_match.group("name")
    else:
        # ลองหาจังหวัดจาก list โดยไม่มี prefix (เช่น "เชียงใหม่", "ภูเก็ต")
        prov_noprefix = _PROVINCE_NO_PREFIX_RE.search(text_no_postal)
        if prov_noprefix:
            result["province"] = prov_noprefix.group("name")
        else:
            # ลองหา "กรุงเทพฯ" หรือ "กรุงเทพมหานคร" โดยไม่มี prefix
            bk_match = _BANGKOK_SHORT_RE.search(text_no_postal)
            if bk_match:
                result["province"] = bk_match.group("name")

    return result


def parse_thai_address_column(series: pd.Series) -> pd.DataFrame:
    """ใช้ parse_thai_address กับทุกค่าใน pandas Series คืน DataFrame ที่แยกคอลัมน์แล้ว.

    Parameters
    ----------
    series : pd.Series
        Series ของที่อยู่ไทยที่ต้องการแยกวิเคราะห์

    Returns
    -------
    pd.DataFrame
        DataFrame ที่มีคอลัมน์: house_number, moo, subdistrict, district,
        province, postal_code

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(["123 หมู่ 4 ตำบลบางบัว อำเภอบางบัว จังหวัดกรุงเทพมหานคร 10230"])
    >>> df = parse_thai_address_column(s)
    >>> list(df.columns)
    ['house_number', 'moo', 'subdistrict', 'district', 'province', 'postal_code']
    """
    results = [parse_thai_address(v) for v in series]
    return pd.DataFrame(
        results,
        columns=[
            "house_number",
            "moo",
            "subdistrict",
            "district",
            "province",
            "postal_code",
        ],
    )
