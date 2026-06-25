"""Column type detection — จำแนกประเภทคอลัมน์ โดยเน้นการแยกข้อความไทย/อังกฤษ/ผสม."""

from __future__ import annotations

import re
from enum import Enum

import pandas as pd

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


def _name_hints_id(series: pd.Series) -> bool:
    """ชื่อคอลัมน์บอกใบ้ว่าเป็น ID หรือไม่ (เช่น 'id', 'user_id')."""
    name = str(series.name).lower() if series.name is not None else ""
    return name == "id" or name.endswith("_id") or name.endswith("id")


def _looks_like_id(series: pd.Series, non_null: pd.Series) -> bool:
    """เดาว่าเป็นคอลัมน์ตัวระบุ (ID) เชิงตัวเลข — ต้องไม่ซ้ำเกือบทั้งหมด และชื่อบอกใบ้.

    เราไม่จัดคอลัมน์ตัวเลขทั่วไปที่บังเอิญไม่ซ้ำให้เป็น ID เพราะค่าวัดหลายอย่าง
    (ราคา, น้ำหนัก) ก็ไม่ซ้ำได้ จึงต้องอาศัยชื่อคอลัมน์เป็นสัญญาณ
    """
    n = len(non_null)
    if n < 5:
        return False
    if non_null.nunique() / n < 0.95:
        return False
    return _name_hints_id(series)


def _looks_like_string_id(series: pd.Series, non_null: pd.Series, str_sample: list[str]) -> bool:
    """เดาว่าเป็น ID แบบสตริง (รหัส/UUID) — ไม่ซ้ำ, token เดี่ยว, สั้น, ไม่ใช่ข้อความไทยยาว."""
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

    ลำดับการตัดสิน: empty -> datetime -> numeric -> (id) -> text(thai/eng/mixed)
    -> categorical.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return ColumnType.EMPTY

    # --- datetime ---
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

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

    # --- วิเคราะห์สคริปต์ของข้อความด้วยสัดส่วนรวม (mean ต่อเซลล์) ---
    n = len(str_sample)
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
    nunique = non_null.nunique()
    unique_ratio = nunique / len(non_null)
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
    # ลอง parse — ถ้าเป็น Thai month ให้แปลงก่อน
    has_thai = any(thai_date_re.search(v) for v in values)
    if has_thai:
        # แปลง Thai month names เป็นตัวเลขก่อน parse
        return True  # ถ้ามี Thai month names มากพอ ถือว่าเป็น datetime
    parsed = pd.to_datetime(pd.Series(values), errors="coerce", format="mixed")
    return bool(parsed.notna().mean() > 0.90)


def detect_all(df: pd.DataFrame) -> dict[str, ColumnType]:
    """จำแนกทุกคอลัมน์ใน DataFrame คืน mapping ชื่อคอลัมน์ -> ColumnType."""
    return {str(col): detect_column_type(df[col]) for col in df.columns}


__all__ = [
    "ColumnType",
    "script_ratio",
    "is_thai_text",
    "detect_column_type",
    "detect_all",
    "normalize_phone_number",
    "is_phone_number",
    "clean_phone_string",
]
