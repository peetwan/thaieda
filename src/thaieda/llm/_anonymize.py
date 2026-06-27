"""การทำให้ข้อมูลไม่ระบุตัวบุคคลได้ (anonymization) — ลบ PII ก่อนส่งให้ LLM (v0.9).

โมดูลนี้ตรวจหาและแทนที่ข้อมูลที่ระบุตัวบุคคลได้ (PII) 3 ประเภท:
  1. ชื่อบุคคล/องค์กร/สถานที่ — ใช้ NER จาก ``thaieda.ner`` (PER/ORG/LOC)
  2. เบอร์โทรศัพท์ไทย — ใช้ regex (10 หลัก ขึ้นต้น 0 หรือ +66)
  3. เลขบัตรประชาชนไทย — ใช้ regex รูปแบบ ``X-XXXX-XXXXX-XX-X``

การแทนที่ใช้ "token" แบบคงที่: PII แต่ละค่าที่ไม่ซ้ำกันจะได้ token ของตัวเอง
เช่น ชื่อ "สมชาย" → ``[NAME_1]`` เบอร์ "0812345678" → ``[PHONE_1]`` เป็นต้น
token เดียวกันใช้สำหรับค่าเดียวกันเสมอ (ทำให้ LLM ยังเห็นรูปแบบได้โดยไม่เห็นค่าจริง)

หลักการ:
  * Vectorized — ประมวลผลค่าไม่ซ้ำ (unique values) แทนทีละแถว
  * NER เป็น optional — ถ้าไม่มี pythainlp จะข้ามการแทนที่ชื่อ แต่ยังทำเบอร์/บัตรได้
  * ไม่มี silent fallback — ถ้า NER ไม่พร้อม จะบันทึกใน token_map[\"_ner_available\"] = False
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

# ----------------------------------------------------------------------------
# รูปแบบ regex สำหรับ PII
# ----------------------------------------------------------------------------
# เลขบัตรประชาชนไทย: X-XXXX-XXXXX-XX-X (1-4-5-2-1 รวม 13 หลัก)
_IDCARD_RE = re.compile(r"\b\d{1}-\d{4}-\d{5}-\d{2}-\d{1}\b")

# เบอร์โทรไทยในข้อความ — ครอบคลุมรูปแบบที่พบบ่อย:
#   0812345678, 08-1234-5678, 081-234-5678, +66812345678, +66-812-345-678
_PHONE_RE = re.compile(
    r"(?:"
    r"\+66[-\s]?\d{2}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{1}"  # +66 format
    r"|0\d{2}[-\s]?\d{4}[-\s]?\d{3,4}"  # 08-1234-5678 / 08-1234-567
    r"|0\d{2}[-\s]?\d{3}[-\s]?\d{4}"  # 081-234-5678 (3-3-4)
    r"|0\d{3}[-\s]?\d{3}[-\s]?\d{3,4}"  # 0812-345-6789
    r"|0\d{9}"  # 0812345678 (10 หลักติดกัน)
    r")"
)

# placeholder token ที่ขั้น regex (เบอร์/บัตร) ใส่ไว้ เช่น "[PHONE_1]", "[IDCARD_2]"
# ใช้กัน NER ไม่ให้แทนที่ token เหล่านี้ซ้ำ (NER มักจับ "[PHONE_1]" เป็น LOCATION/URL)
_TOKEN_PLACEHOLDER_RE = re.compile(r"^\[[A-Z]+_\d+\]$")

# ประเภท NER → ป้าย token
# thaieda.ner ใช้ป้ายแบบ pythainlp: PERSON, LOCATION, ORGANIZATION
_NER_TYPE_TOKEN: dict[str, str] = {
    "PERSON": "NAME",
    "LOCATION": "LOC",
    "ORGANIZATION": "ORG",
}


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def anonymize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """ทำให้ DataFrame ไม่ระบุตัวบุคคลได้ — แทนที่ PII ด้วย token.

    ตรวจหาและแทนที่:
      * ชื่อบุคคล/องค์กร/สถานที่ ด้วย NER (ถ้ามี pythainlp + backend)
      * เบอร์โทรศัพท์ไทย ด้วย regex
      * เลขบัตรประชาชนไทย ด้วย regex

    แต่ละ PII ที่ไม่ซ้ำจะได้ token ของตัวเอง เช่น ``[NAME_1]``, ``[PHONE_1]``, ``[IDCARD_1]``
    PII ค่าเดียวกันจะได้ token เดียวกันเสมอ (คงความสัมพันธ์ในข้อมูลได้)

    Args:
        df: DataFrame ที่จะทำให้ไม่ระบุตัวบุคคลได้.

    Returns:
        (df_safe, token_map):
            df_safe — DataFrame ที่ PII ถูกแทนที่ด้วย token แล้ว
            token_map — {original_text: token} พร้อม key พิเศษ ``_ner_available``

    Raises:
        ไม่ raise — ถ้า NER ไม่พร้อมจะข้ามการแทนที่ชื่อ และบันทึกใน token_map
    """
    df_safe = df.copy()
    token_map: dict[str, str] = {}
    counters: dict[str, int] = defaultdict(int)
    ner_available = False

    for col in df_safe.columns:
        series = df_safe[col]
        # ประมวลผลเฉพาะคอลัมน์ข้อความ (object หรือ string dtype)
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        str_series = series.astype(str)
        # ทำงานบน unique values เพื่อความเร็ว (ไม่ประมวลผลทุกแถว)
        unique_vals = str_series.dropna().unique()

        # สร้าง mapping ค่าเดิม → ค่าใหม่ (เฉพาะที่เปลี่ยน)
        val_map: dict[str, str] = {}

        for val in unique_vals:
            if not isinstance(val, str) or not val:
                continue
            new_val = val

            # 1. แทนที่เลขบัตรประชาชน
            new_val = _IDCARD_RE.sub(
                lambda m: _get_or_create_token(m.group(0), "IDCARD", token_map, counters),
                new_val,
            )

            # 2. แทนที่เบอร์โทรศัพท์
            new_val = _PHONE_RE.sub(
                lambda m: _get_or_create_token(m.group(0), "PHONE", token_map, counters),
                new_val,
            )

            if new_val != val:
                val_map[val] = new_val

        # แทนที่ค่าใน Series แบบ vectorized (map เฉพาะค่าที่เปลี่ยน)
        if val_map:
            str_series = _apply_value_map(str_series, val_map)

        # 3. แทนที่ชื่อบุคคล/องค์กร/สถานที่ด้วย NER (ถ้าพร้อม)
        try:
            str_series, _ner_ok = _replace_ner_entities(str_series, col, token_map, counters)
            ner_available = ner_available or _ner_ok
        except ImportError:
            # NER ไม่พร้อม — ข้ามการแทนที่ชื่อ (ยังได้เบอร์/บัตรจาก regex ข้างต้น)
            pass

        df_safe[col] = str_series

    token_map["_ner_available"] = str(ner_available).lower()
    return df_safe, token_map


# ----------------------------------------------------------------------------
# helper
# ----------------------------------------------------------------------------
def _get_or_create_token(
    pii_text: str,
    token_type: str,
    token_map: dict[str, str],
    counters: dict[str, int],
) -> str:
    """คืน token ที่มีอยู่ หรือสร้างใหม่ สำหรับ PII แต่ละค่า (ค่าเดียวกัน → token เดียวกัน)."""
    if pii_text in token_map:
        return token_map[pii_text]
    counters[token_type] += 1
    token = f"[{token_type}_{counters[token_type]}]"
    token_map[pii_text] = token
    return token


def _apply_value_map(series: pd.Series, val_map: dict[str, str]) -> pd.Series:
    """แทนที่ค่าใน Series ตาม mapping dict (vectorized — ประมวลผลเฉพาะค่าที่เปลี่ยน).

    ทำงานเฉพาะกับค่าสตริงที่อยู่ใน ``val_map`` — ค่าอื่น ๆ ไม่ถูกแตะ
    ใช้ default-argument binding เพื่อกัน B023 (loop-variable binding trap)
    """

    def _replace(x: str, _m: dict[str, str] = val_map) -> str:
        return _m.get(x, x) if isinstance(x, str) else x

    return series.map(_replace)


def _replace_ner_entities(
    series: pd.Series,
    column_name: str,
    token_map: dict[str, str],
    counters: dict[str, int],
) -> tuple[pd.Series, bool]:
    """ตรวจหาและแทนที่ named entities (PER/ORG/LOC) ในคอลัมน์ข้อความ.

    ใช้ ``thaieda.ner.extract_entities`` ในการสกัด entity จากคอลัมน์
    แล้วแทนทฺทา entity text แต่ตัวด้วย token แบบ literal string replace (vectorized)

    Returns:
        (series ที่แทนที่แล้ว, ner_used=True)

    Raises:
        ImportError: ถ้าไม่มี NER engine (pythainlp + backend)
    """
    from thaieda.ner import extract_entities  # lazy import — optional dependency [ner]

    # สกัด entities จากคอลัมน์ (ประมวลผลทุกเซลล์ — ไม่ sample)
    non_null = series.dropna()
    if len(non_null) == 0:
        return series, False

    result = extract_entities(non_null, max_sample=len(non_null))

    local_keys: list[str] = []
    for etype, pairs in result.top_entities.items():
        token_type = _NER_TYPE_TOKEN.get(etype)
        if token_type is None:
            continue
        for entity_text, _count in pairs:
            # ข้าม placeholder ที่ขั้น regex ใส่ไว้แล้ว (เช่น "[PHONE_1]") — ไม่แทนซ้ำ
            # ป้องกัน token_map เสีย (key="[PHONE_1]" → value="[LOC_1]") และคงความ invertible
            if _TOKEN_PLACEHOLDER_RE.match(entity_text):
                continue
            if not entity_text:
                continue
            if entity_text not in token_map:
                counters[token_type] += 1
                token = f"[{token_type}_{counters[token_type]}]"
                token_map[entity_text] = token
            local_keys.append(entity_text)

    if local_keys:
        import re

        # เรียงตามความยาวเพื่อป้องกันการจับคู่ส่วนย่อยก่อนส่วนเต็ม
        local_keys = sorted(list(set(local_keys)), key=len, reverse=True)
        pattern = "|".join(re.escape(k) for k in local_keys)
        series = series.str.replace(pattern, lambda m: token_map[m.group(0)], regex=True)
        ner_used = True

    return series, ner_used


__all__ = ["anonymize_dataframe"]
