"""Data I/O — อ่านไฟล์ CSV/JSON อัตโนมัติ พร้อมตรวจ encoding และ format.

โมดูลนี้เป็น "ประตูทางเข้า" ของ pipeline: รับพาธไฟล์แล้วคืน DataFrame โดย
ไม่ต้องให้ผู้ใช้ระบุ encoding/format เอง — เดาให้อัตโนมัติแต่ fail loudly ถ้าเดาไม่ได้

หลักการ:
  * encoding: ลอง utf-8 ก่อน (พบบ่อยสุด) แล้วค่อยถอยไป tis-620/cp874/cp1252 (ไทยเก่า)
  * format: เดาจากนามสกุลไฟล์ (.csv/.json/.jsonl/.ndjson) ไม่รู้จัก → ลอง CSV ก่อนแล้ว JSON
  * chardet เป็น optional (thaieda[detect]) — ใช้เป็นตัวช่วยเดา encoding ถ้าติดตั้งไว้
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ลำดับ encoding ที่ลองถอดรหัส (utf-8 ก่อนเสมอ แล้วค่อย legacy ไทย แล้ว Western)
_ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8", "tis-620", "cp874", "cp1252")

# นามสกุลไฟล์ -> format
_EXT_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}

# format ที่รองรับ
_VALID_FORMATS: frozenset[str] = frozenset({"csv", "json", "jsonl"})


def _can_decode(raw: bytes, encoding: str) -> bool:
    """ตรวจว่าถอดรหัสไบต์ทั้งหมดด้วย encoding นี้ได้โดยไม่ error ไหม."""
    try:
        raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return False
    return True


def _normalize_encoding_name(name: str) -> str:
    """ปรับชื่อ encoding ให้เป็นมาตรฐาน (เช่น ascii → utf-8 เพราะ utf-8 ครอบคลุม ascii)."""
    lowered = name.strip().lower()
    if lowered in ("ascii", "us-ascii"):
        return "utf-8"
    if lowered in ("iso-8859-11", "iso8859-11", "tis620", "tis-620"):
        return "tis-620"
    if lowered in ("windows-874", "cp874"):
        return "cp874"
    if lowered in ("windows-1252", "cp1252"):
        return "cp1252"
    return lowered


def detect_encoding(path: str | Path) -> str:
    """ตรวจ encoding ของไฟล์ — ลอง utf-8 ก่อน, แล้ว tis-620, cp874, cp1252.

    ถ้าติดตั้ง chardet (thaieda[detect]) จะใช้เป็นตัวช่วยเดาก่อน (ถ้ามั่นใจ ≥0.8 และถอดได้จริง)
    ไฟล์ว่างคืน 'utf-8' ตามค่าเริ่มต้น

    Raises:
        FileNotFoundError: ถ้าไม่พบไฟล์.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

    raw = p.read_bytes()
    if not raw:
        return "utf-8"

    # ตัวช่วยเดาด้วย chardet ถ้ามี — ยอมรับเมื่อมั่นใจสูงและถอดรหัสได้จริง
    try:
        import chardet  # lazy import — optional dependency [detect]

        guess = chardet.detect(raw)
        enc = guess.get("encoding")
        confidence = guess.get("confidence") or 0.0
        if enc and confidence >= 0.8:
            normalized = _normalize_encoding_name(enc)
            if _can_decode(raw, normalized):
                return normalized
    except ImportError:
        pass

    # ถอยไปลองทีละ encoding ตามลำดับความน่าจะเป็น
    for enc in _ENCODING_CANDIDATES:
        if _can_decode(raw, enc):
            return enc

    # ไม่มี encoding ใดถอดได้ทั้งหมด — คืน utf-8 ให้ผู้เรียกจัดการ error ต่อ
    return "utf-8"


def _sniff_format(path: str | Path) -> str:
    """เดา format จากเนื้อหา 4KB แรก เมื่อไม่รู้จากนามสกุล — '[' → json, หลาย '{' บรรทัด → jsonl."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return "csv"
    text = head.decode("utf-8", errors="ignore").lstrip()
    if text.startswith("["):
        return "json"
    if text.startswith("{"):
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) > 1 and all(ln.lstrip().startswith("{") for ln in lines):
            return "jsonl"
        return "json"
    return "csv"


def detect_format(path: str | Path) -> str:
    """ตรวจประเภทไฟล์จากนามสกุล — คืน 'csv', 'json' หรือ 'jsonl'.

    .csv/.tsv → csv, .json → json, .jsonl/.ndjson → jsonl
    นามสกุลที่ไม่รู้จักหรือไม่มีนามสกุล → เดาจากเนื้อหา ('[' → json, หลาย '{' บรรทัด → jsonl,
    อย่างอื่น → csv) — ไฟล์ที่ไม่มีอยู่จริงคืน 'csv'
    """
    ext = Path(path).suffix.lower()
    if ext in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[ext]
    return _sniff_format(path)


def _read_csv(path: Path, encoding: str) -> pd.DataFrame:
    """อ่านไฟล์ CSV/TSV — ตรวจตัวคั่น tab จากนามสกุล .tsv."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, encoding=encoding, sep=sep)


def _read_json_lines(path: Path, encoding: str) -> pd.DataFrame:
    """อ่าน JSON Lines (หนึ่ง JSON object ต่อหนึ่งบรรทัด)."""
    return pd.read_json(path, lines=True, encoding=encoding)


def _read_json_standard(path: Path, encoding: str) -> pd.DataFrame:
    """อ่าน JSON มาตรฐาน (array ของ records หรือ dict ของคอลัมน์)."""
    return pd.read_json(path, encoding=encoding)


def _read_json_any(path: Path, encoding: str) -> pd.DataFrame:
    """อ่าน JSON โดยเดารูปแบบ — ลอง JSON มาตรฐาน (array/object) ก่อน แล้วถอยไป JSONL.

    standard ก่อนเพราะ lines=True จะ "อ่านผ่าน" array แบบเงียบ ๆ แต่ได้ผลผิด (เช่นคอลัมน์ 0,1)
    ส่วน standard จะ fail ชัดเจนบนไฟล์ JSONL จึงถอยไป lines=True ได้อย่างปลอดภัย
    """
    try:
        return _read_json_standard(path, encoding)
    except (ValueError, UnicodeDecodeError):
        return _read_json_lines(path, encoding)


def _read_with_format(path: Path, fmt: str, encoding: str) -> pd.DataFrame:
    """อ่านไฟล์ตาม format ที่ระบุ (csv/json/jsonl)."""
    if fmt == "csv":
        return _read_csv(path, encoding)
    if fmt == "jsonl":
        return _read_json_lines(path, encoding)
    if fmt == "json":
        return _read_json_any(path, encoding)
    raise ValueError(
        f"format ไม่รองรับ: {fmt!r} — รองรับเฉพาะ auto, csv, json, jsonl"
    )


def read_data(
    path: str | Path, format: str = "auto", encoding: str = "auto"
) -> pd.DataFrame:
    """อ่านไฟล์ CSV/JSON อัตโนมัติ — ตรวจ encoding และ format ให้เอง.

    Args:
        path: พาธไฟล์ที่ต้องการอ่าน.
        format: "auto" (เดาจากนามสกุล), "csv", "json" หรือ "jsonl".
        encoding: "auto" (ลอง utf-8/tis-620/cp874/cp1252) หรือ encoding เฉพาะ.

    Returns:
        pandas DataFrame.

    Raises:
        FileNotFoundError: ถ้าไม่พบไฟล์.
        ValueError: ถ้า format ไม่รองรับ หรืออ่านไฟล์ไม่สำเร็จทุกวิธีที่ลอง.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์: {p}")

    if format != "auto" and format not in _VALID_FORMATS:
        raise ValueError(
            f"format ไม่รองรับ: {format!r} — รองรับเฉพาะ auto, csv, json, jsonl"
        )

    enc = detect_encoding(p) if encoding == "auto" else encoding
    auto_format = format == "auto"
    fmt = detect_format(p) if auto_format else format

    # ข้อความบอก encoding ที่ลอง — ช่วยผู้ใช้รู้ว่าควรระบุ --encoding อะไรถ้าอ่านไม่ออก
    if encoding == "auto":
        tried_enc = f"encoding ที่ลอง: {', '.join(_ENCODING_CANDIDATES)} (เลือก {enc})"
    else:
        tried_enc = f"encoding: {enc}"

    try:
        return _read_with_format(p, fmt, enc)
    except Exception as exc:  # noqa: BLE001 — เก็บ error ไว้ก่อน ลอง format อื่นถ้าเดาเอง
        if not auto_format:
            raise ValueError(f"อ่านไฟล์ {p} แบบ {fmt} ไม่สำเร็จ ({tried_enc}): {exc}") from exc
        # เดา format เอง — ลอง format อื่นที่เหลือ (เช่น ไม่มีนามสกุล → ลอง csv แล้ว json)
        for alt in ("json", "csv"):
            if alt == fmt:
                continue
            try:
                return _read_with_format(p, alt, enc)
            except Exception:  # noqa: BLE001 — ลองตัวถัดไป
                continue
        raise ValueError(
            f"อ่านไฟล์ {p} ไม่สำเร็จ (ลองทั้ง CSV และ JSON แล้ว, {tried_enc}): {exc}"
        ) from exc


__all__ = [
    "read_data",
    "detect_encoding",
    "detect_format",
]
