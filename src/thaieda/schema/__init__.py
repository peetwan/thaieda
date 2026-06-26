"""Multi-file schema discovery — ค้นหาความสัมพันธ์ระหว่างหลายตาราง (v0.5).

โมดูลนี้ "อ่านชุดไฟล์หลายไฟล์เป็นฐานข้อมูลเดียว" แล้วเดาว่าตารางไหนเชื่อมกับตารางไหน
ผ่านคอลัมน์คีย์ (PK/FK) โดยอาศัยทั้งชื่อคอลัมน์และค่าจริงในข้อมูล — เพื่อสร้างแผนผัง
ความสัมพันธ์ (ER diagram) และตรวจหาข้อมูลกำพร้า (orphan) ที่อ้างถึงคีย์ที่ไม่มีอยู่จริง

หลักการสำคัญ (เหมือนโมดูลอื่นของ ThaiEDA):
  * โลจิกล้วน — ไม่มี Jinja/HTML (การเรนเดอร์อยู่ใน report/)
  * normalize ค่าคีย์ก่อนเทียบเสมอ: เลขไทย→อารบิก, ลบอักขระล่องหน, ตัด ".0" ที่ค้างจาก float
  * ป้องกัน false positive ของ date↔date: ถ้าทั้งสองฝั่ง "ไม่ unique" จะไม่เชื่อมกัน
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from thaieda.detect import _THAI_DIGIT_MAP, _name_hints_id, detect_all

# ----------------------------------------------------------------------------
# ค่าคงที่ (เกณฑ์ตัดสิน)
# ----------------------------------------------------------------------------
# สัดส่วนค่าไม่ซ้ำที่ถือว่าคอลัมน์ "เกือบunique" — เป็นคีย์หลัก (PK) ได้
_UNIQUE_RATIO = 0.95
# จำนวนแถวขั้นต่ำที่ทำให้ uniqueness "มีความหมาย" (กันตารางจิ๋วที่บังเอิญไม่ซ้ำ)
_MIN_UNIQUE_ROWS = 5
# overlap ต่ำกว่านี้ = ปฏิเสธความสัมพันธ์ (ชื่อตรงแต่ค่าไม่เกี่ยวกัน)
_REJECT_OVERLAP = 0.5
# overlap ตั้งแต่นี้ = ยืนยันมั่นใจ (ต่ำกว่าแต่ >= reject = ความมั่นใจปานกลาง)
_CONFIRM_OVERLAP = 0.9
# จำนวนตัวอย่างค่ากำพร้าสูงสุดที่เก็บไว้แสดง
_MAX_ORPHAN_EXAMPLES = 5
# นามสกุลไฟล์ที่รองรับเมื่อสแกนไดเรกทอรี
_SUPPORTED_EXTS = (".csv", ".tsv", ".json", ".jsonl", ".ndjson")

# อักขระล่องหน (zero-width / BOM) ที่ต้องลบก่อนเทียบคีย์ — ทำให้ join พังแบบเงียบ ๆ
# U+200B/C/D = zero-width space/non-joiner/joiner, U+2060 = word joiner, U+FEFF = BOM
_ZW_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
# float ที่ค้างจากการอ่าน int ที่มี NaN (เช่น "3827.0" → "3827")
_FLOAT_INT_RE = re.compile(r"^(-?\d+)\.0+$")


# ----------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ----------------------------------------------------------------------------
@dataclass
class KeyCandidate:
    """คอลัมน์ที่น่าจะเป็นคีย์ (PK/FK) ของตารางหนึ่ง."""

    table: str  # ชื่อไฟล์ (ตัดนามสกุล)
    column: str
    is_unique: bool  # nunique/nrows >= 0.95 → เป็น PK candidate
    null_ratio: float
    cardinality: int  # จำนวนค่าไม่ซ้ำ
    name_hint: bool  # ชื่อบอกใบ้ว่าเป็น id (จาก _name_hints_id)
    dtype: str  # ชนิดข้อมูล pandas

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "column": self.column,
            "is_unique": self.is_unique,
            "null_ratio": self.null_ratio,
            "cardinality": self.cardinality,
            "name_hint": self.name_hint,
            "dtype": self.dtype,
        }


@dataclass
class Relationship:
    """ความสัมพันธ์หนึ่งคู่ระหว่างตาราง (child.FK → parent.PK)."""

    from_table: str  # ตารางลูก (ฝั่ง FK)
    from_column: str
    to_table: str  # ตารางแม่ (ฝั่ง PK)
    to_column: str
    match_method: str  # "name" | "value" | "both"
    overlap_ratio: float  # |distinct(FK) ∩ distinct(PK)| / |distinct(FK)|
    orphan_count: int  # ค่า FK (distinct) ที่ไม่มีใน PK
    orphan_ratio: float  # orphan_count / |distinct(FK)|
    cardinality: str  # "1:1" | "1:N"
    confidence: float  # 0-1 (ถ่วงน้ำหนักชื่อ+ค่า)
    description_th: str  # คำอธิบายความสัมพันธ์ภาษาไทย
    is_validated: bool  # True ถ้าตรวจ value overlap แล้ว

    def to_dict(self) -> dict:
        return {
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
            "match_method": self.match_method,
            "overlap_ratio": self.overlap_ratio,
            "orphan_count": self.orphan_count,
            "orphan_ratio": self.orphan_ratio,
            "cardinality": self.cardinality,
            "confidence": self.confidence,
            "description_th": self.description_th,
            "is_validated": self.is_validated,
        }


@dataclass
class TableProfile:
    """สรุปโครงสร้างของตารางหนึ่ง (ไฟล์หนึ่งไฟล์)."""

    name: str  # ชื่อไฟล์ (ตัดนามสกุล)
    file_path: str
    row_count: int
    column_count: int
    columns: list[str]
    column_types: dict[str, str]  # ชื่อคอลัมน์ -> ColumnType value
    key_candidates: list[KeyCandidate]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "column_types": self.column_types,
            "key_candidates": [k.to_dict() for k in self.key_candidates],
            "notes": self.notes,
        }


@dataclass
class DatasetProfile:
    """ผลวิเคราะห์ชุดข้อมูลหลายตาราง — ตาราง + ความสัมพันธ์ + ข้อมูลกำพร้า."""

    tables: list[TableProfile]
    relationships: list[Relationship]
    notes: list[str] = field(default_factory=list)
    orphan_findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "table_count": len(self.tables),
            "relationship_count": len(self.relationships),
            "tables": [t.to_dict() for t in self.tables],
            "relationships": [r.to_dict() for r in self.relationships],
            "orphan_findings": self.orphan_findings,
            "notes": self.notes,
        }

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        """ส่งออกเป็น JSON string (เขียนไฟล์ด้วยถ้าระบุ path)."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_mermaid(self) -> str:
        """สร้างข้อความ Mermaid erDiagram จากตาราง + ความสัมพันธ์.

        แต่ละตารางเป็น entity พร้อมคอลัมน์ (ทำเครื่องหมาย PK/FK) และเส้นความสัมพันธ์
        ใช้รูปแบบ `PARENT ||--o{ CHILD : "fk"` (1:N) หรือ `||--||` (1:1)
        ชื่อ entity/attribute ถูก sanitize ให้เหลือ [A-Za-z0-9_] เพื่อไม่ให้ Mermaid พัง
        """
        lines: list[str] = ["erDiagram"]

        # คอลัมน์ FK ต่อตาราง (จากความสัมพันธ์ที่ตารางเป็นฝั่งลูก)
        fk_map: dict[str, set[str]] = defaultdict(set)
        for rel in self.relationships:
            fk_map[rel.from_table].add(rel.from_column)

        for t in self.tables:
            pk_cols = {k.column for k in t.key_candidates if k.is_unique}
            fk_cols = fk_map.get(t.name, set())
            ent = _sanitize_ident(t.name)
            lines.append(f"    {ent} {{")
            for col in t.columns:
                typ = _mermaid_type(t.column_types.get(col, ""))
                attr = _sanitize_ident(col)
                if col in pk_cols:
                    marker = " PK"
                elif col in fk_cols:
                    marker = " FK"
                else:
                    marker = ""
                lines.append(f"        {typ} {attr}{marker}")
            lines.append("    }")

        for rel in self.relationships:
            parent = _sanitize_ident(rel.to_table)
            child = _sanitize_ident(rel.from_table)
            conn = "||--||" if rel.cardinality == "1:1" else "||--o{"
            label = _sanitize_ident(rel.from_column)
            lines.append(f'    {parent} {conn} {child} : "{label}"')

        return "\n".join(lines)


# ----------------------------------------------------------------------------
# helper: normalize ค่าคีย์
# ----------------------------------------------------------------------------
def _normalize_key_series(non_null: pd.Series) -> pd.Series:
    """แปลงค่าคีย์เป็นสตริงมาตรฐานสำหรับเทียบ (แบบ vectorized เพื่อรองรับข้อมูลล้านแถว).

    ขั้นตอน: → str, เลขไทย→อารบิก, ลบอักขระล่องหน, ตัดช่องว่าง, ตัด ".0" ที่ค้างจาก float
    """
    s = non_null.astype(str)
    s = s.str.translate(_THAI_DIGIT_MAP)
    s = s.str.replace(_ZW_RE, "", regex=True)
    s = s.str.strip()
    # callable แทนสตริง backreference (r"\1") — Arrow regex engine ใน pandas 3.x ไม่รองรับ \1
    s = s.str.replace(_FLOAT_INT_RE, lambda m: m.group(1), regex=True)
    return s


def _distinct_norm(
    non_null: pd.Series,
    sample_size: int | None,
    cache: dict[tuple[str, str], set[str]],
    cache_key: tuple[str, str],
) -> set[str]:
    """คืนเซ็ตของค่าคีย์ที่ normalize แล้ว (ไม่ซ้ำ) — memoize ต่อ (ตาราง, คอลัมน์)."""
    if cache_key in cache:
        return cache[cache_key]
    series = non_null
    if sample_size is not None and len(series) > sample_size:
        series = series.head(sample_size)
    norm = _normalize_key_series(series)
    distinct = set(norm.unique().tolist())
    distinct.discard("")
    cache[cache_key] = distinct
    return distinct


def _overlap_stats(child_d: set[str], parent_d: set[str]) -> tuple[float, int, float, list[str]]:
    """คำนวณ overlap / orphan ระหว่างค่าคีย์ฝั่งลูก (FK) กับฝั่งแม่ (PK)."""
    if not child_d:
        return 0.0, 0, 0.0, []
    inter = child_d & parent_d
    overlap = len(inter) / len(child_d)
    orphans = child_d - parent_d
    orphan_count = len(orphans)
    orphan_ratio = orphan_count / len(child_d)
    examples = sorted(orphans)[:_MAX_ORPHAN_EXAMPLES]
    return overlap, orphan_count, orphan_ratio, examples


# ----------------------------------------------------------------------------
# โครงสร้างภายในระหว่างจับคู่
# ----------------------------------------------------------------------------
@dataclass
class _ColInfo:
    """ข้อมูลคอลัมน์หนึ่งระหว่างขั้นตอนจับคู่ความสัมพันธ์."""

    table: str
    column: str
    non_null: pd.Series
    n: int
    nunique: int
    unique: bool
    is_bool: bool


def _col_info(table: str, column: str, series: pd.Series) -> _ColInfo | None:
    """สร้าง _ColInfo — คืน None ถ้าคอลัมน์ว่างทั้งหมด."""
    is_bool = bool(pd.api.types.is_bool_dtype(series))
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return None
    nun = int(non_null.nunique())
    unique = (not is_bool) and n >= _MIN_UNIQUE_ROWS and nun > 1 and (nun / n) >= _UNIQUE_RATIO
    return _ColInfo(table, column, non_null, n, nun, unique, is_bool)


def _evaluate_pair(
    a: _ColInfo,
    b: _ColInfo,
    validate: bool,
    sample_size: int | None,
    cache: dict[tuple[str, str], set[str]],
) -> Relationship | None:
    """ตัดสินว่าคอลัมน์ a, b (ชื่อตรงกัน) เป็นความสัมพันธ์หรือไม่ + ทิศทาง + ความมั่นใจ.

    กฎ:
      * ทั้งคู่ "ไม่ unique" → ไม่เชื่อม (กัน date↔date / หมวดหมู่ false positive)
      * ฝั่ง unique = แม่ (PK/to_), ฝั่งไม่ unique = ลูก (FK/from_) → 1:N
      * ทั้งคู่ unique → ฝั่งค่าไม่ซ้ำมากกว่าเป็นแม่; ถ้าเท่ากัน → 1:1
      * ถ้า validate: overlap < 0.5 ปฏิเสธ; ใช้ทิศทางที่ค่า overlap สูงกว่า
    """
    # bool ทั้งคู่ไม่มีทางเป็นคีย์ (มีแค่ True/False)
    if a.is_bool and b.is_bool:
        return None
    if not a.unique and not b.unique:
        return None

    # --- ตัดสินทิศทางเชิงโครงสร้าง ---
    if a.unique and not b.unique:
        parent, child, card = a, b, "1:N"
    elif b.unique and not a.unique:
        parent, child, card = b, a, "1:N"
    elif a.nunique == b.nunique:
        card = "1:1"
        # ทิศทางเริ่มต้นแบบ deterministic (เรียงชื่อตาราง) — อาจสลับด้วย overlap ภายหลัง
        parent, child = (a, b) if a.table <= b.table else (b, a)
    elif a.nunique > b.nunique:
        parent, child, card = a, b, "1:N"
    else:
        parent, child, card = b, a, "1:N"

    overlap = 0.0
    orphan_count = 0
    orphan_ratio = 0.0
    is_validated = False
    method = "name"

    if validate:
        child_d = _distinct_norm(child.non_null, sample_size, cache, (child.table, child.column))
        parent_d = _distinct_norm(
            parent.non_null, sample_size, cache, (parent.table, parent.column)
        )
        overlap, orphan_count, orphan_ratio, _ = _overlap_stats(child_d, parent_d)

        # ทิศทางกำกวม (1:1) หรือ overlap ต่ำ → ลองสลับทิศทางถ้าได้ค่าดีกว่า
        if card == "1:1" or overlap < _REJECT_OVERLAP:
            rev_overlap, rev_orphan, rev_ratio, _ = _overlap_stats(parent_d, child_d)
            if rev_overlap > overlap:
                parent, child = child, parent
                child_d, parent_d = parent_d, child_d
                overlap, orphan_count, orphan_ratio = rev_overlap, rev_orphan, rev_ratio

        if overlap < _REJECT_OVERLAP:
            return None
        is_validated = True
        method = "both"

    confidence = round(min(1.0, 0.4 + 0.6 * overlap), 3) if is_validated else 0.5
    description_th = _describe_relationship(
        child.table, child.column, parent.table, card, overlap, is_validated
    )

    return Relationship(
        from_table=child.table,
        from_column=child.column,
        to_table=parent.table,
        to_column=parent.column,
        match_method=method,
        overlap_ratio=round(overlap, 4),
        orphan_count=orphan_count,
        orphan_ratio=round(orphan_ratio, 4),
        cardinality=card,
        confidence=confidence,
        description_th=description_th,
        is_validated=is_validated,
    )


def _describe_relationship(
    child_table: str,
    child_column: str,
    parent_table: str,
    cardinality: str,
    overlap: float,
    is_validated: bool,
) -> str:
    """สร้างคำอธิบายความสัมพันธ์ภาษาไทย."""
    cov = f", ครอบคลุม {overlap * 100:.1f}%" if is_validated else ", จับคู่ด้วยชื่อคอลัมน์"
    return (
        f"ตาราง {child_table} อ้างอิงไป {parent_table} ผ่านคอลัมน์ {child_column} ({cardinality}{cov})"
    )


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def discover_keys(df: pd.DataFrame, table_name: str) -> list[KeyCandidate]:
    """ค้นหาคอลัมน์ที่น่าจะเป็นคีย์ (PK/FK) ในตาราง.

    คืน KeyCandidate สำหรับคอลัมน์ที่ "ชื่อบอกใบ้ว่าเป็น id" หรือ "เกือบ unique" (>=95%
    และมีข้อมูลมากพอ) — ตัดคอลัมน์บูลีนและคอลัมน์ค่าเดียว (constant) ออก
    """
    out: list[KeyCandidate] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            continue  # ตัดบูลีน — เป็นได้แค่ True/False ไม่ใช่คีย์
        non_null = series.dropna()
        n = len(non_null)
        if n == 0:
            continue
        nun = int(non_null.nunique())
        if nun <= 1:
            continue  # คอลัมน์ค่าเดียว ไม่ใช่คีย์
        is_unique = (nun / n) >= _UNIQUE_RATIO
        name_hint = _name_hints_id(series)
        # รวมเฉพาะที่ชื่อบอกใบ้ หรือ unique อย่างมีนัย (n มากพอ) — กันค่าไม่ซ้ำโดยบังเอิญในตารางจิ๋ว
        if not (name_hint or (is_unique and n >= _MIN_UNIQUE_ROWS)):
            continue
        out.append(
            KeyCandidate(
                table=table_name,
                column=str(col),
                is_unique=is_unique,
                null_ratio=round(float(series.isna().mean()), 4),
                cardinality=nun,
                name_hint=name_hint,
                dtype=str(series.dtype),
            )
        )
    return out


def match_relationships(
    tables: dict[str, pd.DataFrame],
    table_profiles: dict[str, TableProfile],
    *,
    validate_values: bool = True,
    sample_size: int | None = None,
) -> list[Relationship]:
    """จับคู่ความสัมพันธ์ระหว่างตาราง โดยอาศัยชื่อคอลัมน์ + ค่าจริง.

    ขั้นตอน:
      1. จับกลุ่มคอลัมน์ที่ชื่อ (normalize lower().strip()) ตรงกันข้ามตาราง
      2. ตัดสินทิศทาง: ฝั่ง unique = แม่ (PK), ฝั่งไม่ unique = ลูก (FK)
         - ทั้งคู่ไม่ unique → ไม่เชื่อม (กัน date↔date false positive)
      3. ถ้า validate_values: เทียบ overlap ของค่าจริง (normalize เลขไทย/อักขระล่องหน)
         - overlap < 0.5 ปฏิเสธ, >= 0.5 ยอมรับ (มั่นใจตาม overlap), นับ orphan
      4. สร้างคำอธิบายภาษาไทย + เรียงตามความมั่นใจ

    Args:
        tables: ชื่อตาราง -> DataFrame.
        table_profiles: ชื่อตาราง -> TableProfile (ใช้ชื่อคีย์/ใบ้).
        validate_values: ตรวจ value overlap ด้วยหรือไม่ (ปิดเพื่อความเร็ว).
        sample_size: จำกัดจำนวนแถวที่นำมาเทียบค่า (None = ใช้ค่าทั้งหมด).

    Returns:
        รายการ Relationship เรียงตามความมั่นใจ (มากไปน้อย).
    """
    # จับกลุ่มตามชื่อคอลัมน์ที่ normalize แล้ว
    name_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tname, df in tables.items():
        for col in df.columns:
            name_map[str(col).strip().lower()].append((tname, str(col)))

    cache: dict[tuple[str, str], set[str]] = {}
    relationships: list[Relationship] = []
    seen: set[tuple[str, str, str, str]] = set()

    for occ in name_map.values():
        if len(occ) < 2:
            continue
        infos: list[_ColInfo] = []
        for tname, col in occ:
            info = _col_info(tname, col, tables[tname][col])
            if info is not None:
                infos.append(info)
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                rel = _evaluate_pair(infos[i], infos[j], validate_values, sample_size, cache)
                if rel is None:
                    continue
                key = (rel.from_table, rel.from_column, rel.to_table, rel.to_column)
                if key in seen:
                    continue
                seen.add(key)
                relationships.append(rel)

    relationships.sort(key=lambda r: (-r.confidence, r.from_table, r.to_table, r.from_column))
    return relationships


def _orphan_findings(relationships: list[Relationship]) -> list[str]:
    """สร้างข้อความ orphan (ภาษาไทย) สำหรับความสัมพันธ์ที่มีค่ากำพร้า."""
    findings: list[str] = []
    for rel in relationships:
        if rel.orphan_count <= 0:
            continue
        findings.append(
            f"ตาราง {rel.from_table} มีค่า {rel.from_column} ที่ไม่มีใน {rel.to_table} "
            f"จำนวน {rel.orphan_count:,} ค่า ({rel.orphan_ratio * 100:.1f}% ของค่าไม่ซ้ำ) — "
            f"ข้อมูลกำพร้า อาจกระทบการ join/วิเคราะห์"
        )
    return findings


def _resolve_paths(paths: list[str] | str) -> list[Path]:
    """แปลง input (ไดเรกทอรี / ไฟล์เดียว / list ของไฟล์) เป็นรายการไฟล์ที่มีอยู่จริง."""
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            return [
                f
                for f in sorted(p.iterdir())
                if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTS
            ]
        return [p]
    return [Path(x) for x in paths]


def _unique_table_name(stem: str, used: set[str]) -> str:
    """คืนชื่อตารางที่ไม่ชนกับที่มีอยู่ (เติม _2, _3 ถ้าซ้ำ)."""
    if stem not in used:
        return stem
    i = 2
    while f"{stem}_{i}" in used:
        i += 1
    return f"{stem}_{i}"


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    """แจ้งความคืบหน้าถ้ามี callback."""
    if progress is not None:
        progress(message)


def profile_dataset(
    paths: list[str] | str,
    *,
    lang: str = "th",
    validate_values: bool = True,
    max_file_size_mb: float = 500,
    sample_size: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> DatasetProfile:
    """วิเคราะห์หลายไฟล์พร้อมกัน ค้นหาความสัมพันธ์ระหว่างตาราง.

    ขั้นตอน:
      1. ถ้า paths เป็นไดเรกทอรี → หาไฟล์ .csv/.json/.jsonl/.ndjson ทั้งหมด
      2. อ่านแต่ละไฟล์ด้วย read_data()
      3. ตรวจประเภทคอลัมน์ด้วย detect_all() + discover_keys()
      4. match_relationships() รวมทุกตาราง
      5. สร้าง orphan_findings สำหรับความสัมพันธ์ที่มีค่ากำพร้า
      6. คืน DatasetProfile

    Args:
        paths: ไดเรกทอรี, ไฟล์เดียว, หรือ list ของไฟล์.
        lang: ภาษาของหมายเหตุ (สงวนไว้สำหรับอนาคต — คำอธิบายเป็นภาษาไทยเสมอ).
        validate_values: ตรวจ value overlap (ปิดด้วย False เพื่อความเร็ว).
        max_file_size_mb: ข้ามไฟล์ที่ใหญ่กว่านี้ (เมกะไบต์).
        sample_size: จำกัดแถวที่นำมาเทียบค่า (None = ทั้งหมด).
        progress: callback(ข้อความ) สำหรับแสดงความคืบหน้า.

    Returns:
        DatasetProfile พร้อมตาราง + ความสัมพันธ์ + ข้อมูลกำพร้า.
    """
    from thaieda.io import read_data

    file_list = _resolve_paths(paths)
    tables: dict[str, pd.DataFrame] = {}
    profiles_map: dict[str, TableProfile] = {}
    profiles_list: list[TableProfile] = []
    notes: list[str] = []

    for p in file_list:
        if not p.is_file():
            notes.append(f"ไม่พบไฟล์: {p}")
            continue
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0.0
        if size_mb > max_file_size_mb:
            notes.append(f"ข้ามไฟล์ {p.name} ({size_mb:.0f}MB > {max_file_size_mb:.0f}MB)")
            continue

        _emit(progress, f"อ่าน {p.name} ...")
        try:
            df = read_data(p)
        except Exception as exc:  # noqa: BLE001 — ไฟล์เสียหนึ่งไฟล์ไม่ควรล้มทั้งชุด
            notes.append(f"อ่านไฟล์ {p.name} ไม่สำเร็จ: {exc}")
            continue
        if df.empty:
            notes.append(f"ไฟล์ {p.name} ว่างเปล่า — ข้าม")
            continue

        name = _unique_table_name(p.stem, set(profiles_map))
        _emit(progress, f"ตรวจโครงสร้าง {name} ...")
        col_types = detect_all(df)
        keys = discover_keys(df, name)
        profile = TableProfile(
            name=name,
            file_path=str(p),
            row_count=int(len(df)),
            column_count=int(len(df.columns)),
            columns=[str(c) for c in df.columns],
            column_types={k: v.value for k, v in col_types.items()},
            key_candidates=keys,
        )
        tables[name] = df
        profiles_map[name] = profile
        profiles_list.append(profile)

    if len(profiles_list) < 2:
        notes.append("มีตารางน้อยกว่า 2 ตาราง — ไม่สามารถค้นหาความสัมพันธ์ระหว่างตารางได้")

    _emit(progress, "จับคู่ความสัมพันธ์ระหว่างตาราง ...")
    relationships = match_relationships(
        tables, profiles_map, validate_values=validate_values, sample_size=sample_size
    )
    orphan_findings = _orphan_findings(relationships)

    return DatasetProfile(
        tables=profiles_list,
        relationships=relationships,
        notes=notes,
        orphan_findings=orphan_findings,
    )


# ----------------------------------------------------------------------------
# helper: sanitize สำหรับ Mermaid
# ----------------------------------------------------------------------------
_NON_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")

# pandas dtype / ColumnType → ชนิดแบบสั้นสำหรับ Mermaid (ห้ามมีช่องว่าง)
_MERMAID_TYPE: dict[str, str] = {
    "numeric": "number",
    "categorical": "string",
    "thai_text": "string",
    "english_text": "string",
    "mixed_text": "string",
    "datetime": "datetime",
    "id": "id",
    "phone_number": "string",
    "empty": "string",
}


def _sanitize_ident(name: str) -> str:
    """แปลงชื่อ entity/attribute ให้เหลือเฉพาะ [A-Za-z0-9_] (กัน Mermaid พัง)."""
    cleaned = _NON_IDENT_RE.sub("_", str(name)).strip("_")
    if not cleaned:
        return "col"
    if cleaned[0].isdigit():
        cleaned = "t_" + cleaned
    return cleaned


def _mermaid_type(column_type: str) -> str:
    """แปลง ColumnType value เป็นชนิดสั้นสำหรับ Mermaid."""
    return _MERMAID_TYPE.get(column_type, "string")


__all__ = [
    "KeyCandidate",
    "Relationship",
    "TableProfile",
    "DatasetProfile",
    "discover_keys",
    "match_relationships",
    "profile_dataset",
]
