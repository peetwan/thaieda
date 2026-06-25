"""Named Entity Recognition — สกัดชื่อเฉพาะ (คน/สถานที่/องค์กร) จากคอลัมน์ข้อความไทย.

เป็นฟีเจอร์เฉพาะภาษาไทยที่ AutoEDA ทั่วไปไม่มี — ใช้ pythainlp.tag.NER ตัดชื่อเฉพาะออกมา
แล้วสรุปว่าในคอลัมน์มีคน/สถานที่/องค์กร อะไรบ้าง ปรากฏกี่ครั้ง

หลักการ: import pythainlp แบบ lazy, ไม่มี fallback แบบเงียบ ๆ — ถ้าไม่มี NER engine
จะ fail loudly พร้อมคำแนะนำติดตั้ง (ยกเว้นคอลัมน์ว่างที่คืนผลเปล่าได้โดยไม่ต้องโหลด engine)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd

from thaieda.tokenize import Tokenizer

# ----------------------------------------------------------------------------
# ค่าคงที่
# ----------------------------------------------------------------------------
# ลำดับ NER engine ที่จะลอง (thainer = CRF เบากว่า, thainer-v2 = transformers แม่นกว่าแต่หนัก)
_NER_ENGINE_PREFERENCE = ("thainer", "thainer-v2")

# จำนวนเซลล์แรกที่เก็บ entity ราย ๆ เซลล์ไว้แสดงเป็นตัวอย่างในรายงาน
_MAX_SAMPLE_CELLS = 10

_NER_INSTALL_HINT = (
    "Thai NER requires pip install thaieda[ner] (the 'pythainlp' package with a NER engine "
    "backend, e.g. 'python-crfsuite'). No working NER engine found."
)


# ----------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ----------------------------------------------------------------------------
@dataclass
class NEREntity:
    """ชื่อเฉพาะหนึ่งรายการที่สกัดได้จากข้อความ."""

    text: str  # ข้อความของ entity
    entity_type: str  # ประเภท เช่น PERSON, LOCATION, ORGANIZATION
    start: int  # ตำแหน่งอักขระเริ่มต้น (0-based) ในเซลล์
    end: int  # ตำแหน่งอักขระสิ้นสุด (exclusive)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
        }


@dataclass
class NERResult:
    """สรุปการสกัดชื่อเฉพาะของคอลัมน์หนึ่ง."""

    column: str
    total_entities: int
    entity_counts: dict[str, int]  # {ประเภท: จำนวน}
    top_entities: dict[str, list[tuple[str, int]]]  # {ประเภท: [(ข้อความ, จำนวน), ...]}
    sample_entities: list[list[NEREntity]] = field(default_factory=list)  # entity รายเซลล์ N เซลล์แรก
    engine_used: str = ""

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "total_entities": self.total_entities,
            "entity_counts": self.entity_counts,
            "top_entities": {
                etype: [list(pair) for pair in pairs] for etype, pairs in self.top_entities.items()
            },
            "sample_entities": [[e.to_dict() for e in cell] for cell in self.sample_entities],
            "engine_used": self.engine_used,
        }


# ----------------------------------------------------------------------------
# helper ทั่วไป
# ----------------------------------------------------------------------------
def _col_name(series: pd.Series) -> str:
    return str(series.name) if series.name is not None else ""


def _iob_to_entities(tagged: list[tuple[str, str]]) -> list[NEREntity]:
    """แปลงผลลัพธ์ NER แบบ IOB (รายการ (คำ, ป้าย)) เป็นรายการ NEREntity พร้อมตำแหน่งอักขระ.

    ป้ายเป็นรูปแบบ IOB เช่น 'B-PERSON' (เริ่ม entity), 'I-PERSON' (ต่อ entity), 'O' (ไม่ใช่)
    ตำแหน่งอักขระคำนวณจากการสะสมความยาวคำตามลำดับ (คำของ NER ต่อกันได้ข้อความเดิม)
    เป็นฟังก์ชันบริสุทธิ์ (ไม่พึ่ง engine) จึงทดสอบแยกได้
    """
    entities: list[NEREntity] = []
    pos = 0
    cur_words: list[str] = []
    cur_type: str | None = None
    cur_start = 0

    def flush() -> None:
        nonlocal cur_words, cur_type
        if cur_type is not None and cur_words:
            text = "".join(cur_words)
            entities.append(NEREntity(text, cur_type, cur_start, cur_start + len(text)))
        cur_words = []
        cur_type = None

    for word, label in tagged:
        if label.startswith("B-"):
            flush()
            cur_type = label[2:]
            cur_start = pos
            cur_words = [word]
        elif label.startswith("I-"):
            etype = label[2:]
            if cur_type == etype:
                cur_words.append(word)
            else:
                # I- ที่ไม่มี B- นำหน้า (ผลจาก tagger ไม่สมบูรณ์) — เริ่ม entity ใหม่แบบผ่อนปรน
                flush()
                cur_type = etype
                cur_start = pos
                cur_words = [word]
        else:
            flush()
        pos += len(word)

    flush()
    return entities


# ----------------------------------------------------------------------------
# การจัดการ NER engine (lazy, แคชไว้)
# ----------------------------------------------------------------------------
# แคช engine ที่สร้างแล้ว: {ชื่อ engine: NER object} — กันการโหลดโมเดลซ้ำต่อคอลัมน์
_NER_CACHE: dict[str, object] = {}


def _get_ner(engine: str | None = None) -> tuple[object, str]:
    """สร้าง (หรือดึงจากแคช) NER engine — คืน (engine object, ชื่อ engine).

    ถ้า engine เป็น None จะลองตาม _NER_ENGINE_PREFERENCE และใช้ตัวแรกที่ backend พร้อม
    Raises:
        ImportError: ถ้าไม่มี pythainlp หรือไม่มี NER backend ที่ใช้ได้เลย.
    """
    try:
        from pythainlp.tag import NER
    except ImportError as exc:
        raise ImportError(_NER_INSTALL_HINT) from exc

    candidates = (engine,) if engine is not None else _NER_ENGINE_PREFERENCE
    last_error: Exception | None = None
    for name in candidates:
        if name in _NER_CACHE:
            return _NER_CACHE[name], name
        try:
            obj = NER(engine=name)
        except ImportError as exc:
            # backend ของ engine นี้ไม่ได้ติดตั้ง — ลองตัวถัดไป
            last_error = exc
            continue
        _NER_CACHE[name] = obj
        return obj, name

    raise ImportError(_NER_INSTALL_HINT) from last_error


def ner_available() -> bool:
    """คืน True ถ้ามี NER engine ที่ใช้งานได้ (pythainlp + backend) — ใช้ตัดสินใจในรายงาน.

    หมายเหตุ: การเรียกครั้งแรกอาจโหลด/ดาวน์โหลดโมเดล NER (ถ้ามี backend ติดตั้ง)
    """
    try:
        _get_ner()
    except ImportError:
        return False
    return True


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def extract_entities(
    series: pd.Series,
    tokenizer: Tokenizer | None = None,
    max_sample: int = 1000,
    top_n: int = 20,
    random_state: int = 42,
    engine: str | None = None,
) -> NERResult:
    """สกัด named entities จากคอลัมน์ข้อความไทย แล้วสรุปจำนวน/ตัวอย่างตามประเภท.

    Args:
        series: คอลัมน์ข้อความ.
        tokenizer: รับไว้เพื่อความสม่ำเสมอของ API (NER ตัดคำเองภายใน จึงไม่ถูกใช้).
        max_sample: จำนวนเซลล์สูงสุดที่นำมาวิเคราะห์ (สุ่มถ้าเกิน).
        top_n: จำนวน entity ยอดนิยมที่เก็บต่อประเภท.
        random_state: seed สำหรับการสุ่มตัวอย่าง (ผลทำซ้ำได้).
        engine: ระบุ NER engine เอง (เช่น 'thainer', 'thainer-v2') หรือ None = เลือกอัตโนมัติ.

    Returns:
        NERResult — คอลัมน์ว่าง (ไม่มีข้อความ) คืนผลเปล่าโดยไม่ต้องโหลด engine.

    Raises:
        ImportError: เมื่อคอลัมน์มีข้อความแต่ไม่มี NER engine ที่ใช้ได้ (ติดตั้ง thaieda[ner]).
    """
    col = _col_name(series)
    non_null = series.dropna().astype(str)

    if len(non_null) > max_sample:
        sample = non_null.sample(n=max_sample, random_state=random_state)
    else:
        sample = non_null

    # คอลัมน์ว่าง — คืนผลเปล่าได้เลย ไม่ต้องโหลดโมเดล (ทำให้ทดสอบ edge case ได้โดยไม่ต้องมี backend)
    if len(sample) == 0:
        return NERResult(
            column=col,
            total_entities=0,
            entity_counts={},
            top_entities={},
            sample_entities=[],
            engine_used="",
        )

    ner, engine_name = _get_ner(engine)

    type_counts: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    sample_entities: list[list[NEREntity]] = []

    for i, text in enumerate(sample):
        try:
            tagged = ner.tag(text)  # type: ignore[attr-defined]
            entities = _iob_to_entities(tagged)
        except Exception:  # noqa: BLE001 — เซลล์เดียวพังไม่ควรล้มทั้งคอลัมน์
            entities = []
        if i < _MAX_SAMPLE_CELLS:
            sample_entities.append(entities)
        for ent in entities:
            type_counts[ent.entity_type] += 1
            by_type[ent.entity_type][ent.text] += 1

    top_entities = {etype: counter.most_common(top_n) for etype, counter in by_type.items()}

    return NERResult(
        column=col,
        total_entities=int(sum(type_counts.values())),
        entity_counts=dict(type_counts),
        top_entities=top_entities,
        sample_entities=sample_entities,
        engine_used=engine_name,
    )


__all__ = [
    "NEREntity",
    "NERResult",
    "extract_entities",
    "ner_available",
]
