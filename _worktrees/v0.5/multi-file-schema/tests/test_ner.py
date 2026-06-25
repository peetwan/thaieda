"""ทดสอบ thaieda.ner — การสกัด named entities จากข้อความไทย.

NER engine ของ pythainlp ต้องมี backend (python-crfsuite / transformers) จึงจะรันได้จริง
จึงทดสอบตรรกะการแปลง IOB แยกจาก engine (เป็นฟังก์ชันบริสุทธิ์) และ skip เทสที่ต้องใช้ engine
เมื่อ backend ไม่พร้อม
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from thaieda.ner import (
    NEREntity,
    NERResult,
    _iob_to_entities,
    extract_entities,
    ner_available,
)

_ner_ok = ner_available()
requires_ner = pytest.mark.skipif(not _ner_ok, reason="NER engine backend not installed")
requires_no_ner = pytest.mark.skipif(_ner_ok, reason="NER engine backend is installed")


# --------------------------------------------------------- IOB -> entities (pure)
def test_iob_to_entities_basic():
    tagged = [
        ("นาย", "B-PERSON"),
        ("สมชาย", "I-PERSON"),
        (" ", "O"),
        ("เดินทาง", "O"),
        ("ไป", "O"),
        ("กรุงเทพ", "B-LOCATION"),
    ]
    ents = _iob_to_entities(tagged)
    assert len(ents) == 2
    person, location = ents
    assert person.text == "นายสมชาย"
    assert person.entity_type == "PERSON"
    assert person.start == 0
    assert person.end == len("นายสมชาย")
    assert location.text == "กรุงเทพ"
    assert location.entity_type == "LOCATION"
    # ตำแหน่งของ "กรุงเทพ": นาย(3)+สมชาย(5)+' '(1)+เดินทาง(7)+ไป(2) = 18
    assert location.start == 18
    assert location.end == 25


def test_iob_to_entities_empty_and_all_o():
    assert _iob_to_entities([]) == []
    assert _iob_to_entities([("ก", "O"), ("ข", "O")]) == []


def test_iob_to_entities_orphan_inside_tag_is_lenient():
    # I- ที่ไม่มี B- นำหน้า ต้องถูกตีความเป็น entity ใหม่ (ผ่อนปรน)
    tagged = [("ปารีส", "I-LOCATION")]
    ents = _iob_to_entities(tagged)
    assert len(ents) == 1
    assert ents[0].entity_type == "LOCATION"
    assert ents[0].text == "ปารีส"


def test_iob_to_entities_consecutive_entities():
    tagged = [
        ("สมชาย", "B-PERSON"),
        ("สมหญิง", "B-PERSON"),
    ]
    ents = _iob_to_entities(tagged)
    assert len(ents) == 2
    assert all(e.entity_type == "PERSON" for e in ents)
    assert ents[1].start == len("สมชาย")


# --------------------------------------------------------- dataclasses
def test_ner_entity_to_dict():
    e = NEREntity(text="สมชาย", entity_type="PERSON", start=0, end=5)
    d = e.to_dict()
    assert d == {"text": "สมชาย", "entity_type": "PERSON", "start": 0, "end": 5}


def test_ner_result_to_dict():
    e = NEREntity("สมชาย", "PERSON", 0, 5)
    result = NERResult(
        column="name",
        total_entities=1,
        entity_counts={"PERSON": 1},
        top_entities={"PERSON": [("สมชาย", 1)]},
        sample_entities=[[e]],
        engine_used="thainer",
    )
    d = result.to_dict()
    assert d["column"] == "name"
    assert d["total_entities"] == 1
    assert d["entity_counts"] == {"PERSON": 1}
    assert d["top_entities"]["PERSON"] == [["สมชาย", 1]]
    assert d["sample_entities"] == [[e.to_dict()]]
    assert d["engine_used"] == "thainer"


# --------------------------------------------------------- extract_entities (edge)
def test_extract_entities_empty_series_no_engine_needed():
    # คอลัมน์ว่าง -> ผลเปล่า โดยไม่ต้องโหลด NER engine (ทำงานได้แม้ไม่มี backend)
    s = pd.Series([None, None], dtype="object", name="name")
    result = extract_entities(s)
    assert isinstance(result, NERResult)
    assert result.total_entities == 0
    assert result.entity_counts == {}
    assert result.column == "name"


@requires_no_ner
def test_extract_entities_without_backend_raises():
    # มีข้อความแต่ไม่มี NER backend -> ต้อง fail loudly
    s = pd.Series(["นายสมชายไปกรุงเทพ"], name="text")
    with pytest.raises(ImportError):
        extract_entities(s)


def test_ner_available_returns_false_without_pythainlp(monkeypatch):
    monkeypatch.setitem(sys.modules, "pythainlp", None)
    monkeypatch.setitem(sys.modules, "pythainlp.tag", None)
    # ล้างแคชเพื่อบังคับให้สร้างใหม่
    import thaieda.ner as ner_mod

    monkeypatch.setattr(ner_mod, "_NER_CACHE", {})
    assert ner_available() is False


# --------------------------------------------------------- real extraction (needs backend)
@requires_ner
def test_extract_entities_real():
    s = pd.Series(
        [
            "นายสมชาย เดินทางไปกรุงเทพมหานคร",
            "บริษัท ปตท. จำกัด ตั้งอยู่ที่กรุงเทพ",
        ],
        name="text",
    )
    result = extract_entities(s)
    assert isinstance(result, NERResult)
    assert result.total_entities >= 1
    assert result.engine_used
    # ทุก entity ต้องมีประเภทและข้อความ
    for cell in result.sample_entities:
        for ent in cell:
            assert ent.text
            assert ent.entity_type
