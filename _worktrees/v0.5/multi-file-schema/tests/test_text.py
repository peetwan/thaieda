"""ทดสอบ thaieda.text — text_metrics, stopwords, n-grams, sampling."""

from __future__ import annotations

import pandas as pd
import pytest

from thaieda.text import DEFAULT_THAI_STOPWORDS, TextMetrics, text_metrics
from thaieda.tokenize import available_engines, get_tokenizer

_HAS_ENGINE = len(available_engines()) > 0
pytestmark = pytest.mark.skipif(not _HAS_ENGINE, reason="ไม่มี Thai tokenizer engine ติดตั้ง")


@pytest.fixture
def tokenizer():
    return get_tokenizer("auto")


def test_text_metrics_basic(tokenizer):
    s = pd.Series(["อาหารอร่อยมาก", "บริการดีมาก", "ราคาแพงไป"])
    m = text_metrics(s, tokenizer)
    assert isinstance(m, TextMetrics)
    assert m.total_cells == 3
    assert m.non_null_cells == 3
    assert m.sampled_cells == 3
    assert m.avg_char_length > 0
    assert m.total_tokens > 0
    assert m.engine_used.startswith("pythainlp") or m.engine_used in ("nlpo3", "attacut")


def test_text_metrics_excludes_stopwords(tokenizer):
    # "และ", "ที่", "ของ" เป็น stopword — ไม่ควรอยู่ใน top_tokens
    s = pd.Series(
        [
            "แมว และ หมา ที่ บ้าน ของ ฉัน",
            "แมว และ หมา ที่ สวน ของ เธอ",
            "แมว และ หมา ที่ ตลาด ของ เขา",
        ]
    )
    m = text_metrics(s, tokenizer)
    top_words = [w for w, _ in m.top_tokens]
    for sw in ("และ", "ที่", "ของ"):
        assert sw not in top_words
    # คำที่มีความหมายอย่าง "แมว" ควรติด top
    assert "แมว" in top_words


def test_text_metrics_ngrams(tokenizer):
    s = pd.Series(["เครื่องซักผ้า ราคาดี", "เครื่องซักผ้า ราคาดี", "เครื่องซักผ้า ราคาดี"])
    m = text_metrics(s, tokenizer)
    # bigram ต้องเป็นคู่คำ (มีช่องว่างคั่นจากการ join token)
    assert len(m.top_bigrams) >= 1
    top_bigram, count = m.top_bigrams[0]
    assert " " in top_bigram
    assert count == 3


def test_text_metrics_custom_stopwords(tokenizer):
    s = pd.Series(["กาแฟ อร่อย", "กาแฟ ดี", "กาแฟ ร้อน"])
    # กำหนดให้ "กาแฟ" เป็น stopword เอง
    m = text_metrics(s, tokenizer, stopwords=frozenset({"กาแฟ"}))
    top_words = [w for w, _ in m.top_tokens]
    assert "กาแฟ" not in top_words


def test_text_metrics_sampling(tokenizer):
    # ซีรีส์ใหญ่กว่า max_sample -> ต้องสุ่มและรายงานตามจริง
    big = pd.Series(["ข้อความทดสอบ"] * 100)
    m = text_metrics(big, tokenizer, max_sample=10)
    assert m.non_null_cells == 100
    assert m.sampled_cells == 10
    assert m.sampled_cells < m.non_null_cells


def test_text_metrics_empty_column(tokenizer):
    s = pd.Series([None, None], dtype="object")
    m = text_metrics(s, tokenizer)
    assert m.non_null_cells == 0
    assert m.total_tokens == 0
    assert m.top_tokens == []


def test_default_stopwords_nonempty():
    assert len(DEFAULT_THAI_STOPWORDS) > 10
    assert "และ" in DEFAULT_THAI_STOPWORDS
