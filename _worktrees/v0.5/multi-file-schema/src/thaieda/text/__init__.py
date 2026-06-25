"""Text metrics — สถิติข้อความสำหรับคอลัมน์ภาษาไทย (ใช้ตัวตัดคำผ่าน adapter)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from thaieda.tokenize import Tokenizer

# ----------------------------------------------------------------------------
# stopword ภาษาไทยพื้นฐาน (ปรับแต่งได้ผ่านพารามิเตอร์)
# ----------------------------------------------------------------------------
DEFAULT_THAI_STOPWORDS: frozenset[str] = frozenset(
    {
        "และ",
        "หรือ",
        "ที่",
        "ของ",
        "ใน",
        "การ",
        "เป็น",
        "ได้",
        "ไม่",
        "มี",
        "จะ",
        "ก็",
        "ให้",
        "นี้",
        "แล้ว",
        "จาก",
        "ทำ",
        "อยู่",
        "คน",
        "เพราะ",
        "ว่า",
        "ขอ",
        "ผม",
        "คุณ",
        "กับ",
        "ความ",
        "ด้วย",
        "แต่",
        "ต้อง",
        "เรา",
        "มา",
        "ไป",
        "อย่าง",
        "นั้น",
        "ๆ",
        "โดย",
        "ๆๆ",
        "เมื่อ",
        "ถ้า",
        "ซึ่ง",
    }
)


@dataclass
class TextMetrics:
    """สถิติข้อความของคอลัมน์หนึ่ง."""

    total_cells: int
    non_null_cells: int
    sampled_cells: int
    avg_char_length: float
    avg_token_length: float
    avg_word_length: float
    median_char_length: float
    min_char_length: int
    max_char_length: int
    total_tokens: int
    unique_tokens: int
    top_tokens: list[tuple[str, int]] = field(default_factory=list)
    top_bigrams: list[tuple[str, int]] = field(default_factory=list)
    top_trigrams: list[tuple[str, int]] = field(default_factory=list)
    engine_used: str = ""

    def to_dict(self) -> dict:
        return {
            "total_cells": self.total_cells,
            "non_null_cells": self.non_null_cells,
            "sampled_cells": self.sampled_cells,
            "avg_char_length": round(self.avg_char_length, 2),
            "avg_token_length": round(self.avg_token_length, 2),
            "avg_word_length": round(self.avg_word_length, 2),
            "median_char_length": round(self.median_char_length, 2),
            "min_char_length": self.min_char_length,
            "max_char_length": self.max_char_length,
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_tokens,
            "top_tokens": self.top_tokens,
            "top_bigrams": self.top_bigrams,
            "top_trigrams": self.top_trigrams,
            "engine_used": self.engine_used,
        }


def _is_meaningful(token: str, stopwords: frozenset[str]) -> bool:
    """True ถ้า token ควรนับ (ไม่ใช่ stopword, ไม่ใช่ช่องว่าง, มีความยาว)."""
    t = token.strip()
    if not t:
        return False
    if t in stopwords:
        return False
    # ตัด token ที่เป็นเครื่องหมายวรรคตอนล้วน
    return any(c.isalnum() for c in t)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """สร้าง n-gram จากรายการ token."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def text_metrics(
    series: pd.Series,
    tokenizer: Tokenizer,
    max_sample: int = 5000,
    stopwords: frozenset[str] | None = None,
    top_n: int = 20,
    random_state: int = 42,
) -> TextMetrics:
    """คำนวณสถิติข้อความของคอลัมน์ โดยใช้ tokenizer ที่ให้มา.

    คอลัมน์ที่ใหญ่กว่า max_sample จะถูกสุ่มตัวอย่าง และรายงาน sampled_cells ตามจริง
    n-gram คำนวณจาก token (ไม่ใช่การตัดอักขระ) และ top_tokens ตัด stopword ออก
    """
    if stopwords is None:
        stopwords = DEFAULT_THAI_STOPWORDS

    total_cells = len(series)
    non_null = series.dropna().astype(str)
    non_null_cells = len(non_null)

    if non_null_cells == 0:
        return TextMetrics(
            total_cells=total_cells,
            non_null_cells=0,
            sampled_cells=0,
            avg_char_length=0.0,
            avg_token_length=0.0,
            avg_word_length=0.0,
            median_char_length=0.0,
            min_char_length=0,
            max_char_length=0,
            total_tokens=0,
            unique_tokens=0,
            engine_used=getattr(tokenizer, "name", "unknown"),
        )

    # สุ่มตัวอย่างถ้าคอลัมน์ใหญ่เกิน max_sample
    if non_null_cells > max_sample:
        sample = non_null.sample(n=max_sample, random_state=random_state)
    else:
        sample = non_null
    sampled_cells = len(sample)

    char_lengths: list[int] = []
    token_counts_per_cell: list[int] = []
    token_counter: Counter[str] = Counter()
    all_token_count = 0
    unique_tokens_set: set[str] = set()
    bigram_counter: Counter[tuple[str, ...]] = Counter()
    trigram_counter: Counter[tuple[str, ...]] = Counter()

    for value in sample:
        char_lengths.append(len(value))
        tokens = tokenizer.tokenize(value)
        # token ที่ไม่ใช่ช่องว่างล้วน
        clean_tokens = [t for t in tokens if t.strip()]
        token_counts_per_cell.append(len(clean_tokens))
        all_token_count += len(clean_tokens)
        unique_tokens_set.update(clean_tokens)

        # top tokens — ตัด stopword
        for t in clean_tokens:
            if _is_meaningful(t, stopwords):
                token_counter[t] += 1

        # n-grams — จาก token ที่มีความหมาย (ตัด stopword) เพื่อ phrase ที่สื่อความ
        meaningful = [t for t in clean_tokens if _is_meaningful(t, stopwords)]
        for bg in _ngrams(meaningful, 2):
            bigram_counter[bg] += 1
        for tg in _ngrams(meaningful, 3):
            trigram_counter[tg] += 1

    char_series = pd.Series(char_lengths)
    avg_tokens = (sum(token_counts_per_cell) / sampled_cells) if sampled_cells else 0.0

    return TextMetrics(
        total_cells=total_cells,
        non_null_cells=non_null_cells,
        sampled_cells=sampled_cells,
        avg_char_length=float(char_series.mean()),
        avg_token_length=avg_tokens,
        avg_word_length=avg_tokens,  # สำหรับไทย คำ ≈ token
        median_char_length=float(char_series.median()),
        min_char_length=int(char_series.min()),
        max_char_length=int(char_series.max()),
        total_tokens=all_token_count,
        unique_tokens=len(unique_tokens_set),
        top_tokens=token_counter.most_common(top_n),
        top_bigrams=[(" ".join(bg), c) for bg, c in bigram_counter.most_common(top_n)],
        top_trigrams=[(" ".join(tg), c) for tg, c in trigram_counter.most_common(top_n)],
        engine_used=getattr(tokenizer, "name", "unknown"),
    )


def tfidf_top_terms(
    series: pd.Series,
    tokenizer: Tokenizer,
    target_series: pd.Series | None = None,
    top_n: int = 20,
    stopwords: frozenset[str] | None = None,
) -> list[tuple[str, float]] | dict[str, list[tuple[str, float]]]:
    """คำนวณ TF-IDF top terms.

    ถ้าไม่ระบุ target_series: ถือว่าแต่ละเซลล์เป็นหนึ่งเอกสาร คืน term เด่นโดยรวม.
    ถ้าระบุ target_series: จัดกลุ่มข้อความตามคลาสของ target แล้วคืน top terms ต่อคลาส
    (เปรียบเทียบว่าคำไหนเด่นในคลาสไหน) — คืนเป็น dict {class_value: [(term, score), ...]}.
    """
    if stopwords is None:
        stopwords = DEFAULT_THAI_STOPWORDS

    def tokenize_clean(text: str) -> list[str]:
        return [t for t in tokenizer.tokenize(text) if _is_meaningful(t, stopwords)]

    if target_series is None:
        # แต่ละเซลล์เป็นหนึ่งเอกสาร
        docs = [tokenize_clean(str(v)) for v in series.dropna()]
        return _tfidf_over_docs(docs, top_n)

    # จัดกลุ่มตาม target: แต่ละคลาสเป็นหนึ่งเอกสารใหญ่
    df = pd.DataFrame({"text": series, "target": target_series}).dropna()
    classes = sorted(df["target"].unique(), key=str)
    class_docs: list[list[str]] = []
    for cls in classes:
        toks: list[str] = []
        for v in df.loc[df["target"] == cls, "text"]:
            toks.extend(tokenize_clean(str(v)))
        class_docs.append(toks)

    n_docs = len(class_docs)
    # document frequency ของแต่ละ term ในระดับคลาส
    df_counts: Counter[str] = Counter()
    for doc in class_docs:
        for term in set(doc):
            df_counts[term] += 1

    result: dict[str, list[tuple[str, float]]] = {}
    for cls, doc in zip(classes, class_docs, strict=True):
        tf = Counter(doc)
        total = sum(tf.values()) or 1
        scores: list[tuple[str, float]] = []
        for term, count in tf.items():
            idf = math.log((1 + n_docs) / (1 + df_counts[term])) + 1.0
            scores.append((term, (count / total) * idf))
        scores.sort(key=lambda x: x[1], reverse=True)
        result[str(cls)] = scores[:top_n]
    return result


def _tfidf_over_docs(docs: list[list[str]], top_n: int) -> list[tuple[str, float]]:
    """TF-IDF รวมทุกเอกสาร — คืน term ที่มีคะแนน TF-IDF รวมสูงสุด."""
    n_docs = len(docs)
    if n_docs == 0:
        return []
    df_counts: Counter[str] = Counter()
    for doc in docs:
        for term in set(doc):
            df_counts[term] += 1

    agg: dict[str, float] = {}
    for doc in docs:
        tf = Counter(doc)
        total = sum(tf.values()) or 1
        for term, count in tf.items():
            idf = math.log((1 + n_docs) / (1 + df_counts[term])) + 1.0
            agg[term] = agg.get(term, 0.0) + (count / total) * idf

    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


__all__ = [
    "TextMetrics",
    "DEFAULT_THAI_STOPWORDS",
    "text_metrics",
    "tfidf_top_terms",
]
