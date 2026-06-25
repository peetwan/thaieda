"""ทดสอบ language detection และการปรับ pipeline ตามภาษา."""

from __future__ import annotations

import pandas as pd

from thaieda.detect import ColumnType, _detect_language
from thaieda.quality import run_quality_checks
from thaieda.report import _detect_data_type, profile


def test_detect_language_thai():
    df = pd.DataFrame({"name": ["กาแฟ", "ชาไทย", "ขนม"], "price": [10, 20, 30]})
    info = _detect_language(df)
    assert info["language"] == "thai"
    assert info["columns"]["name"] == "thai"
    assert info["thai_ratio"] > 0.30


def test_detect_language_english():
    df = pd.DataFrame({"name": ["coffee", "tea", "cake"], "price": [10, 20, 30]})
    info = _detect_language(df)
    assert info["language"] == "english"
    assert info["columns"]["name"] == "english"


def test_detect_language_mixed():
    df = pd.DataFrame({"name": ["กาแฟ Coffee", "ชา Tea", "Cake เค้ก"]})
    info = _detect_language(df)
    assert info["language"] == "mixed"
    assert info["columns"]["name"] == "mixed"


def test_detect_language_numeric():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    info = _detect_language(df)
    assert info["language"] == "numeric"
    assert set(info["columns"].values()) == {"numeric"}


def test_detect_language_returns_confidence_and_column_details():
    df = pd.DataFrame({"name": ["กาแฟ", "ชาไทย", "ขนม"], "price": [10, 20, 30]})
    info = _detect_language(df)
    assert 0.0 <= info["confidence"] <= 1.0
    assert info["sample_rows"] == 3
    assert info["column_details"]["name"]["language"] == "thai"
    assert 0.0 <= info["column_details"]["name"]["confidence"] <= 1.0
    assert info["column_details"]["name"]["thai_vowel_tone_chars"] > 0


def test_detect_language_column_level_when_thai_text_is_small_but_important():
    df = pd.DataFrame(
        {
            "product_name": ["กาแฟ", "ชาไทย", "ขนม"],
            "sku": ["SKU001", "SKU002", "SKU003"],
            "price": [60, 45, 30],
            "updated_at": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    info = _detect_language(df)
    assert info["language"] == "thai"
    assert info["columns"]["product_name"] == "thai"
    assert info["columns"]["sku"] == "numeric"


def test_detect_language_mixed_cell_with_common_thai_words_and_rating_text():
    df = pd.DataFrame({"review": ["อร่อยมาก 5/5 stars", "ดีครับ 4 stars", "ไม่ดี 1/5 stars"]})
    info = _detect_language(df)
    assert info["language"] == "mixed"
    assert info["columns"]["review"] == "mixed"
    assert info["column_details"]["review"]["common_thai_word_hits"] >= 3


def test_detect_language_handles_zero_width_spaces():
    df = pd.DataFrame({"name": ["สม\u200bชาย", "ไทย\u200bดี", "กาแฟ"]})
    info = _detect_language(df)
    assert info["language"] == "thai"
    assert info["columns"]["name"] == "thai"
    assert info["column_details"]["name"]["zero_width_chars"] == 2


def test_detect_language_samples_first_500_rows():
    df = pd.DataFrame({"text": ["hello"] * 500 + ["ภาษาไทย"] * 500})
    info = _detect_language(df)
    assert info["sample_rows"] == 500
    assert info["language"] == "english"


def test_english_quality_skips_thai_specific_checks():
    df = pd.DataFrame({"name": ["ana", "bob", "cid"], "year": [2567, 2568, 2569]})
    issues = run_quality_checks(
        df,
        {"name": ColumnType.CATEGORICAL, "year": ColumnType.NUMERIC},
    )
    checks = {i.check_name for i in issues}
    assert "buddhist_era" not in checks
    assert "thai_numerals" not in checks


def test_thai_quality_runs_thai_specific_checks():
    df = pd.DataFrame(
        {"name": ["ก", "ข", "ค"], "year": [2567, 2568, 2569], "price": ["๑", "๒", "๓"]}
    )
    issues = run_quality_checks(
        df,
        {
            "name": ColumnType.THAI_TEXT,
            "year": ColumnType.NUMERIC,
            "price": ColumnType.CATEGORICAL,
        },
    )
    checks = {i.check_name for i in issues}
    assert "buddhist_era" in checks
    assert "thai_numerals" in checks


def test_report_data_type_contains_language_info():
    info = _detect_data_type(pd.DataFrame({"name": ["coffee", "tea"], "price": [10, 20]}))
    assert info["language"]["language"] == "english"
    assert info["show_thai_specific"] is False
    assert "ข้าม Thai-specific" in info["language_impact"]


def test_report_html_shows_detected_language_and_hides_thai_section_for_english():
    html = profile(
        pd.DataFrame({"name": ["coffee", "tea"], "price": [10, 20]}),
        make_charts=False,
    ).to_html()
    assert "ภาษาข้อมูลที่ตรวจพบ" in html
    assert "อังกฤษ" in html
    assert "คำแนะนำเฉพาะข้อมูลไทย" not in html
