"""ทดสอบ thaieda.report — ProfileReport, profile(), to_html/to_dict/to_json/_repr_html_."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from thaieda.report import (
    ProfileReport,
    _group_insights_by_column,
    _is_row_removing_op,
    _space_thai_latin,
    profile,
)


def test_group_insights_by_column_merges_and_orders_by_severity():
    items = [
        {
            "severity": "info",
            "description_th": "คอลัมน์ 'age': มีค่าผิดปกติเล็กน้อย",
            "title_th": "outlier",
            "category_label": "การกระจาย",
        },
        {
            "severity": "critical",
            "description_th": "คอลัมน์ 'age': ค่าหาย 60%",
            "title_th": "missing",
            "category_label": "ความสมบูรณ์",
        },
        {
            "severity": "warning",
            "description_th": "ภาพรวมชุดข้อมูลมีคอลัมน์ซ้ำ",
            "title_th": "dup",
            "category_label": "โครงสร้าง",
        },
    ]
    cards = _group_insights_by_column(items)
    # การ์ด 'age' รวม 2 ข้อค้นพบ และจัดเรียง critical ก่อน
    age_card = next(c for c in cards if c["column"] == "age")
    assert len(age_card["findings"]) == 2
    assert age_card["severity"] == "critical"
    assert age_card["findings"][0]["severity"] == "critical"
    # คำนำ "คอลัมน์ 'age':" ถูกตัดออกจาก description
    assert not age_card["findings"][0]["description_th"].startswith("คอลัมน์ 'age'")
    # การ์ดวิกฤตอยู่ก่อนการ์ด warning (จัดเรียงตามความรุนแรง)
    assert cards[0]["severity"] == "critical"
    # ข้อค้นพบที่ไม่ผูกคอลัมน์เป็นการ์ดเดี่ยว
    standalone = [c for c in cards if c["column"] == ""]
    assert len(standalone) == 1


def test_is_row_removing_op_distinguishes_rows_from_cells():
    assert _is_row_removing_op("remove_duplicate_rows", "(entire df)") is True
    # handle_missing_values ลบแถวเฉพาะตอน drop ทั้ง DataFrame
    assert _is_row_removing_op("handle_missing_values", "(entire df)") is True
    # รายคอลัมน์ = การเติมค่า (impute) ไม่ใช่การลบแถว
    assert _is_row_removing_op("handle_missing_values", "age") is False
    assert _is_row_removing_op("strip_whitespace", "name") is False


def test_cleaning_summary_separates_rows_removed_from_values_changed():
    df = pd.DataFrame({"name": ["  Alice ", "BOB", "  Alice ", "bob", "Carol"] * 4})
    html = profile(df, clean=True, lang="th").to_html()
    # หน่วยต้องแยกชัด: "ค่าที่แก้ไข" (เซลล์) กับ "แถวที่ถูกลบ" (แถว) — ไม่เรียกรวมว่า "เซลล์"
    assert "ค่าที่แก้ไข" in html
    assert "แถวที่ถูกลบ" in html


def test_space_thai_latin_inserts_space_around_english():
    # คำอังกฤษที่ติดอักษรไทยต้องถูกคั่นด้วยช่องว่างทั้งสองด้าน
    assert _space_thai_latin("สหสัมพันธ์ Pearsonสูง") == "สหสัมพันธ์ Pearson สูง"
    assert _space_thai_latin("ค่าPM25") == "ค่า PM25"


def test_space_thai_latin_leaves_numbers_and_quoted_names():
    # ตัวเลข/เครื่องหมาย และชื่อคอลัมน์ในเครื่องหมายคำพูด ต้องไม่ถูกแตะ
    assert _space_thai_latin("r=0.95, n=1,234") == "r=0.95, n=1,234"
    assert _space_thai_latin("คอลัมน์ 'col_a' ดี") == "คอลัมน์ 'col_a' ดี"


def test_space_thai_latin_spaces_closing_bracket_before_thai():
    # วงเล็บที่ปิดท้ายโทเคนอังกฤษและชนอักษรไทย ต้องถูกคั่นด้วยช่องว่าง
    assert _space_thai_latin("Spearman (non-linear)สูง") == "Spearman (non-linear) สูง"
    assert _space_thai_latin("(non-linear)เป็นลบ") == "(non-linear) เป็นลบ"


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "review": ["อาหารอร่อยมาก", "ร้านนี้ดี​แต่ราคาแพง", "12345", "สวัสดีครับ"],
            "rating": [5, 3, 4, 5],
            "year": [2567, 2024, 2568, 2023],
            "price": ["๑๒๐", "150", "๒๐๐", "300"],
        }
    )


def test_profile_returns_report(sample_df):
    r = profile(sample_df)
    assert isinstance(r, ProfileReport)


def test_profile_runs_analysis(sample_df):
    r = profile(sample_df)
    assert r.column_types  # ตรวจประเภทแล้ว
    assert "rows" in r.overview
    assert r.overview["rows"] == 4
    assert r.overview["columns"] == 4


def test_to_html_returns_sections(sample_df):
    r = profile(sample_df)
    html = r.to_html()
    assert isinstance(html, str)
    assert len(html) > 1000
    # ต้องเป็น HTML self-contained
    assert "<!DOCTYPE html>" in html
    # ส่วนสำคัญต้องอยู่ครบ (ภาษาไทยเป็น default)
    assert "ภาพรวม" in html  # overview
    assert "ปัญหาคุณภาพข้อมูล" in html  # quality issues
    assert "รายละเอียดคอลัมน์" in html  # column details
    # ไม่พึ่งทรัพยากรภายนอก
    assert "<style>" in html


def test_to_html_writes_file(sample_df, tmp_path):
    r = profile(sample_df)
    out = tmp_path / "report.html"
    html = r.to_html(str(out))
    assert out.is_file()
    assert out.read_text(encoding="utf-8") == html


def test_to_dict_structure(sample_df):
    r = profile(sample_df)
    d = r.to_dict()
    assert isinstance(d, dict)
    for key in ("overview", "column_types", "quality_issues", "columns"):
        assert key in d
    assert d["column_types"]["review"] == "thai_text"
    # ปัญหา Buddhist Era ต้องถูกตรวจพบในคอลัมน์ year
    checks = {i["check_name"] for i in d["quality_issues"]}
    assert "buddhist_era" in checks


def test_to_json(sample_df, tmp_path):
    r = profile(sample_df)
    out = tmp_path / "report.json"
    text = r.to_json(str(out))
    parsed = json.loads(text)
    assert parsed["overview"]["rows"] == 4
    assert out.is_file()


def test_repr_html(sample_df):
    r = profile(sample_df)
    html = r._repr_html_()
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html


def test_quality_issues_detected(sample_df):
    r = profile(sample_df)
    checks = {i.check_name for i in r.quality_issues}
    # ตัวอย่างข้อมูลมี: Buddhist Era, Thai numerals, zero-width
    assert "buddhist_era" in checks
    assert "thai_numerals" in checks
    assert "zero_width_chars" in checks


def test_english_report(sample_df):
    r = ProfileReport(sample_df, lang="en")
    html = r.to_html()
    assert "Overview" in html
    assert "Data Quality Issues" in html


def test_lazy_import_from_package():
    # thaieda.profile ต้อง import ได้แบบ lazy
    import thaieda

    assert callable(thaieda.profile)
    assert thaieda.ProfileReport is not None


# ------------------------------------------------------------- insights (v0.3)
def test_report_has_insights(sample_df):
    from thaieda.insight import InsightSummary

    r = profile(sample_df)
    assert isinstance(r.insights, InsightSummary)
    assert r.insights.total_insights >= 1
    assert r.insights.executive_summary_th


def test_insights_in_html(sample_df):
    r = profile(sample_df)
    html = r.to_html()
    assert "ข้อค้นพบสำคัญ" in html  # auto insights heading
    assert "บทสรุปผู้บริหาร" in html  # executive summary label


def test_insights_in_to_dict(sample_df):
    r = profile(sample_df)
    d = r.to_dict()
    assert "insights" in d
    assert d["insights"]["total_insights"] >= 1
    assert "executive_summary_th" in d["insights"]


# ------------------------------------------------------------- cleaning diff (v0.3)
def test_clean_flag_produces_diff(sample_df):
    r = profile(sample_df, clean=True)
    # ต้องมีการทำความสะอาดจริง (zero-width / เลขไทย ในข้อมูลตัวอย่าง)
    assert len(r.cleaning_diff) >= 1
    ops = {c.operation for c in r.cleaning_diff}
    assert "normalize_thai_numerals" in ops or "remove_zero_width_chars" in ops


def test_clean_flag_cleans_dataframe(sample_df):
    r = profile(sample_df, clean=True)
    r.run()
    # เลขไทยในคอลัมน์ price ต้องถูกแปลง (v2.0 pipeline อาจ coerce เป็น numeric)
    assert "๑๒๐" not in r.df["price"].astype(str).tolist()
    assert 120 in r.df["price"].tolist()


def test_clean_quality_comparison_in_dict(sample_df):
    r = profile(sample_df, clean=True)
    assert r.quality_comparison is not None
    assert r.quality_comparison["score_after"] >= 0
    d = r.to_dict()
    assert "quality_comparison" in d
    assert "quality_issues_before" in d


def test_clean_diff_in_html_and_dict(sample_df):
    r = profile(sample_df, clean=True)
    html = r.to_html()
    assert "การทำความสะอาด" in html
    d = r.to_dict()
    assert "cleaning_diff" in d
    assert len(d["cleaning_diff"]) >= 1


def test_no_clean_no_diff(sample_df):
    r = profile(sample_df, clean=False)
    assert r.cleaning_diff == []
    d = r.to_dict()
    assert "cleaning_diff" not in d
