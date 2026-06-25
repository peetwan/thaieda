"""ทดสอบ thaieda.report — ProfileReport, profile(), to_html/to_dict/to_json/_repr_html_."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from thaieda.report import ProfileReport, profile


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
    # เลขไทยในคอลัมน์ price ต้องถูกแปลงเป็นเลขอารบิกแล้ว
    assert "๑๒๐" not in list(r.df["price"])
    assert "120" in list(r.df["price"])


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
