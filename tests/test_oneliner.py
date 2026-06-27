"""ทดสอบ one-liner API — thaieda.run() / thaieda.EDA() / EDAResult (v0.9).

ทดสอบ:
  1. run() คืน EDAResult ที่มี report ครบ
  2. EDAResult properties (overview, insights, quality_issues, cleaned_df)
  3. clean=True ทำความสะอาดข้อมูลจริง
  4. make_charts=False ข้ามกราฟ
  5. run() กับ target_column
  6. to_html / to_dict / to_json ของ EDAResult
  7. EDA() เป็น alias ของ run()
  8. run(llm=True) — mock LLM
  9. run(llm=True) ส่ง insights ให้ LLM
  10. TypeError เมื่อ df ไม่ใช่ DataFrame
  11. backward compat — API เดิมยังใช้ได้
  12. lang="en" สร้างรายงานภาษาอังกฤษ
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

import thaieda


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """DataFrame ตัวอย่างที่มีข้อความไทย + ตัวเลข + ปัญหาที่ตรวจได้."""
    return pd.DataFrame(
        {
            "review": ["อาหารอร่อยมาก", "ร้านนี้ดี​แต่ราคาแพง", "12345", "สวัสดีครับ"],
            "rating": [5, 3, 4, 5],
            "year": [2567, 2024, 2568, 2023],
            "price": ["๑๒๐", "150", "๒๐๐", "300"],
        }
    )


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    """DataFrame เล็กสำหรับทดสอบเร็ว ๆ."""
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "C"],
            "value": [10, 20, 15, 30],
        }
    )


# ----------------------------------------------------------------------------
# 1. run() คืน EDAResult ที่มี report ครบ
# ----------------------------------------------------------------------------
class TestRunReturnsResult:
    """ทดสอบว่า run() คืน EDAResult ที่ถูกต้อง."""

    def test_run_returns_eda_result(self, simple_df):
        """run() ต้องคืน EDAResult."""
        result = thaieda.run(simple_df)
        assert isinstance(result, thaieda.EDAResult)

    def test_run_has_report(self, simple_df):
        """EDAResult.report ต้องเป็น ProfileReport."""
        from thaieda.report import ProfileReport

        result = thaieda.run(simple_df)
        assert isinstance(result.report, ProfileReport)

    def test_run_report_already_executed(self, simple_df):
        """report ที่ได้ต้องรันการวิเคราะห์แล้ว (ไม่ต้องเรียก .run() อีก)."""
        result = thaieda.run(simple_df)
        # เข้าถึง property ได้โดยไม่ error = รันแล้ว
        assert result.report.column_types
        assert "rows" in result.report.overview

    def test_run_overview_rows(self, simple_df):
        """overview ต้องบอกจำนวนแถวที่ถูกต้อง."""
        result = thaieda.run(simple_df)
        assert result.overview["rows"] == 4
        assert result.overview["columns"] == 2


# ----------------------------------------------------------------------------
# 2. EDAResult properties
# ----------------------------------------------------------------------------
class TestEDAResultProperties:
    """ทดสอบ property ของ EDAResult."""

    def test_cleaned_df_returns_dataframe(self, simple_df):
        """cleaned_df ต้องเป็น DataFrame."""
        result = thaieda.run(simple_df)
        assert isinstance(result.cleaned_df, pd.DataFrame)

    def test_insights_property(self, sample_df):
        """insights ต้องเป็น InsightSummary หรือ None."""
        from thaieda.insight import InsightSummary

        result = thaieda.run(sample_df)
        assert isinstance(result.insights, InsightSummary)
        assert result.insights.total_insights >= 1

    def test_quality_issues_property(self, sample_df):
        """quality_issues ต้องเป็น list ของ QualityIssue."""
        result = thaieda.run(sample_df)
        issues = result.quality_issues
        assert isinstance(issues, list)
        assert len(issues) > 0  # ข้อมูลตัวอย่างมีปัญหา (Buddhist Era, เลขไทย, ฯลฯ)

    def test_anomalies_property(self, sample_df):
        """anomalies ต้องเป็น list."""
        result = thaieda.run(sample_df)
        assert isinstance(result.anomalies, list)

    def test_notes_collected(self, sample_df):
        """notes ต้องเป็น list (อาจว่าง)."""
        result = thaieda.run(sample_df, make_charts=False)
        assert isinstance(result.notes, list)


# ----------------------------------------------------------------------------
# 3. clean=True ทำความสะอาดข้อมูลจริง
# ----------------------------------------------------------------------------
class TestCleanOption:
    """ทดสอบการทำความสะอาด."""

    def test_clean_true_cleans_dataframe(self, sample_df):
        """clean=True ต้องทำความสะอาดข้อมูล (เลขไทย → อารบิก + coerce numeric)."""
        result = thaieda.run(sample_df, clean=True)
        # เลขไทยในคอลัมน์ price ต้องถูกแปลง (v2.0 pipeline อาจ coerce เป็น numeric)
        assert "๑๒๐" not in result.cleaned_df["price"].astype(str).tolist()
        assert 120 in result.cleaned_df["price"].tolist()

    def test_clean_false_preserves_data(self, sample_df):
        """clean=False ต้องไม่แก้ข้อมูล."""
        result = thaieda.run(sample_df, clean=False)
        # เลขไทยยังอยู่
        assert "๑๒๐" in list(result.cleaned_df["price"])

    def test_clean_has_cleaning_diff(self, sample_df):
        """clean=True ต้องมี cleaning_diff."""
        result = thaieda.run(sample_df, clean=True)
        assert len(result.report.cleaning_diff) >= 1

    def test_clean_has_cleaning_report(self, sample_df):
        """clean=True ต้องมี cleaning_report จาก clean() v2.0."""
        result = thaieda.run(sample_df, clean=True, make_charts=False, narrative=False)
        assert result.cleaning_report is not None
        assert result.report.cleaning_plan is not None

    def test_clean_has_quality_comparison(self, sample_df):
        """clean=True ต้องมี quality_comparison ก่อน/หลัง."""
        result = thaieda.run(sample_df, clean=True, make_charts=False, narrative=False)
        assert result.quality_comparison is not None
        assert result.quality_score is not None
        assert "score" in result.quality_score

    def test_clean_duplicate_rows_are_visible_in_overview(self):
        df = pd.DataFrame({"value": ["a", "a", "b"]})
        result = thaieda.run(
            df,
            clean=True,
            make_charts=False,
            timeseries=False,
            insights_engine=False,
        )
        assert result.overview["rows"] == 2
        assert result.overview["rows_before_cleaning"] == 3
        assert result.overview["rows_after_cleaning"] == 2
        assert result.overview["rows_removed_by_cleaning"] == 1
        assert any("duplicate rows" in note for note in result.notes)


# ----------------------------------------------------------------------------
# 4. make_charts=False ข้ามกราฟ
# ----------------------------------------------------------------------------
class TestNoCharts:
    """ทดสอบการข้ามกราฟ."""

    def test_no_charts_faster(self, simple_df):
        """make_charts=False ต้องไม่สร้างกราฟ — รันได้โดยไม่ error."""
        result = thaieda.run(simple_df, make_charts=False)
        assert isinstance(result, thaieda.EDAResult)


# ----------------------------------------------------------------------------
# 5. run() กับ target_column
# ----------------------------------------------------------------------------
class TestTargetColumn:
    """ทดสอบการวิเคราะห์ตัวแปรเป้าหมาย."""

    def test_target_column_runs(self, sample_df):
        """ระบุ target_column ต้องวิเคราะห์ความสัมพันธ์ได้."""
        result = thaieda.run(sample_df, target_column="rating")
        assert isinstance(result, thaieda.EDAResult)
        # target_associations ต้องถูกคำนวณ
        assert isinstance(result.report.target_associations, list)

    def test_invalid_target_raises(self, simple_df):
        """target_column ไม่มีใน DataFrame → KeyError."""
        with pytest.raises(KeyError, match="not found"):
            thaieda.run(simple_df, target_column="nonexistent")


# ----------------------------------------------------------------------------
# 6. to_html / to_dict / to_json ของ EDAResult
# ----------------------------------------------------------------------------
class TestExport:
    """ทดสอบการส่งออกผลลัพธ์ผ่าน EDAResult."""

    def test_to_html(self, simple_df, tmp_path):
        """EDAResult.to_html() ต้องสร้าง HTML ได้."""
        result = thaieda.run(simple_df, make_charts=False)
        html = result.to_html()
        assert "<!DOCTYPE html>" in html

    def test_to_html_writes_file(self, simple_df, tmp_path):
        """EDAResult.to_html(path) ต้องเขียนไฟล์ได้."""
        result = thaieda.run(simple_df, make_charts=False)
        out = tmp_path / "report.html"
        result.to_html(str(out))
        assert out.is_file()

    def test_to_dict(self, simple_df):
        """EDAResult.to_dict() ต้องส่งออก dict ที่มีโครงสร้างครบ."""
        result = thaieda.run(simple_df, make_charts=False)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "overview" in d
        assert "column_types" in d

    def test_to_json(self, simple_df, tmp_path):
        """EDAResult.to_json(path) ต้องเขียน JSON ได้."""
        result = thaieda.run(simple_df, make_charts=False)
        out = tmp_path / "report.json"
        text = result.to_json(str(out))
        parsed = json.loads(text)
        assert parsed["overview"]["rows"] == 4
        assert out.is_file()


# ----------------------------------------------------------------------------
# 7. EDA() เป็น alias ของ run()
# ----------------------------------------------------------------------------
class TestEDAAlias:
    """ทดสอบว่า EDA() เป็น alias ของ run()."""

    def test_eda_is_run(self):
        """EDA ต้องเป็นฟังก์ชันเดียวกับ run."""
        assert thaieda.EDA is thaieda.run

    def test_eda_returns_eda_result(self, simple_df):
        """EDA() ต้องคืน EDAResult."""
        result = thaieda.EDA(simple_df, make_charts=False)
        assert isinstance(result, thaieda.EDAResult)


# ----------------------------------------------------------------------------
# 8. run(llm=True) — mock LLM
# ----------------------------------------------------------------------------
class TestLLMOption:
    """ทดสอบการเรียก LLM ผ่าน run(llm=True)."""

    @patch("thaieda.llm._provider._call_openai")
    def test_llm_returns_response(self, mock_openai, simple_df):
        """llm=True ต้องเรียก LLM และเก็บผลใน llm_response."""
        mock_openai.return_value = "วิเคราะห์เสร็จแล้ว"
        result = thaieda.run(
            simple_df, make_charts=False, llm=True, provider="openai", model="gpt-4o-mini"
        )
        assert result.llm_response == "วิเคราะห์เสร็จแล้ว"
        mock_openai.assert_called_once()

    def test_llm_false_no_response(self, simple_df):
        """llm=False (default) ต้องไม่เรียก LLM."""
        result = thaieda.run(simple_df, make_charts=False)
        assert result.llm_response is None

    @patch("thaieda.llm._provider._call_openai")
    def test_llm_passes_insights(self, mock_openai, sample_df):
        """llm=True ต้องส่ง insights ให้ LLM."""
        mock_openai.return_value = "ok"
        thaieda.run(sample_df, make_charts=False, llm=True, provider="openai")
        mock_openai.assert_called_once()
        # prompt ที่ส่งให้ LLM ต้องมีข้อค้นพบ
        prompt = mock_openai.call_args[0][0]  # first positional arg
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ----------------------------------------------------------------------------
# 9. TypeError เมื่อ df ไม่ใช่ DataFrame
# ----------------------------------------------------------------------------
class TestInputValidation:
    """ทดสอบการตรวจสอบ input."""

    def test_non_dataframe_raises(self):
        """ส่ง list แทน DataFrame → TypeError."""
        with pytest.raises(TypeError, match="DataFrame"):
            thaieda.run([1, 2, 3])

    def test_none_raises(self):
        """ส่ง None → TypeError."""
        with pytest.raises(TypeError, match="DataFrame"):
            thaieda.run(None)


# ----------------------------------------------------------------------------
# 10. Backward compatibility — API เดิมยังใช้ได้
# ----------------------------------------------------------------------------
class TestBackwardCompat:
    """ทดสอบว่า API เดิมยังใช้ได้หลังเพิ่ม run()."""

    def test_profile_still_callable(self, sample_df):
        """thaieda.profile ยังใช้ได้."""
        r = thaieda.profile(sample_df)
        assert r is not None

    def test_profile_report_still_importable(self, sample_df):
        """thaieda.ProfileReport ยัง import ได้."""
        from thaieda.report import ProfileReport

        r = ProfileReport(sample_df)
        r.run()
        assert r.overview["rows"] == 4

    def test_discover_insights_still_callable(self, sample_df):
        """thaieda.discover_insights ยังใช้ได้."""
        from thaieda.detect import detect_all

        result = thaieda.discover_insights(sample_df, detect_all(sample_df))
        assert result is not None

    def test_all_old_exports_present(self):
        """__all__ ต้องมี export เดิมครบ."""
        old_exports = [
            "profile",
            "ProfileReport",
            "extract_entities",
            "analyze_target",
            "generate_insights",
            "Insight",
            "InsightSummary",
            "discover_insights",
            "InsightCard",
            "InsightEngineResult",
            "Perspective",
            "analyze_timeseries",
            "analyze_dataframe_timeseries",
            "detect_timeseries_columns",
            "TimeseriesResult",
            "TimeseriesComponent",
            "read_data",
            "detect_encoding",
            "detect_format",
            "profile_dataset",
            "DatasetProfile",
            "Relationship",
            "KeyCandidate",
            "TableProfile",
            "DatasetReport",
            "__version__",
        ]
        for name in old_exports:
            assert name in thaieda.__all__, f"export เดิมหายไป: {name}"


# ----------------------------------------------------------------------------
# 11. lang="en" สร้างรายงานภาษาอังกฤษ
# ----------------------------------------------------------------------------
class TestLanguageOption:
    """ทดสอบการเลือกภาษาของรายงาน."""

    def test_english_report(self, sample_df):
        """lang='en' ต้องสร้างรายงานภาษาอังกฤษ."""
        result = thaieda.run(sample_df, lang="en", make_charts=False)
        html = result.to_html()
        assert "Overview" in html
        assert "Data Quality Issues" in html

    def test_thai_report_default(self, sample_df):
        """lang='th' (default) ต้องสร้างรายงานภาษาไทย."""
        result = thaieda.run(sample_df, make_charts=False)
        html = result.to_html()
        assert "ภาพรวม" in html


# ----------------------------------------------------------------------------
# 12. clean + downcast บนคอลัมน์วันที่แบบ category (regression)
# ----------------------------------------------------------------------------
class TestCategoricalDatetimeDowncast:
    """หลัง downcast คอลัมน์ date อาจเป็น category — one-liner ต้องไม่ crash."""

    def test_clean_with_downcast_categorical_date(self):
        """หลัง downcast คอลัมน์ date อาจเป็น category — one-liner ต้องไม่ crash."""
        df = pd.DataFrame(
            {
                "date": (["2024-01-01", "2024-02-01", "2023-09-12"] * 34)[:100],
                "city": (["กรุงเทพ", "เชียงใหม่", "ภูเก็ต"] * 34)[:100],
                "amount": list(range(100)),
            }
        )
        result = thaieda.run(
            df,
            clean=True,
            downcast=True,
            make_charts=False,
            narrative=False,
            timeseries=False,
            insights_engine=False,
        )
        assert result.overview["rows"] == 100
        assert result.quality_comparison is not None
