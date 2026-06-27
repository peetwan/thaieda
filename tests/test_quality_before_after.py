"""ทดสอบ quality ก่อน/หลัง clean และ cleaning_report บน EDAResult."""

from __future__ import annotations

import pandas as pd

import thaieda
from thaieda.detect import detect_all
from thaieda.quality import run_quality_checks


def test_quality_score_improves_after_clean_with_zero_width():
    """zero-width ถูกแก้โดย clean → คะแนนหลัง clean สูงกว่าก่อน."""
    df = pd.DataFrame(
        {
            "text": ["ปกติ", "มีอักขระ​ซ่อน", "ปกติ2"],
            "value": [1, 2, 3],
        }
    )
    result = thaieda.run(df, clean=True, make_charts=False, narrative=False)
    comparison = result.quality_comparison
    assert comparison is not None
    assert comparison["score_after"] >= comparison["score_before"]
    before_names = {i.check_name for i in result.quality_issues_before}
    assert (
        "zero_width_chars" in before_names
        or comparison["score_before"] <= comparison["score_after"]
    )


def test_cleaning_report_on_eda_result():
    """EDAResult.cleaning_report ต้องมีเมื่อ clean=True."""
    df = pd.DataFrame({"a": ["  hello  ", "world"], "b": [1, 2]})
    result = thaieda.run(df, clean=True, make_charts=False, narrative=False)
    assert result.cleaning_report is not None
    assert result.cleaning_report.rows_before >= 2


def test_quality_comparison_fixed_checks():
    """fixed_checks ต้องมีรายการที่หายไปหลัง clean."""
    df = pd.DataFrame({"price": ["๑๐๐", "200"], "qty": [1, 2]})
    result = thaieda.run(df, clean=True, make_charts=False, narrative=False)
    assert result.quality_comparison is not None
    assert "score_before" in result.quality_comparison
    assert "score_after" in result.quality_comparison


def test_no_quality_comparison_when_clean_false():
    """clean=False ไม่มี quality_comparison."""
    df = pd.DataFrame({"a": [1, 2]})
    result = thaieda.run(df, clean=False, make_charts=False, narrative=False)
    assert result.quality_comparison is None


def test_duplicate_rows_quality_before_clean():
    """duplicate_rows ต้องถูกตรวจก่อน clean."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    col_types = detect_all(df)
    issues = run_quality_checks(df, col_types)
    dup = [i for i in issues if i.check_name == "duplicate_rows"]
    assert len(dup) == 1
    assert dup[0].count == 2
