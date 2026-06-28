"""Regression tests for data-integrity edge cases found in the 2026-06 bug hunt."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import thaieda
from thaieda.clean import remove_duplicate_rows
from thaieda.compare import compare_reports


def _fast_run(df: pd.DataFrame, **kwargs):
    return thaieda.run(
        df,
        make_charts=False,
        timeseries=False,
        insights_engine=False,
        narrative=False,
        **kwargs,
    )


def test_run_duplicate_column_names_raise_clear_error() -> None:
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])

    with pytest.raises(ValueError, match="duplicate column name"):
        _fast_run(df, clean=False)


def test_dedup_does_not_merge_bool_and_int() -> None:
    df = pd.DataFrame({"mixed": [1, "two", 3.0, None, "สี่", True]})

    out, result = remove_duplicate_rows(df)

    assert result.rows_affected == 0
    assert len(out) == 6
    assert out["mixed"].tolist()[-1] is True


def test_dedup_mixed_native_and_object_key_path() -> None:
    # คอลัมน์ตัวเลข (hash แบบ native) + คอลัมน์ object (tokenize ทีละค่า) ต้องทำงานร่วมกัน
    # ได้ถูกต้อง: แถวที่ทุกคอลัมน์เท่ากันจริงเท่านั้นที่เป็นแถวซ้ำ
    df = pd.DataFrame(
        {
            "num": [1, 1, 1, 2],
            "obj": ["1", 1, "1", "1"],  # str "1" vs int 1 ต้องไม่ถือว่าซ้ำกัน
        }
    )

    out, result = remove_duplicate_rows(df)

    # แถว 0 และ 2 เหมือนกันทุกคอลัมน์ (num=1, obj="1") → ซ้ำ 1 แถว; แถว 1 (obj=int 1) ต่างชนิด
    assert result.rows_affected == 1
    assert len(out) == 3


def test_dedup_preserves_all_nan_rows() -> None:
    df = pd.DataFrame({"a": [np.nan] * 100})

    out, result = remove_duplicate_rows(df)

    assert result.rows_affected == 0
    assert len(out) == 100


def test_run_clean_true_preserves_all_nan_row_count() -> None:
    result = _fast_run(pd.DataFrame({"a": [np.nan] * 25}), clean=True)

    assert result.overview["rows"] == 25
    assert result.overview["rows_removed_by_cleaning"] == 0


def test_duplicate_removal_note_respects_thai_language() -> None:
    result = _fast_run(pd.DataFrame({"value": ["a", "a", "b"]}), clean=True, lang="th")

    assert result.overview["rows_removed_by_cleaning"] == 1
    assert any("ลบแถวซ้ำ" in note for note in result.notes)


def test_inf_values_do_not_emit_runtime_warning_and_are_reported() -> None:
    df = pd.DataFrame({"x": [1.0, np.inf, -np.inf, np.nan, 5.0]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = _fast_run(df, clean=False)

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any(issue.check_name == "infinite_values" for issue in result.quality_issues)


def test_compare_reports_accepts_profile_report_and_eda_result() -> None:
    result_a = _fast_run(pd.DataFrame({"x": [1, 2, 3]}), clean=False)
    result_b = _fast_run(pd.DataFrame({"x": [1, 2, 9]}), clean=False)

    html_from_results = compare_reports(result_a, result_b, lang="en")
    html_from_reports = compare_reports(result_a.report, result_b.report, lang="en")

    assert "ThaiEDA Dataset Comparison" in html_from_results
    assert "ThaiEDA Dataset Comparison" in html_from_reports
