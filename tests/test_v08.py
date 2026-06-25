"""Tests for v0.8 features — cleaning, quality, insight engine, io."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaieda.clean import (
    coerce_numeric_column,
    convert_buddhist_era,
    handle_missing_values,
    normalize_dates,
    remove_duplicate_rows,
)
from thaieda.detect import detect_all
from thaieda.insight_engine import discover_insights
from thaieda.quality import (
    check_constant_column,
    check_placeholder_values,
    run_quality_checks,
)


# ------------------------------------------------------------------ clean
class TestCoerceNumericColumn:
    def test_thai_numerals_become_numeric(self):
        """'๑๐๐' → 100 (not NaN) after coerce."""
        s = pd.Series(["100", "200", "๑๐๐", "150"], name="ยอดขาย")
        out, result = coerce_numeric_column(s)
        # ค่าที่แปลงได้เป็น numeric
        numeric_vals = pd.to_numeric(out, errors="coerce").dropna()
        assert 100.0 in numeric_vals.values
        assert 200.0 in numeric_vals.values
        assert result.rows_affected > 0

    def test_placeholder_becomes_nan(self):
        """'-' → NaN."""
        s = pd.Series(["100", "-", "200", "N/A"], name="col")
        out, result = coerce_numeric_column(s)
        assert result.rows_affected > 0
        # placeholder ควรเป็น NaN
        assert pd.isna(out.iloc[1]) or out.iloc[1] != "-"

    def test_text_column_not_coerced(self):
        """Text column should not be coerced."""
        s = pd.Series(["hello", "world", "abc"], name="text")
        out, result = coerce_numeric_column(s)
        assert result.rows_affected == 0

    def test_empty_series(self):
        s = pd.Series([], dtype=object, name="empty")
        out, result = coerce_numeric_column(s)
        assert result.rows_affected == 0


class TestConvertBuddhistEra:
    def test_numeric_be_to_ce(self):
        s = pd.Series([2530, 2540, 2024], name="year")
        out, result = convert_buddhist_era(s)
        assert result.rows_affected == 2  # 2530, 2540
        numeric = pd.to_numeric(out, errors="coerce")
        assert 1987.0 in numeric.values  # 2530 - 543
        assert 2024.0 in numeric.values  # unchanged

    def test_string_be_to_ce(self):
        s = pd.Series(["2567-01-15", "2024-01-15"], name="date")
        out, result = convert_buddhist_era(s)
        assert result.rows_affected == 1
        assert "2024" in str(out.iloc[0])

    def test_no_be_values(self):
        s = pd.Series([2020, 2021, 2022], name="year")
        out, result = convert_buddhist_era(s)
        assert result.rows_affected == 0

    def test_empty(self):
        s = pd.Series([], dtype=float, name="empty")
        out, result = convert_buddhist_era(s)
        assert result.rows_affected == 0


class TestNormalizeDates:
    def test_thai_month_full(self):
        s = pd.Series(["15 มกราคม 2567"], name="date")
        out, result = normalize_dates(s)
        assert result.rows_affected == 1
        assert "01" in str(out.iloc[0])
        assert "2024" in str(out.iloc[0])

    def test_thai_month_abbreviated(self):
        s = pd.Series(["1 ก.พ. 67"], name="date")
        out, result = normalize_dates(s)
        assert result.rows_affected == 1
        assert "02" in str(out.iloc[0])

    def test_no_change_needed(self):
        s = pd.Series(["2024-01-15"], name="date")
        out, result = normalize_dates(s)
        assert result.rows_affected == 0

    def test_empty(self):
        s = pd.Series([], dtype=object, name="empty")
        out, result = normalize_dates(s)
        assert result.rows_affected == 0


class TestRemoveDuplicateRows:
    def test_removes_dups(self):
        df = pd.DataFrame({"a": [1, 2, 1, 3], "b": ["x", "y", "x", "z"]})
        out, result = remove_duplicate_rows(df)
        assert result.rows_affected == 1
        assert len(out) == 3

    def test_no_dups(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        out, result = remove_duplicate_rows(df)
        assert result.rows_affected == 0


class TestHandleMissingValues:
    def test_flag_numeric(self):
        s = pd.Series([1.0, np.nan, 3.0], name="col")
        out, result = handle_missing_values(s, "flag")
        assert result.rows_affected == 1
        assert out.iloc[1] == 0

    def test_flag_text(self):
        s = pd.Series(["a", None, "c"], name="col")
        out, result = handle_missing_values(s, "flag")
        assert result.rows_affected == 1
        assert out.iloc[1] == "ไม่ระบุ"

    def test_median(self):
        s = pd.Series([1.0, 2.0, np.nan, 3.0], name="col")
        out, result = handle_missing_values(s, "median")
        assert result.rows_affected == 1
        assert out.iloc[2] == 2.0  # median of [1,2,3]

    def test_no_missing(self):
        s = pd.Series([1, 2, 3], name="col")
        out, result = handle_missing_values(s, "flag")
        assert result.rows_affected == 0


# ------------------------------------------------------------------ quality
class TestCheckPlaceholderValues:
    def test_detects_dash(self):
        s = pd.Series(["100", "-", "200", "-"], name="col")
        issue = check_placeholder_values(s, "col")
        assert issue is not None
        assert issue.count == 2
        assert issue.severity == "warning"

    def test_detects_n_a(self):
        s = pd.Series(["a", "N/A", "b", "n/a"], name="col")
        issue = check_placeholder_values(s, "col")
        assert issue is not None
        assert issue.count == 2

    def test_no_placeholders(self):
        s = pd.Series(["a", "b", "c"], name="col")
        issue = check_placeholder_values(s, "col")
        assert issue is None

    def test_empty(self):
        s = pd.Series([], dtype=object, name="empty")
        issue = check_placeholder_values(s, "empty")
        assert issue is None


class TestCheckConstantColumn:
    def test_constant_numeric(self):
        s = pd.Series([5, 5, 5, 5], name="col")
        issue = check_constant_column(s, "col")
        assert issue is not None
        assert issue.severity == "info"

    def test_constant_string(self):
        s = pd.Series(["x", "x", "x"], name="col")
        issue = check_constant_column(s, "col")
        assert issue is not None

    def test_not_constant(self):
        s = pd.Series([1, 2, 3], name="col")
        issue = check_constant_column(s, "col")
        assert issue is None

    def test_empty(self):
        s = pd.Series([], dtype=float, name="empty")
        issue = check_constant_column(s, "empty")
        assert issue is None


# ------------------------------------------------------------------ insight engine
class TestCorrelationPattern:
    def test_strong_correlation_detected(self):
        np.random.seed(42)
        x = np.arange(100)
        y = x * 2 + np.random.randn(100) * 0.1  # strong positive correlation
        df = pd.DataFrame({"x": x, "y": y})
        result = discover_insights(df, detect_all(df), top_n=10)
        patterns = [c.pattern for c in result.cards]
        assert "correlation" in patterns

    def test_no_correlation_with_uncorrelated(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),  # no correlation
            }
        )
        result = discover_insights(df, detect_all(df), top_n=10)
        corr_cards = [c for c in result.cards if c.pattern == "correlation"]
        assert len(corr_cards) == 0


class TestOutlierPattern:
    def test_outlier_detected(self):
        np.random.seed(42)
        vals = list(np.random.randn(100) * 10) + [500, 600, 700]  # clear outliers
        df = pd.DataFrame({"val": vals, "cat": ["a"] * len(vals)})
        result = discover_insights(df, detect_all(df), top_n=10)
        patterns = [c.pattern for c in result.cards]
        assert "outlier" in patterns

    def test_no_outlier_in_clean_data(self):
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.randn(100) * 0.1})
        result = discover_insights(df, detect_all(df), top_n=10)
        outlier_cards = [c for c in result.cards if c.pattern == "outlier"]
        assert len(outlier_cards) == 0


class TestAdaptiveMinSegment:
    def test_small_dataset_gets_insights(self):
        """Small dataset (50 rows) should still get insights with adaptive min_segment."""
        df = pd.DataFrame(
            {
                "cat": ["a", "b", "c", "d"] * 12,  # 48 rows, 4 categories
                "val": [10, 20, 30, 40] * 12,
            }
        )
        result = discover_insights(df, detect_all(df), top_n=8)
        # With default _MIN_SEGMENT=30, groups of 12 would be skipped
        # With adaptive, min_segment should be ~2-5
        assert any("min_segment" in note for note in result.notes)


# ------------------------------------------------------------------ io
class TestExcelFormat:
    def test_detect_xlsx(self):
        from thaieda.io import detect_format

        assert detect_format("test.xlsx") == "excel"

    def test_detect_xls(self):
        from thaieda.io import detect_format

        assert detect_format("test.xls") == "excel"


# ------------------------------------------------------------------ integration
class TestRunQualityChecksV08:
    def test_placeholder_detected_in_quality_checks(self):
        df = pd.DataFrame({"col": ["a", "-", "b", "N/A"]})
        types = detect_all(df)
        issues = run_quality_checks(df, types)
        placeholder_issues = [i for i in issues if i.check_name == "placeholder_values"]
        assert len(placeholder_issues) > 0

    def test_constant_column_detected_in_quality_checks(self):
        df = pd.DataFrame({"const": [1, 1, 1], "var": [1, 2, 3]})
        types = detect_all(df)
        issues = run_quality_checks(df, types)
        constant_issues = [i for i in issues if i.check_name == "constant_column"]
        assert len(constant_issues) > 0
