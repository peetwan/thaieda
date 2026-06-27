"""ทดสอบ clean() + CleaningReport — DataFrame-level cleaning pipeline (v2.0)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import thaieda
from thaieda.clean import CleaningReport, clean


class TestCleanBasics:
    """พฤติกรรมพื้นฐานของ clean()."""

    def test_returns_df_and_report(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        out, report = clean(df)
        assert isinstance(out, pd.DataFrame)
        assert isinstance(report, CleaningReport)

    def test_callable_subpackage(self):
        # thaieda.clean(df) ต้องเรียกได้แม้ thaieda.clean เป็นชื่อ subpackage
        df = pd.DataFrame({"a": [1, 2, 3]})
        out, report = thaieda.clean(df)
        assert isinstance(out, pd.DataFrame)

    def test_typeerror_on_non_dataframe(self):
        with pytest.raises(TypeError):
            clean([1, 2, 3])

    def test_valueerror_on_bad_missing_strategy(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError):
            clean(df, handle_missing="bogus")

    def test_original_df_not_mutated(self):
        df = pd.DataFrame({"a": ["  x  ", "y"], "b": [1, 2]})
        original = df.copy(deep=True)
        clean(df)
        pd.testing.assert_frame_equal(df, original)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        out, report = clean(df)
        assert out.empty
        assert report.rows_before == 0


class TestCleanOperations:
    """ทดสอบว่า operation ต่าง ๆ ทำงานใน pipeline."""

    def test_strips_whitespace(self):
        df = pd.DataFrame({"name": ["  สมชาย  ", "สมหญิง"]})
        out, _ = clean(df, downcast=False, remove_duplicates=False)
        assert out["name"].iloc[0] == "สมชาย"

    def test_converts_thai_numerals(self):
        df = pd.DataFrame({"qty": ["๑๐", "๒๐", "๓๐"]})
        out, _ = clean(df, downcast=False, remove_duplicates=False)
        assert out["qty"].tolist() == [10, 20, 30]

    def test_keeps_thai_numerals_when_disabled(self):
        df = pd.DataFrame({"qty": ["๑๐", "๒๐", "๓๐"]})
        out, _ = clean(df, fix_numerals=False, downcast=False, remove_duplicates=False)
        assert "๑" in str(out["qty"].iloc[0])

    def test_converts_currency(self):
        df = pd.DataFrame({"price": ["฿1,200", "฿2,500", "฿3,000"]})
        out, _ = clean(df, downcast=False, remove_duplicates=False)
        assert out["price"].tolist() == [1200, 2500, 3000]

    def test_converts_buddhist_era_dates(self):
        df = pd.DataFrame({"reg_date": ["2567-01-15", "2566-05-20", "2567-12-01"]})
        out, _ = clean(df, downcast=False, remove_duplicates=False)
        assert out["reg_date"].iloc[0].startswith("2024")

    def test_does_not_corrupt_numeric_year_in_price_column(self):
        # ราคา 2500 (numeric) ต้องไม่ถูกแปลงเป็น พ.ศ. → 1957
        df = pd.DataFrame({"price": [2500, 2550, 2560], "name": ["a", "b", "c"]})
        out, _ = clean(df, downcast=False, remove_duplicates=False)
        assert out["price"].tolist() == [2500, 2550, 2560]

    def test_does_not_convert_dates_when_disabled(self):
        df = pd.DataFrame({"reg_date": ["2567-01-15", "2566-05-20"]})
        out, _ = clean(df, fix_dates=False, downcast=False, remove_duplicates=False)
        assert "2567" in str(out["reg_date"].iloc[0])

    def test_custom_operations_list(self):
        # ระบุเฉพาะ whitespace → เลขไทยไม่ถูกแปลง
        df = pd.DataFrame({"qty": ["๑๐", " x "]})
        out, _ = clean(
            df, operations=["whitespace"], fix_dates=False, downcast=False, remove_duplicates=False
        )
        assert "๑" in str(out["qty"].iloc[0])
        assert out["qty"].iloc[1] == "x"


class TestCleanDuplicates:
    def test_removes_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        out, report = clean(df, downcast=False)
        assert len(out) == 2
        assert report.rows_before == 3
        assert report.rows_after == 2

    def test_keeps_duplicates_when_disabled(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        out, _ = clean(df, remove_duplicates=False, downcast=False)
        assert len(out) == 3


class TestCleanMissing:
    def test_flag_fills_missing(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"]})
        out, _ = clean(df, downcast=False, remove_duplicates=False)
        assert out["a"].isna().sum() == 1
        assert out["b"].isna().sum() == 1

    def test_drop_removes_rows(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "z"]})
        out, report = clean(df, handle_missing="drop", downcast=False, remove_duplicates=False)
        assert len(out) == 2

    def test_median_strategy(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "x": ["p", "q", "r"]})
        out, _ = clean(df, handle_missing="median", downcast=False, remove_duplicates=False)
        assert out["a"].iloc[1] == 2.0


class TestCleanDowncast:
    def test_downcast_changes_dtypes(self):
        df = pd.DataFrame({"i": pd.Series([1, 2, 3], dtype="int64")})
        out, _ = clean(df, downcast=True, remove_duplicates=False)
        assert out["i"].dtype.itemsize < 8

    def test_downcast_false_keeps_int64(self):
        df = pd.DataFrame({"i": pd.Series([1, 2, 3], dtype="int64")})
        out, _ = clean(df, downcast=False, remove_duplicates=False)
        assert out["i"].dtype == "int64"

    def test_downcast_reduces_memory(self):
        df = pd.DataFrame({"i": pd.Series(range(500), dtype="int64"), "c": ["a", "b"] * 250})
        with_dc, _ = clean(df, downcast=True, remove_duplicates=False)
        without_dc, _ = clean(df, downcast=False, remove_duplicates=False)
        assert with_dc.memory_usage(deep=True).sum() < without_dc.memory_usage(deep=True).sum()


class TestCleaningReport:
    def test_report_fields(self):
        df = pd.DataFrame({"name": ["  x  ", "y", "z"]})
        _, report = clean(df, downcast=False, remove_duplicates=False)
        assert report.rows_before == 3
        assert report.rows_after == 3
        assert isinstance(report.columns_affected, list)
        assert isinstance(report.total_changes, int)
        assert isinstance(report.warnings, list)

    def test_columns_affected(self):
        df = pd.DataFrame({"name": ["  x  ", "y"], "untouched": [1, 2]})
        _, report = clean(df, downcast=False, remove_duplicates=False)
        assert "name" in report.columns_affected

    def test_to_dict(self):
        df = pd.DataFrame({"name": ["  x  ", "y"]})
        _, report = clean(df, downcast=False, remove_duplicates=False)
        d = report.to_dict()
        assert d["rows_before"] == 2
        assert "operations_run" in d

    def test_to_json_string_and_file(self, tmp_path):
        df = pd.DataFrame({"name": ["  x  ", "y"]})
        _, report = clean(df, downcast=False, remove_duplicates=False)
        s = report.to_json()
        assert json.loads(s)["rows_before"] == 2
        p = tmp_path / "r.json"
        report.to_json(str(p))
        assert json.loads(p.read_text(encoding="utf-8"))["rows_before"] == 2

    def test_summary_th(self):
        df = pd.DataFrame({"price": ["฿1,000", "฿2,000"]})
        _, report = clean(df, downcast=False, remove_duplicates=False)
        summary = report.summary_th()
        assert "สรุปการทำความสะอาด" in summary
        assert "price" in summary

    def test_high_na_warning(self):
        # คอลัมน์ขาดข้อมูล > 40% → มีคำเตือน
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan, 5.0], "b": [1, 2, 3, 4, 5]})
        _, report = clean(df, downcast=False, remove_duplicates=False)
        assert len(report.warnings) > 0
