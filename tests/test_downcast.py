"""ทดสอบ downcast_dtypes — ลด dtype เพื่อประหยัด memory (v2.0)."""

from __future__ import annotations

import pandas as pd

from thaieda.io import downcast_dtypes, read_data


class TestDowncastDtypes:
    """ทดสอบการลด dtype."""

    def test_int64_downcast(self):
        df = pd.DataFrame({"x": pd.Series([1, 2, 3], dtype="int64")})
        out, _ = downcast_dtypes(df)
        assert out["x"].dtype.itemsize < 8  # เล็กกว่า int64
        assert out["x"].tolist() == [1, 2, 3]

    def test_int_range_picks_small_type(self):
        df = pd.DataFrame({"x": pd.Series([1, 2, 3], dtype="int64")})
        out, _ = downcast_dtypes(df)
        assert out["x"].dtype == "int8"

    def test_float64_to_float32(self):
        df = pd.DataFrame({"y": pd.Series([1.5, 2.5, 3.5], dtype="float64")})
        out, _ = downcast_dtypes(df)
        assert out["y"].dtype == "float32"

    def test_object_low_cardinality_to_category(self):
        df = pd.DataFrame({"c": ["a", "b", "a", "b"] * 25})
        out, _ = downcast_dtypes(df)
        assert isinstance(out["c"].dtype, pd.CategoricalDtype)

    def test_object_high_cardinality_stays_object(self):
        df = pd.DataFrame({"c": [f"v{i}" for i in range(100)]})
        out, _ = downcast_dtypes(df)
        # pandas 3.x อาจแปลง object→str แทนที่จะคง object ไว้ — ยอมรับทั้งคู่
        assert pd.api.types.is_object_dtype(out["c"]) or pd.api.types.is_string_dtype(out["c"])

    def test_bool_unchanged(self):
        df = pd.DataFrame({"b": [True, False, True]})
        out, changed = downcast_dtypes(df)
        assert out["b"].dtype == "bool"
        assert "b" not in changed["columns_changed"]

    def test_datetime_unchanged(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-01-02"])})
        out, changed = downcast_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(out["d"])
        assert "d" not in changed["columns_changed"]

    def test_report_has_required_keys(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        _, report = downcast_dtypes(df)
        for key in (
            "memory_before_mb",
            "memory_after_mb",
            "reduction_pct",
            "columns_changed",
            "n_columns_changed",
        ):
            assert key in report

    def test_memory_reduced(self):
        df = pd.DataFrame(
            {
                "i": pd.Series(range(1000), dtype="int64"),
                "f": pd.Series([1.5] * 1000, dtype="float64"),
                "c": ["a", "b"] * 500,
            }
        )
        _, report = downcast_dtypes(df)
        assert report["memory_after_bytes"] < report["memory_before_bytes"]
        assert report["reduction_pct"] > 0

    def test_values_preserved(self):
        df = pd.DataFrame({"i": [10, 20, 30], "f": [1.0, 2.0, 3.0], "c": ["x", "y", "x"]})
        out, _ = downcast_dtypes(df)
        assert out["i"].tolist() == [10, 20, 30]
        assert out["f"].tolist() == [1.0, 2.0, 3.0]
        assert list(out["c"].astype(str)) == ["x", "y", "x"]

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        out, report = downcast_dtypes(df)
        assert out.empty
        assert report["n_columns_changed"] == 0

    def test_original_not_mutated(self):
        df = pd.DataFrame({"x": pd.Series([1, 2, 3], dtype="int64")})
        before = str(df["x"].dtype)
        downcast_dtypes(df)
        assert str(df["x"].dtype) == before

    def test_already_category_unchanged(self):
        df = pd.DataFrame({"c": pd.Series(["a", "b", "a"], dtype="category")})
        out, changed = downcast_dtypes(df)
        assert isinstance(out["c"].dtype, pd.CategoricalDtype)
        assert "c" not in changed["columns_changed"]


class TestReadDataDowncast:
    """ทดสอบ read_data(downcast=...) — v2.0."""

    def test_read_data_downcast_true(self, tmp_path):
        p = tmp_path / "d.csv"
        df = pd.DataFrame({"i": range(50), "c": ["a", "b"] * 25})
        df.to_csv(p, index=False)
        loaded = read_data(p, downcast=True)
        assert loaded["i"].dtype.itemsize < 8
        assert isinstance(loaded["c"].dtype, pd.CategoricalDtype)

    def test_read_data_downcast_false_default(self, tmp_path):
        p = tmp_path / "d.csv"
        df = pd.DataFrame({"i": range(50)})
        df.to_csv(p, index=False)
        loaded = read_data(p)
        assert loaded["i"].dtype == "int64"
