"""ทดสอบการเปรียบเทียบชุดข้อมูลสองชุดแบบ side-by-side."""

from __future__ import annotations

import warnings

import pandas as pd

import thaieda
from thaieda.compare import compare_datasets, compare_reports


def test_same_dataframes_no_diffs() -> None:
    """DataFrame เหมือนกันต้องไม่มี schema/row/categorical diff และค่าสถิติเท่ากัน."""
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, 4],
            "cat": ["ก", "ข", "ก", "ค"],
        }
    )

    result = compare_datasets(df, df.copy())

    assert result["schema_diff"]["columns_only_in_A"] == []
    assert result["schema_diff"]["columns_only_in_B"] == []
    assert result["schema_diff"]["type_changes"] == []
    assert result["row_count"]["diff"] == 0
    assert result["categorical_drift"] == {}
    assert result["numeric_stats_diff"]["num"]["mean_diff"] == 0.0
    assert result["numeric_stats_diff"]["num"]["std_diff"] == 0.0
    assert result["missing_diff"]["num"]["diff"] == 0
    assert result["missing_diff"]["cat"]["diff"] == 0

    drift = result["distribution_drift"]["num"]
    if drift["method"] == "ks_2samp":
        assert drift["ks_statistic"] == 0.0
    else:
        assert drift["method"] == "mean_std"
        assert drift["cohens_d"] == 0.0


def test_top_level_compare_alias() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = thaieda.compare(df, df.copy())
    assert result["row_count"]["diff"] == 0


def test_infinite_values_do_not_emit_runtime_warning() -> None:
    df_a = pd.DataFrame({"x": [1.0, 2.0, float("inf")]})
    df_b = pd.DataFrame({"x": [1.0, 3.0, float("-inf")]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = compare_datasets(df_a, df_b)

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert result["numeric_stats_diff"]["x"]["A"]["count"] == 2
    assert result["numeric_stats_diff"]["x"]["B"]["count"] == 2


def test_different_row_counts_shows_diff() -> None:
    """จำนวนแถวต่างกันต้องแสดง diff."""
    df_a = pd.DataFrame({"x": [1, 2, 3]})
    df_b = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

    result = compare_datasets(df_a, df_b)

    assert result["row_count"]["A"] == 3
    assert result["row_count"]["B"] == 5
    assert result["row_count"]["diff"] == 2


def test_different_columns_shows_schema_diff() -> None:
    """คอลัมน์ที่มีเฉพาะชุดใดชุดหนึ่งต้องอยู่ใน schema diff."""
    df_a = pd.DataFrame({"x": [1], "only_a": [10]})
    df_b = pd.DataFrame({"x": [1], "only_b": [20]})

    result = compare_datasets(df_a, df_b)

    assert result["schema_diff"]["columns_only_in_A"] == ["only_a"]
    assert result["schema_diff"]["columns_only_in_B"] == ["only_b"]


def test_numeric_drift_ks_stat_detected() -> None:
    """คอลัมน์ตัวเลขที่ distribution ต่างกันมากต้องตรวจพบ drift."""
    df_a = pd.DataFrame({"score": list(range(100))})
    df_b = pd.DataFrame({"score": list(range(1000, 1100))})

    result = compare_datasets(df_a, df_b)
    drift = result["distribution_drift"]["score"]

    assert drift["drift_detected"] is True
    if drift["method"] == "ks_2samp":
        assert drift["ks_statistic"] > 0.9
        assert drift["p_value"] < 0.001
    else:
        assert drift["method"] == "mean_std"
        assert drift["cohens_d"] > 1.0
        assert "scipy" in drift["note"]


def test_categorical_drift_frequency_shift_detected() -> None:
    """ความถี่ของค่าหมวดหมู่เปลี่ยนต้องแสดง frequency shift."""
    df_a = pd.DataFrame({"segment": ["A", "A", "A", "B"]})
    df_b = pd.DataFrame({"segment": ["A", "B", "B", "B"]})

    result = compare_datasets(df_a, df_b)
    cat = result["categorical_drift"]["segment"]

    shifts = {item["value"]: item for item in cat["top_values"]}
    assert shifts["A"]["freq_A"] == 0.75
    assert shifts["A"]["freq_B"] == 0.25
    assert shifts["A"]["shift"] == -0.5
    assert shifts["B"]["shift"] == 0.5


def test_different_labels_are_used() -> None:
    """labels ที่กำหนดเองต้องถูกใช้ใน key และ HTML report."""
    df_a = pd.DataFrame({"x": [1, 2], "old_col": ["a", "b"]})
    df_b = pd.DataFrame({"x": [2, 3], "new_col": ["c", "d"]})

    result = compare_datasets(df_a, df_b, labels=("before", "after"))

    assert result["labels"] == ["before", "after"]
    assert result["schema_diff"]["columns_only_in_before"] == ["old_col"]
    assert result["schema_diff"]["columns_only_in_after"] == ["new_col"]
    assert "before" in result["row_count"]
    assert "after" in result["row_count"]

    html = compare_reports(df_a, df_b, labels=("before", "after"), lang="th")
    assert "before" in html
    assert "after" in html
    assert "รายงานเปรียบเทียบชุดข้อมูล ThaiEDA" in html


def test_empty_dataframe_edge_case() -> None:
    """DataFrame ว่างต้องไม่ error และคืนโครงสร้างผลลัพธ์ครบ."""
    df_a = pd.DataFrame()
    df_b = pd.DataFrame()

    result = compare_datasets(df_a, df_b)

    assert result["schema_diff"]["columns_only_in_A"] == []
    assert result["schema_diff"]["columns_only_in_B"] == []
    assert result["schema_diff"]["type_changes"] == []
    assert result["row_count"] == {"A": 0, "B": 0, "diff": 0}
    assert result["numeric_stats_diff"] == {}
    assert result["missing_diff"] == {}
    assert result["distribution_drift"] == {}
    assert result["categorical_drift"] == {}

    html = compare_reports(df_a, df_b)
    assert "<!DOCTYPE html>" in html
    assert "ไม่พบความแตกต่าง" in html


def test_empty_dataframe_with_columns_edge_case() -> None:
    """DataFrame ว่างแต่มีคอลัมน์ต้องคำนวณ missing/stat ได้โดยไม่ error."""
    df_a = pd.DataFrame({"num": pd.Series(dtype="float64"), "cat": pd.Series(dtype="object")})
    df_b = pd.DataFrame({"num": pd.Series(dtype="float64"), "cat": pd.Series(dtype="object")})

    result = compare_datasets(df_a, df_b)

    assert result["row_count"]["diff"] == 0
    assert result["missing_diff"]["num"]["diff"] == 0
    assert result["missing_diff"]["cat"]["diff"] == 0
    assert result["numeric_stats_diff"]["num"]["A"]["count"] == 0
    assert result["numeric_stats_diff"]["num"]["A"]["mean"] is None
    assert result["distribution_drift"] == {}
    assert result["categorical_drift"] == {}


def test_compare_reports_english_html() -> None:
    """compare_reports ต้องสร้าง HTML ภาษาอังกฤษได้."""
    df_a = pd.DataFrame({"x": [1, 2]})
    df_b = pd.DataFrame({"x": [1, 3]})

    html = compare_reports(df_a, df_b, lang="en")

    assert "ThaiEDA Dataset Comparison" in html
    assert "Numeric Stats Diff" in html
    assert "Distribution Drift" in html
