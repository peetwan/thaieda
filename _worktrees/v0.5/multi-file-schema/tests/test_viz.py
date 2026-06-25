"""ทดสอบ thaieda.viz — กราฟชุดใหม่ (correlation/box/violin/missing) และ auto_visualize.

กราฟทุกชนิดคืนค่าเป็นสตริง base64 ของ PNG จึงตรวจด้วยลายเซ็นไฟล์ PNG
"""

from __future__ import annotations

import base64

import numpy as np
import pandas as pd

from thaieda.detect import ColumnType
from thaieda.viz import (
    auto_select_charts,
    auto_visualize,
    create_acf_plot,
    create_boxplot,
    create_category_bar,
    create_correlation_heatmap,
    create_decomposition_plot,
    create_distribution_histogram,
    create_missing_heatmap,
    create_missing_matrix,
    create_scatter_matrix,
    create_timeseries_plot,
    create_violinplot,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_png_base64(s: str) -> None:
    """ยืนยันว่า s เป็นสตริง base64 ที่ถอดออกมาแล้วเป็นไฟล์ PNG."""
    assert isinstance(s, str)
    assert s != ""
    raw = base64.b64decode(s)
    assert raw[:8] == _PNG_MAGIC


def _sample_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 60
    df = pd.DataFrame(
        {
            "price": rng.normal(100, 10, n),
            "qty": rng.integers(1, 50, n).astype(float),
            "score": rng.normal(3, 1, n),
            "review": ["อาหารอร่อยมาก", "ร้านนี้ดี", "ราคาแพง", "บริการดี"] * (n // 4),
        }
    )
    # ใส่ค่าว่างในสองคอลัมน์เพื่อให้ missing matrix/heatmap ทำงาน
    df.loc[0:9, "qty"] = np.nan
    df.loc[5:14, "score"] = np.nan
    return df


def _column_types() -> dict:
    return {
        "price": ColumnType.NUMERIC,
        "qty": ColumnType.NUMERIC,
        "score": ColumnType.NUMERIC,
        "review": ColumnType.THAI_TEXT,
    }


# ------------------------------------------------------------- correlation
def test_create_correlation_heatmap_returns_png():
    _assert_png_base64(create_correlation_heatmap(_sample_df()))


def test_correlation_heatmap_empty_when_too_few_numeric():
    df = pd.DataFrame({"only": [1.0, 2.0, 3.0], "text": ["ก", "ข", "ค"]})
    assert create_correlation_heatmap(df) == ""


# ------------------------------------------------------------- box / violin
def test_create_boxplot_returns_png():
    _assert_png_base64(create_boxplot(_sample_df()))


def test_create_violinplot_returns_png():
    _assert_png_base64(create_violinplot(_sample_df()))


def test_boxplot_empty_when_no_numeric():
    df = pd.DataFrame({"text": ["ก", "ข", "ค"]})
    assert create_boxplot(df) == ""
    assert create_violinplot(df) == ""


# ------------------------------------------------------------- distribution
def test_create_distribution_histogram_returns_png():
    _assert_png_base64(create_distribution_histogram(_sample_df()["price"], title="price"))


def test_distribution_histogram_empty_when_no_values():
    assert create_distribution_histogram(pd.Series(["ก", "ข"]), title="x") == ""


# ------------------------------------------------------------- missing
def test_create_missing_matrix_returns_png():
    _assert_png_base64(create_missing_matrix(_sample_df()))


def test_create_missing_heatmap_returns_png():
    _assert_png_base64(create_missing_heatmap(_sample_df()))


def test_missing_heatmap_empty_when_one_missing_column():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]})
    assert create_missing_heatmap(df) == ""


# ------------------------------------------------------------- auto_visualize
def test_auto_visualize_returns_dict_of_charts():
    charts = auto_visualize(_sample_df(), _column_types())
    assert isinstance(charts, dict)
    # กราฟระดับชุดข้อมูลต้องมีครบ (มีตัวเลข >1 คอลัมน์ และมีค่าว่าง 2 คอลัมน์)
    for key in (
        "correlation_heatmap",
        "boxplot",
        "violinplot",
        "missing_matrix",
        "missing_heatmap",
    ):
        assert key in charts
        _assert_png_base64(charts[key])
    # ฮิสโทแกรมต่อคอลัมน์ตัวเลข + ความยาวต่อคอลัมน์ข้อความ
    assert "distribution::price" in charts
    assert "length_hist::review" in charts
    for img in charts.values():
        _assert_png_base64(img)


def test_auto_visualize_no_missing_skips_missing_charts():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(5, 2, 30)})
    types = {"a": ColumnType.NUMERIC, "b": ColumnType.NUMERIC}
    charts = auto_visualize(df, types)
    assert "missing_matrix" not in charts
    assert "missing_heatmap" not in charts
    assert "correlation_heatmap" in charts


def test_auto_visualize_accepts_string_column_types():
    # column_types อาจส่งเป็น string value แทน ColumnType enum ได้
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    charts = auto_visualize(df, {"a": "numeric", "b": "numeric"})
    assert "correlation_heatmap" in charts


def test_auto_visualize_empty_when_no_charts_possible():
    df = pd.DataFrame({"text": ["ก", "ข", "ค"]})
    charts = auto_visualize(df, {"text": ColumnType.CATEGORICAL})
    # ไม่มีตัวเลข ไม่มีค่าว่าง ไม่มีคอลัมน์ข้อความที่เข้าเกณฑ์ -> dict ว่าง
    assert charts == {}


# ------------------------------------------------------------- scatter matrix
def test_create_scatter_matrix_returns_png():
    _assert_png_base64(create_scatter_matrix(_sample_df()))


def test_scatter_matrix_empty_when_too_few_numeric():
    df = pd.DataFrame({"only": [1.0, 2.0, 3.0], "text": ["ก", "ข", "ค"]})
    assert create_scatter_matrix(df) == ""


# ------------------------------------------------------------- category bar
def test_create_category_bar_returns_png():
    s = pd.Series(["แดง", "เขียว", "แดง", "น้ำเงิน", "แดง", "เขียว"])
    _assert_png_base64(create_category_bar(s, title="color"))


def test_category_bar_empty_when_no_values():
    assert create_category_bar(pd.Series([None, None], dtype="object")) == ""


# ------------------------------------------------------------- auto_select_charts
def test_auto_select_charts_picks_by_type():
    charts = auto_select_charts(_sample_df())
    # ตัวเลข >1 คอลัมน์ -> heatmap + scatter matrix; การแจกแจง -> box/violin/hist
    for key in ("correlation_heatmap", "scatter_matrix", "boxplot", "violinplot"):
        assert key in charts
        _assert_png_base64(charts[key])
    assert "distribution::price" in charts
    # ค่าว่าง 2 คอลัมน์ -> missing matrix + heatmap
    assert "missing_matrix" in charts
    assert "missing_heatmap" in charts
    # คอลัมน์ข้อความถูกตรวจหาเองและสร้างฮิสโทแกรมความยาว
    assert "length_hist::review" in charts
    for img in charts.values():
        _assert_png_base64(img)


def test_auto_select_charts_categorical_value_counts():
    df = pd.DataFrame({"grade": ["A", "B", "A", "C", "B", "A", "C", "B"]})
    charts = auto_select_charts(df, text_columns=[])
    assert "valuecounts::grade" in charts
    _assert_png_base64(charts["valuecounts::grade"])


def test_auto_select_charts_no_numeric_no_scatter():
    df = pd.DataFrame({"only": [1.0, 2.0, 3.0, 4.0]})
    charts = auto_select_charts(df, text_columns=[])
    # ตัวเลขคอลัมน์เดียว -> ไม่มี scatter matrix / correlation
    assert "scatter_matrix" not in charts
    assert "correlation_heatmap" not in charts


# ------------------------------------------------------------- timeseries plots
def _ts_series() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.Series([float(i) + (i % 7) for i in range(60)], index=idx, name="sales")


def test_create_timeseries_plot_returns_png():
    _assert_png_base64(create_timeseries_plot(_ts_series(), title="sales"))


def test_timeseries_plot_works_without_datetime_index():
    s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    _assert_png_base64(create_timeseries_plot(s))


def test_timeseries_plot_empty_when_no_values():
    assert create_timeseries_plot(pd.Series(["ก", "ข"])) == ""


def test_create_decomposition_plot_returns_png():
    components = {
        "trend": [float(i) for i in range(30)],
        "seasonal": [float(i % 7) for i in range(30)],
        "residual": [0.1 * (i % 3) for i in range(30)],
    }
    _assert_png_base64(create_decomposition_plot(components, title="decomp"))


def test_decomposition_plot_empty_when_no_components():
    assert create_decomposition_plot({}) == ""


def test_create_acf_plot_returns_png():
    _assert_png_base64(create_acf_plot(_ts_series(), lags=20))


def test_acf_plot_empty_when_too_few_points():
    assert create_acf_plot(pd.Series([1.0])) == ""
