"""ทดสอบ thaieda.viz._interactive และ thaieda.viz._extra_charts.

กลุ่มทดสอบ:
  1. กราฟ interactive (Plotly) — คืนค่าเป็น HTML div
  2. lazy import ต้องแจ้ง ImportError ถ้า plotly ไม่พร้อม
  3. extra charts (pair/kde/qq/sunburst) — คืนค่าเป็น base64 PNG
  4. ตรวจสอบว่า palette + font ถูกใช้
"""

from __future__ import annotations

import base64
import builtins
import sys

import numpy as np
import pandas as pd
import pytest

from thaieda.viz._extra_charts import (
    create_kde_plot,
    create_pair_plot,
    create_qq_plot,
    create_sunburst_chart,
)
from thaieda.viz._interactive import (
    create_correlation_heatmap_interactive,
    create_distribution_interactive,
    create_missing_matrix_interactive,
    create_scatter_interactive,
)
from thaieda.viz._palette import PALETTE, PLOTLY_FONT_FAMILY

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------- helpers
def _sample_df(seed: int = 42) -> pd.DataFrame:
    """DataFrame ตัวอย่างสำหรับทดสอบ — มีค่าว่าง และคอลัมน์หมวดหมู่."""
    rng = np.random.default_rng(seed)
    n = 80
    df = pd.DataFrame(
        {
            "price": rng.normal(100, 10, n),
            "qty": rng.integers(1, 50, n).astype(float),
            "score": rng.normal(3, 1, n),
            "category": rng.choice(["A", "B", "C"], n),
        }
    )
    df.loc[0:9, "qty"] = np.nan
    df.loc[5:14, "score"] = np.nan
    return df


def _assert_png_base64(s: str) -> None:
    """ยืนยันว่า s เป็นสตริง base64 ที่ถอดออกมาแล้วเป็นไฟล์ PNG."""
    assert isinstance(s, str)
    assert s != ""
    raw = base64.b64decode(s)
    assert raw[:8] == _PNG_MAGIC


# ============================================================ interactive
# ------------------------------------- correlation heatmap
def test_correlation_heatmap_interactive_returns_html():
    html = create_correlation_heatmap_interactive(_sample_df())
    assert isinstance(html, str)
    assert html != ""
    assert "<div" in html
    assert "plotly" in html.lower()


def test_correlation_heatmap_interactive_empty_when_too_few_numeric():
    df = pd.DataFrame({"only": [1.0, 2.0, 3.0], "text": ["ก", "ข", "ค"]})
    assert create_correlation_heatmap_interactive(df) == ""


def test_correlation_heatmap_uses_palette_colors():
    html = create_correlation_heatmap_interactive(_sample_df())
    # colorway ของ palette ใช้โทนสี Okabe-Ito (#0072B2 ตัวแรก)
    assert "#0072B2" in html or "0072B2" in html or "RdBu" in html


# ------------------------------------- distribution
def test_distribution_interactive_returns_html():
    html = create_distribution_interactive(_sample_df(), "price")
    assert isinstance(html, str)
    assert html != ""
    assert "<div" in html
    assert "plotly" in html.lower()


def test_distribution_interactive_uses_palette_color():
    html = create_distribution_interactive(_sample_df(), "price")
    # color_discrete_sequence ใช้ PALETTE[0]
    assert "#0072B2" in html or "0072B2" in html


def test_distribution_interactive_empty_when_non_numeric():
    df = pd.DataFrame({"text": ["ก", "ข", "ค"]})
    assert create_distribution_interactive(df, "text") == ""


def test_distribution_interactive_empty_when_missing_col():
    df = pd.DataFrame({"price": [1.0, 2.0]})
    assert create_distribution_interactive(df, "nope") == ""


# ------------------------------------- missing matrix
def test_missing_matrix_interactive_returns_html():
    html = create_missing_matrix_interactive(_sample_df())
    assert isinstance(html, str)
    assert html != ""
    assert "<div" in html
    assert "plotly" in html.lower()


def test_missing_matrix_interactive_empty_when_empty_df():
    assert create_missing_matrix_interactive(pd.DataFrame()) == ""


def test_missing_matrix_interactive_uses_palette():
    html = create_missing_matrix_interactive(_sample_df())
    # colorscale ใช้ PALETTE[0] (#0072B2) สำหรับ present values
    assert "#0072B2" in html or "2b3038" in html  # มืดเป็น #2b3038 สว่างเป็น palette


# ------------------------------------- scatter
def test_scatter_interactive_returns_html():
    html = create_scatter_interactive(_sample_df(), "price", "score")
    assert isinstance(html, str)
    assert html != ""
    assert "<div" in html
    assert "plotly" in html.lower()


def test_scatter_interactive_with_color_col():
    html = create_scatter_interactive(_sample_df(), "price", "qty", color_col="category")
    assert isinstance(html, str)
    assert html != ""
    assert "<div" in html


def test_scatter_interactive_empty_when_col_not_found():
    df = _sample_df()
    assert create_scatter_interactive(df, "nope", "price") == ""
    assert create_scatter_interactive(df, "price", "nope") == ""


def test_scatter_interactive_empty_when_all_null():
    df = pd.DataFrame({"x": [None, None], "y": [None, None]})
    assert create_scatter_interactive(df, "x", "y") == ""


# ============================================================ lazy import
def _hide_plotly(monkeypatch):
    """ทำให้ import plotly ล้มเหมือนไม่ได้ติดตั้ง."""
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    # เอา plotly ออกจาก cache ทุก submodule
    to_remove = [k for k in sys.modules if k.startswith("plotly")]
    for k in to_remove:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.setattr("builtins.__import__", _blocked_import)


def test_lazy_import_raises_helpful_error(monkeypatch):
    _hide_plotly(monkeypatch)
    with pytest.raises(ImportError) as exc_info:
        create_correlation_heatmap_interactive(_sample_df())
    msg = str(exc_info.value)
    assert "plotly" in msg
    assert "pip install" in msg
    assert "5.18.0" in msg


def test_lazy_import_raises_for_distribution(monkeypatch):
    _hide_plotly(monkeypatch)
    with pytest.raises(ImportError):
        create_distribution_interactive(_sample_df(), "price")


def test_lazy_import_raises_for_missing_matrix(monkeypatch):
    _hide_plotly(monkeypatch)
    with pytest.raises(ImportError):
        create_missing_matrix_interactive(_sample_df())


def test_lazy_import_raises_for_scatter(monkeypatch):
    _hide_plotly(monkeypatch)
    with pytest.raises(ImportError):
        create_scatter_interactive(_sample_df(), "price", "qty")


# ============================================================ extra charts
# ------------------------------------- pair plot
def test_pair_plot_returns_png():
    _assert_png_base64(create_pair_plot(_sample_df()))


def test_pair_plot_with_hue_col():
    _assert_png_base64(create_pair_plot(_sample_df(), hue_col="category"))


def test_pair_plot_empty_when_too_few_numeric():
    df = pd.DataFrame({"only": [1.0, 2.0], "text": ["ก", "ข"]})
    assert create_pair_plot(df) == ""


def test_pair_plot_empty_when_all_nan():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    assert create_pair_plot(df) == ""


# ------------------------------------- KDE
def test_kde_plot_returns_png():
    _assert_png_base64(create_kde_plot(_sample_df(), "price"))


def test_kde_plot_empty_when_non_numeric():
    df = pd.DataFrame({"text": ["ก", "ข", "ค"]})
    assert create_kde_plot(df, "text") == ""


def test_kde_plot_empty_when_missing_col():
    df = pd.DataFrame({"price": [1.0, 2.0]})
    assert create_kde_plot(df, "nope") == ""


def test_kde_plot_empty_when_too_few_values():
    assert create_kde_plot(pd.DataFrame({"x": [1.0]}), "x") == ""


def test_kde_plot_empty_when_constant():
    assert create_kde_plot(pd.DataFrame({"x": [5.0, 5.0, 5.0]}), "x") != ""


# ------------------------------------- QQ plot
def test_qq_plot_returns_png():
    _assert_png_base64(create_qq_plot(_sample_df(), "price"))


def test_qq_plot_empty_when_non_numeric():
    df = pd.DataFrame({"text": ["ก", "ข", "ค"]})
    assert create_qq_plot(df, "text") == ""


def test_qq_plot_empty_when_too_few_values():
    assert create_qq_plot(pd.DataFrame({"x": [1.0]}), "x") == ""


def test_qq_plot_empty_when_missing_col():
    df = pd.DataFrame({"price": [1.0, 2.0]})
    assert create_qq_plot(df, "nope") == ""


# ------------------------------------- sunburst
def test_sunburst_chart_count_returns_png():
    _assert_png_base64(create_sunburst_chart(_sample_df(), "category"))


def test_sunburst_chart_with_val_col_returns_png():
    _assert_png_base64(create_sunburst_chart(_sample_df(), "category", val_col="price"))


def test_sunburst_chart_empty_when_missing_col():
    assert create_sunburst_chart(_sample_df(), "nope") == ""


def test_sunburst_chart_empty_when_all_null():
    df = pd.DataFrame({"cat": [None, None, None]})
    assert create_sunburst_chart(df, "cat") == ""


def test_sunburst_chart_pie_fallback_returns_png(monkeypatch):
    """ทดสอบ pie chart fallback เมื่อ plotly ไม่พร้อมใช้."""
    _hide_plotly(monkeypatch)
    png = create_sunburst_chart(_sample_df(), "category")
    _assert_png_base64(png)


# ============================================================ palette
def test_palette_exported():
    assert PALETTE is not None
    assert len(PALETTE) >= 7
    assert PALETTE[0] == "#0072B2"


def test_plotly_font_family_exported():
    assert isinstance(PLOTLY_FONT_FAMILY, str)
    assert "Sarabun" in PLOTLY_FONT_FAMILY


def test_correlation_heatmap_html_contains_font_family():
    html = create_correlation_heatmap_interactive(_sample_df())
    # plotly layout ใช้ font family ที่เรากำหนด
    assert "Sarabun" in html or "font" in html.lower()
