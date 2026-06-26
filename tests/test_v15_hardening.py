"""ทดสอบการ hardening v1.5 — insight overflow cap, chart/table budget, high-NA, viz skips.

ครอบคลุม defect P1 (insight cap), P2 (HTML bloat), P3 (viz O(n^2) skips), Q1 (high-NA clean).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaieda.clean import handle_missing_values
from thaieda.insight import generate_insights
from thaieda.quality import QualityIssue
from thaieda.report import ProfileReport
from thaieda.viz import create_correlation_heatmap, create_scatter_matrix


def _qi(i: int, severity: str) -> QualityIssue:
    return QualityIssue(
        check_name="whitespace",
        severity=severity,
        column=f"col{i}",
        count=5,
        percentage=2.0,
        description="d",
        description_th="พบช่องว่าง",
        suggestion="s",
        suggestion_th="แก้ช่องว่าง",
    )


# ---------------------------------------------------------------- P1: insight cap
def test_insights_capped_to_max():
    df = pd.DataFrame({"a": [1, 2, 3]})
    issues = [_qi(i, "warning") for i in range(391)] + [_qi(1000 + i, "info") for i in range(288)]
    summary = generate_insights(df, issues, [], {})
    assert summary.total_insights == 30
    assert summary.total_generated == 679
    assert len(summary.insights) == 30
    # exec summary บอกจำนวนทั้งหมด
    assert "679" in summary.executive_summary_th


def test_cap_never_drops_critical():
    df = pd.DataFrame({"a": [1, 2, 3]})
    issues = [_qi(i, "critical") for i in range(35)]
    summary = generate_insights(df, issues, [], {})
    # critical ทั้ง 35 ต้องอยู่ครบ แม้เกิน max_insights=30
    assert summary.critical_count == 35
    assert summary.total_insights == 35


def test_cap_prioritizes_warning_over_info():
    df = pd.DataFrame({"a": [1, 2, 3]})
    issues = [_qi(i, "warning") for i in range(40)] + [_qi(1000 + i, "info") for i in range(40)]
    summary = generate_insights(df, issues, [], {}, max_insights=30)
    # โควตา 30 ควรเต็มด้วย warning ก่อน info ถูกตัดหมด
    assert summary.warning_count == 30
    assert summary.info_count == 0


def test_cap_can_be_disabled():
    df = pd.DataFrame({"a": [1, 2, 3]})
    issues = [_qi(i, "warning") for i in range(100)]
    summary = generate_insights(df, issues, [], {}, max_insights=0)
    assert summary.total_insights == 100
    assert summary.total_generated == 100


def test_total_generated_in_to_dict():
    df = pd.DataFrame({"a": [1, 2, 3]})
    summary = generate_insights(df, [_qi(0, "warning")], [], {})
    d = summary.to_dict()
    assert d["total_generated"] == 1
    assert d["total_insights"] == 1


# ---------------------------------------------------------------- Q1: high-NA clean
def test_mostly_missing_skips_fill():
    # > 80% ค่าว่าง — ไม่เติม แต่ flag เป็น mostly_missing
    s = pd.Series([1.0] + [np.nan] * 9, name="pm25")
    out, result = handle_missing_values(s, "flag")
    assert "mostly_missing" in result.description_th
    # ค่าว่างยังคงเป็น NaN (ไม่ถูก fill เป็น 0)
    assert int(out.isna().sum()) == 9


def test_high_na_warns_but_fills():
    # 50% ค่าว่าง — ยังเติม แต่แนบคำเตือน > 40%
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan], name="x")
    out, result = handle_missing_values(s, "flag")
    assert "> 40%" in result.description_th
    assert int(out.isna().sum()) == 0  # ถูกเติมแล้ว


def test_normal_na_unchanged_behavior():
    # < 40% ค่าว่าง — พฤติกรรมเดิม ไม่มีคำเตือน
    s = pd.Series([1.0, np.nan, 3.0], name="col")
    out, result = handle_missing_values(s, "flag")
    assert out.iloc[1] == 0
    assert "40%" not in result.description_th
    assert "mostly_missing" not in result.description_th


# ---------------------------------------------------------------- P3: viz skips
def _wide_numeric_df(n_cols: int, rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({f"n{i}": rng.normal(0, 1, rows) for i in range(n_cols)})


def test_correlation_heatmap_skipped_when_too_wide():
    assert create_correlation_heatmap(_wide_numeric_df(35)) == ""
    # ยังวาดได้เมื่อไม่กว้างเกิน
    assert create_correlation_heatmap(_wide_numeric_df(5)) != ""


def test_scatter_matrix_skipped_when_too_wide():
    assert create_scatter_matrix(_wide_numeric_df(60)) == ""
    assert create_scatter_matrix(_wide_numeric_df(4)) != ""


# ---------------------------------------------------------------- P2: chart budget
def test_chart_budget_enforced_on_wide_report():
    rng = np.random.default_rng(1)
    data = {f"num{i}": rng.normal(i, 1.0, 300) for i in range(70)}
    for j in range(5):
        data[f"cat{j}"] = rng.choice(list("abcde"), 300)
    df = pd.DataFrame(data)
    rep = ProfileReport(df, make_charts=True, timeseries=False, insights_engine=False).run()
    assert rep._count_embedded_charts() <= 40
    html = rep.to_html()
    assert len(html.encode("utf-8")) < 2_000_000  # < 2MB
    assert html.startswith("<!DOCTYPE")  # ไม่มี whitespace นำหน้า (กัน quirks mode)


def test_timeseries_charts_size_budget_under_2mb():
    # อนุกรมเวลาหลายคอลัมน์สร้างกราฟ decomposition ขนาดใหญ่ — ต้องคุมขนาดให้ < 2MB
    rng = np.random.default_rng(3)
    n = 2000
    data = {"date": pd.date_range("2020-01-01", periods=n, freq="h")}
    for i in range(15):
        data[f"metric{i}"] = np.cumsum(rng.normal(0, 1, n)) + i * 10
    df = pd.DataFrame(data)
    rep = ProfileReport(df, make_charts=True, timeseries=True, insights_engine=False).run()
    html = rep.to_html()
    assert rep._count_embedded_charts() <= 40
    assert rep._embedded_chart_bytes() <= 1_600_000
    assert len(html.encode("utf-8")) < 2_000_000


def test_column_summary_table_when_many_columns():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({f"c{i}": rng.normal(0, 1, 100) for i in range(70)})
    rep = ProfileReport(df, make_charts=False, timeseries=False, insights_engine=False).run()
    html = rep.to_html()
    # > 60 คอลัมน์ → สรุปเป็นตาราง แทนการ์ดรายคอลัมน์
    assert "สรุปเป็นตารางแทนการ์ด" in html
