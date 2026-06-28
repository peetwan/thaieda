"""ทดสอบ thaieda.timeseries — การวิเคราะห์อนุกรมเวลา (trend/seasonality/gap/anomaly)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from thaieda.timeseries import (
    TimeseriesComponent,
    TimeseriesResult,
    analyze_dataframe_timeseries,
    analyze_timeseries,
    detect_timeseries_columns,
    is_panel_time_axis,
)


def _statsmodels_installed() -> bool:
    return importlib.util.find_spec("statsmodels") is not None


def _daily_series(values, start="2024-01-01") -> pd.Series:
    """สร้าง series รายวันที่มี DatetimeIndex."""
    idx = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="metric")


# --------------------------------------------------------------- detect columns
def test_detect_timeseries_columns_finds_datetime():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "value": range(30),
            "name": ["ก"] * 30,
        }
    )
    cols = detect_timeseries_columns(df)
    assert "date" in cols
    assert "value" not in cols


def test_detect_timeseries_columns_empty_when_no_datetime():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert detect_timeseries_columns(df) == {}


def test_detect_timeseries_columns_needs_enough_points():
    # มี datetime แค่ 2 จุด (< _MIN_TS_POINTS) -> ไม่ถือเป็น timeseries
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "v": [1, 2]})
    assert detect_timeseries_columns(df) == {}


# --------------------------------------------------------------- trend detection
def test_increasing_trend_detected():
    series = _daily_series([float(i) for i in range(60)])
    result = analyze_timeseries(series, engine="basic")
    assert result.is_timeseries
    assert result.has_trend
    assert result.trend_direction == "increasing"
    assert result.trend_direction_th == "เพิ่มขึ้น"


def test_decreasing_trend_detected():
    series = _daily_series([float(100 - i) for i in range(60)])
    result = analyze_timeseries(series, engine="basic")
    assert result.has_trend
    assert result.trend_direction == "decreasing"


def test_no_trend_for_flat_noise():
    rng = np.random.default_rng(0)
    series = _daily_series(rng.normal(50, 1, 60))
    result = analyze_timeseries(series, engine="basic")
    assert result.trend_direction == "stable"
    assert not result.has_trend


# --------------------------------------------------------------- seasonality
def test_weekly_seasonality_detected():
    # สัญญาณรอบ 7 วัน ชัดเจน + noise เล็กน้อย
    rng = np.random.default_rng(1)
    base = np.tile([0, 5, 10, 5, 0, -5, -10], 12).astype("float64")
    base += rng.normal(0, 0.3, base.size)
    series = _daily_series(base)
    result = analyze_timeseries(series, engine="basic")
    assert result.has_seasonality
    assert result.seasonal_period == 7
    assert "รอบ 7" in result.seasonal_period_th


def test_frequency_detection_daily():
    series = _daily_series([float(i) for i in range(20)])
    result = analyze_timeseries(series, engine="basic")
    assert result.frequency == "D"
    assert result.frequency_th == "รายวัน"


# --------------------------------------------------------------- gaps
def test_time_gap_detected():
    # ข้อมูลรายวัน แต่ข้ามช่วงกลางไป 10 วัน
    dates = list(pd.date_range("2024-01-01", periods=15, freq="D"))
    dates += list(pd.date_range("2024-01-26", periods=15, freq="D"))
    series = pd.Series(range(30), index=pd.DatetimeIndex(dates), name="v")
    result = analyze_timeseries(series, engine="basic")
    assert result.gap_count >= 1
    assert len(result.gaps) == result.gap_count
    assert isinstance(result.gaps[0], tuple)


def test_no_gap_for_continuous_daily():
    series = _daily_series([float(i) for i in range(40)])
    result = analyze_timeseries(series, engine="basic")
    assert result.gap_count == 0


# --------------------------------------------------------------- anomalies / spike
def test_spike_anomaly_detected():
    values = [10.0] * 60
    values[30] = 500.0  # spike เดี่ยว
    series = _daily_series(values)
    result = analyze_timeseries(series, engine="basic")
    assert len(result.anomalies) >= 1


# --------------------------------------------------------------- edge cases
def test_too_few_points_returns_not_timeseries():
    series = _daily_series([1.0, 2.0, 3.0])
    result = analyze_timeseries(series, engine="basic")
    assert not result.is_timeseries
    assert "น้อยเกินไป" in result.insights[0]


def test_result_to_dict_is_compact():
    series = _daily_series([float(i) for i in range(40)])
    result = analyze_timeseries(series, engine="basic")
    d = result.to_dict()
    assert "components" not in d  # อาเรย์หนักไม่อยู่ใน to_dict
    assert d["column"] == "metric"
    assert d["has_trend"] is True
    assert "stats" in d
    assert isinstance(d["insights"], list)


def test_components_present_in_result():
    series = _daily_series([float(i) for i in range(40)])
    result = analyze_timeseries(series, engine="basic")
    assert set(result.components.keys()) == {"trend", "seasonal", "residual"}
    assert isinstance(result.components["trend"], TimeseriesComponent)
    assert len(result.components["trend"].values) == 40


def test_random_walk_insight():
    rng = np.random.default_rng(5)
    series = _daily_series(rng.normal(0, 1, 60))
    result = analyze_timeseries(series, engine="basic")
    assert any("random walk" in s or "นิ่ง" in s for s in result.insights)


# --------------------------------------------------------------- engine handling
def test_basic_engine_used_label():
    series = _daily_series([float(i) for i in range(40)])
    result = analyze_timeseries(series, engine="basic")
    assert result.engine_used == "basic"


def test_invalid_engine_raises():
    series = _daily_series([float(i) for i in range(40)])
    with pytest.raises(ValueError):
        analyze_timeseries(series, engine="nope")


@pytest.mark.skipif(not _statsmodels_installed(), reason="ต้องมี statsmodels")
def test_statsmodels_engine_decomposes():
    rng = np.random.default_rng(2)
    base = np.tile([0, 5, 10, 5, 0, -5, -10], 12).astype("float64")
    base += rng.normal(0, 0.3, base.size) + np.arange(base.size) * 0.1
    series = _daily_series(base)
    result = analyze_timeseries(series, engine="statsmodels")
    assert result.engine_used == "statsmodels"
    assert len(result.components["trend"].values) == len(base)


# --------------------------------------------------------------- dataframe-level
def test_analyze_dataframe_timeseries_auto():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "sales": [float(i) for i in range(60)],
            "label": ["ก"] * 60,
        }
    )
    results = analyze_dataframe_timeseries(df, engine="basic")
    assert "sales" in results
    assert results["sales"].has_trend


def test_analyze_dataframe_timeseries_no_datetime():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert analyze_dataframe_timeseries(df) == {}


def test_analyze_dataframe_timeseries_explicit_col():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=40, freq="D"),
            "y": [float(100 - i) for i in range(40)],
        }
    )
    results = analyze_dataframe_timeseries(df, time_col="ts", engine="basic")
    assert "y" in results
    assert results["y"].trend_direction == "decreasing"


# --------------------------------------------------------------- panel/snapshot gate
def test_is_panel_time_axis_true_for_snapshot():
    # 100 แถว แต่มีเพียง 3 ช่วงเวลา (panel/snapshot ของหลายสถานี) → ไม่ใช่ TS รายแถว
    stamps = pd.to_datetime(["2026-06-28"] * 40 + ["2026-06-29"] * 40 + ["2026-06-30"] * 20)
    assert is_panel_time_axis(stamps) is True


def test_is_panel_time_axis_false_for_real_series():
    stamps = pd.date_range("2024-01-01", periods=60, freq="D")
    assert is_panel_time_axis(stamps) is False


def test_analyze_dataframe_timeseries_skips_panel_data():
    # ข้อมูล snapshot: หลายสถานี ณ เวลาเดียว → ไม่ควรวิเคราะห์เป็นอนุกรมเวลา
    df = pd.DataFrame(
        {
            "date": ["2026-06-28"] * 30 + ["2026-06-29"] * 30,
            "pm25": [float(i % 7) for i in range(60)],
        }
    )
    assert analyze_dataframe_timeseries(df, engine="basic") == {}


def test_result_dataclass_basic_shape():
    series = _daily_series([float(i) for i in range(30)])
    result = analyze_timeseries(series, engine="basic")
    assert isinstance(result, TimeseriesResult)
    assert result.stats["mean"] is not None
    assert "autocorr_lag1" in result.stats
