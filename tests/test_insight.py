"""ทดสอบ thaieda.insight — generate_insights, Insight, InsightSummary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaieda.analysis import TargetAssociation
from thaieda.anomaly import AnomalyIssue, detect_anomalies
from thaieda.clean import CleaningResult
from thaieda.detect import ColumnType, detect_all
from thaieda.insight import Insight, InsightSummary, generate_insights
from thaieda.quality import QualityIssue, run_quality_checks
from thaieda.timeseries import analyze_dataframe_timeseries


def _quality_issue(check_name="buddhist_era", severity="critical", column="year", count=3):
    return QualityIssue(
        check_name=check_name,
        severity=severity,
        column=column,
        count=count,
        percentage=30.0,
        description="desc en",
        description_th="คำอธิบายภาษาไทย",
        suggestion="fix it",
        suggestion_th="แก้ไขด้วยวิธีนี้",
    )


def _anomaly_issue(check_name="numeric_outliers", anomaly_type="statistical", column="price"):
    return AnomalyIssue(
        check_name=check_name,
        severity="warning",
        column=column,
        anomaly_type=anomaly_type,
        count=12,
        percentage=4.0,
        description="outliers",
        description_th="พบค่าผิดปกติ",
        suggestion="check",
        suggestion_th="ตรวจสอบค่าเหล่านี้",
    )


# ------------------------------------------------------------- basic structure
def test_generate_insights_returns_summary():
    df = pd.DataFrame({"a": [1, 2, 3]})
    summary = generate_insights(df, [], [], {})
    assert isinstance(summary, InsightSummary)
    assert summary.total_insights == 0
    assert "3 แถว" in summary.executive_summary_th
    assert "พร้อมนำไปวิเคราะห์" in summary.executive_summary_th


def test_quality_and_anomaly_become_insights():
    df = pd.DataFrame({"year": [2567, 2024, 2568], "price": list(range(3))})
    summary = generate_insights(
        df, [_quality_issue()], [_anomaly_issue()], {}
    )
    assert summary.total_insights == 2
    assert summary.critical_count == 1
    assert summary.warning_count == 1
    categories = {i.category for i in summary.insights}
    assert "quality" in categories
    assert "anomaly" in categories


def test_insights_sorted_critical_first():
    df = pd.DataFrame({"a": [1, 2, 3]})
    issues = [
        _quality_issue(check_name="whitespace", severity="info", column="a"),
        _quality_issue(check_name="buddhist_era", severity="critical", column="year"),
        _quality_issue(check_name="thai_numerals", severity="warning", column="id"),
    ]
    summary = generate_insights(df, issues, [], {})
    severities = [i.severity for i in summary.insights]
    assert severities == ["critical", "warning", "info"]


def test_executive_summary_mentions_counts():
    df = pd.DataFrame({"year": [2567, 2024], "price": [1, 2]})
    summary = generate_insights(df, [_quality_issue()], [_anomaly_issue()], {})
    assert "2 แถว × 2 คอลัมน์" in summary.executive_summary_th
    assert "ปัญหาคุณภาพ" in summary.executive_summary_th
    assert "ความผิดปกติ" in summary.executive_summary_th


# ------------------------------------------------------------- structural insights
def test_high_cardinality_text_insight():
    # คอลัมน์ข้อความที่ทุกค่าไม่ซ้ำ (25 แถว) -> insight หมวด text
    df = pd.DataFrame({"note": [f"ข้อความที่ {i}" for i in range(25)]})
    column_types = {"note": ColumnType.THAI_TEXT}
    summary = generate_insights(df, [], [], {}, column_types=column_types)
    text_insights = [i for i in summary.insights if i.category == "text"]
    assert len(text_insights) == 1
    assert "note" in text_insights[0].description_th


def test_comissing_insight_mnar():
    # คอลัมน์ a, b ว่างพร้อมกันในแถวเดียวกัน -> สหสัมพันธ์การขาดหายสูง
    a = [None] * 10 + list(range(10))
    b = [None] * 10 + list(range(100, 110))
    df = pd.DataFrame({"a": a, "b": b})
    summary = generate_insights(df, [], [], {})
    structure = [i for i in summary.insights if i.category == "structure"]
    assert any("พร้อมกัน" in i.title_th for i in structure)
    assert any("MNAR" in i.description_th for i in structure)


def test_cleaning_insight():
    results = [
        CleaningResult(
            operation="normalize_encoding",
            rows_affected=5,
            column="text",
            description_th="แก้ encoding",
        ),
        CleaningResult(
            operation="remove_zero_width_chars",
            rows_affected=3,
            column="text",
            description_th="ลบ zw",
        ),
        CleaningResult(operation="strip_whitespace", rows_affected=0, column="text"),
    ]
    df = pd.DataFrame({"text": ["a", "b"]})
    summary = generate_insights(df, [], [], {}, cleaning_results=results)
    cleaning = [i for i in summary.insights if i.title_th == "สรุปการทำความสะอาดข้อมูล"]
    assert len(cleaning) == 1
    assert "8 เซลล์" in cleaning[0].description_th  # 5 + 3


def test_target_insight_strong_correlation():
    assoc = TargetAssociation(
        column="size",
        target="price",
        association_type="correlation",
        score=0.85,
        p_value=0.001,
        description_th="สหสัมพันธ์สูง",
    )
    df = pd.DataFrame({"size": [1, 2], "price": [10, 20]})
    summary = generate_insights(df, [], [], {}, target_associations=[assoc])
    target = [i for i in summary.insights if i.category == "target"]
    assert len(target) == 1
    assert "size" in target[0].description_th


def test_weak_correlation_no_target_insight():
    assoc = TargetAssociation(
        column="noise",
        target="price",
        association_type="correlation",
        score=0.05,
        p_value=float("nan"),
        description_th="อ่อน",
    )
    df = pd.DataFrame({"noise": [1, 2], "price": [10, 20]})
    summary = generate_insights(df, [], [], {}, target_associations=[assoc])
    assert not [i for i in summary.insights if i.category == "target"]


# ------------------------------------------------------------- integration
def test_integration_with_real_checks():
    df = pd.DataFrame(
        {
            "review": ["อาหารอร่อยมาก", "ร้านนี้ดี​แต่แพง", "12345", "สวัสดีครับ"],
            "year": [2567, 2024, 2568, 2023],
            "price": ["๑๒๐", "150", "๒๐๐", "300"],
        }
    )
    column_types = detect_all(df)
    quality = run_quality_checks(df, column_types)
    anomalies = detect_anomalies(df, column_types, None)
    summary = generate_insights(
        df, quality, anomalies, {}, column_types=column_types
    )
    assert summary.total_insights >= 1
    # ต้องมี insight เกี่ยวกับ พ.ศ. (buddhist_era ถูกตรวจพบ)
    assert any("พุทธศักราช" in i.title_th for i in summary.insights)


# ------------------------------------------------------------- distribution insights
def test_skewness_insight():
    rng = np.random.default_rng(0)
    # การกระจายเบ้ขวาแรง (exponential)
    df = pd.DataFrame({"amount": rng.exponential(2.0, 200)})
    column_types = {"amount": ColumnType.NUMERIC}
    summary = generate_insights(df, [], [], {}, column_types=column_types)
    dist = [i for i in summary.insights if i.category == "distribution"]
    assert any("เบ้" in i.title_th for i in dist)


def test_no_skewness_for_symmetric():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 200)})
    column_types = {"x": ColumnType.NUMERIC}
    summary = generate_insights(df, [], [], {}, column_types=column_types)
    dist = [i for i in summary.insights if i.category == "distribution"]
    assert not any("เบ้มาก" in i.title_th for i in dist)


def test_bimodal_insight():
    rng = np.random.default_rng(1)
    # สองกลุ่มแยกชัดเจน
    values = np.concatenate([rng.normal(0, 1, 150), rng.normal(20, 1, 150)])
    df = pd.DataFrame({"v": values})
    summary = generate_insights(df, [], [], {}, column_types={"v": ColumnType.NUMERIC})
    dist = [i for i in summary.insights if i.category == "distribution"]
    assert any("bimodal" in i.title_th for i in dist)


# ------------------------------------------------------------- correlation insights
def test_correlation_insight_high_pair():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 100)
    df = pd.DataFrame({"a": x, "b": x * 2 + rng.normal(0, 0.01, 100), "c": rng.normal(0, 1, 100)})
    summary = generate_insights(df, [], [], {})
    structure = [i for i in summary.insights if i.title_th == "คู่คอลัมน์สหสัมพันธ์สูง"]
    assert len(structure) >= 1
    assert "a" in structure[0].description_th and "b" in structure[0].description_th


# ------------------------------------------------------------- duplicate rows
def test_duplicate_row_insight():
    df = pd.DataFrame({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]})
    summary = generate_insights(df, [], [], {})
    dup = [i for i in summary.insights if i.title_th == "พบแถวซ้ำ"]
    assert len(dup) == 1
    assert "แถว" in dup[0].description_th


def test_no_duplicate_insight_when_unique():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    summary = generate_insights(df, [], [], {})
    assert not [i for i in summary.insights if i.title_th == "พบแถวซ้ำ"]


# ------------------------------------------------------------- type mismatch
def test_type_mismatch_insight():
    # คอลัมน์หมวดหมู่ที่จริง ๆ เป็นตัวเลข 85% (ปน 'N/A' บ้าง)
    vals = [str(i) for i in range(85)] + ["N/A"] * 15
    df = pd.DataFrame({"code": vals})
    column_types = {"code": ColumnType.CATEGORICAL}
    summary = generate_insights(df, [], [], {}, column_types=column_types)
    mismatch = [i for i in summary.insights if i.title_th == "คอลัมน์เก็บตัวเลขเป็นข้อความ"]
    assert len(mismatch) == 1


# ------------------------------------------------------------- timeseries insights
def test_timeseries_insights_trend():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "sales": [float(i) for i in range(60)],
        }
    )
    ts = analyze_dataframe_timeseries(df, engine="basic")
    summary = generate_insights(df, [], [], {}, timeseries_results=ts)
    tsi = [i for i in summary.insights if i.category == "timeseries"]
    assert any("แนวโน้ม" in i.title_th for i in tsi)
    assert "อนุกรมเวลา" in summary.executive_summary_th


def test_timeseries_insights_gap_warning():
    dates = list(pd.date_range("2024-01-01", periods=20, freq="D"))
    dates += list(pd.date_range("2024-03-01", periods=20, freq="D"))
    df = pd.DataFrame({"date": dates, "v": [float(i) for i in range(40)]})
    ts = analyze_dataframe_timeseries(df, engine="basic")
    summary = generate_insights(df, [], [], {}, timeseries_results=ts)
    tsi = [i for i in summary.insights if i.category == "timeseries"]
    assert any("ช่องว่าง" in i.title_th for i in tsi)


# ------------------------------------------------------------- to_dict
def test_to_dict_structure():
    df = pd.DataFrame({"year": [2567, 2024]})
    summary = generate_insights(df, [_quality_issue()], [], {})
    d = summary.to_dict()
    assert d["total_insights"] == 1
    assert d["critical_count"] == 1
    assert isinstance(d["insights"], list)
    assert "executive_summary_th" in d
    insight = d["insights"][0]
    for key in ("category", "severity", "title_th", "description_th", "recommendation_th"):
        assert key in insight


def test_insight_dataclass_to_dict():
    ins = Insight("quality", "critical", "หัวข้อ", "คำอธิบาย", "คำแนะนำ")
    d = ins.to_dict()
    assert d == {
        "category": "quality",
        "severity": "critical",
        "title_th": "หัวข้อ",
        "description_th": "คำอธิบาย",
        "recommendation_th": "คำแนะนำ",
    }
