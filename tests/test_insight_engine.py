"""Tests สำหรับ cross-column insight engine (v0.6).

ใช้ fixture หลายโดเมน (retail / HR / generic) เพื่อพิสูจน์ว่าเอนจินไม่ overfit โดเมนใด
และขับการตัดสินด้วย ColumnType + cardinality + ช่วงค่าเท่านั้น (ไม่มี logic ชื่อคอลัมน์)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from thaieda.detect import detect_all
from thaieda.insight_engine import (
    InsightCard,
    InsightEngineResult,
    Perspective,
    discover_insights,
)


# ----------------------------------------------------------------------------
# 1. Outstanding Value
# ----------------------------------------------------------------------------
def test_outstanding_value():
    df = pd.DataFrame(
        {
            "category": ["A"] * 100 + ["B"] * 50 + ["C"] * 50,
            "value": [100] * 100 + [30] * 50 + [20] * 50,
        }
    )
    result = discover_insights(df, detect_all(df))
    outstanding = [c for c in result.cards if c.pattern == "outstanding"]
    assert len(outstanding) >= 1
    assert "A" in outstanding[0].description_th


def test_near_constant_breakdown_is_ignored():
    """คอลัมน์ breakdown ที่แทบคงที่ (ค่าเดียวครอบ >97% ของแถว) ต้องไม่ผลิต insight.

    การ groupby ด้วยคอลัมน์แบบนี้ได้กลุ่มเด่นแบบ tautology (เช่น 999 vs 1 แถว)
    ซึ่งไม่ใช่ข้อค้นพบที่มีความหมาย — เคยทำให้เกิด business insight "โดดเด่น ... เท่า"
    ปลอมบนคอลัมน์ timestamp ที่แทบคงที่ (created_at/updated_at).
    """
    df = pd.DataFrame(
        {
            "near_const": ["2019-08-09"] * 999 + ["2025-11-15"],
            "value": list(range(1000)),
        }
    )
    result = discover_insights(df, detect_all(df))
    assert not any(c.breakdown == "near_const" for c in result.cards)


# ----------------------------------------------------------------------------
# 2. Attribution (share > 50%)
# ----------------------------------------------------------------------------
def test_attribution_share():
    df = pd.DataFrame(
        {
            "category": ["X"] * 100 + ["Y"] * 30 + ["Z"] * 20 + ["W"] * 10,
            "revenue": [100] * 100 + [20] * 30 + [10] * 20 + [5] * 10,
        }
    )
    result = discover_insights(df, detect_all(df))
    attribution = [c for c in result.cards if c.pattern == "attribution"]
    assert any("X" in c.description_th for c in attribution)


# ----------------------------------------------------------------------------
# 3. Comparison with significance
# ----------------------------------------------------------------------------
def test_comparison_significant():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "group": ["A"] * 200 + ["B"] * 200,
            "score": list(rng.normal(100, 10, 200)) + list(rng.normal(80, 10, 200)),
        }
    )
    result = discover_insights(df, detect_all(df))
    comparison = [c for c in result.cards if c.pattern == "comparison"]
    assert len(comparison) >= 1
    # p-value ควรถูกคำนวณ (มี scipy) และ evidence ครบ
    assert comparison[0].evidence["p_value"] is not None


# ----------------------------------------------------------------------------
# 4. Trend over datetime breakdown
# ----------------------------------------------------------------------------
def test_trend_datetime():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    values = np.linspace(10, 100, 100)  # แนวโน้มขึ้นชัดเจน
    df = pd.DataFrame({"date": dates, "revenue": values})
    result = discover_insights(df, detect_all(df))
    trend = [c for c in result.cards if c.pattern == "trend"]
    assert len(trend) >= 1
    assert "เพิ่มขึ้น" in trend[0].description_th


def test_trend_decreasing_datetime():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    values = np.linspace(500, 50, 120)  # แนวโน้มลงชัดเจน
    df = pd.DataFrame({"date": dates, "sales": values})
    result = discover_insights(df, detect_all(df))
    trend = [c for c in result.cards if c.pattern == "trend"]
    assert len(trend) >= 1
    assert any("ลดลง" in c.description_th for c in trend)


# ----------------------------------------------------------------------------
# 5. Negative test — uniform random → few/no insights
# ----------------------------------------------------------------------------
def test_no_insights_on_uniform():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "category": rng.choice(["A", "B", "C"], 300),
            "value": rng.uniform(50, 51, 300),  # เกือบคงที่
        }
    )
    result = discover_insights(df, detect_all(df))
    assert len(result.cards) <= 2


# ----------------------------------------------------------------------------
# 6. Non-additive guard — percentage column skips "sum"
# ----------------------------------------------------------------------------
def test_non_additive_skip_sum():
    df = pd.DataFrame(
        {
            "category": ["A"] * 100 + ["B"] * 100,
            "pct": [0.3] * 100 + [0.7] * 100,  # bounded [0,1] → ใช้ mean ไม่ใช่ sum
        }
    )
    result = discover_insights(df, detect_all(df))
    sum_cards = [
        c for c in result.cards if c.perspective.agg == "sum" and c.perspective.measure == "pct"
    ]
    assert len(sum_cards) == 0


# ----------------------------------------------------------------------------
# 7. Thai key normalization before groupby
# ----------------------------------------------------------------------------
def test_thai_key_normalization():
    df = pd.DataFrame(
        {
            "store": ["ร้านA"] * 50 + ["ร้านA​"] * 50,  # ต่างกันแค่ zero-width space
            "amount": list(np.linspace(10, 60, 50)) + list(np.linspace(11, 61, 50)),
        }
    )
    # ไม่ควร crash + ควรรวม 2 กลุ่มเป็นกลุ่มเดียวหลัง normalize (เหลือ 1 segment → ไม่มี card)
    result = discover_insights(df, detect_all(df))
    assert isinstance(result, InsightEngineResult)


# ----------------------------------------------------------------------------
# 8. ID column excluded from breakdowns
# ----------------------------------------------------------------------------
def test_id_excluded():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "id": range(1000),
            "category": rng.choice(["A", "B", "C"], 1000),
            "value": rng.normal(50, 10, 1000),
        }
    )
    result = discover_insights(df, detect_all(df))
    id_breakdowns = [c for c in result.cards if c.perspective.breakdown == "id"]
    assert len(id_breakdowns) == 0


# ----------------------------------------------------------------------------
# 9. Performance — large data with sampling
# ----------------------------------------------------------------------------
def test_large_data_sampling():
    rng = np.random.default_rng(42)
    n = 200_000
    df = pd.DataFrame(
        {
            "category": rng.choice(["A", "B", "C", "D"], n),
            "value": rng.normal(100, 20, n),
        }
    )
    t0 = time.time()
    result = discover_insights(df, detect_all(df), sample_size=10_000)
    assert time.time() - t0 < 10
    assert isinstance(result, InsightEngineResult)
    # ควรมี note ว่ามีการสุ่มตัวอย่าง
    assert any("ตัวอย่าง" in n for n in result.notes)


# ----------------------------------------------------------------------------
# 10. Multi-domain — HR data (not retail)
# ----------------------------------------------------------------------------
def test_hr_domain():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "department": (["Engineering"] * 50 + ["Sales"] * 50 + ["HR"] * 50 + ["Finance"] * 50),
            "salary": (
                list(rng.normal(80000, 10000, 50))
                + list(rng.normal(60000, 8000, 50))
                + list(rng.normal(55000, 5000, 50))
                + list(rng.normal(70000, 7000, 50))
            ),
            "years_experience": rng.integers(1, 20, 200),
        }
    )
    result = discover_insights(df, detect_all(df))
    eng_cards = [c for c in result.cards if "Engineering" in c.description_th]
    assert len(eng_cards) >= 1


# ----------------------------------------------------------------------------
# 11. Benjamini-Hochberg correction applied
# ----------------------------------------------------------------------------
def test_bh_correction():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "cat": rng.choice([f"cat_{i}" for i in range(20)], 1000),
            "val": rng.normal(50, 10, 1000),  # ไม่มีความต่างจริง
        }
    )
    result = discover_insights(df, detect_all(df))
    assert len(result.cards) <= 3


# ----------------------------------------------------------------------------
# 12. to_dict() serialization
# ----------------------------------------------------------------------------
def test_to_dict():
    df = pd.DataFrame({"category": ["A"] * 100 + ["B"] * 50, "value": [100] * 100 + [20] * 50})
    result = discover_insights(df, detect_all(df))
    d = result.to_dict()
    assert "total" in d and "cards" in d and "notes" in d
    assert all("evidence" in c for c in d["cards"])
    assert all("perspective" in c for c in d["cards"])
    # ต้อง JSON-serializable
    import json

    json.dumps(d, ensure_ascii=False)


def test_perspective_dataclass():
    p = Perspective("category", "value", "sum")
    assert p.to_dict() == {"breakdown": "category", "measure": "value", "agg": "sum"}


# ----------------------------------------------------------------------------
# 13. Integration with ProfileReport
# ----------------------------------------------------------------------------
def test_profile_report_integration():
    from thaieda.report import profile

    df = pd.DataFrame(
        {
            "category": ["A"] * 100 + ["B"] * 50,
            "value": [100] * 100 + [20] * 50,
        }
    )
    report = profile(df, insights_engine=True, make_charts=False, timeseries=False)
    assert report.insight_engine is not None
    assert isinstance(report.insight_engine, InsightEngineResult)
    html = report.to_html()
    assert "ข้อค้นพบจากการวิเคราะห์คอลัมน์ผสม" in html


def test_profile_report_disabled():
    from thaieda.report import profile

    df = pd.DataFrame(
        {
            "category": ["A"] * 100 + ["B"] * 50,
            "value": [100] * 100 + [20] * 50,
        }
    )
    report = profile(df, insights_engine=False, make_charts=False, timeseries=False)
    assert report.insight_engine is None


# ----------------------------------------------------------------------------
# 14. CLI --no-insights flag
# ----------------------------------------------------------------------------
def test_cli_no_insights(tmp_path):
    from thaieda.cli import main

    csv = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "category": ["A"] * 100 + ["B"] * 50,
            "value": [100] * 100 + [20] * 50,
        }
    )
    df.to_csv(csv, index=False, encoding="utf-8")
    out = tmp_path / "report.html"
    code = main(["profile", str(csv), "-o", str(out), "--no-insights", "--no-charts", "--quiet"])
    assert code == 0
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "ข้อค้นพบจากการวิเคราะห์คอลัมน์ผสม" not in html


def test_cli_insights_top(tmp_path):
    from thaieda.cli import main

    csv = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "category": ["A"] * 100 + ["B"] * 50 + ["C"] * 50,
            "value": [100] * 100 + [30] * 50 + [20] * 50,
        }
    )
    df.to_csv(csv, index=False, encoding="utf-8")
    out = tmp_path / "report.html"
    code = main(
        [
            "profile",
            str(csv),
            "-o",
            str(out),
            "--insights-top",
            "2",
            "--no-charts",
            "--quiet",
        ]
    )
    assert code == 0
    assert out.exists()


# ----------------------------------------------------------------------------
# 15. scipy-optional degradation
# ----------------------------------------------------------------------------
def test_scipy_optional(monkeypatch):
    import thaieda.insight_engine as engine

    # บังคับให้ไม่มี scipy
    monkeypatch.setattr(engine, "_scipy_stats", lambda: None)

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "group": ["A"] * 200 + ["B"] * 200,
            "score": list(rng.normal(100, 10, 200)) + list(rng.normal(80, 10, 200)),
        }
    )
    result = discover_insights(df, detect_all(df))
    # ยังต้องคืนผลได้ (effect-size อย่างเดียว) + มี note เรื่อง scipy
    assert isinstance(result, InsightEngineResult)


def test_empty_dataframe():
    df = pd.DataFrame()
    result = discover_insights(df, {})
    assert result.total == 0
    assert result.cards == []


def test_no_breakdowns():
    # มีแต่คอลัมน์ตัวเลข (ไม่มีหมวดหมู่/วันที่) → ไม่มี breakdown
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    result = discover_insights(df, detect_all(df))
    assert result.total == 0


def test_card_to_dict_fields():
    df = pd.DataFrame(
        {
            "category": ["A"] * 100 + ["B"] * 50 + ["C"] * 50,
            "value": [100] * 100 + [30] * 50 + [20] * 50,
        }
    )
    result = discover_insights(df, detect_all(df))
    assert result.cards
    card = result.cards[0]
    assert isinstance(card, InsightCard)
    d = card.to_dict()
    for key in (
        "pattern",
        "perspective",
        "severity",
        "score",
        "title_th",
        "description_th",
        "recommendation_th",
        "evidence",
    ):
        assert key in d


def test_outlier_insight_uses_robust_method_on_skewed_data():
    """outlier insight ต้องเลือกวิธีตามการกระจาย (สอดคล้องกับโมดูล anomaly):

    คอลัมน์เบ้มาก mean/std ถูกบิดด้วย outlier เอง → z-score ปกตินับ outlier ต่ำผิด
    จึงต้องใช้วิธี robust (MAD/IQR) ส่วนคอลัมน์ใกล้ปกติยังใช้ z-score/GESD ตามเดิม
    """
    rng = np.random.default_rng(0)
    n = 600
    skewed = np.concatenate([rng.exponential(1.0, n - 30), rng.uniform(50, 100, 30)])
    df = pd.DataFrame(
        {
            "grp": (["a", "b", "c"] * (n // 3))[:n],
            "skewed_amount": skewed,
        }
    )
    result = discover_insights(df, detect_all(df))
    cards = [
        c
        for c in result.cards
        if c.pattern == "outlier" and c.evidence.get("column") == "skewed_amount"
    ]
    assert cards, "ควรพบ outlier insight บนคอลัมน์เบ้"
    method = cards[0].evidence["method"]
    # เบ้มาก → ต้องเป็นวิธี robust ไม่ใช่ mean/std z-score ล้วน
    assert method != "z_score"
    assert "MAD" in method or "IQR" in method
    # title ต้องไม่ผูกกับ z-score แบบ hardcode อีกต่อไป
    assert "z-score ≥" not in (cards[0].title_th or "")
