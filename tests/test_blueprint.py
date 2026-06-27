"""Tests for EDA Blueprint features — ml_tabular, leakage, lang=en, grouped findings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaieda.report import ProfileReport, _detect_data_type, profile


def _ml_df():
    rng = np.random.default_rng(42)
    n = 200
    user_ctr = rng.uniform(0, 0.3, n)
    clicked = (user_ctr + rng.normal(0, 0.05, n) > 0.12).astype(int)
    # Strong leakage proxy — nearly deterministic from target
    historical_user_ctr = clicked.astype(float) * 0.5 + rng.normal(0, 0.001, n)
    return pd.DataFrame(
        {
            "ad_id": [f"ad_{i}" for i in range(n)],
            "user_id": [f"u_{i % 50}" for i in range(n)],
            "impression_hour": rng.integers(0, 24, n),
            "campaign_type": rng.choice(["A", "B", "C"], n),
            "historical_user_ctr": historical_user_ctr,
            "user_ctr": user_ctr,
            "clicked": clicked,
        }
    )


def test_detect_ml_tabular_with_target():
    dt = _detect_data_type(_ml_df(), target_column="clicked", lang="en")
    assert dt["key"] == "ml_tabular"


def test_leakage_wired_to_report():
    r = profile(
        _ml_df(),
        target_column="clicked",
        clean=False,
        make_charts=False,
        timeseries=False,
        insights_engine=False,
    )
    assert len(r._leakage_findings) >= 1
    features = {f["feature"] for f in r._leakage_findings}
    assert "historical_user_ctr" in features or "user_ctr" in features
    d = r.to_dict()
    assert "leakage" in d["target_analysis"]
    assert d["target_analysis"]["baseline"]["is_binary"] is True


def test_en_report_content():
    r = ProfileReport(
        _ml_df(), lang="en", target_column="clicked", make_charts=False, timeseries=False
    )
    html = r.to_html()
    assert "Executive Summary" in html
    assert "Modeling Blueprint" in html


def test_th_blueprint_report_content():
    r = profile(
        _ml_df(),
        lang="th",
        target_column="clicked",
        report_mode="blueprint",
        clean=True,
        handle_missing="flag",
        make_charts=False,
        timeseries=False,
        insights_engine=False,
    )
    html = r.to_html()
    assert 'lang="th"' in html
    assert "บทสรุปผู้บริหาร" in html
    assert "แผนสร้างโมเดล" in html
    assert "สิ่งที่ควรทำก่อน" in html
    assert "Executive Summary" not in html
    assert ">Modeling Blueprint<" not in html
    assert "Overview</button>" not in html
    assert "ภาพรวม</button>" in html


def test_grouped_placeholder_findings():
    df = pd.DataFrame(
        {
            "a": ["-", "1", "-", "2"],
            "b": ["-", "x", "-", "y"],
            "c": ["-", "p", "-", "q"],
            "d": ["-", "m", "-", "n"],
            "clicked": [0, 1, 0, 1],
        }
    )
    r = profile(df, target_column="clicked", clean=False, make_charts=False, timeseries=False)
    findings = r._top_findings()
    grouped = [f for f in findings if f.get("group_count", 0) > 1]
    assert grouped


def test_report_mode_blueprint():
    r = profile(
        _ml_df(),
        lang="en",
        target_column="clicked",
        report_mode="blueprint",
        clean=True,
        handle_missing="flag",
        make_charts=False,
        timeseries=False,
        insights_engine=False,
    )
    assert r.report_mode == "blueprint"
    assert r.to_dict()["report_mode"] == "blueprint"


def test_raw_missing_preserved_with_flag():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"], "clicked": [0, 1, 0]})
    r = profile(df, clean=True, handle_missing="flag", make_charts=False, timeseries=False)
    assert r.overview.get("raw_missing_pct", 0) > 0
    assert r.df["a"].isna().any()
    assert r.df["b"].isna().any()


def test_binary_target_baseline_uses_positive_class():
    """Bool/0-1 targets: positive rate = P(True/1), not majority-class share."""
    from thaieda.report import _build_target_baseline

    df = pd.DataFrame({"clicked": [False, False, True, True, True]})
    bl = _build_target_baseline(df, "clicked", lang="en")
    assert bl is not None
    assert bl["positive_rate_pct"] == 60.0


def test_handle_missing_flag_does_not_fill():
    from thaieda.clean import handle_missing_values

    s = pd.Series(["a", None, "c"], name="col")
    out, result = handle_missing_values(s, "flag")
    assert result.rows_affected == 1
    assert pd.isna(out.iloc[1])


def test_senior_citizen_not_flagged_as_thai_id():
    """Binary 0/1 column named SeniorCitizen must not trigger Thai ID check."""
    from thaieda.quality._thai_id import check_thai_id

    series = pd.Series([0, 1, 0, 1, 0, 1] * 20, name="SeniorCitizen")
    assert check_thai_id(series, "SeniorCitizen", is_id_type=False) is None


def test_ticket_strings_not_flagged_as_mixed_dates():
    """High-cardinality ticket IDs must not be flagged as mixed date formats."""
    from thaieda.anomaly import detect_column_anomalies
    from thaieda.detect import detect_all

    tickets = [f"CA-2017-{100000 + i}" for i in range(100)]
    df = pd.DataFrame({"Ticket": tickets})
    ctypes = detect_all(df)
    issues = detect_column_anomalies(df, ctypes)
    mixed = [i for i in issues if i.check_name == "mixed_date_formats"]
    assert mixed == []


def test_proxy_leakage_tier_b_with_name_hint():
    """Proxy column with name hint + moderate correlation should be Tier B."""
    from thaieda.insight_engine._leakage import detect_target_leakage

    rng = np.random.default_rng(0)
    n = 300
    target = rng.integers(0, 2, n)
    # corr ~0.7 with target — below Tier A threshold
    noise = rng.normal(0, 1, n)
    historical_user_ctr = target.astype(float) * 2.0 + noise * 0.8
    df = pd.DataFrame(
        {"historical_user_ctr": historical_user_ctr, "noise_feat": noise, "clicked": target}
    )
    res = detect_target_leakage(df, "clicked")
    proxy = [
        d for d in res if d["feature"] == "historical_user_ctr" and d["kind"] == "suspected_proxy"
    ]
    assert proxy
    assert proxy[0]["tier"] == "warning"
    assert all(d["feature"] != "noise_feat" for d in res if d["kind"] == "suspected_proxy")


def test_verdict_differs_clean_vs_dirty():
    """Executive verdict should reflect dataset health signals."""
    clean_df = pd.DataFrame(
        {"age": np.arange(50), "score": np.linspace(0, 1, 50), "label": [0, 1] * 25}
    )
    dirty_df = pd.DataFrame(
        {
            "a": [np.nan] * 40 + list(range(10)),
            "b": [np.nan] * 40 + list(range(10)),
            "label": [0, 1] * 25,
        }
    )
    r_clean = profile(
        clean_df, target_column="label", make_charts=False, timeseries=False, insights_engine=False
    )
    r_dirty = profile(
        dirty_df, target_column="label", make_charts=False, timeseries=False, insights_engine=False
    )
    s_clean = r_clean._build_report_summary()
    s_dirty = r_dirty._build_report_summary()
    assert s_clean["status"] == "good"
    assert s_dirty["status"] in {"critical", "warning"}
    assert s_clean["verdict"] != s_dirty["verdict"]
