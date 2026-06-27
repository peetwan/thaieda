"""ทดสอบ insight patterns ใหม่ — Simpson's paradox + target leakage (v2.0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaieda.insight_engine import detect_simpsons_paradox, detect_target_leakage


def _rng(seed=0):
    return np.random.RandomState(seed)


def _simpson_frame(seed=0):
    """สร้างข้อมูลที่มี Simpson's paradox: ภายในกลุ่ม corr ติดลบ, ภาพรวม corr บวก."""
    rng = _rng(seed)
    frames = []
    for i, center in enumerate([0, 5, 10]):
        x = rng.normal(center, 1.0, size=80)
        y = -2.0 * x + 8.0 * center + rng.normal(0, 1.0, size=80)
        frames.append(pd.DataFrame({"x": x, "y": y, "seg": f"s{i}"}))
    return pd.concat(frames, ignore_index=True)


class TestSimpsonsParadox:
    def test_detects_reversal(self):
        df = _simpson_frame()
        res = detect_simpsons_paradox(df, target_col="y", group_col="x", subgroup_col="seg")
        assert len(res) == 1
        assert res[0]["overall_direction"] == "positive"
        assert res[0]["n_reversed"] >= 2

    def test_auto_subgroup_search(self):
        df = _simpson_frame()
        res = detect_simpsons_paradox(df, target_col="y", group_col="x")
        assert len(res) >= 1

    def test_negative_case_no_paradox(self):
        rng = _rng(1)
        df = pd.DataFrame({"x": rng.normal(size=200), "g": rng.choice(["a", "b"], 200)})
        df["y"] = df["x"] * 2 + rng.normal(size=200)
        res = detect_simpsons_paradox(df, target_col="y", group_col="x", subgroup_col="g")
        assert res == []

    def test_categorical_group(self):
        # group เป็นหมวดหมู่ 2 ระดับ; ภาพรวม A>B แต่ในแต่ละ subgroup B>A
        rng = _rng(2)
        rows = []
        # subgroup ที่มีสัดส่วน A/B ต่างกัน → paradox
        for seg, base, na, nb in [("p", 0, 70, 10), ("q", 50, 10, 70)]:
            for _ in range(na):
                rows.append({"grp": "A", "seg": seg, "y": base + 5 + rng.normal()})
            for _ in range(nb):
                rows.append({"grp": "B", "seg": seg, "y": base + 10 + rng.normal()})
        df = pd.DataFrame(rows)
        res = detect_simpsons_paradox(df, target_col="y", group_col="grp", subgroup_col="seg")
        # ภายในแต่ละ seg, B > A; ภาพรวมอาจ A > B เพราะ A กระจุกใน seg ค่าต่ำ
        assert isinstance(res, list)

    def test_keyerror_missing_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(KeyError):
            detect_simpsons_paradox(df, target_col="missing", group_col="a")

    def test_insufficient_data_returns_empty(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "seg": ["a", "b"]})
        res = detect_simpsons_paradox(df, target_col="y", group_col="x", subgroup_col="seg")
        assert res == []

    def test_finding_is_json_serializable(self):
        import json

        df = _simpson_frame()
        res = detect_simpsons_paradox(df, target_col="y", group_col="x", subgroup_col="seg")
        json.dumps(res)  # ต้องไม่ throw


class TestTargetLeakage:
    def test_detects_duplicate(self):
        rng = _rng(3)
        t = rng.normal(size=200)
        df = pd.DataFrame({"dup": t, "rand": rng.normal(size=200), "target": t})
        res = detect_target_leakage(df, "target")
        kinds = {(d["feature"], d["kind"]) for d in res}
        assert ("dup", "duplicate") in kinds

    def test_detects_high_correlation(self):
        rng = _rng(4)
        t = rng.normal(size=200)
        df = pd.DataFrame({"hi": t * 3 + rng.normal(0, 0.001, 200), "target": t})
        res = detect_target_leakage(df, "target")
        assert any(d["kind"] == "high_correlation" for d in res)

    def test_detects_deterministic_mapping(self):
        rng = _rng(5)
        # feature → target เป็นฟังก์ชัน (แต่ละ feature value map ไป target เดียว)
        feat = rng.choice(["a", "b", "c", "d"], size=200)
        mapping = {"a": "win", "b": "lose", "c": "win", "d": "lose"}
        target = pd.Series(feat).map(mapping)
        df = pd.DataFrame({"feat": feat, "noise": rng.normal(size=200), "target": target})
        res = detect_target_leakage(df, "target")
        assert any(d["feature"] == "feat" and d["kind"] == "deterministic_mapping" for d in res)

    def test_detects_near_perfect_separation(self):
        rng = _rng(6)
        df = pd.DataFrame(
            {
                "sep": [1.0] * 100 + [100.0] * 100,
                "noise": rng.normal(size=200),
                "label": ["A"] * 100 + ["B"] * 100,
            }
        )
        res = detect_target_leakage(df, "label")
        assert any(d["feature"] == "sep" and d["kind"] == "near_perfect_separation" for d in res)

    def test_innocent_feature_not_flagged(self):
        rng = _rng(7)
        df = pd.DataFrame({"rand": rng.normal(size=200), "target": rng.normal(size=200)})
        res = detect_target_leakage(df, "target")
        assert all(d["feature"] != "rand" for d in res)

    def test_keyerror_missing_target(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(KeyError):
            detect_target_leakage(df, "nope")

    def test_small_df_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3], "target": [1, 2, 3]})
        assert detect_target_leakage(df, "target") == []

    def test_results_sorted_by_score(self):
        rng = _rng(8)
        t = rng.normal(size=200)
        df = pd.DataFrame(
            {
                "dup": t,
                "hi": t * 2 + rng.normal(0, 0.5, 200),
                "target": t,
            }
        )
        res = detect_target_leakage(df, "target")
        scores = [d["score"] for d in res]
        assert scores == sorted(scores, reverse=True)

    def test_suspected_proxy_has_tier_and_severity(self):
        rng = _rng(9)
        n = 250
        target = rng.randint(0, 2, n)
        feat = target.astype(float) * 1.5 + rng.normal(0, 0.9, n)
        df = pd.DataFrame({"historical_rate": feat, "target": target})
        res = detect_target_leakage(df, "target")
        proxy = [d for d in res if d["kind"] == "suspected_proxy"]
        if proxy:
            assert proxy[0]["tier"] == "warning"
            assert proxy[0]["severity"] == "warning"

    def test_tier_a_has_critical_tier(self):
        rng = _rng(10)
        t = rng.normal(size=200)
        df = pd.DataFrame({"dup": t, "target": t})
        res = detect_target_leakage(df, "target")
        assert res[0]["tier"] == "critical"
        assert res[0]["severity"] == "critical"
