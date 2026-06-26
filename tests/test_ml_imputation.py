"""ทดสอบ ML imputation — handle_missing_values('ml') + ml_impute(df) + guardrails (v2.0)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaieda.clean import handle_missing_values
from thaieda.clean._impute import ml_impute


def _rng(seed=0):
    return np.random.RandomState(seed)


class TestHandleMissingML:
    """ทดสอบ strategy='ml' ระดับ Series เดี่ยว (degrade → median/mode)."""

    def test_ml_numeric_fills_with_median(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], name="x")
        out, res = handle_missing_values(s, "ml")
        assert out.isna().sum() == 0
        assert out.iloc[2] == s.median()
        assert res.rows_affected == 1

    def test_ml_categorical_fills_with_mode(self):
        s = pd.Series(["a", "a", None, "b"], name="c")
        out, res = handle_missing_values(s, "ml")
        assert out.isna().sum() == 0
        assert out.iloc[2] == "a"

    def test_ml_no_missing_noop(self):
        s = pd.Series([1, 2, 3], name="x")
        out, res = handle_missing_values(s, "ml")
        assert res.rows_affected == 0
        assert out.tolist() == [1, 2, 3]

    def test_ml_audit_description(self):
        s = pd.Series([1.0, np.nan, 3.0], name="x")
        _, res = handle_missing_values(s, "ml")
        assert "ML" in res.description_th


class TestMlImpute:
    """ทดสอบ ml_impute(df) — ระบบ 3 ระดับ + guardrails."""

    def test_no_missing_returns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out, results, warnings = ml_impute(df)
        assert results == []
        assert out.equals(df)

    def test_mcar_uses_median(self):
        rng = _rng()
        df = pd.DataFrame(
            {"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.normal(size=100)}
        )
        df.loc[df.sample(20, random_state=1).index, "a"] = np.nan
        out, results, _ = ml_impute(df, mechanism="MCAR")
        assert out["a"].isna().sum() == 0
        assert any("MCAR" in r.description_th for r in results)

    def test_mar_uses_iterative_imputer(self):
        rng = _rng()
        a = rng.normal(size=200)
        b = a * 0.8 + rng.normal(0, 0.3, size=200)
        c = rng.normal(size=200)
        df = pd.DataFrame({"a": a, "b": b, "c": c})
        df.loc[df.sample(40, random_state=2).index, "a"] = np.nan
        out, results, _ = ml_impute(df, mechanism="MAR_likely")
        assert out["a"].isna().sum() == 0
        assert any("IterativeImputer" in r.description_th for r in results)

    def test_mnar_flags_only_no_fill(self):
        rng = _rng()
        df = pd.DataFrame(
            {"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.normal(size=100)}
        )
        df.loc[df.sample(60, random_state=3).index, "a"] = np.nan
        out, results, warnings = ml_impute(df, mechanism="MNAR_likely")
        # MNAR → ไม่เติมค่า
        assert out["a"].isna().sum() == 60
        assert any("MNAR" in w for w in warnings)
        assert all(r.rows_affected == 0 for r in results)

    def test_categorical_uses_mode_fallback(self):
        rng = _rng()
        df = pd.DataFrame(
            {
                "num": rng.normal(size=100),
                "cat": rng.choice(["x", "y", "z"], size=100),
                "num2": rng.normal(size=100),
            }
        )
        df.loc[df.sample(20, random_state=4).index, "cat"] = None
        out, results, _ = ml_impute(df, mechanism="MAR_likely")
        assert out["cat"].isna().sum() == 0
        assert any(r.column == "cat" for r in results)

    def test_guardrail_caps_predictors(self):
        rng = _rng()
        # 30 คอลัมน์ตัวเลข > เพดาน 20
        data = {f"f{i}": rng.normal(size=80) for i in range(30)}
        df = pd.DataFrame(data)
        df.loc[df.sample(15, random_state=5).index, "f0"] = np.nan
        out, results, _ = ml_impute(df, mechanism="MAR_likely", max_predictors=20)
        assert out["f0"].isna().sum() == 0  # ยังเติมได้แม้จำกัด predictor

    def test_guardrail_subsamples_rows(self):
        rng = _rng()
        df = pd.DataFrame(
            {"a": rng.normal(size=500), "b": rng.normal(size=500), "c": rng.normal(size=500)}
        )
        df.loc[df.sample(100, random_state=6).index, "a"] = np.nan
        out, results, _ = ml_impute(df, mechanism="MAR_likely", max_train_rows=100)
        assert out["a"].isna().sum() == 0
        assert any("subsample" in r.description_th for r in results)

    def test_length_preserved(self):
        rng = _rng()
        df = pd.DataFrame(
            {"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.normal(size=100)}
        )
        df.loc[df.sample(20, random_state=7).index, "a"] = np.nan
        out, _, _ = ml_impute(df, mechanism="MAR_likely")
        assert len(out) == len(df)

    def test_audit_records_rows_affected(self):
        rng = _rng()
        df = pd.DataFrame(
            {"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.normal(size=100)}
        )
        df.loc[df.sample(25, random_state=8).index, "a"] = np.nan
        _, results, _ = ml_impute(df, mechanism="MAR_likely")
        a_results = [r for r in results if r.column == "a"]
        assert a_results and a_results[0].rows_affected == 25

    def test_auto_mechanism_detection(self):
        # ข้อมูลพอสำหรับ detect_missing_mechanism (>=50 rows, >=3 cols)
        rng = _rng()
        df = pd.DataFrame(
            {"a": rng.normal(size=120), "b": rng.normal(size=120), "c": rng.normal(size=120)}
        )
        df.loc[df.sample(20, random_state=9).index, "a"] = np.nan
        out, results, _ = ml_impute(df)  # mechanism=None → auto
        assert out["a"].isna().sum() == 0

    def test_original_not_mutated(self):
        rng = _rng()
        df = pd.DataFrame(
            {"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.normal(size=100)}
        )
        df.loc[df.sample(20, random_state=10).index, "a"] = np.nan
        na_before = int(df["a"].isna().sum())
        ml_impute(df, mechanism="MAR_likely")
        assert int(df["a"].isna().sum()) == na_before

    def test_clean_integration_ml(self):
        import thaieda

        rng = _rng()
        df = pd.DataFrame(
            {"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.normal(size=100)}
        )
        df.loc[df.sample(20, random_state=11).index, "a"] = np.nan
        out, report = thaieda.clean(
            df, handle_missing="ml", remove_duplicates=False, downcast=False
        )
        assert out["a"].isna().sum() == 0
