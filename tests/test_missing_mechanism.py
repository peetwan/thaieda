"""Test missing data mechanism detection — v1.8."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaieda.quality import detect_missing_mechanism


class TestMCAR:
    """Missing Completely at Random."""

    def test_mcar_random_missing(self):
        """Missing กระจายสุ่ม → MCAR."""
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
            "d": np.random.randn(n),
        })
        # สุ่มทำ missing ~5% แบบ independent
        for col in df.columns:
            mask = np.random.random(n) < 0.05
            df.loc[mask, col] = np.nan

        result = detect_missing_mechanism(df)
        assert result is not None
        assert result.mechanism == "MCAR"
        assert result.missing_pct > 0

    def test_no_missing_returns_none(self):
        """ไม่มี missing → None."""
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100),
            "c": range(100),
        })
        result = detect_missing_mechanism(df)
        assert result is None


class TestMAR:
    """Missing at Random — missing pattern correlated with observed values."""

    def test_mar_missing_correlated_with_value(self):
        """Missing ใน col A correlated กับค่าใน col B → MAR_likely."""
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "value": np.random.randn(n) * 10 + 50,
            "age": np.random.randint(20, 60, size=n),
            "score": np.random.uniform(0, 100, size=n),
            "income": np.random.randn(n) * 1000 + 30000,
        })
        # age > 40 มักมี missing ใน value (MAR pattern with numeric predictor)
        old_mask = df["age"] > 40
        miss_mask = np.random.random(n) < 0.6
        df.loc[old_mask & miss_mask, "value"] = np.nan

        result = detect_missing_mechanism(df)
        assert result is not None
        # ต้องไม่เป็น MCAR เพราะมี pattern กับ age
        assert result.mechanism != "MCAR"


class TestMNAR:
    """Missing Not at Random."""

    def test_mnar_all_columns_high_missing(self):
        """ทุกคอลัมน์มี missing สูง → MNAR_likely."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
        })
        # ทำ missing 70% ทุกคอลัมน์
        for col in df.columns:
            mask = np.random.random(n) < 0.7
            df.loc[mask, col] = np.nan

        result = detect_missing_mechanism(df)
        assert result is not None
        assert result.mechanism == "MNAR_likely"
        assert result.missing_pct > 50


class TestInsufficientData:
    """ข้อมูลไม่พอ."""

    def test_too_few_rows(self):
        """น้อยกว่า 50 แถว → None."""
        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5],
            "b": [1, np.nan, 3, 4, 5],
            "c": [1, 2, 3, np.nan, 5],
        })
        result = detect_missing_mechanism(df)
        assert result is None

    def test_too_few_cols(self):
        """น้อยกว่า 3 คอลัมน์ → None."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
        })
        df.loc[df.index[:5], "a"] = np.nan
        result = detect_missing_mechanism(df)
        assert result is None

    def test_to_dict_structure(self):
        """to_dict ต้องมี fields ครบ."""
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
            "c": np.random.randn(n),
        })
        mask = np.random.random(n) < 0.05
        df.loc[mask, "a"] = np.nan

        result = detect_missing_mechanism(df)
        assert result is not None
        d = result.to_dict()
        assert "mechanism" in d
        assert "missing_pct" in d
        assert "description_th" in d
        assert "evidence" in d