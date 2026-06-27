"""Test Spearman rank correlation in insight engine — v1.8."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaieda.detect import ColumnType
from thaieda.insight_engine import _detect_strong_correlations


class TestSpearmanCorrelation:
    """Spearman จับ non-linear monotonic relationships ที่ Pearson พลาด."""

    def test_spearman_catches_nonlinear_monotonic(self):
        """y = x⁵ สำหรับ x ~ N(0,1) → Pearson ≈ 0.49 แต่ Spearman = 1.0."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = x**5  # monotonic แต่ non-linear มาก — Pearson ต่ำ แต่ Spearman สูง
        df = pd.DataFrame({"x": x, "y": y})
        measures = {
            "x": {"type": ColumnType.NUMERIC},
            "y": {"type": ColumnType.NUMERIC},
        }
        candidates = _detect_strong_correlations(df, measures)
        # ต้องเจอ correlation ระหว่าง x และ y
        assert len(candidates) >= 1
        corr_candidate = [c for c in candidates if c["pattern"] == "correlation"]
        assert len(corr_candidate) >= 1
        # method ต้องเป็น spearman เพราะ Pearson จะต่ำกว่า 0.7
        assert corr_candidate[0]["evidence"]["method"] == "spearman"

    def test_pearson_still_used_for_linear(self):
        """ข้อมูล linear → ต้องใช้ Pearson."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = 2 * x + 1 + np.random.randn(200) * 0.1  # strong linear
        df = pd.DataFrame({"x": x, "y": y})
        measures = {
            "x": {"type": ColumnType.NUMERIC},
            "y": {"type": ColumnType.NUMERIC},
        }
        candidates = _detect_strong_correlations(df, measures)
        assert len(candidates) >= 1
        corr = [c for c in candidates if c["pattern"] == "correlation"]
        assert len(corr) >= 1
        assert corr[0]["evidence"]["method"] == "pearson"

    def test_no_correlation_when_independent(self):
        """ข้อมูลอิสระ → ไม่มี correlation candidate."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)  # อิสระกัน
        df = pd.DataFrame({"x": x, "y": y})
        measures = {
            "x": {"type": ColumnType.NUMERIC},
            "y": {"type": ColumnType.NUMERIC},
        }
        candidates = _detect_strong_correlations(df, measures)
        corr = [c for c in candidates if c["pattern"] == "correlation"]
        assert len(corr) == 0
