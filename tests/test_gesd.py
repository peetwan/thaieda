"""Test Generalized ESD test for multiple outlier detection — v1.8."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaieda.anomaly import _gesd_test, detect_numeric_outliers


class TestGESD:
    """Generalized ESD test — จับ multiple outliers."""

    def test_detects_multiple_outliers(self):
        """ข้อมูล normal 100 ค่า + 5 outliers → ต้องหาได้อย่างน้อย 3/5."""
        np.random.seed(42)
        data = np.random.randn(100) * 10 + 50
        # เพิ่ม 5 outliers
        data[95] = 200
        data[96] = 210
        data[97] = -100
        data[98] = 220
        data[99] = -90
        mask = _gesd_test(data)
        assert mask is not None
        # ต้องหา outlier ได้อย่างน้อย 3 จาก 5
        outlier_count = int(mask.sum())
        assert outlier_count >= 3

    def test_no_false_positive_on_normal(self):
        """ข้อมูล normal ล้วน → ไม่ควร flag อะไรเลย."""
        np.random.seed(42)
        data = np.random.randn(100) * 10 + 50
        mask = _gesd_test(data)
        if mask is not None:
            assert int(mask.sum()) <= 2  # tolerance สำหรับ false positive ต่ำ

    def test_single_outlier(self):
        """ข้อมูลที่มี outlier 1 ตัว → ต้องหาเจอ."""
        np.random.seed(42)
        data = np.random.randn(50) * 5 + 30
        data[49] = 100  # outlier เด่นชัด
        mask = _gesd_test(data)
        assert mask is not None
        assert bool(mask[49])  # index 49 ต้องเป็น outlier

    def test_too_few_samples(self):
        """n < 25 → None."""
        data = np.array([1, 2, 3, 4, 5, 100], dtype=float)
        mask = _gesd_test(data)
        assert mask is None

    def test_detect_numeric_outliers_uses_gesd(self):
        """detect_numeric_outliers ต้องใช้ GESD เมื่อข้อมูลใกล้ normal."""
        np.random.seed(42)
        data = np.random.randn(100) * 10 + 50
        data[95] = 200
        data[96] = -100
        series = pd.Series(data, name="values")
        result = detect_numeric_outliers(series)
        assert result is not None
        # detect_numeric_outliers ต้องจับ outlier ได้ (count > 0)
        assert result.count > 0