"""Test Cramér's V effect size — v1.8."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaieda.analysis import _cramers_v, analyze_target


class TestCramersV:
    """Cramér's V effect size สำหรับ categorical association."""

    def test_perfect_association(self):
        """2x2 table ที่ทุกค่าอยู่ในแนวทแยง → V ≈ 1.0."""
        observed = np.array([[50, 0], [0, 50]], dtype=float)
        v = _cramers_v(observed)
        assert v > 0.9  # ใกล้ 1.0

    def test_no_association(self):
        """Independent variables → V ≈ 0."""
        observed = np.array([[25, 25], [25, 25]], dtype=float)
        v = _cramers_v(observed)
        assert v < 0.1  # ใกล้ 0

    def test_v_in_range_0_1(self):
        """V ต้องอยู่ในช่วง [0, 1]."""
        np.random.seed(42)
        observed = np.random.randint(10, 50, size=(3, 4)).astype(float)
        v = _cramers_v(observed)
        assert 0.0 <= v <= 1.0

    def test_associate_includes_effect_size(self):
        """analyze_target ต้องคืน effect_size สำหรับ chi_square."""
        np.random.seed(42)
        n = 200
        # สร้าง categorical × categorical ที่มี association
        a = np.random.choice(["X", "Y"], size=n)
        b = np.where(a == "X", np.random.choice(["P", "Q"], size=n, p=[0.8, 0.2]),
                      np.random.choice(["P", "Q"], size=n, p=[0.2, 0.8]))
        df = pd.DataFrame({"a": a, "b": b})
        results = analyze_target(df, "b")
        chi_results = [r for r in results if r.association_type == "chi_square"]
        if chi_results:
            assert chi_results[0].effect_size > 0  # ต้องมี effect_size > 0