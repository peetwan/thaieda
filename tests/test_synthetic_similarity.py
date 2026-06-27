"""Test synthetic data statistical similarity — v1.9.3.

ทดสอบฟังก์ชันใหม่:
  - _detect_spike: ตรวจจับ zero-inflated / spike-at-value
  - _gen_spike_mixture: สร้างข้อมูลแบบ spike + tail
  - _gen_quantile_sample: empirical quantile sampling + noise
  - _fit_distributions: fit 6 distributions (normal/lognormal/expon/gamma/weibull/uniform)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaieda.llm._synthetic import (
    _detect_spike,
    _fit_distributions,
    _gen_quantile_sample,
    _gen_spike_mixture,
    generate_synthetic_data,
)


class TestDetectSpike:
    """ตรวจจับ spike (zero-inflated)."""

    def test_detects_zero_spike(self):
        """90% เป็น 0 → ตรวจจับได้."""
        np.random.seed(42)
        vals = np.array([0] * 180 + list(np.random.lognormal(3, 1, 20)))
        spike_val, spike_rate, tail = _detect_spike(vals)
        assert spike_val == 0.0
        assert spike_rate > 0.8
        assert len(tail) == 20

    def test_no_spike_normal(self):
        """ข้อมูล normal ไม่มี spike."""
        np.random.seed(42)
        vals = np.random.randn(200)
        spike_val, spike_rate, tail = _detect_spike(vals)
        assert spike_val is None
        assert spike_rate == 0.0

    def test_spike_at_nonzero(self):
        """spike ที่ค่าอื่นที่ไม่ใช่ 0."""
        vals = np.array([99] * 150 + [1, 2, 3, 4, 5, 10, 20, 30, 40, 50])
        spike_val, spike_rate, tail = _detect_spike(vals)
        assert spike_val == 99.0
        assert spike_rate > 0.9

    def test_low_spike_rate_not_detected(self):
        """spike < 25% → ไม่นับเป็น spike."""
        vals = np.array([0] * 20 + list(range(1, 81)))
        spike_val, spike_rate, tail = _detect_spike(vals)
        assert spike_val is None


class TestSpikeMixture:
    """spike + tail mixture."""

    def test_preserves_spike_rate(self):
        """spike_rate ใกล้เคียงข้อมูลจริง."""
        np.random.seed(42)
        vals = np.array([0] * 180 + list(np.random.lognormal(3, 1, 20)))
        spike_val, spike_rate, tail = _detect_spike(vals)
        result = _gen_spike_mixture(spike_val, spike_rate, tail, 1000, vals)

        # นับจำนวน 0 ในผลลัพธ์
        zero_rate = (result == 0).mean()
        assert abs(zero_rate - spike_rate) < 0.1  # ±10%

    def test_tail_mean_close(self):
        """mean ของ tail ใกล้เคียง tail จริง."""
        np.random.seed(42)
        tail_vals = np.random.lognormal(3, 1, 100)
        vals = np.array([0] * 300 + list(tail_vals))
        spike_val, spike_rate, tail = _detect_spike(vals)
        result = _gen_spike_mixture(spike_val, spike_rate, tail, 2000, vals)

        # เปรียบเทียบ mean ของส่วนที่ไม่ใช่ 0
        nonzero_result = result[result != 0]
        real_tail_mean = tail_vals.mean()
        synth_tail_mean = nonzero_result.mean()
        diff_pct = abs(real_tail_mean - synth_tail_mean) / real_tail_mean * 100
        assert diff_pct < 30  # tolerance สำหรับ lognormal


class TestQuantileSample:
    """quantile sampling + noise."""

    def test_preserves_mean(self):
        """mean ใกล้เคียง."""
        np.random.seed(42)
        vals = np.random.randn(500) * 10 + 50
        result = _gen_quantile_sample(vals, 1000)
        diff_pct = abs(vals.mean() - result.mean()) / abs(vals.mean()) * 100
        assert diff_pct < 10

    def test_preserves_std(self):
        """std ใกล้เคียง."""
        np.random.seed(42)
        vals = np.random.randn(500) * 10 + 50
        result = _gen_quantile_sample(vals, 1000)
        diff_pct = abs(vals.std() - result.std()) / vals.std() * 100
        assert diff_pct < 15

    def test_no_exact_values(self):
        """ไม่มีค่าตรงกับข้อมูลจริงทั้งหมด (privacy) — ใช้ข้อมูลที่มี std พอ."""
        np.random.seed(42)
        vals = np.random.randn(200) * 10 + 50  # std=10 → noise 0.1 พอให้ต่างจากค่าจริง
        result = _gen_quantile_sample(vals, 200)
        exact_matches = sum(1 for v in result if float(v) in set(vals.tolist()))
        # อนุญาตให้มีบางค่าตรงได้ แต่ต้องไม่เกิน 60%
        assert exact_matches < 120

    def test_non_negative_stays_non_negative(self):
        """ข้อมูลไม่ติดลบ → ผลลัพธ์ไม่ติดลบ."""
        np.random.seed(42)
        vals = np.abs(np.random.randn(500))
        result = _gen_quantile_sample(vals, 200)
        assert (result >= 0).all()


class TestFitDistributions:
    """fit 6 distributions."""

    def test_fits_normal(self):
        """ข้อมูล normal → normal fit ดีที่สุด."""
        from scipy import stats as st

        np.random.seed(42)
        vals = np.random.randn(200)
        candidates = _fit_distributions(vals, st)
        assert len(candidates) >= 1
        best = max(candidates, key=lambda x: x[2])
        assert best[0] == "normal"

    def test_fits_gamma(self):
        """ข้อมูล gamma → gamma อยู่ใน candidates."""
        from scipy import stats as st

        np.random.seed(42)
        vals = st.gamma.rvs(2, size=200)
        candidates = _fit_distributions(vals, st)
        dist_names = [c[0] for c in candidates]
        assert "gamma" in dist_names

    def test_handles_negative_values(self):
        """ข้อมูลติดลบ → ไม่ fit gamma/weibull/lognormal."""
        from scipy import stats as st

        np.random.seed(42)
        vals = np.random.randn(100)  # มีค่าติดลบ
        candidates = _fit_distributions(vals, st)
        dist_names = [c[0] for c in candidates]
        assert "gamma" not in dist_names
        assert "weibull" not in dist_names
        assert "lognormal" not in dist_names


class TestEndToEndSkewed:
    """end-to-end กับข้อมูลเฉียงจริง."""

    def test_zero_inflated_column(self):
        """คอลัมน์ zero-inflated → mean ใกล้เคียง."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "capital_gain": [0] * 180 + list(np.random.lognormal(5, 2, 20)),
            }
        )
        synth = generate_synthetic_data(df, random_seed=2)
        real_mean = df["capital_gain"].mean()
        synth_mean = synth["capital_gain"].mean()
        diff_pct = abs(real_mean - synth_mean) / abs(real_mean) * 100
        # ก่อนปรับ: 226% หลังปรับ: ควรต่ำกว่า 50%
        assert diff_pct < 50

    def test_heavily_skewed_column(self):
        """คอลัมน์เฉียงมาก → mean ใกล้เคียง."""
        np.random.seed(42)
        # count data แบบเฉียงขวา
        vals = np.random.negative_binomial(1, 0.1, 500)
        df = pd.DataFrame({"counts": vals})
        synth = generate_synthetic_data(df, random_seed=42)
        real_mean = df["counts"].mean()
        synth_mean = synth["counts"].mean()
        diff_pct = abs(real_mean - synth_mean) / real_mean * 100
        # ก่อนปรับ: 59% หลังปรับ: ควรต่ำกว่า 15%
        assert diff_pct < 15

    def test_bounded_count_column(self):
        """ข้อมูล count จำกัด (เช่น SibSp 0-8) → mean ใกล้เคียง."""
        np.random.seed(42)
        vals = np.random.choice([0, 0, 0, 1, 1, 2, 3, 4, 0, 1], 200)
        df = pd.DataFrame({"sibsp": vals})
        synth = generate_synthetic_data(df, random_seed=42)
        real_mean = df["sibsp"].mean()
        synth_mean = synth["sibsp"].mean()
        diff_pct = abs(real_mean - synth_mean) / real_mean * 100
        # ก่อนปรับ: 38% หลังปรับ: ควรต่ำกว่า 20%
        assert diff_pct < 20
