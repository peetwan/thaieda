"""Test distribution fitting + KS test — v1.8."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaieda.quality import fit_distributions


class TestNormalDistribution:
    """ข้อมูลที่เป็น normal distribution."""

    def test_normal_data_fits_normal(self):
        """ข้อมูล normal → best_fit ต้องเป็น normal."""
        np.random.seed(42)
        data = np.random.randn(500) * 10 + 50
        series = pd.Series(data, name="values")
        result = fit_distributions(series, "values")
        assert result is not None
        assert result.best_fit == "normal"
        assert result.p_value > 0.05  # fit ดี
        assert "mean" in result.parameters
        assert "std" in result.parameters

    def test_normal_parameters_close(self):
        """Parameters ที่ fit ต้องใกล้ค่าจริง."""
        np.random.seed(42)
        mu, sigma = 100, 15
        data = np.random.randn(1000) * sigma + mu
        series = pd.Series(data, name="score")
        result = fit_distributions(series, "score")
        assert result is not None
        assert result.best_fit == "normal"
        assert abs(result.parameters["mean"] - mu) < 5
        assert abs(result.parameters["std"] - sigma) < 3


class TestUniformDistribution:
    """ข้อมูลที่เป็น uniform distribution."""

    def test_uniform_data_fits_uniform(self):
        """ข้อมูล uniform → best_fit ต้องเป็น uniform."""
        np.random.seed(42)
        data = np.random.uniform(0, 100, 500)
        series = pd.Series(data, name="uniform_data")
        result = fit_distributions(series, "uniform_data")
        assert result is not None
        assert result.best_fit == "uniform"
        assert result.p_value > 0.05


class TestExponentialDistribution:
    """ข้อมูลที่เป็น exponential distribution."""

    def test_exponential_data_fits_exponential(self):
        """ข้อมูล exponential → best_fit ต้องเป็น exponential."""
        np.random.seed(42)
        data = np.random.exponential(scale=50, size=500)
        series = pd.Series(data, name="exp_data")
        result = fit_distributions(series, "exp_data")
        assert result is not None
        assert result.best_fit == "exponential"
        assert result.p_value > 0.05


class TestEdgeCases:
    """กรณีพิเศษ."""

    def test_too_few_samples(self):
        """น้อยกว่า 30 ค่า → None."""
        series = pd.Series([1, 2, 3, 4, 5], name="small")
        result = fit_distributions(series, "small")
        assert result is None

    def test_constant_column(self):
        """ค่าคงที่ → None (std = 0)."""
        series = pd.Series([5.0] * 100, name="constant")
        result = fit_distributions(series, "constant")
        assert result is None

    def test_to_dict_structure(self):
        """to_dict ต้องมี fields ครบ."""
        np.random.seed(42)
        data = np.random.randn(200)
        series = pd.Series(data, name="test_col")
        result = fit_distributions(series, "test_col")
        assert result is not None
        d = result.to_dict()
        assert "column" in d
        assert "best_fit" in d
        assert "ks_statistic" in d
        assert "p_value" in d
        assert "parameters" in d
        assert "all_fits" in d
        assert len(d["all_fits"]) >= 1  # ต้องมีอย่างน้อย 1 distribution

    def test_all_fits_contain_multiple(self):
        """ข้อมูล normal ต้อง fit ได้อย่างน้อย normal + uniform."""
        np.random.seed(42)
        data = np.abs(np.random.randn(500))  # ค่าบวก → lognormal/exponential ลองได้
        series = pd.Series(data, name="positive")
        result = fit_distributions(series, "positive")
        assert result is not None
        # ต้องมีอย่างน้อย 3 distributions (normal, lognormal, exponential, uniform)
        assert len(result.all_fits) >= 3

    def test_description_th_present(self):
        """description_th ต้องมีค่า."""
        np.random.seed(42)
        data = np.random.randn(200)
        series = pd.Series(data, name="col_x")
        result = fit_distributions(series, "col_x")
        assert result is not None
        assert result.description_th  # ไม่ว่าง
        assert "col_x" in result.description_th or "การกระจาย" in result.description_th
