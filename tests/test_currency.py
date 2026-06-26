"""ทดสอบ normalize_currency — แปลงคอลัมน์สกุลเงินเป็นตัวเลข (v2.0)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from thaieda.clean import available_operations, normalize_currency


class TestNormalizeCurrency:
    """ทดสอบการแปลงคอลัมน์สกุลเงิน."""

    def test_strips_baht_and_commas(self):
        s = pd.Series(["฿1,200", "฿2,500", "฿3,000"], name="price")
        out, res = normalize_currency(s)
        assert out.tolist() == [1200, 2500, 3000]
        assert res.rows_affected == 3
        assert res.operation == "normalize_currency"

    def test_strips_dollar(self):
        s = pd.Series(["$10.50", "$20.00", "$5.25"], name="usd")
        out, _ = normalize_currency(s)
        assert np.allclose(out.to_numpy(dtype=float), [10.50, 20.00, 5.25])

    def test_strips_euro_pound_yen_rupee(self):
        s = pd.Series(["€5", "£6", "¥7", "₹8"], name="cur")
        out, _ = normalize_currency(s)
        assert out.tolist() == [5, 6, 7, 8]

    def test_strips_currency_words(self):
        s = pd.Series(["1,000 บาท", "2,000 บาท", "3,000 บาท"], name="amt")
        out, res = normalize_currency(s)
        assert out.tolist() == [1000, 2000, 3000]
        assert res.rows_affected == 3

    def test_preserves_nan(self):
        s = pd.Series(["฿100", None, "฿300"], name="p")
        out, _ = normalize_currency(s)
        assert out.iloc[0] == 100
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == 300

    def test_numeric_column_untouched(self):
        s = pd.Series([1.0, 2.0, 3.0], name="already_num")
        out, res = normalize_currency(s)
        assert res.rows_affected == 0
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_detection_threshold_below_10pct(self):
        # มีสัญลักษณ์เพียง 1/20 = 5% (< 10%) → ไม่ถือเป็นคอลัมน์สกุลเงิน
        vals = ["apple"] * 19 + ["$5"]
        s = pd.Series(vals, name="text")
        out, res = normalize_currency(s)
        assert res.rows_affected == 0
        assert out.tolist() == vals

    def test_detection_threshold_above_10pct(self):
        vals = ["$5"] * 3 + ["10"] * 17  # 15% มีสัญลักษณ์ → แปลง
        s = pd.Series(vals, name="mixed")
        out, res = normalize_currency(s)
        assert res.rows_affected == 3
        assert pd.api.types.is_numeric_dtype(out)

    def test_result_is_numeric_dtype(self):
        s = pd.Series(["฿1", "฿2", "฿3", "฿4"], name="p")
        out, _ = normalize_currency(s)
        assert pd.api.types.is_numeric_dtype(out)

    def test_non_convertible_becomes_nan(self):
        s = pd.Series(["฿100", "฿free", "฿300"], name="p")
        out, _ = normalize_currency(s)
        assert out.iloc[0] == 100
        assert pd.isna(out.iloc[1])

    def test_empty_column(self):
        s = pd.Series([None, None], name="p", dtype="object")
        out, res = normalize_currency(s)
        assert res.rows_affected == 0
        assert out.isna().all()

    def test_before_after_examples_recorded(self):
        s = pd.Series(["฿1,200", "฿2,500"], name="p")
        _, res = normalize_currency(s)
        assert len(res.before_examples) > 0
        assert len(res.after_examples) > 0

    def test_registered_in_operations(self):
        assert "currency" in available_operations()


def test_currency_with_negative_and_decimals():
    s = pd.Series(["฿1,234.56", "฿0.99", "฿1,000,000"], name="p")
    out, _ = normalize_currency(s)
    assert np.isclose(out.iloc[0], 1234.56)
    assert np.isclose(out.iloc[1], 0.99)
    assert out.iloc[2] == 1_000_000
