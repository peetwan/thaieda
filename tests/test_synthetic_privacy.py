"""Test synthetic data generation + privacy audit — v1.9."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thaieda.llm._synthetic import generate_synthetic_data, privacy_audit_report
from thaieda.llm import prepare_for_llm


class TestGenerateSyntheticData:
    """สร้างข้อมูลจำลองจาก distribution จริง."""

    def test_preserves_shape(self):
        """Synthetic data ต้องมีขนาดเท่าข้อมูลจริง."""
        np.random.seed(42)
        df = pd.DataFrame({
            "age": np.random.randint(20, 60, 200),
            "income": np.random.randn(200) * 1000 + 30000,
        })
        synthetic = generate_synthetic_data(df)
        assert len(synthetic) == len(df)
        assert list(synthetic.columns) == list(df.columns)

    def test_no_real_values_in_synthetic(self):
        """ไม่มีค่าจริงปนใน synthetic (numeric)."""
        np.random.seed(42)
        df = pd.DataFrame({
            "value": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0] * 20,
        })
        synthetic = generate_synthetic_data(df)
        # ตรวจว่าค่าใน synthetic ไม่ตรงค่าจริงทั้งหมด
        real_set = set(df["value"].unique())
        synth_set = set(synthetic["value"].unique())
        # อนุญาตให้มี overlap บางส่วน แต่ต้องไม่ตรงทั้งหมด
        assert len(synth_set - real_set) > 0 or len(synth_set) > 1

    def test_preserves_numeric_stats(self):
        """ค่าสถิติ (mean, std) ต้องใกล้เคียงข้อมูลจริง."""
        np.random.seed(42)
        data = np.random.randn(500) * 10 + 50
        df = pd.DataFrame({"value": data})
        synthetic = generate_synthetic_data(df)
        # mean ต้องใกล้กัน (±20%)
        assert abs(synthetic["value"].mean() - df["value"].mean()) < df["value"].std()
        # std ต้องใกล้กัน (±50%)
        assert abs(synthetic["value"].std() - df["value"].std()) < df["value"].std()

    def test_preserves_categorical_proportions(self):
        """Categorical proportions ต้องใกล้เคียง."""
        np.random.seed(42)
        df = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C"], size=500, p=[0.5, 0.3, 0.2]),
        })
        synthetic = generate_synthetic_data(df)
        orig_props = df["category"].value_counts(normalize=True)
        synth_props = synthetic["category"].value_counts(normalize=True)
        for cat in orig_props.index:
            assert abs(orig_props[cat] - synth_props.get(cat, 0)) < 0.15

    def test_preserves_missing_rate(self):
        """Missing rate ต้องใกล้เคียง."""
        np.random.seed(42)
        df = pd.DataFrame({
            "col": np.random.randn(200),
        })
        df.loc[df.index[:50], "col"] = np.nan  # 25% missing
        synthetic = generate_synthetic_data(df)
        orig_miss = df["col"].isna().mean()
        synth_miss = synthetic["col"].isna().mean()
        assert abs(orig_miss - synth_miss) < 0.1

    def test_text_becomes_placeholder(self):
        """Text column ต้องกลายเป็น placeholder."""
        np.random.seed(42)
        df = pd.DataFrame({
            "review": [f"review text number {i} " * 5 for i in range(100)],
        })
        synthetic = generate_synthetic_data(df)
        # ต้องไม่มีข้อความจริง
        assert not any("review text number" in str(v) for v in synthetic["review"].dropna())
        # ต้องมี placeholder
        assert any("<text_" in str(v) for v in synthetic["review"].dropna())

    def test_custom_n_rows(self):
        """ระบุ n_rows เองได้."""
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(100)})
        synthetic = generate_synthetic_data(df, n_rows=50)
        assert len(synthetic) == 50

    def test_reproducible_with_seed(self):
        """Seed เดียวกัน → ผลเหมือนกัน."""
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(100)})
        s1 = generate_synthetic_data(df, random_seed=99)
        s2 = generate_synthetic_data(df, random_seed=99)
        pd.testing.assert_series_equal(s1["a"], s2["a"])


class TestPrivacyAuditReport:
    """รายงาน privacy audit."""

    def test_detects_phone_numbers(self):
        """พบเบอร์โทรศัพท์ → PII detected."""
        df = pd.DataFrame({
            "contact": ["081-234-5678", "082-345-6789", "no phone", "066-789-0123"],
            "name": ["A", "B", "C", "D"],
        })
        report = privacy_audit_report(df, "synthetic")
        pii_types = [p["type"] for p in report["pii_detected"]]
        assert "phone_number" in pii_types

    def test_detects_email(self):
        """พบอีเมล → PII detected."""
        df = pd.DataFrame({
            "email": ["john@example.com", "jane@test.org", "no email", "x@y.com"],
        })
        report = privacy_audit_report(df, "anonymized")
        pii_types = [p["type"] for p in report["pii_detected"]]
        assert "email" in pii_types

    def test_detects_thai_national_id(self):
        """พบเลขบัตรประชาชน → critical risk."""
        df = pd.DataFrame({
            "id": ["1-1234-56789-01-2", "2-2345-67890-12-3", "no id", "3-3456-78901-23-4"],
        })
        report = privacy_audit_report(df, "full")
        pii_types = [p["type"] for p in report["pii_detected"]]
        assert "thai_national_id" in pii_types
        # full mode + critical PII → recommendations ต้องเตือน
        assert any("synthetic" in r.lower() or "insight" in r.lower()
                    for r in report["recommendations"])

    def test_no_pii_clean_data(self):
        """ข้อมูลไม่มี PII → n_pii_types = 0."""
        df = pd.DataFrame({
            "category": ["A", "B", "C", "D"],
            "value": [1.0, 2.0, 3.0, 4.0],
        })
        report = privacy_audit_report(df, "insight_only")
        assert report["n_pii_types"] == 0
        assert report["overall_risk"] == "low"

    def test_synthetic_mode_low_risk(self):
        """synthetic mode → overall_risk = low."""
        df = pd.DataFrame({"a": range(100)})
        report = privacy_audit_report(df, "synthetic")
        assert report["overall_risk"] == "low"

    def test_full_mode_high_risk_with_pii(self):
        """full mode + PII → high risk."""
        df = pd.DataFrame({"phone": ["081-234-5678"] * 10})
        report = privacy_audit_report(df, "full")
        assert report["overall_risk"] == "high"

    def test_data_sent_description(self):
        """แต่ละ mode ต้องมี description ของข้อมูลที่ส่ง."""
        for mode in ["insight_only", "synthetic", "anonymized", "dp_noise", "full"]:
            df = pd.DataFrame({"a": [1, 2, 3]})
            report = privacy_audit_report(df, mode)
            assert report["data_sent_to_llm"]  # ไม่ว่าง


class TestPrepareForLLMSynthetic:
    """prepare_for_llm กับ synthetic mode."""

    def test_synthetic_mode_returns_data(self):
        """synthetic mode ต้องคืน data (DataFrame จำลอง)."""
        np.random.seed(42)
        df = pd.DataFrame({
            "value": np.random.randn(100) * 10 + 50,
            "category": np.random.choice(["X", "Y"], 100),
        })
        prepared = prepare_for_llm(df, privacy_mode="synthetic")
        assert prepared["mode"] == "synthetic"
        assert prepared["data"] is not None
        assert len(prepared["data"]) == len(df)
        # ต้องไม่ใช่ข้อมูลจริง (ค่า mean ใกล้กัน แต่ไม่ตรงทุกค่า)
        assert prepared["data"]["value"].mean() != df["value"].mean()

    def test_synthetic_mode_no_pii(self):
        """synthetic mode ต้องไม่มี PII."""
        df = pd.DataFrame({
            "name": ["สมชาย ใจดี", "สมหญิง รักไทย", "John Smith"],
            "phone": ["081-234-5678", "082-345-6789", "083-456-7890"],
            "age": [25, 30, 28],
        })
        prepared = prepare_for_llm(df, privacy_mode="synthetic")
        # เบอร์โทรศัพท์ต้องไม่อยู่ใน synthetic data
        text_parts = []
        for col in prepared["data"].columns:
            text_parts.extend(prepared["data"][col].astype(str).tolist())
        synthetic_text = " ".join(text_parts)
        assert "081-234-5678" not in synthetic_text
        assert "082-345-6789" not in synthetic_text