"""Test export_synthetic_data — v1.9.1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thaieda.llm import export_synthetic_data


class TestExportSyntheticData:
    """Export synthetic data เป็นไฟล์."""

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "id": range(200),
                "value": np.random.randn(200) * 10 + 50,
                "category": np.random.choice(["A", "B", "C"], 200),
            }
        )

    def test_export_csv(self, sample_df, tmp_path):
        """export เป็น CSV."""
        path = tmp_path / "synthetic.csv"
        result = export_synthetic_data(sample_df, str(path), include_audit=False)
        assert Path(result["output_path"]).exists()
        assert result["n_rows"] == 200
        assert result["n_cols"] == 3
        # อ่านกลับ
        read = pd.read_csv(path)
        assert len(read) == 200

    def test_export_xlsx(self, sample_df, tmp_path):
        """export เป็น XLSX."""
        path = tmp_path / "synthetic.xlsx"
        result = export_synthetic_data(sample_df, str(path))
        assert Path(result["output_path"]).exists()
        read = pd.read_excel(path)
        assert len(read) == 200

    def test_export_json(self, sample_df, tmp_path):
        """export เป็น JSON."""
        path = tmp_path / "synthetic.json"
        result = export_synthetic_data(sample_df, str(path))
        assert Path(result["output_path"]).exists()
        read = pd.read_json(path)
        assert len(read) == 200

    def test_export_parquet(self, sample_df, tmp_path):
        """export เป็น Parquet."""
        path = tmp_path / "synthetic.parquet"
        result = export_synthetic_data(sample_df, str(path))
        assert Path(result["output_path"]).exists()
        read = pd.read_parquet(path)
        assert len(read) == 200

    def test_invalid_extension_raises(self, sample_df, tmp_path):
        """นามสกุลไม่รองรับ → ValueError."""
        path = tmp_path / "synthetic.txt"
        with pytest.raises(ValueError, match="ไม่รองรับนามสกุล"):
            export_synthetic_data(sample_df, str(path))

    def test_audit_report_attached(self, sample_df, tmp_path):
        """include_audit=True → ไฟล์ audit แนบมา."""
        path = tmp_path / "synthetic.csv"
        result = export_synthetic_data(sample_df, str(path), include_audit=True)
        audit_path = Path(result["audit_path"])
        assert audit_path.exists()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert "overall_risk" in audit
        assert "pii_detected" in audit

    def test_no_audit_when_disabled(self, sample_df, tmp_path):
        """include_audit=False → ไม่มีไฟล์ audit."""
        path = tmp_path / "synthetic.csv"
        result = export_synthetic_data(sample_df, str(path), include_audit=False)
        assert "audit_path" not in result

    def test_custom_n_rows(self, sample_df, tmp_path):
        """ระบุ n_rows เอง."""
        path = tmp_path / "synthetic.csv"
        result = export_synthetic_data(sample_df, str(path), n_rows=50)
        assert result["n_rows"] == 50
        read = pd.read_csv(path)
        assert len(read) == 50

    def test_no_real_values_in_export(self, tmp_path):
        """ข้อมูลที่ export ต้องไม่มี PII จริง."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "phone": [f"081-234-{i:04d}" for i in range(100)],
                "email": [f"user{i}@test.com" for i in range(100)],
            }
        )
        path = tmp_path / "synthetic.csv"
        export_synthetic_data(df, str(path), include_audit=False)
        read = pd.read_csv(path)
        # ไม่มีเบอร์จริงปน
        for v in read["phone"].dropna():
            assert "081-234-" not in str(v)
        # ไม่มีอีเมลจริงปน
        for v in read["email"].dropna():
            assert "@test.com" not in str(v)

    def test_file_size_reported(self, sample_df, tmp_path):
        """file_size_kb ต้อง > 0."""
        path = tmp_path / "synthetic.csv"
        result = export_synthetic_data(sample_df, str(path), include_audit=False)
        assert result["file_size_kb"] > 0

    def test_reproducible_with_seed(self, tmp_path):
        """seed เดียวกัน → ไฟล์เหมือนกัน."""
        np.random.seed(42)
        df = pd.DataFrame({"v": np.random.randn(100)})
        p1 = tmp_path / "s1.csv"
        p2 = tmp_path / "s2.csv"
        export_synthetic_data(df, str(p1), random_seed=99, include_audit=False)
        export_synthetic_data(df, str(p2), random_seed=99, include_audit=False)
        r1 = pd.read_csv(p1)
        r2 = pd.read_csv(p2)
        pd.testing.assert_series_equal(r1["v"], r2["v"])
