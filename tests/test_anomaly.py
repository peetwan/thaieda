"""ทดสอบ thaieda.anomaly — การตรวจจับความผิดปกติ (สถิติ/ข้อความ/การเข้ารหัส/หมวดหมู่)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from thaieda.anomaly import (
    AnomalyIssue,
    AnomalySummary,
    detect_anomalies,
    detect_anomalies_all,
    detect_categorical_anomalies,
    detect_column_anomalies,
    detect_isolation_forest,
    detect_lof,
    detect_numeric_outliers,
    detect_text_anomalies,
    detect_thai_mojibake,
    detect_thai_text_anomalies,
    sklearn_available,
)
from thaieda.detect import ColumnType

# ข้อความ mojibake: ไบต์ UTF-8 ของ "สวัสดี" ถูกถอดเป็น Latin-1
MOJIBAKE = "สวัสดี".encode().decode("latin-1")
# mojibake แบบ cp874/tis-620 ของ "สวัสดี" (อักษรไทยทุกตัวกลายเป็น "เ"+x)
MOJIBAKE_CP874 = "สวัสดี".encode().decode("cp874")

# ต้องมี scikit-learn จึงจะทดสอบว่าวิธี ML จับ outlier ได้จริง
sklearn_installed = sklearn_available()
requires_sklearn = pytest.mark.skipif(not sklearn_installed, reason="scikit-learn not installed")


# ----------------------------------------------------------- numeric outliers
def test_numeric_outliers_detected():
    s = pd.Series([10, 11, 12, 10, 11, 9, 10, 12, 11, 1000])
    issue = detect_numeric_outliers(s)
    assert issue is not None
    assert issue.check_name == "numeric_outliers"
    assert issue.anomaly_type == "statistical"
    assert issue.count == 1
    # ตำแหน่งแถวของค่า 1000 คือ index 9
    assert 9 in issue.indices


def test_numeric_outliers_none_when_clean():
    s = pd.Series([10, 11, 12, 10, 11, 9, 10, 12, 11, 10])
    assert detect_numeric_outliers(s) is None


def test_numeric_outliers_too_few_values():
    s = pd.Series([1, 2, 1000])  # น้อยกว่าขั้นต่ำ
    assert detect_numeric_outliers(s) is None


def test_numeric_outliers_indices_capped():
    # มี outlier มากกว่า 20 ตัว — ต้องเก็บ index ไม่เกิน 20
    s = pd.Series([i % 5 for i in range(200)] + [10_000] * 25)
    issue = detect_numeric_outliers(s)
    assert issue is not None
    assert issue.count == 25
    assert len(issue.indices) <= 20


# ------------------------------------------------------------- text anomalies
def test_text_length_anomaly_detected():
    s = pd.Series(["สวัสดีครับ"] * 15 + ["x"])
    issues = detect_text_anomalies(s, tokenizer=None)
    names = {i.check_name for i in issues}
    assert "text_length_anomaly" in names


def test_mojibake_detected():
    s = pd.Series(["ปกติ", MOJIBAKE, "ข้อความ"])
    issues = detect_text_anomalies(s, tokenizer=None)
    encoding = [i for i in issues if i.check_name == "encoding_mojibake"]
    assert encoding
    assert encoding[0].severity == "critical"
    assert encoding[0].anomaly_type == "encoding"
    assert encoding[0].count == 1


def test_garbled_text_detected():
    s = pd.Series(["ปกติ", "เสีย�หาย", "ดี"])
    issues = detect_text_anomalies(s, tokenizer=None)
    names = {i.check_name for i in issues}
    assert "garbled_text" in names


def test_excessive_repetition_detected():
    s = pd.Series(["55555", "ปกติ", "ดีมาก"])
    issues = detect_text_anomalies(s, tokenizer=None)
    names = {i.check_name for i in issues}
    assert "excessive_repetition" in names


# -------------------------------------------------------- thai text anomalies
def test_tone_mark_stacking_detected():
    s = pd.Series(["น้้ำ", "ปกติ", "อร่อย"])
    issues = detect_thai_text_anomalies(s)
    names = {i.check_name for i in issues}
    assert "tone_mark_stacking" in names


def test_invalid_thai_sequence_detected():
    # ขึ้นต้นด้วยวรรณยุกต์ (combining mark ลอย ไม่มีพยัญชนะฐาน)
    s = pd.Series(["้ำ", "ปกติ", "ดี"])
    issues = detect_thai_text_anomalies(s)
    names = {i.check_name for i in issues}
    assert "invalid_thai_sequence" in names


# --------------------------------------------------------- categorical anomalies
def test_fuzzy_duplicates_detected():
    s = pd.Series(["กรุงเทพ"] * 5 + ["กรุงเทพฯ"] * 3 + ["เชียงใหม่"] * 4)
    issues = detect_categorical_anomalies(s)
    fuzzy = [i for i in issues if i.check_name == "fuzzy_duplicates"]
    assert fuzzy
    assert fuzzy[0].severity == "warning"


def test_fuzzy_duplicates_skips_distinct_codes():
    """รหัส/รุ่น (model/part number) ที่สตริงคล้ายกันสูงแต่เป็นคนละค่าจริง

    เช่น 'EMB-145LR' vs 'EMB-145', 'CL-600-2B19' vs 'CL-600-2C10' — ต่างกันโดย
    ตั้งใจ ไม่ใช่ typo จึงต้องไม่ถูก flag เป็น near-duplicate (เคยเป็น false positive
    บนคอลัมน์ model/manufacturer ของชุดข้อมูลเครื่องบิน nycflights13).
    """
    models = (
        ["EMB-145LR"] * 30
        + ["EMB-145"] * 25
        + ["CL-600-2B19"] * 20
        + ["CL-600-2C10"] * 18
        + ["737-924ER"] * 15
        + ["737-924"] * 12
    )
    issues = detect_categorical_anomalies(pd.Series(models))
    assert not any(i.check_name == "fuzzy_duplicates" for i in issues)


def test_fuzzy_duplicates_skips_distinct_word_phrases():
    """วลีหลายคำที่ต่างกันคนละคำจริง ต้องไม่ถูก flag เป็น near-duplicate.

    'Fixed wing multi engine' vs 'Fixed wing single engine' ใช้คำส่วนใหญ่ร่วมกัน
    ทำให้ similarity สูง แต่ 'multi'/'single' เป็นคนละคำ — เป็นคนละหมวด ไม่ใช่ typo.
    """
    phrases = ["Fixed wing multi engine"] * 40 + ["Fixed wing single engine"] * 35
    issues = detect_categorical_anomalies(pd.Series(phrases))
    assert not any(i.check_name == "fuzzy_duplicates" for i in issues)


def test_fuzzy_duplicates_still_detects_word_typo():
    """typo ระดับตัวอักษรในวลี (คำที่ต่างยังคล้ายกัน) ต้องยังถูกจับเป็น near-duplicate."""
    s = pd.Series(["San Francisco"] * 20 + ["San Fransisco"] * 3 + ["Boston"] * 10)
    issues = detect_categorical_anomalies(s)
    assert any(i.check_name == "fuzzy_duplicates" for i in issues)


def test_rare_categories_detected():
    s = pd.Series(["A"] * 100 + ["B"] * 100 + ["typo_x"])
    issues = detect_categorical_anomalies(s)
    names = {i.check_name for i in issues}
    assert "rare_categories" in names


def test_case_inconsistency_detected():
    s = pd.Series(["Bangkok"] * 5 + ["bangkok"] * 3 + ["Phuket"] * 4)
    issues = detect_categorical_anomalies(s)
    names = {i.check_name for i in issues}
    assert "case_inconsistency" in names


def test_categorical_clean_no_anomalies():
    s = pd.Series(["แดง", "เขียว", "น้ำเงิน", "แดง", "เขียว"])
    assert detect_categorical_anomalies(s) == []


# ------------------------------------------------------------- column anomalies
def test_constant_column_detected():
    df = pd.DataFrame({"const": ["x"] * 10})
    issues = detect_column_anomalies(df, {"const": ColumnType.CATEGORICAL})
    names = {i.check_name for i in issues}
    assert "constant_column" in names


def test_high_null_spike_detected():
    df = pd.DataFrame({"sparse": [1, None, None, None, None, None]})
    issues = detect_column_anomalies(df, {"sparse": ColumnType.NUMERIC})
    names = {i.check_name for i in issues}
    assert "high_null_spike" in names


def test_type_mixing_detected():
    df = pd.DataFrame({"amount": ["10", "20", "30", "40", "50", "60", "70", "N/A"]})
    issues = detect_column_anomalies(df, {"amount": ColumnType.CATEGORICAL})
    names = {i.check_name for i in issues}
    assert "type_mixing" in names


# ------------------------------------------------------------- detect_anomalies
def test_detect_anomalies_returns_sorted():
    df = pd.DataFrame(
        {
            "rating": [5, 3, 4, 5, 4, 3, 5, 4, 3, 100],  # numeric outlier -> warning
            "const": ["x"] * 10,  # constant column -> info
        }
    )
    issues = detect_anomalies(df)
    assert all(isinstance(i, AnomalyIssue) for i in issues)
    severities = [i.severity for i in issues]
    assert "warning" in severities
    assert "info" in severities
    rank = {"critical": 0, "warning": 1, "info": 2}
    assert severities == sorted(severities, key=lambda s: rank[s])


def test_detect_anomalies_default_column_types():
    # ไม่ส่ง column_types -> ต้องเรียก detect_all ให้เอง
    df = pd.DataFrame({"price": [100, 100, 100, 100, 100, 5000]})
    issues = detect_anomalies(df)
    assert any(i.check_name == "numeric_outliers" for i in issues)


def test_detect_anomalies_skips_text_without_tokenizer():
    # ไม่มี tokenizer -> ข้ามการตรวจข้อความทั่วไป (mojibake) แต่ยังตรวจไทยเฉพาะทาง
    df = pd.DataFrame({"text": ["ปกติ"] * 5 + [MOJIBAKE]})
    issues = detect_anomalies(df, tokenizer=None)
    assert all(i.check_name != "encoding_mojibake" for i in issues)


def test_detect_anomalies_clean_data_no_critical():
    df = pd.DataFrame(
        {
            "name": ["สมชาย", "สมหญิง", "สมศักดิ์", "สมพร", "สมคิด", "สมบัติ"],
            "age": [25, 30, 35, 28, 40, 33],
        }
    )
    issues = detect_anomalies(df)
    assert all(i.severity != "critical" for i in issues)


def test_anomaly_issue_to_dict():
    s = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 999])
    issue = detect_numeric_outliers(s)
    assert issue is not None
    d = issue.to_dict()
    for key in ("check_name", "severity", "column", "anomaly_type", "count", "indices"):
        assert key in d


# ------------------------------------------------------------- ML-based outliers
def _numeric_with_outliers(n: int = 130, seed: int = 42) -> pd.Series:
    """ค่าปกติ n ตัว (รอบ ๆ 50) แล้วเติม outlier สุดขั้วเข้าไป (รวม > 100 แถว)."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(50, 1.5, n)
    return pd.Series(list(normal) + [500.0, -400.0, 1000.0])


@requires_sklearn
def test_isolation_forest_detects_outliers():
    s = _numeric_with_outliers()
    issue = detect_isolation_forest(s)
    assert issue is not None
    assert issue.check_name == "isolation_forest"
    assert issue.anomaly_type == "statistical"
    assert issue.count >= 1
    # ตำแหน่งของ outlier ที่เติมไว้ (3 ตัวท้าย)
    n = len(s)
    assert any(idx in issue.indices for idx in (n - 1, n - 2, n - 3))
    # คะแนนความผิดปกติถูกแนบในตัวอย่าง
    assert any("score=" in ex for ex in issue.examples)


@requires_sklearn
def test_lof_detects_outliers():
    s = _numeric_with_outliers()
    issue = detect_lof(s)
    assert issue is not None
    assert issue.check_name == "local_outlier_factor"
    assert issue.anomaly_type == "statistical"
    assert issue.count >= 1
    assert any("LOF=" in ex for ex in issue.examples)


@requires_sklearn
def test_ml_methods_skip_when_too_many_flagged():
    # การกระจายแบบ uniform ไม่มี outlier จริง — contamination='auto' อาจ flag จำนวนมาก
    # guard ต้องตัดผลที่ flag เกิน ~20% ทิ้ง (คืน None) เพื่อลด noise
    s = pd.Series(range(200))
    assert detect_isolation_forest(s) is None


@requires_sklearn
def test_ml_methods_skip_small_samples():
    # <=100 แถว -> วิธี ML ต้องคืน None (ผลไม่น่าเชื่อถือ)
    s = pd.Series([1.0] * 50 + [9999.0])
    assert detect_isolation_forest(s) is None
    assert detect_lof(s) is None


def test_ml_methods_skip_gracefully_without_sklearn(monkeypatch):
    # จำลองว่าไม่มี scikit-learn -> ต้องคืน None อย่างสุภาพ ไม่ crash
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
    monkeypatch.setitem(sys.modules, "sklearn.neighbors", None)
    s = _numeric_with_outliers()
    assert detect_isolation_forest(s) is None
    assert detect_lof(s) is None
    assert sklearn_available() is False


def test_detect_anomalies_runs_ml_when_available():
    if not sklearn_installed:
        pytest.skip("scikit-learn not installed")
    df = pd.DataFrame({"v": _numeric_with_outliers()})
    issues = detect_anomalies(df)
    names = {i.check_name for i in issues}
    # มีทั้งวิธีเชิงสถิติและวิธี ML อย่างน้อยหนึ่งวิธี
    assert "isolation_forest" in names or "local_outlier_factor" in names


# ------------------------------------------------------------- thai mojibake
def test_thai_mojibake_latin1_detected():
    # ตัวอย่างจากโจทย์: "สวัสดี" UTF-8 ถูกถอดเป็น latin-1
    s = pd.Series(["ปกติ", "à¸ªà¸§à¸±à¸ªà¸”à¸µ", "ดีมาก"])
    issue = detect_thai_mojibake(s)
    assert issue is not None
    assert issue.check_name == "thai_mojibake"
    assert issue.anomaly_type == "encoding"
    assert issue.severity == "critical"
    assert issue.count == 1


def test_thai_mojibake_cp874_detected():
    s = pd.Series(["ปกติ", MOJIBAKE_CP874, "ข้อความ"])
    issue = detect_thai_mojibake(s)
    assert issue is not None
    assert issue.count == 1


def test_thai_mojibake_no_false_positive_on_real_thai():
    # คำไทยจริงที่ขึ้นต้นด้วย "เธ"/"เน" ต้องไม่ถูก flag
    s = pd.Series(["เธอเป็นเพื่อนของเนื้อคู่", "ปกติ", "อร่อยมาก"])
    assert detect_thai_mojibake(s) is None


def test_thai_mojibake_integrated_in_text_anomalies():
    s = pd.Series(["ปกติ", "à¸ªà¸§à¸±à¸ªà¸”à¸µ", "ดี"])
    issues = detect_text_anomalies(s, tokenizer=None)
    names = {i.check_name for i in issues}
    assert "thai_mojibake" in names


# ------------------------------------------------ unified single-column API
def test_detect_anomalies_series_auto_numeric():
    s = pd.Series([10, 11, 12, 10, 11, 9, 10, 12, 11, 1000], name="rating")
    summary = detect_anomalies(s)  # method="auto"
    assert isinstance(summary, AnomalySummary)
    assert summary.column == "rating"
    assert summary.method.startswith("auto")
    assert summary.total_anomalies >= 1
    assert summary.anomaly_rate > 0
    assert summary.issues


def test_detect_anomalies_series_explicit_method():
    # inlier เกาะกลุ่มแน่น + outlier ปานกลาง เพื่อให้ z-score เกิน 3σ (ไม่โดน std ของตัวเองกลบ)
    s = pd.Series([10] * 20 + [11] * 20 + [13])
    summary = detect_anomalies(s, method="zscore")
    assert isinstance(summary, AnomalySummary)
    assert summary.method == "zscore"
    assert all(i.check_name == "numeric_outliers_zscore" for i in summary.issues)
    assert summary.total_anomalies == 1


def test_detect_anomalies_series_returns_none_when_clean():
    s = pd.Series([10, 11, 12, 10, 11, 9, 10, 12, 11, 10])
    assert detect_anomalies(s, method="iqr") is None


def test_detect_anomalies_series_invalid_method_raises():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    with pytest.raises(ValueError):
        detect_anomalies(s, method="bogus")


def test_detect_anomalies_series_categorical():
    s = pd.Series(["A"] * 100 + ["B"] * 100 + ["typo_x"], name="cat")
    summary = detect_anomalies(s)
    assert summary is not None
    assert summary.method == "auto:categorical"
    assert summary.total_anomalies >= 1


def test_detect_anomalies_summary_to_dict():
    s = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 999])
    summary = detect_anomalies(s, method="mad")
    assert summary is not None
    d = summary.to_dict()
    for key in ("column", "method", "total_anomalies", "anomaly_rate", "issues"):
        assert key in d
    assert isinstance(d["issues"], list)


def test_detect_anomalies_all_returns_dict():
    df = pd.DataFrame(
        {
            "rating": [5, 3, 4, 5, 4, 3, 5, 4, 3, 100],
            "clean": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    result = detect_anomalies_all(df)
    assert set(result) == {"rating", "clean"}
    assert isinstance(result["rating"], AnomalySummary)
    # คอลัมน์ที่ไม่มี outlier -> None
    assert result["clean"] is None


def test_detect_anomalies_dataframe_still_returns_list():
    # ความเข้ากันได้ย้อนหลัง: ส่ง DataFrame ต้องได้ list[AnomalyIssue] เหมือนเดิม
    df = pd.DataFrame({"rating": [5, 3, 4, 5, 4, 3, 5, 4, 3, 100], "const": ["x"] * 10})
    issues = detect_anomalies(df)
    assert isinstance(issues, list)
    assert all(isinstance(i, AnomalyIssue) for i in issues)
