"""ทดสอบ thaieda.anomaly — การตรวจจับความผิดปกติ (สถิติ/ข้อความ/การเข้ารหัส/หมวดหมู่)."""

from __future__ import annotations

import pandas as pd

from thaieda.anomaly import (
    AnomalyIssue,
    detect_anomalies,
    detect_categorical_anomalies,
    detect_column_anomalies,
    detect_numeric_outliers,
    detect_text_anomalies,
    detect_thai_text_anomalies,
)
from thaieda.detect import ColumnType

# ข้อความ mojibake: ไบต์ UTF-8 ของ "สวัสดี" ถูกถอดเป็น Latin-1
MOJIBAKE = "สวัสดี".encode().decode("latin-1")


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
