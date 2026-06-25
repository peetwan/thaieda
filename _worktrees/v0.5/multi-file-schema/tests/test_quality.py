"""ทดสอบ thaieda.quality — การตรวจคุณภาพข้อมูลเฉพาะภาษาไทย (the moat)."""

from __future__ import annotations

import pandas as pd

from thaieda.detect import ColumnType
from thaieda.quality import (
    QualityIssue,
    check_buddhist_era,
    check_normalization,
    check_script_composition,
    check_thai_numerals,
    check_zero_width,
    run_quality_checks,
)


# ----------------------------------------------------------- Buddhist Era
def test_buddhist_era_detected():
    s = pd.Series([2567, 2568, 2566])
    issue = check_buddhist_era(s, "year")
    assert issue is not None
    assert issue.check_name == "buddhist_era"
    assert issue.count == 3


def test_buddhist_era_mixed_is_critical():
    s = pd.Series([2567, 2024, 2568, 2023])
    issue = check_buddhist_era(s, "year")
    assert issue is not None
    assert issue.severity == "critical"  # ปนกันระหว่าง พ.ศ. และ ค.ศ.
    assert "2567" in issue.examples


def test_buddhist_era_not_flagged_for_normal_ce():
    s = pd.Series([2020, 2021, 2022, 2023])
    assert check_buddhist_era(s, "year") is None


def test_buddhist_era_in_date_strings():
    s = pd.Series(["2567-01-01", "2568-05-12", "2566-12-31"])
    issue = check_buddhist_era(s, "date")
    assert issue is not None
    assert issue.count == 3


# ----------------------------------------------------------- Thai numerals
def test_thai_numerals_detected():
    s = pd.Series(["๑๒๓", "123", "๔๕๖", "789"])
    issue = check_thai_numerals(s, "price")
    assert issue is not None
    assert issue.check_name == "thai_numerals"
    assert issue.count == 2  # 2 เซลล์มีเลขไทย
    # ปนกับเลขอารบิก => warning
    assert issue.severity == "warning"


def test_thai_numerals_not_flagged_when_none():
    s = pd.Series(["123", "456", "789"])
    assert check_thai_numerals(s, "x") is None


def test_thai_numerals_examples_shown():
    s = pd.Series(["ราคา ๑๐๐ บาท", "200"])
    issue = check_thai_numerals(s, "price")
    assert issue is not None
    assert any("๑๐๐" in ex for ex in issue.examples)


# ------------------------------------------------------- zero-width chars
def test_zero_width_detected():
    s = pd.Series(["ปกติ", "มีอักขระ​ซ่อน", "ปกติ2"])
    issue = check_zero_width(s, "text")
    assert issue is not None
    assert issue.check_name == "zero_width_chars"
    assert issue.severity == "critical"
    assert issue.count == 1


def test_zero_width_examples_visible_via_repr():
    s = pd.Series(["a​b"])
    issue = check_zero_width(s, "text")
    assert issue is not None
    # ตัวอย่างต้องแสดงอักขระล่องหนให้มองเห็นได้ (ผ่าน repr)
    assert any("\\u200b" in ex for ex in issue.examples)


def test_zero_width_bom_and_zwnj():
    s = pd.Series(["﻿hello", "a‌b", "clean"])
    issue = check_zero_width(s, "text")
    assert issue is not None
    assert issue.count == 2


def test_zero_width_none_when_clean():
    s = pd.Series(["สะอาด", "ไม่มีปัญหา"])
    assert check_zero_width(s, "text") is None


# ----------------------------------------------------- repeated-char spam
def test_repeated_char_spam_detected():
    s = pd.Series(["55555", "ๆๆๆ", "ปกติ"])
    issue = check_normalization(s, "comment")
    assert issue is not None
    assert issue.count == 2
    assert "repeated-char spam" in issue.description


def test_normalization_duplicate_tone_marks():
    # วรรณยุกต์ซ้อนกัน (ไม้โทสองตัว)
    s = pd.Series(["น้้ำ", "ปกติ"])
    issue = check_normalization(s, "text")
    assert issue is not None
    assert "duplicate tone marks" in issue.description


# ---------------------------------------------------- script composition
def test_script_composition_mislabeled_thai():
    # คอลัมน์ที่คาดว่าเป็นไทย แต่มีไทยแค่ ~10%
    s = pd.Series(["hello world foo", "bar baz qux", "lorem ipsum dolor", "ดี"])
    issue = check_script_composition(s, "thai_col", expected_thai=True)
    assert issue is not None
    assert issue.check_name == "mislabeled_thai_column"
    assert issue.severity == "warning"


def test_script_composition_ok_when_mostly_thai():
    s = pd.Series(["อาหารอร่อย", "บริการดี", "ราคาถูก"])
    assert check_script_composition(s, "thai_col", expected_thai=True) is None


def test_script_composition_no_check_when_not_expected_thai():
    s = pd.Series(["hello", "world"])
    assert check_script_composition(s, "eng", expected_thai=False) is None


# --------------------------------------------------------- run_quality_checks
def test_run_quality_checks_sorted_by_severity():
    df = pd.DataFrame(
        {
            "year": [2567, 2024, 2568],  # critical (BE mixed)
            "text": ["a​b", "clean", "ok"],  # critical (zero-width)
            "price": ["๑๒๓", "456", "789"],  # warning (thai numerals)
        }
    )
    types = {
        "year": ColumnType.NUMERIC,
        "text": ColumnType.THAI_TEXT,
        "price": ColumnType.CATEGORICAL,
    }
    issues = run_quality_checks(df, types)
    assert len(issues) >= 3
    assert all(isinstance(i, QualityIssue) for i in issues)
    # critical ต้องมาก่อน warning/info
    severities = [i.severity for i in issues]
    sev_rank = {"critical": 0, "warning": 1, "info": 2}
    assert severities == sorted(severities, key=lambda s: sev_rank[s])


def test_run_quality_checks_clean_data_few_issues():
    df = pd.DataFrame({"name": ["ana", "bob", "cid"], "age": [20, 30, 40]})
    types = {"name": ColumnType.CATEGORICAL, "age": ColumnType.NUMERIC}
    issues = run_quality_checks(df, types)
    # ข้อมูลสะอาด ไม่ควรมีปัญหาวิกฤต
    assert all(i.severity != "critical" for i in issues)
