"""ทดสอบ thaieda.quality._score — compute_quality_score คำนวณคะแนนคุณภาพ (0–100).

ครอบคลุม:
  - ไม่มีปัญหา → คะแนน 100, เกรด A
  - ปัญหา critical ทั้งหมด → คะแนนต่ำ, เกรด F
  - ปัญหาผสม → คะแนนกลาง
  - กรณีพิเศษ: list ว่าง, None, จำนวนคอลัมน์เป็น 0, ค่าลบ
  - ตรวจสอบช่วงคะแนนเสมอ 0–100
  - ตรวจสอบเกณฑ์เกรด: A≥90, B≥80, C≥70, D≥60, F<60
"""

from __future__ import annotations

import random

import pytest

from thaieda.quality import QualityIssue, compute_quality_score


# ------------------------------------------------------------------------------
# helper — สร้าง QualityIssue สำหรับทดสอบ
# ------------------------------------------------------------------------------
def _issue(severity: str, column: str = "col") -> QualityIssue:
    """สร้าง QualityIssue สำหรับทดสอบด้วย severity ที่กำหนด."""
    return QualityIssue(
        check_name="test",
        severity=severity,
        column=column,
        count=1,
        percentage=10.0,
        description="test issue",
        description_th="ปัญหาทดสอบ",
    )


def _warnings(n: int) -> list[QualityIssue]:
    """สร้าง list ของ warning issues จำนวน n รายการ."""
    return [_issue("warning") for _ in range(n)]


# ------------------------------------------------------------------------------
# กรณีหลัก
# ------------------------------------------------------------------------------
def test_no_issues_perfect_score():
    """ไม่มีปัญหาเลย → คะแนน 100, เกรด A."""
    result = compute_quality_score([], n_columns=5, n_rows=100)
    assert result["score"] == 100
    assert result["grade"] == "A"
    assert result["breakdown"]["critical_count"] == 0
    assert result["breakdown"]["warning_count"] == 0
    assert result["breakdown"]["info_count"] == 0
    assert result["breakdown"]["weighted_score"] == 0.0


def test_all_critical_low_score():
    """ปัญหา critical ทั้งหมด → คะแนนต่ำ, เกรด F."""
    issues = [_issue("critical") for _ in range(5)]
    result = compute_quality_score(issues, n_columns=2, n_rows=10)
    # magnitude penalty = 5 × 3 × (10/100) = 1.5, ratio = 1.5/2 = 0.75 → score 25
    assert result["score"] < 60
    assert result["grade"] == "F"
    assert result["breakdown"]["critical_count"] == 5
    assert result["breakdown"]["weighted_score"] == 15.0


def test_mixed_issues_middle_score():
    """ปัญหาผสม → คะแนนกลาง."""
    issues = (
        [_issue("critical") for _ in range(2)]
        + [_issue("warning") for _ in range(3)]
        + [_issue("info") for _ in range(5)]
    )
    result = compute_quality_score(issues, n_columns=5, n_rows=100)
    # penalty = (2×3 + 3×1 + 5×0.2) × (10/100) = 1.0, ratio = 1.0/5 = 0.2 → score 80
    assert 60 <= result["score"] <= 90
    assert result["breakdown"]["critical_count"] == 2
    assert result["breakdown"]["warning_count"] == 3
    assert result["breakdown"]["info_count"] == 5


def test_info_only_high_score():
    """ปัญหา info อย่างเดียว → คะแนนใกล้เคียง 100."""
    issues = [_issue("info") for _ in range(5)]
    result = compute_quality_score(issues, n_columns=10, n_rows=100)
    # penalty = 5×0.2×(10/100) = 0.1, ratio = 0.1/10 = 0.01 → score 99
    assert result["score"] >= 90
    assert result["grade"] == "A"


# ------------------------------------------------------------------------------
# กรณีพิเศษ (edge cases)
# ------------------------------------------------------------------------------
def test_empty_list_is_perfect():
    """list ว่าง → คะแนน 100, เกรด A."""
    result = compute_quality_score([], n_columns=3, n_rows=50)
    assert result["score"] == 100
    assert result["grade"] == "A"


def test_none_issues_raises_type_error():
    """None → raise TypeError (no silent fallback)."""
    with pytest.raises(TypeError):
        compute_quality_score(None, n_columns=5, n_rows=100)  # type: ignore[arg-type]


def test_zero_columns_no_issues():
    """ไม่มีคอลัมน์ และไม่มีปัญหา → คะแนน 100."""
    result = compute_quality_score([], n_columns=0, n_rows=0)
    assert result["score"] == 100
    assert result["grade"] == "A"


def test_zero_columns_with_issues():
    """ไม่มีคอลัมน์ แต่มีปัญหา → คะแนนถูกหัก (capacity คอลัมน์ขั้นต่ำ = 1)."""
    issues = [_issue("critical")]
    result = compute_quality_score(issues, n_columns=0, n_rows=0)
    # magnitude penalty = 3 × (10/100) = 0.3, ratio = 0.3/max(0,1) = 0.3 → score 70
    assert result["score"] == 70
    assert result["grade"] == "C"


def test_zero_rows_with_issues():
    """คะแนนเป็นสัดส่วน (intensive) — ไม่ขึ้นกับจำนวนแถว."""
    issues = [_issue("critical")]
    result = compute_quality_score(issues, n_columns=5, n_rows=0)
    # magnitude penalty = 3 × (10/100) = 0.3, ratio = 0.3/5 = 0.06 → score 94
    assert result["score"] == 94
    assert result["grade"] == "A"


def test_negative_columns_raises_value_error():
    """n_columns ลบ → raise ValueError."""
    with pytest.raises(ValueError):
        compute_quality_score([], n_columns=-1, n_rows=10)


def test_negative_rows_raises_value_error():
    """n_rows ลบ → raise ValueError."""
    with pytest.raises(ValueError):
        compute_quality_score([], n_columns=5, n_rows=-1)


def test_unknown_severity_raises_value_error():
    """severity ที่ไม่รู้จัก → raise ValueError."""
    bad = QualityIssue(
        check_name="x",
        severity="bogus",
        column="c",
        count=1,
        percentage=1.0,
        description="d",
        description_th="ด",
    )
    with pytest.raises(ValueError):
        compute_quality_score([bad], n_columns=1, n_rows=1)


def test_non_qualityissue_member_raises_type_error():
    """สมาชิกที่ไม่ใช่ QualityIssue → raise TypeError."""
    with pytest.raises(TypeError):
        compute_quality_score(["not an issue"], n_columns=1, n_rows=1)  # type: ignore[list-item]


# ------------------------------------------------------------------------------
# ตรวจสอบเกณฑ์เกรด: A≥90, B≥80, C≥70, D≥60, F<60
# ใช้ dataset ขนาดเท่ากัน (10 คอลัมน์, 100 แถว → capacity = 100)
# ปรับจำนวน warning เพื่อควบคุม ratio
# ------------------------------------------------------------------------------
def test_grade_A_threshold():
    """ratio = 0.01 → score 99 → A."""
    result = compute_quality_score(_warnings(1), n_columns=10, n_rows=100)
    assert result["score"] >= 90
    assert result["grade"] == "A"


def test_grade_B_threshold():
    """ratio = 0.15 → score 85 → B."""
    result = compute_quality_score(_warnings(15), n_columns=10, n_rows=100)
    assert 80 <= result["score"] < 90
    assert result["grade"] == "B"


def test_grade_C_threshold():
    """ratio = 0.25 → score 75 → C."""
    result = compute_quality_score(_warnings(25), n_columns=10, n_rows=100)
    assert 70 <= result["score"] < 80
    assert result["grade"] == "C"


def test_grade_D_threshold():
    """ratio = 0.35 → score 65 → D."""
    result = compute_quality_score(_warnings(35), n_columns=10, n_rows=100)
    assert 60 <= result["score"] < 70
    assert result["grade"] == "D"


def test_grade_F_threshold():
    """ratio = 0.50 → score 50 → F."""
    result = compute_quality_score(_warnings(50), n_columns=10, n_rows=100)
    assert result["score"] < 60
    assert result["grade"] == "F"


def test_grade_F_at_zero():
    """ratio > 1 → score 0 → F."""
    result = compute_quality_score(_warnings(200), n_columns=10, n_rows=100)
    assert result["score"] == 0
    assert result["grade"] == "F"


# ------------------------------------------------------------------------------
# ตรวจสอบว่าคะแนนอยู่ในช่วง 0–100 เสมอ (fuzz)
# ------------------------------------------------------------------------------
def test_score_always_in_range():
    """คะแนนต้องอยู่ในช่วง 0–100 เสมอ แม้ input สุ่ม."""
    rng = random.Random(42)
    for _ in range(500):
        n_crit = rng.randint(0, 30)
        n_warn = rng.randint(0, 30)
        n_info = rng.randint(0, 30)
        issues = (
            [_issue("critical") for _ in range(n_crit)]
            + [_issue("warning") for _ in range(n_warn)]
            + [_issue("info") for _ in range(n_info)]
        )
        n_cols = rng.randint(0, 20)
        n_rows = rng.randint(0, 1000)
        result = compute_quality_score(issues, n_columns=n_cols, n_rows=n_rows)
        assert 0 <= result["score"] <= 100, (n_cols, n_rows, n_crit, n_warn, n_info)
        assert result["grade"] in {"A", "B", "C", "D", "F"}
