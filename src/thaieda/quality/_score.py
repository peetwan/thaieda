"""คำนวณคะแนนคุณภาพข้อมูล (0–100) สำหรับ ThaiEDA.

ให้คะแนนเดียวที่สรุปคุณภาพของ dataset โดยรวม พร้อมเกรด (A/B/C/D/F)
และรายละเอียดการให้คะแนน (breakdown) เพื่อใช้ในรายงานและ UI

หลักการให้คะแนน (v2.3):
  - ถ่วงน้ำหนักตาม severity: critical ×3, warning ×1, info ×0.2
  - **ถ่วงน้ำหนักตามขนาดของปัญหา (magnitude)**: penalty ของแต่ละ issue
    = น้ำหนัก severity × สัดส่วนที่กระทบ (percentage / 100)
    → คอลัมน์ว่าง 100% หรือค่าหาย 77% หักคะแนนมากกว่าปัญหาเล็ก ๆ มาก
  - normalize ด้วย "จำนวนคอลัมน์" เท่านั้น (ไม่หารด้วย sqrt(จำนวนแถว) อีกต่อไป)
    เพราะวัดปัญหาเป็นสัดส่วน (intensive) แล้ว — dataset ใหญ่จึงไม่ได้คะแนนสูงฟรี ๆ
  - คะแนน 100 = ไม่มีปัญหาเลย, 0 = มีปัญหารุนแรงครอบคลุมทั้ง dataset
  - เกรด: A ≥90, B ≥80, C ≥70, D ≥60, F <60
"""

from __future__ import annotations

from typing import TypedDict

from thaieda.quality import QualityIssue

# ------------------------------------------------------------------------------
# น้ำหนัก severity — critical รุนแรงที่สุด
# ------------------------------------------------------------------------------
_SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 3.0,
    "warning": 1.0,
    "info": 0.2,
}

# เกรดตามช่วงคะแนน (เรียงจากสูงไปต่ำ)
_GRADE_THRESHOLDS: tuple[tuple[int, str], ...] = (
    (90, "A"),
    (80, "B"),
    (70, "C"),
    (60, "D"),
)


# ------------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ------------------------------------------------------------------------------
class QualityBreakdown(TypedDict):
    """รายละเอียดส่วนประกอบของคะแนนคุณภาพ."""

    critical_count: int
    warning_count: int
    info_count: int
    weighted_score: float


class QualityScoreResult(TypedDict):
    """ผลลัพธ์คะแนนคุณภาพข้อมูล พร้อมเกรดและรายละเอียด."""

    score: int
    grade: str
    breakdown: QualityBreakdown


# ------------------------------------------------------------------------------
# helper
# ------------------------------------------------------------------------------
def _grade_for(score: int) -> str:
    """แปลงคะแนน 0–100 เป็นเกรด A/B/C/D/F.

    เกรด A ≥90, B ≥80, C ≥70, D ≥60, F <60
    """
    for threshold, grade in _GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


# ------------------------------------------------------------------------------
# public API
# ------------------------------------------------------------------------------
def compute_quality_score(
    quality_issues: list[QualityIssue],
    n_columns: int,
    n_rows: int,
) -> QualityScoreResult:
    """คำนวณคะแนนคุณภาพข้อมูลแบบ composite (0–100).

    ใช้น้ำหนัก severity (critical ×3, warning ×1, info ×0.2) คูณด้วยสัดส่วน
    ที่ปัญหากระทบ (percentage/100) แล้ว normalize ด้วยจำนวนคอลัมน์
    คืน 100 ถ้าไม่มีปัญหา และเข้าใกล้ 0 ถ้ามีปัญหารุนแรงครอบคลุมทั้ง dataset

    Args:
        quality_issues: รายการ QualityIssue ที่พบจาก ``run_quality_checks``
        n_columns: จำนวนคอลัมน์ของ dataset (ต้องไม่ติดลบ)
        n_rows: จำนวนแถวของ dataset (ต้องไม่ติดลบ)

    Returns:
        dict ที่มี ``score`` (int 0–100), ``grade`` (A/B/C/D/F),
        และ ``breakdown`` (จำนวนแต่ละ severity + weighted_score ก่อน normalize)

    Raises:
        TypeError: ถ้า ``quality_issues`` เป็น ``None`` หรือมีสมาชิกที่ไม่ใช่ ``QualityIssue``
        ValueError: ถ้า ``n_columns`` หรือ ``n_rows`` ติดลบ หรือพบ severity ที่ไม่รู้จัก
    """
    # ตรวจอาร์กิวเมนต์ — no silent fallbacks
    if quality_issues is None:
        raise TypeError("quality_issues ต้องเป็น list ของ QualityIssue ไม่ใช่ None")
    if n_columns < 0 or n_rows < 0:
        raise ValueError(
            f"n_columns และ n_rows ต้องไม่ติดลบ (ได้ n_columns={n_columns}, n_rows={n_rows})"
        )

    # นับตาม severity + รวม penalty ที่ถ่วงน้ำหนักด้วยขนาดของปัญหา (magnitude)
    critical_count = 0
    warning_count = 0
    info_count = 0
    weighted_count = 0.0  # ผลรวมน้ำหนัก severity (นับ issue — ใช้รายงานใน breakdown)
    magnitude_penalty = 0.0  # ผลรวม penalty = น้ำหนัก × สัดส่วนที่กระทบ (ใช้คิดคะแนน)
    for issue in quality_issues:
        if not isinstance(issue, QualityIssue):
            raise TypeError(
                f"สมาชิกใน quality_issues ต้องเป็น QualityIssue ไม่ใช่ {type(issue).__name__}"
            )
        sev = issue.severity
        if sev not in _SEVERITY_WEIGHTS:
            raise ValueError(f"severity ไม่รู้จัก: {sev!r} (ต้องเป็น critical / warning / info)")
        if sev == "critical":
            critical_count += 1
        elif sev == "warning":
            warning_count += 1
        else:
            info_count += 1
        weight = _SEVERITY_WEIGHTS[sev]
        weighted_count += weight
        # สัดส่วนที่กระทบ 0–1 (clamp percentage ให้อยู่ในช่วง [0,100])
        impact = min(max(issue.percentage, 0.0), 100.0) / 100.0
        magnitude_penalty += weight * impact

    weighted_score = weighted_count

    # คำนวณคะแนน normalized — หารด้วยจำนวนคอลัมน์เท่านั้น (penalty เป็นสัดส่วนแล้ว)
    if magnitude_penalty == 0.0:
        score = 100
    else:
        ratio = magnitude_penalty / max(n_columns, 1)
        # score = 100 × (1 − ratio) แต่ไม่ต่ำกว่า 0 และไม่เกิน 100
        raw = 100.0 * max(0.0, 1.0 - ratio)
        score = int(round(raw))
        score = max(0, min(100, score))

    return QualityScoreResult(
        score=score,
        grade=_grade_for(score),
        breakdown=QualityBreakdown(
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
            weighted_score=weighted_score,
        ),
    )


def _count_by_severity(issues: list[QualityIssue]) -> QualityBreakdown:
    """นับจำนวน issue ตาม severity."""
    critical_count = sum(1 for i in issues if i.severity == "critical")
    warning_count = sum(1 for i in issues if i.severity == "warning")
    info_count = sum(1 for i in issues if i.severity == "info")
    weighted_score = (
        critical_count * _SEVERITY_WEIGHTS["critical"]
        + warning_count * _SEVERITY_WEIGHTS["warning"]
        + info_count * _SEVERITY_WEIGHTS["info"]
    )
    return QualityBreakdown(
        critical_count=critical_count,
        warning_count=warning_count,
        info_count=info_count,
        weighted_score=weighted_score,
    )


class QualityComparisonResult(TypedDict):
    """เปรียบเทียบคุณภาพข้อมูลก่อน/หลังทำความสะอาด."""

    before: QualityBreakdown
    after: QualityBreakdown
    score_before: int
    score_after: int
    grade_before: str
    grade_after: str
    fixed_checks: list[str]


def compute_quality_comparison(
    issues_before: list[QualityIssue],
    issues_after: list[QualityIssue],
    n_columns: int,
    n_rows: int,
) -> QualityComparisonResult:
    """เปรียบเทียบคะแนนและรายการ check ก่อน/หลัง clean."""
    score_before = compute_quality_score(issues_before, n_columns, n_rows)
    score_after = compute_quality_score(issues_after, n_columns, n_rows)
    before_keys = {(i.check_name, i.column) for i in issues_before}
    after_keys = {(i.check_name, i.column) for i in issues_after}
    fixed_checks = sorted(f"{check}:{col}" for check, col in before_keys - after_keys)
    return QualityComparisonResult(
        before=_count_by_severity(issues_before),
        after=_count_by_severity(issues_after),
        score_before=score_before["score"],
        score_after=score_after["score"],
        grade_before=score_before["grade"],
        grade_after=score_after["grade"],
        fixed_checks=fixed_checks,
    )


__all__ = [
    "QualityBreakdown",
    "QualityComparisonResult",
    "QualityScoreResult",
    "compute_quality_comparison",
    "compute_quality_score",
]
