"""Golden eval — regression บน dirty dataset fixtures.

แต่ละ CSV ใน tests/fixtures/dirty_datasets/ มี golden JSON คู่กัน
ที่บันทึก score ก่อน/หลัง clean, issues, fixed_checks และ cleaning audit
เพื่อจับ regression เมื่อแก้ quality/clean pipeline

อัปเดต golden หลังเปลี่ยนโดยตั้งใจ::
    python tests/fixtures/dirty_datasets/regenerate_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import thaieda

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "dirty_datasets"
GOLDEN_DIR = FIXTURES_DIR / "golden"


def _load_golden_paths() -> list[Path]:
    if not GOLDEN_DIR.is_dir():
        return []
    return sorted(GOLDEN_DIR.glob("*.json"))


def _issue_tuple(issue: dict[str, Any]) -> tuple[str, str, str, int]:
    return (issue["check_name"], issue["column"], issue["severity"], issue["count"])


def _run_eval(name: str) -> dict[str, Any]:
    csv_path = FIXTURES_DIR / f"{name}.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    result = thaieda.run(df, clean=True, make_charts=False, narrative=False)
    comparison = result.quality_comparison
    assert comparison is not None

    cleaning_report = result.cleaning_report
    plan = result.report.cleaning_plan
    return {
        "rows_before_cleaning": result.report.overview["rows_before_cleaning"],
        "rows_after_cleaning": result.report.overview["rows_after_cleaning"],
        "score_before": comparison["score_before"],
        "score_after": comparison["score_after"],
        "grade_before": comparison["grade_before"],
        "grade_after": comparison["grade_after"],
        "before_counts": comparison["before"],
        "after_counts": comparison["after"],
        "fixed_checks": sorted(comparison["fixed_checks"]),
        "issues_before": [
            {
                "check_name": i.check_name,
                "column": i.column,
                "severity": i.severity,
                "count": i.count,
            }
            for i in result.quality_issues_before
        ],
        "issues_after": [
            {
                "check_name": i.check_name,
                "column": i.column,
                "severity": i.severity,
                "count": i.count,
            }
            for i in result.quality_issues
        ],
        "cleaning_plan_actions": sorted(plan.actions if plan is not None else []),
        "cleaning_operations": sorted({op.operation for op in cleaning_report.operations_run})
        if cleaning_report is not None
        else [],
    }


@pytest.mark.parametrize(
    "golden_path",
    _load_golden_paths(),
    ids=[p.stem for p in _load_golden_paths()],
)
def test_dirty_dataset_matches_golden(golden_path: Path) -> None:
    """ผล run(clean=True) ต้องตรง golden ทุก dataset."""
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    name = golden["name"]
    actual = _run_eval(name)

    assert actual["rows_before_cleaning"] == golden["rows_before_cleaning"]
    assert actual["rows_after_cleaning"] == golden["rows_after_cleaning"]
    assert actual["score_before"] == golden["score_before"]
    assert actual["score_after"] == golden["score_after"]
    assert actual["grade_before"] == golden["grade_before"]
    assert actual["grade_after"] == golden["grade_after"]
    assert actual["before_counts"] == golden["before_counts"]
    assert actual["after_counts"] == golden["after_counts"]
    assert actual["fixed_checks"] == golden["fixed_checks"]

    assert [_issue_tuple(i) for i in actual["issues_before"]] == [
        _issue_tuple(i) for i in golden["issues_before"]
    ]
    assert [_issue_tuple(i) for i in actual["issues_after"]] == [
        _issue_tuple(i) for i in golden["issues_after"]
    ]

    assert actual["cleaning_plan_actions"] == sorted(golden["cleaning_plan_actions"])
    assert actual["cleaning_operations"] == golden["cleaning_operations"]

    assert actual["score_after"] >= actual["score_before"]


def test_all_fixtures_have_csv_and_golden() -> None:
    """ทุก CSV ต้องมี golden คู่กัน (และกลับกัน)."""
    csv_names = {p.stem for p in FIXTURES_DIR.glob("*.csv")}
    golden_names = {p.stem for p in GOLDEN_DIR.glob("*.json")}
    assert csv_names == golden_names
    assert len(csv_names) >= 5


@pytest.mark.parametrize(
    "name,expected_improvement",
    [
        ("thai_text_dirty", 50),
        ("full_dirty", 1),
        ("registry_dirty", 1),
    ],
)
def test_dirty_datasets_improve_score(name: str, expected_improvement: int) -> None:
    """Dataset สกปรกต้องได้คะแนนหลัง clean ดีกว่าก่อนอย่างมีนัย."""
    golden = json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))
    delta = golden["score_after"] - golden["score_before"]
    assert delta >= expected_improvement
