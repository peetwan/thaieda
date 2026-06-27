"""Regenerate golden JSON สำหรับ dirty dataset eval — รันเมื่อเปลี่ยน scoring/clean โดยตั้งใจ.

Usage:
    python tests/fixtures/dirty_datasets/regenerate_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import thaieda

FIXTURES_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = FIXTURES_DIR / "golden"


def _build_golden(name: str, df: pd.DataFrame) -> dict:
    result = thaieda.run(df, clean=True, make_charts=False, narrative=False)
    comparison = result.quality_comparison
    if comparison is None:
        raise RuntimeError(f"quality_comparison missing for {name}")

    cleaning_report = result.cleaning_report
    plan = result.report.cleaning_plan
    return {
        "name": name,
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
                "check_name": issue.check_name,
                "column": issue.column,
                "severity": issue.severity,
                "count": issue.count,
            }
            for issue in result.quality_issues_before
        ],
        "issues_after": [
            {
                "check_name": issue.check_name,
                "column": issue.column,
                "severity": issue.severity,
                "count": issue.count,
            }
            for issue in result.quality_issues
        ],
        "cleaning_plan_actions": plan.actions if plan is not None else [],
        "cleaning_operations": sorted({op.operation for op in cleaning_report.operations_run})
        if cleaning_report is not None
        else [],
    }


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(FIXTURES_DIR.glob("*.csv")):
        name = csv_path.stem
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        golden = _build_golden(name, df)
        out_path = GOLDEN_DIR / f"{name}.json"
        out_path.write_text(
            json.dumps(golden, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{name}: score {golden['score_before']} -> {golden['score_after']} "
            f"({len(golden['fixed_checks'])} fixed)"
        )


if __name__ == "__main__":
    main()
