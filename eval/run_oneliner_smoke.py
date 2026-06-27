"""One-liner smoke eval on eval/fixtures — dev/CI helper."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd

import thaieda

EVAL = Path(__file__).resolve().parent.parent / "eval" / "fixtures"


def _run_one(name: str, df: pd.DataFrame) -> dict:
    result = thaieda.run(
        df,
        clean=True,
        make_charts=False,
        narrative=False,
        timeseries=False,
        insights_engine=False,
    )
    comp = result.quality_comparison
    return {
        "name": name,
        "rows": result.overview.get("rows"),
        "cols": result.overview.get("columns"),
        "score_before": comp["score_before"] if comp else None,
        "score_after": comp["score_after"] if comp else result.quality_score.get("score"),
        "grade_after": comp["grade_after"] if comp else result.quality_score.get("grade"),
        "issues_before": len(result.quality_issues_before),
        "issues_after": len(result.quality_issues),
        "fixed_checks": len(comp["fixed_checks"]) if comp else 0,
        "cleaning_ops": len(result.cleaning_report.operations_run) if result.cleaning_report else 0,
        "notes": len(result.notes),
        "ok": True,
    }


def main() -> int:
    targets: list[tuple[str, Path | pd.DataFrame]] = [
        ("dirty-thai-labeled", EVAL / "dirty-thai-labeled.csv"),
        ("clean-thai", EVAL / "clean-thai.csv"),
        ("superstore", EVAL / "superstore.csv"),
    ]
    for csv in sorted((EVAL / "coffee-chain").glob("*.csv")):
        targets.append((f"coffee-chain/{csv.stem}", csv))

    rows: list[dict] = []
    failed = 0
    for name, src in targets:
        try:
            df = pd.read_csv(src, encoding="utf-8-sig")
            rows.append(_run_one(name, df))
            print(f"OK  {name:30} score {rows[-1]['score_before']}->{rows[-1]['score_after']} ({rows[-1]['grade_after']})")
        except Exception as exc:
            failed += 1
            rows.append({"name": name, "ok": False, "error": str(exc)})
            print(f"FAIL {name:30} {exc}")
            traceback.print_exc()

    print()
    print(f"Summary: {len(rows) - failed}/{len(rows)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
