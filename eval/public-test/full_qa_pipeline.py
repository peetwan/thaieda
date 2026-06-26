"""ThaiEDA Full Pipeline QA — run every dataset, capture errors/warnings/HTML.

วิ่งทุก dataset ผ่าน run(df) และเก็บ:
- execution time, errors, warnings, HTML report path
- profile insights count, cross-column insights count
- ตรวจสอบ HTML output ว่ามีข้อความว่าง/ขาด section ไหม
"""

import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

# ใช้ source จาก src/ ไม่ใช่ installed package — insert หลัง site-packages
_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from thaieda import run, EDA  # noqa: E402

DATASETS_DIR = Path(__file__).parent.parent.parent / "data-example" / "public-datasets"
OUTPUT_DIR = Path(__file__).parent / "qa-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ทุก dataset พร้อม meta
DATASETS = [
    {
        "name": "titanic",
        "file": "titanic.csv",
        "rows_expect": 891,
        "domain": "passenger survival",
        "has_dates": False,
        "has_text": True,  # Name, Ticket, Cabin
        "has_thai": False,
    },
    {
        "name": "telco-churn",
        "file": "telco-churn.csv",
        "rows_expect": 7043,
        "domain": "telecom churn",
        "has_dates": False,
        "has_text": True,  # customerID
        "has_thai": False,
    },
    {
        "name": "wine-quality",
        "file": "winequality-red.csv",
        "rows_expect": 1599,
        "domain": "wine chemistry",
        "has_dates": False,
        "has_text": False,
        "has_thai": False,
    },
    {
        "name": "california-housing",
        "file": "california-housing.csv",
        "rows_expect": 20640,
        "domain": "housing census",
        "has_dates": False,
        "has_text": True,  # ocean_proximity
        "has_thai": False,
    },
    {
        "name": "superstore",
        "file": "superstore.csv",
        "rows_expect": 10800,
        "domain": "retail sales",
        "has_dates": True,  # Order Date, Ship Date
        "has_text": True,
        "has_thai": False,
    },
    {
        "name": "adult",
        "file": "adult.csv",
        "rows_expect": 32561,
        "domain": "census demographics",
        "has_dates": False,
        "has_text": True,
        "has_thai": False,
    },
    {
        "name": "bank-marketing",
        "file": "bank-marketing.csv",
        "rows_expect": 41188,
        "domain": "bank marketing",
        "has_dates": False,
        "has_text": True,
        "has_thai": False,
        "sep": ";",  # UCI bank uses semicolon
    },
    {
        "name": "online-retail",
        "file": "online-retail.csv",
        "rows_expect": 541909,
        "domain": "e-commerce transactions",
        "has_dates": True,  # InvoiceDate
        "has_text": True,  # Description
        "has_thai": False,
    },
]

# เพิ่ม dirty-thai ด้วย (Thai-specific)
DATASETS.append(
    {
        "name": "dirty-thai-retail",
        "file": "../dirty-thai-retail.csv",
        "rows_expect": 2,
        "domain": "thai retail (dirty)",
        "has_dates": True,
        "has_text": True,
        "has_thai": True,
    }
)


def run_one_dataset(ds_meta):
    """Run ThaiEDA pipeline บน dataset เดียว แล้วเก็บผล."""
    name = ds_meta["name"]
    csv_path = DATASETS_DIR / ds_meta["file"]
    sep = ds_meta.get("sep", ",")

    result = {
        "name": name,
        "file": str(csv_path),
        "domain": ds_meta["domain"],
        "has_dates": ds_meta["has_dates"],
        "has_text": ds_meta["has_text"],
        "has_thai": ds_meta["has_thai"],
        "status": "pending",
        "error": None,
        "traceback": None,
        "rows": None,
        "cols": None,
        "col_types": None,
        "insights_count": None,
        "cross_insights_count": None,
        "html_size_bytes": None,
        "html_has_empty_sections": [],
        "warnings_captured": [],
        "time_seconds": None,
    }

    print(f"\n{'='*60}")
    print(f"  RUNNING: {name} ({ds_meta['domain']})", flush=True)
    print(f"{'='*60}", flush=True)

    # skip ถ้ามี HTML อยู่แล้วและ size > 10KB (รันสำเร็จก่อน hang)
    html_check = OUTPUT_DIR / f"{name}-report.html"
    if html_check.exists() and html_check.stat().st_size > 10000:
        print(f"  SKIP: HTML exists ({html_check.stat().st_size} bytes)", flush=True)
        result["status"] = "ok_skipped"
        result["html_size_bytes"] = html_check.stat().st_size
        result["html_path"] = str(html_check)
        return result

    # อ่านข้อมูล
    try:
        df = pd.read_csv(csv_path, sep=sep, encoding="utf-8")
        result["rows"] = len(df)
        result["cols"] = len(df.columns)
        result["col_types"] = {c: str(df[c].dtype) for c in df.columns}
        print(f"  Loaded: {len(df)} rows × {len(df.columns)} cols", flush=True)
    except Exception as exc:
        result["status"] = "load_error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        return result

    # วิ่ง pipeline พร้อมจับ warnings
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        t0 = time.time()
        try:
            eda_result = run(df, lang="en")
            t1 = time.time()
            result["time_seconds"] = round(t1 - t0, 2)

            # เก็บ insights
            try:
                insights = getattr(eda_result, "insights", [])
                result["insights_count"] = len(insights) if insights else 0
            except Exception:
                result["insights_count"] = 0

            try:
                notes = getattr(eda_result, "notes", {})
                if isinstance(notes, dict):
                    cross = notes.get("cross_column_insights", [])
                    result["cross_insights_count"] = len(cross) if cross else 0
            except Exception:
                pass

            # สร้าง HTML
            html_path = OUTPUT_DIR / f"{name}-report.html"
            try:
                html_str = eda_result.to_html()
                html_path.write_text(html_str, encoding="utf-8")
                result["html_size_bytes"] = len(html_str.encode("utf-8"))
                result["html_path"] = str(html_path)

                # ตรวจหา section ว่างใน HTML
                empty_markers = [
                    "No data available",
                    "No insights found",
                    "undefined",
                    "NaN",
                    "[object Object]",
                    "col-span",
                ]
                for marker in empty_markers:
                    if marker in html_str:
                        result["html_has_empty_sections"].append(marker)

                # ตรวจหา image ที่ base64 แต่ว่าง
                if 'src="data:image/png;base64,"' in html_str or "base64,)" in html_str:
                    result["html_has_empty_sections"].append("empty_base64_image")

            except Exception as exc:
                result["status"] = "html_error"
                result["error"] = f"to_html() failed: {exc}"
                result["traceback"] = traceback.format_exc()
                return result

            result["status"] = "ok"

        except Exception as exc:
            t1 = time.time()
            result["time_seconds"] = round(t1 - t0, 2)
            result["status"] = "pipeline_error"
            result["error"] = str(exc)
            result["traceback"] = traceback.format_exc()
            return result

        # เก็บ warnings
        for w in w_list:
            result["warnings_captured"].append(f"{w.category.__name__}: {w.message}")

    print(f"  Status: {result['status']} | Time: {result['time_seconds']}s")
    print(f"  Insights: {result['insights_count']} | Cross: {result['cross_insights_count']}")
    print(f"  HTML: {result['html_size_bytes']} bytes")

    return result


def main():
    results = []

    for ds in DATASETS:
        try:
            r = run_one_dataset(ds)
            results.append(r)
        except Exception as exc:
            results.append(
                {
                    "name": ds["name"],
                    "status": "harness_error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    # สรุป
    summary_path = OUTPUT_DIR / "qa-summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  QA SUMMARY")
    print("=" * 60)
    ok = sum(1 for r in results if r.get("status") == "ok")
    err = sum(1 for r in results if r.get("status") not in ("ok",))
    print(f"  OK: {ok}/{len(results)} | Errors: {err}")
    print(f"  Summary: {summary_path}")

    # แสดง errors
    for r in results:
        if r.get("status") != "ok":
            print(f"\n  ❌ {r['name']}: {r.get('status')} — {r.get('error', '')[:120]}")

    # แสดง warnings
    for r in results:
        if r.get("warnings_captured"):
            print(f"\n  ⚠ {r['name']}: {len(r['warnings_captured'])} warnings")
            for w in r["warnings_captured"][:3]:
                print(f"    - {w[:120]}")

    return results


if __name__ == "__main__":
    main()