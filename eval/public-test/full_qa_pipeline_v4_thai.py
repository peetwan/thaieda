"""ThaiEDA Full Pipeline QA v4 — Thai & Hybrid datasets.

รัน ThaiEDA pipeline บน 5 datasets ภาษาไทย/ไฮบริด:
1. wongnai-reviews-40k — รีวิวร้านอาหารไทย 40K rows (text + rating)
2. wongnai-train-50k — Wongnai train 40K rows (long Thai reviews)
3. wisesight-sentiment — Thai social media sentiment 26K rows (text + label)
4. thai-ecommerce-15k — Thai e-commerce products 15K rows (text + numeric + dates)
5. thai-restaurant-hybrid-20k — Hybrid 20K rows (Thai review + numeric + dirty dates)

เก็บ: time, insights, quality, anomalies, charts, tables, html size, warnings
"""

import json
import re
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from thaieda import run  # noqa: E402

DATA_DIR = Path(__file__).parent.parent.parent / "data-example" / "thai-datasets"
OUTPUT_DIR = Path(__file__).parent / "thai-qa-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = [
    {"name": "wongnai-reviews-40k", "file": "wongnai-reviews-40k.csv", "domain": "Thai restaurant reviews (HF)", "has_thai": True},
    {"name": "wongnai-train-50k", "file": "wongnai-train-50k.csv", "domain": "Wongnai train reviews", "has_thai": True},
    {"name": "wisesight-sentiment", "file": "wisesight-sentiment.csv", "domain": "Thai social media sentiment", "has_thai": True},
    {"name": "thai-ecommerce-15k", "file": "thai-ecommerce-15k.csv", "domain": "Thai e-commerce products", "has_thai": True},
    {"name": "thai-restaurant-hybrid-20k", "file": "thai-restaurant-hybrid-20k.csv", "domain": "Thai restaurant hybrid (dirty)", "has_thai": True},
]


def scan_html(html_str: str) -> dict:
    charts = len(re.findall(r"data:image/png;base64", html_str))
    tables = len(re.findall(r"<table", html_str))
    h2s = re.findall(r"<h2[^>]*>(.*?)</h2>", html_str, re.DOTALL)
    sections = [re.sub(r"<[^>]+>", "", h).strip()[:50] for h in h2s]
    unrendered = len(re.findall(r"\{\{[^}]+\}\}", html_str))
    raw_nan = len(re.findall(r">NaN<|>nan<|>None<", html_str))
    broken_img = len(re.findall(r'src="data:image/png;base64,\s*"', html_str))
    return {
        "chart_count": charts,
        "table_count": tables,
        "sections": sections,
        "unrendered_jinja": unrendered,
        "raw_nan": raw_nan,
        "broken_images": broken_img,
        "html_size": len(html_str),
    }


results = []
for ds in DATASETS:
    name = ds["name"]
    print(f"\n{'='*60}")
    print(f"  RUNNING: {name} ({ds['domain']})")
    print(f"{'='*60}")

    csv_path = DATA_DIR / ds["file"]
    if not csv_path.exists():
        print(f"  SKIP: {csv_path} not found")
        continue

    t0 = time.perf_counter()
    entry = {**ds, "status": "pending", "error": None, "traceback": None}

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = pd.read_csv(csv_path)
            print(f"  Loaded: {len(df)} rows × {len(df.columns)} cols")

            result = run(df, clean=True, lang="th")
            html = result.to_html()

        elapsed = time.perf_counter() - t0
        scan = scan_html(html)

        html_path = OUTPUT_DIR / f"{name}-report.html"
        html_path.write_text(html, encoding="utf-8")

        entry.update({
            "status": "ok",
            "rows": len(df),
            "cols": len(df.columns),
            "time_seconds": round(elapsed, 2),
            "html_size_bytes": len(html),
            "html_scan": scan,
            "warnings_captured": [str(x.message) for x in w],
            "warning_count": len(w),
        })

        # Extract insight/quality/anomaly counts from result
        try:
            d = result.to_dict()
            ins = d.get("insights")
            if isinstance(ins, dict):
                entry["insights_total"] = ins.get("total_insights", len(ins.get("insights", [])))
            else:
                entry["insights_total"] = len(ins or [])
            entry["quality_issues_count"] = len(d.get("quality_issues", []))
            entry["anomalies_count"] = len(d.get("anomalies", []))
        except Exception:
            pass

        print(f"  Status: ok | Time: {elapsed:.2f}s")
        print(f"  Insights: {entry.get('insights_total','?')} | Quality: {entry.get('quality_issues_count','?')} | Anomalies: {entry.get('anomalies_count','?')}")
        print(f"  Charts: {scan['chart_count']} | Tables: {scan['table_count']}")
        print(f"  Warnings: {len(w)}")
        if scan["unrendered_jinja"] or scan["raw_nan"] or scan["broken_images"]:
            print(f"  ⚠ DEFECTS: jinja={scan['unrendered_jinja']} nan={scan['raw_nan']} broken_img={scan['broken_images']}")

    except Exception as exc:
        import traceback as tb
        elapsed = time.perf_counter() - t0
        entry.update({
            "status": "error",
            "error": str(exc),
            "traceback": tb.format_exc(),
            "time_seconds": round(elapsed, 2),
        })
        print(f"  ERROR: {exc}")

    results.append(entry)

# Save summary
summary_path = OUTPUT_DIR / "qa-summary.json"
summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n{'='*60}")
print(f"  QA SUMMARY v4 (Thai datasets)")
print(f"{'='*60}")
ok = sum(1 for r in results if r["status"] == "ok")
err = sum(1 for r in results if r["status"] == "error")
print(f"  OK: {ok}/{len(results)} | Errors: {err}")
defects = sum(1 for r in results if r.get("html_scan", {}).get("unrendered_jinja", 0) + r.get("html_scan", {}).get("raw_nan", 0) + r.get("html_scan", {}).get("broken_images", 0) > 0)
print(f"  Total defects found: {defects}")
print(f"\n  Summary: {summary_path}")