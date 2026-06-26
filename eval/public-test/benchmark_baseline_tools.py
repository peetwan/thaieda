"""Baseline AutoEDA benchmark — ydata-profiling + sweetviz.

รันบน 6 ตัวแทน datasets (เล็ก-กลาง-ใหญ่ ทั้ง Thai/non-Thai):
1. titanic        — 891 rows × 12 cols (เล็ก, non-Thai)
2. superstore      — 10,800 rows × 21 cols (กลาง, has dates)
3. adult           — 32,561 rows × 15 cols (ใหญ่)
4. dirty-thai-retail — 500 rows × 8 cols (Thai dirty)
5. wisesight-sentiment — 26,737 rows × 2 cols (Thai text)
6. aps-failure     — 16,000 rows × 171 cols (wide table)

วัด: time, HTML size, และบันทึก limitations (crash/timeout/font issues)
"""

import json
import time
import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data-example" / "public-datasets"
THAI_DIR = Path(__file__).parent.parent.parent / "data-example" / "thai-datasets"
OUTPUT_DIR = Path(__file__).parent / "baseline-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = [
    {"name": "titanic", "file": "titanic.csv", "dir": DATA_DIR, "sep": ","},
    {"name": "superstore", "file": "superstore.csv", "dir": DATA_DIR, "sep": ","},
    {"name": "adult", "file": "adult.csv", "dir": DATA_DIR, "sep": ","},
    {"name": "dirty-thai-retail", "file": "../dirty-thai-retail.csv", "dir": DATA_DIR, "sep": ","},
    {"name": "wisesight-sentiment", "file": "wisesight-sentiment.csv", "dir": THAI_DIR, "sep": ","},
    {"name": "aps-failure", "file": "aps-failure-clean.csv", "dir": DATA_DIR, "sep": ","},
]


def run_ydata(df, name):
    """Run ydata-profiling (formerly pandas-profiling)."""
    from ydata_profiling import ProfileReport

    t0 = time.perf_counter()
    result = {"tool": "ydata-profiling", "time_seconds": None, "html_size_bytes": 0,
              "status": "pending", "error": None, "notes": []}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            profile = ProfileReport(df, title=f"{name}", minimal=True, progress_bar=False)
            html = profile.to_html()
            result["html_size_bytes"] = len(html.encode("utf-8"))
            result["status"] = "ok"
            (OUTPUT_DIR / f"{name}-ydata.html").write_text(html, encoding="utf-8")
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:500]
        result["notes"].append(f"crash: {type(exc).__name__}")

    result["time_seconds"] = round(time.perf_counter() - t0, 2)
    return result


def run_sweetviz(df, name):
    """Run sweetviz."""
    import sweetviz as sv

    t0 = time.perf_counter()
    result = {"tool": "sweetviz", "time_seconds": None, "html_size_bytes": 0,
              "status": "pending", "error": None, "notes": []}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = sv.analyze(df)
            html_path = str(OUTPUT_DIR / f"{name}-sweetviz.html")
            report.show_html(html_path, open_browser=False)
            p = Path(html_path)
            result["html_size_bytes"] = p.stat().st_size if p.exists() else 0
            result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:500]
        result["notes"].append(f"crash: {type(exc).__name__}")

    result["time_seconds"] = round(time.perf_counter() - t0, 2)
    return result


def main():
    all_results = []

    for ds in DATASETS:
        name = ds["name"]
        csv_path = ds["dir"] / ds["file"]
        sep = ds.get("sep", ",")

        print(f"\n{'='*60}", flush=True)
        print(f"  BASELINE: {name}", flush=True)
        print(f"{'='*60}", flush=True)

        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found", flush=True)
            continue

        df = pd.read_csv(csv_path, sep=sep, encoding="utf-8")
        print(f"  Loaded: {len(df)} rows × {len(df.columns)} cols", flush=True)

        entry = {"name": name, "rows": len(df), "cols": len(df.columns)}

        # ydata-profiling
        print(f"  Running ydata-profiling...", flush=True)
        ydata_res = run_ydata(df, name)
        print(f"  ydata: {ydata_res['status']} | {ydata_res['time_seconds']}s | {ydata_res['html_size_bytes']:,} bytes", flush=True)
        if ydata_res["error"]:
            print(f"    ERROR: {ydata_res['error'][:100]}", flush=True)
        entry["ydata"] = ydata_res

        # sweetviz
        print(f"  Running sweetviz...", flush=True)
        sv_res = run_sweetviz(df, name)
        print(f"  sweetviz: {sv_res['status']} | {sv_res['time_seconds']}s | {sv_res['html_size_bytes']:,} bytes", flush=True)
        if sv_res["error"]:
            print(f"    ERROR: {sv_res['error'][:100]}", flush=True)
        entry["sweetviz"] = sv_res

        all_results.append(entry)

    # Save
    out_path = OUTPUT_DIR / "baseline-results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"  BASELINE SUMMARY — ydata-profiling vs sweetviz", flush=True)
    print(f"{'='*80}", flush=True)
    header = f"{'Dataset':<25} {'Rows':>7} {'Cols':>5} | {'ydata(s)':>9} {'ydata KB':>9} {'ydata status':>13} | {'sv(s)':>6} {'sv KB':>7} {'sv status':>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in all_results:
        y = r["ydata"]
        s = r["sweetviz"]
        print(f"{r['name']:<25} {r['rows']:>7} {r['cols']:>5} | {y['time_seconds']:>9} {y['html_size_bytes']//1024:>9} {y['status']:>13} | {s['time_seconds']:>6} {s['html_size_bytes']//1024:>7} {s['status']:>10}", flush=True)
    print(f"\n  Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()