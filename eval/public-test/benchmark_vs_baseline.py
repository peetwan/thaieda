"""ThaiEDA vs Baseline Benchmark — เปรียบเทียบ ThaiEDA กับ raw pandas.

วัดทั้ง 14 datasets (public) + 5 datasets (Thai) = 19 datasets:
- Baseline: df.describe(), df.info(), df.isnull().sum() — สิ่งที่ DS ทั่วไปได้จาก pandas
- ThaiEDA: run(df) — full pipeline

เปรียบเทียบ:
1. Time (seconds)
2. Insights found (cross-column patterns)
3. Quality issues detected
4. Anomalies detected
5. Thai-specific issues (BE dates, Thai numerals, ZWSP, mojibake)
6. Charts generated
7. HTML report produced
8. Language detection
9. Column type detection
10. Data type classification
"""

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from thaieda import run  # noqa: E402

PUBLIC_DIR = Path(__file__).parent.parent.parent / "data-example" / "public-datasets"
THAI_DIR = Path(__file__).parent.parent.parent / "data-example" / "thai-datasets"
OUTPUT_DIR = Path(__file__).parent / "benchmark-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = [
    # --- 14 public datasets ---
    {"name": "titanic", "file": "titanic.csv", "dir": PUBLIC_DIR, "domain": "passenger survival", "has_thai": False},
    {"name": "telco-churn", "file": "telco-churn.csv", "dir": PUBLIC_DIR, "domain": "telecom churn", "has_thai": False},
    {"name": "wine-quality", "file": "winequality-red.csv", "dir": PUBLIC_DIR, "domain": "wine chemistry", "has_thai": False},
    {"name": "california-housing", "file": "california-housing.csv", "dir": PUBLIC_DIR, "domain": "housing census", "has_thai": False},
    {"name": "superstore", "file": "superstore.csv", "dir": PUBLIC_DIR, "domain": "retail sales", "has_thai": False},
    {"name": "adult", "file": "adult.csv", "dir": PUBLIC_DIR, "domain": "census demographics", "has_thai": False},
    {"name": "bank-marketing", "file": "bank-marketing.csv", "dir": PUBLIC_DIR, "domain": "bank marketing", "has_thai": False, "sep": ";"},
    {"name": "online-retail", "file": "online-retail.csv", "dir": PUBLIC_DIR, "domain": "e-commerce transactions", "has_thai": False},
    {"name": "dirty-thai-retail", "file": "../dirty-thai-retail.csv", "dir": PUBLIC_DIR, "domain": "thai retail (dirty)", "has_thai": True},
    {"name": "absenteeism", "file": "Absenteeism_at_work.csv", "dir": PUBLIC_DIR, "domain": "workplace absenteeism", "has_thai": False, "sep": ";"},
    {"name": "online-shoppers", "file": "online-shoppers.csv", "dir": PUBLIC_DIR, "domain": "online shopping intention", "has_thai": False},
    {"name": "aps-failure", "file": "aps-failure-clean.csv", "dir": PUBLIC_DIR, "domain": "truck diagnostics", "has_thai": False},
    {"name": "beijing-pm25", "file": "beijing-pm25.csv", "dir": PUBLIC_DIR, "domain": "air quality", "has_thai": False},
    {"name": "bike-sharing", "file": "bike-sharing-hour.csv", "dir": PUBLIC_DIR, "domain": "bike sharing", "has_thai": False},
    # --- 5 Thai datasets ---
    {"name": "wongnai-reviews-40k", "file": "wongnai-reviews-40k.csv", "dir": THAI_DIR, "domain": "Thai restaurant reviews", "has_thai": True},
    {"name": "wongnai-train-50k", "file": "wongnai-train-50k.csv", "dir": THAI_DIR, "domain": "Wongnai train reviews", "has_thai": True},
    {"name": "wisesight-sentiment", "file": "wisesight-sentiment.csv", "dir": THAI_DIR, "domain": "Thai social media sentiment", "has_thai": True},
    {"name": "thai-ecommerce-15k", "file": "thai-ecommerce-15k.csv", "dir": THAI_DIR, "domain": "Thai e-commerce products", "has_thai": True},
    {"name": "thai-restaurant-hybrid-20k", "file": "thai-restaurant-hybrid-20k.csv", "dir": THAI_DIR, "domain": "Thai restaurant hybrid (dirty)", "has_thai": True},
]


def baseline_analysis(df: pd.DataFrame) -> dict:
    """สิ่งที่ DS ทั่วไปทได้จาก raw pandas — describe, info, isnull, dtypes."""
    t0 = time.perf_counter()

    result = {
        "describe_rows": 0,
        "describe_cols": 0,
        "null_counts": 0,
        "dtype_counts": 0,
        "unique_value_counts": 0,
        "thai_issues_found": 0,
        "insights": 0,
        "quality_issues": 0,
        "anomalies": 0,
        "charts": 0,
        "html_report": False,
        "language_detected": False,
        "column_types_detected": 0,
        "data_type_classified": False,
        "be_dates_detected": 0,
        "thai_numerals_detected": 0,
        "zwsp_detected": 0,
        "mojibake_detected": 0,
        "quality_score": None,
    }

    try:
        desc = df.describe(include="all")
        result["describe_rows"] = len(desc)
        result["describe_cols"] = len(desc.columns)
    except Exception:
        pass

    try:
        nulls = df.isnull().sum()
        result["null_counts"] = int(nulls.sum())
    except Exception:
        pass

    try:
        result["dtype_counts"] = len(df.dtypes.unique())
    except Exception:
        pass

    try:
        for col in df.columns:
            if df[col].nunique() < 50:
                result["unique_value_counts"] += 1
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    result["time_seconds"] = round(elapsed, 2)
    return result


def thaieda_analysis(df: pd.DataFrame, lang: str = "en") -> dict:
    """ThaiEDA full pipeline — run(df)."""
    t0 = time.perf_counter()
    result = {
        "insights": 0,
        "quality_issues": 0,
        "anomalies": 0,
        "charts": 0,
        "html_report": False,
        "html_size_bytes": 0,
        "language_detected": False,
        "column_types_detected": 0,
        "data_type_classified": False,
        "be_dates_detected": 0,
        "thai_numerals_detected": 0,
        "zwsp_detected": 0,
        "mojibake_detected": 0,
        "quality_score": None,
        "thai_issues_found": 0,
    }

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        eda = run(df, lang=lang)

    elapsed = time.perf_counter() - t0
    result["time_seconds"] = round(elapsed, 2)

    # insights
    try:
        ins = eda.insights
        if ins and hasattr(ins, "total_insights"):
            result["insights"] = ins.total_insights
        elif ins and isinstance(ins, list):
            result["insights"] = len(ins)
    except Exception:
        pass

    # quality
    try:
        qi = eda.quality_issues
        result["quality_issues"] = len(qi) if qi else 0
    except Exception:
        pass

    # quality score
    try:
        result["quality_score"] = eda.quality_score
    except Exception:
        pass

    # anomalies
    try:
        an = eda.anomalies
        result["anomalies"] = len(an) if an else 0
    except Exception:
        pass

    # HTML report
    try:
        html = eda.to_html()
        result["html_report"] = True
        result["html_size_bytes"] = len(html.encode("utf-8"))
        import re
        result["charts"] = len(re.findall(r"data:image/png;base64", html))
    except Exception:
        pass

    # language detection
    try:
        d = eda.to_dict()
        if d.get("language") or d.get("pre_analysis", {}).get("language"):
            result["language_detected"] = True
        pa = d.get("pre_analysis", {})
        if pa.get("data_type") or pa.get("label"):
            result["data_type_classified"] = True
        result["column_types_detected"] = len(d.get("column_types", {}))
    except Exception:
        pass

    # Thai-specific: count from quality issues
    try:
        qi = eda.quality_issues or []
        for issue in qi:
            msg = str(issue).lower() if isinstance(issue, str) else str(issue.get("message", "")).lower()
            if any(kw in msg for kw in ["พ.ศ.", "buddhist", "be "]):
                result["be_dates_detected"] += 1
            if any(kw in msg for kw in ["เลขไทย", "thai numeral", "numeral"]):
                result["thai_numerals_detected"] += 1
            if any(kw in msg for kw in ["zero-width", "zwsp", "เคาะ"]):
                result["zwsp_detected"] += 1
            if any(kw in msg for kw in ["mojibake", "encoding", "tis-620"]):
                result["mojibake_detected"] += 1
    except Exception:
        pass

    result["thai_issues_found"] = (
        result["be_dates_detected"]
        + result["thai_numerals_detected"]
        + result["zwsp_detected"]
        + result["mojibake_detected"]
    )

    return result


def main():
    all_results = []

    for ds in DATASETS:
        name = ds["name"]
        csv_path = ds["dir"] / ds["file"]
        sep = ds.get("sep", ",")

        print(f"\n{'='*60}", flush=True)
        print(f"  BENCHMARK: {name} ({ds['domain']})", flush=True)
        print(f"{'='*60}", flush=True)

        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found", flush=True)
            all_results.append({"name": name, "status": "missing", "file": str(csv_path)})
            continue

        entry = {
            "name": name,
            "domain": ds["domain"],
            "has_thai": ds["has_thai"],
            "rows": None,
            "cols": None,
            "baseline": None,
            "thaieda": None,
            "status": "pending",
        }

        try:
            df = pd.read_csv(csv_path, sep=sep, encoding="utf-8")
            entry["rows"] = len(df)
            entry["cols"] = len(df.columns)
            print(f"  Loaded: {len(df)} rows × {len(df.columns)} cols", flush=True)
        except Exception as exc:
            print(f"  LOAD ERROR: {exc}", flush=True)
            entry["status"] = "load_error"
            entry["error"] = str(exc)
            all_results.append(entry)
            continue

        # Baseline
        print(f"  Running baseline (raw pandas)...", flush=True)
        baseline = baseline_analysis(df)
        print(f"  Baseline: {baseline['time_seconds']}s | insights={baseline['insights']} | quality={baseline['quality_issues']}", flush=True)

        # ThaiEDA
        lang = "th" if ds["has_thai"] else "en"
        print(f"  Running ThaiEDA (lang={lang})...", flush=True)
        thaieda = thaieda_analysis(df, lang=lang)
        print(f"  ThaiEDA: {thaieda['time_seconds']}s | insights={thaieda['insights']} | quality={thaieda['quality_issues']} | anomalies={thaieda['anomalies']} | charts={thaieda['charts']}", flush=True)
        if thaieda["thai_issues_found"]:
            print(f"  Thai-specific: BE={thaieda['be_dates_detected']} numerals={thaieda['thai_numerals_detected']} zwsp={thaieda['zwsp_detected']} mojibake={thaieda['mojibake_detected']}", flush=True)

        entry["baseline"] = baseline
        entry["thaieda"] = thaieda
        entry["status"] = "ok"
        all_results.append(entry)

    # Save results
    out_path = OUTPUT_DIR / "benchmark-results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    # Print summary table
    print(f"\n{'='*80}", flush=True)
    print(f"  BENCHMARK SUMMARY — ThaiEDA vs Baseline (raw pandas)", flush=True)
    print(f"{'='*80}", flush=True)

    header = f"{'Dataset':<25} {'Rows':>7} {'Cols':>5} {'Base(s)':>8} {'EDA(s)':>8} {'Insights':>9} {'Quality':>8} {'Anomalies':>10} {'Charts':>7} {'Thai':>5}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    total_insights = 0
    total_quality = 0
    total_anomalies = 0
    total_charts = 0
    total_thai_issues = 0
    ok_count = 0

    for r in all_results:
        if r.get("status") != "ok":
            continue
        ok_count += 1
        b = r["baseline"]
        t = r["thaieda"]
        insights = t["insights"]
        quality = t["quality_issues"]
        anomalies = t["anomalies"]
        charts = t["charts"]
        thai = t["thai_issues_found"]

        total_insights += insights
        total_quality += quality
        total_anomalies += anomalies
        total_charts += charts
        total_thai_issues += thai

        print(
            f"{r['name']:<25} {r['rows']:>7} {r['cols']:>5} {b['time_seconds']:>8} {t['time_seconds']:>8} {insights:>9} {quality:>8} {anomalies:>10} {charts:>7} {thai:>5}",
            flush=True,
        )

    print("-" * len(header), flush=True)
    print(f"{'TOTAL':<25} {'':>7} {'':>5} {'':>8} {'':>8} {total_insights:>9} {total_quality:>8} {total_anomalies:>10} {total_charts:>7} {total_thai_issues:>5}", flush=True)
    print(f"\n  Datasets: {ok_count} OK", flush=True)
    print(f"  Baseline total insights: 0 (pandas doesn't find cross-column patterns)", flush=True)
    print(f"  ThaiEDA total insights: {total_insights}", flush=True)
    print(f"  ThaiEDA total quality issues: {total_quality}", flush=True)
    print(f"  ThaiEDA total anomalies: {total_anomalies}", flush=True)
    print(f"  ThaiEDA total charts: {total_charts}", flush=True)
    print(f"  ThaiEDA Thai-specific issues: {total_thai_issues}", flush=True)
    print(f"\n  Results: {out_path}", flush=True)

    return all_results


if __name__ == "__main__":
    main()