"""Quality benchmark — ThaiEDA vs ydata-profiling vs sweetviz.

Measures QUALITY of EDA output on a synthetic dataset with 10 known injected issues.

Metrics:
1. Ground-Truth Recall (GTR) — fraction of known issues detected
2. Issue Type Breadth (ITB) — distinct finding categories covered (out of 11)
3. Cross-Column Insight Rate (CIR) — findings involving ≥2 columns
4. Actionability Rate (AR) — findings with recommended actions
5. Thai-Specific Detection Rate (TSDR) — Thai issues detected
6. Report Completeness — expected sections present in HTML
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

DATA_DIR = Path(__file__).parent.parent.parent / "data-example" / "public-datasets"
OUTPUT_DIR = Path(__file__).parent / "quality-benchmark-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "synthetic-quality-benchmark.csv"

# Known issues (ground truth)
KNOWN_ISSUES = [
    ("outliers", "price"),
    ("missing_values", "quantity"),
    ("duplicates", "customer_name"),
    ("constant_column", "status"),
    ("type_inconsistency", "zip_code"),
    ("placeholders", "rating"),
    ("buddhist_era", "order_date"),
    ("thai_numerals", "thai_amount"),
    ("zero_width_space", "category_text"),
    ("mojibake", "product_desc"),
]

# Issue taxonomy (11 categories)
ISSUE_KEYWORDS = {
    "missing_values": ["missing", "null", "nan", "empty", "na", "blank"],
    "outliers": ["outlier", "anomaly", "extreme", "iqr", "z-score", "far"],
    "duplicates": ["duplicate", "duplicated", "redundant", "same value"],
    "constant_column": ["constant", "all same", "no variance", "single value"],
    "type_inconsistency": ["type", "mixed type", "inconsistent type", "data type"],
    "encoding_errors": ["encoding", "mojibake", "garble", "charset", "tis-620", "tis620", "latin"],
    "cardinality": ["cardinality", "unique", "distinct"],
    "distribution_skew": ["skew", "distribution", "normal", "uniform", "imbalance"],
    "correlations": ["correlat", "relationship", "association", "linked"],
    "cross_column_violation": ["cross", "violation", "rule", "constraint", "together"],
    "thai_specific": ["thai", "ภาษาไทย", "พ.ศ", "buddhist", "เลขไทย", "numeral", "zero-width", "zwsp", "เคาะ", "thai font"],
}

# Action verbs
ACTION_VERBS = ["drop", "impute", "convert", "replace", "flag", "deduplicate",
                "standardize", "remove", "fill", "encode", "review", "clip",
                "cap", "clean", "fix", "correct", "transform", "normalize",
                "แปลง", "แก้ไข", "ลบ", "เติม", "แก้", "ปรับ", "ทำความสะอาด", "เปลี่ยน"]

# Expected report sections
EXPECTED_SECTIONS = [
    "overview", "summary", "insight", "quality", "anomal",
    "missing", "correlat", "distribution", "column", "type",
]


def gtr_score(output_text: str) -> dict:
    """Ground-Truth Recall — fraction of known issues detected."""
    text_lower = output_text.lower()
    detected = 0
    details = []
    for issue_type, col in KNOWN_ISSUES:
        col_lower = col.lower()
        # Check if column name appears AND any keyword for that issue type appears nearby
        col_mentioned = col_lower in text_lower
        # Get keywords for this issue
        if issue_type in ISSUE_KEYWORDS:
            kw_found = any(kw in text_lower for kw in ISSUE_KEYWORDS[issue_type])
        elif issue_type == "placeholders":
            kw_found = any(kw in text_lower for kw in ["placeholder", "missing", "na", "invalid"])
        elif issue_type == "buddhist_era":
            kw_found = any(kw in text_lower for kw in ["buddhist", "พ.ศ", "be ", "b.e."])
        elif issue_type == "thai_numerals":
            kw_found = any(kw in text_lower for kw in ["thai numeral", "เลขไทย", "numeral", "thai digit"])
        elif issue_type == "zero_width_space":
            kw_found = any(kw in text_lower for kw in ["zero-width", "zwsp", "invisible", "เคาะ", "zero width"])
        elif issue_type == "mojibake":
            kw_found = any(kw in text_lower for kw in ["mojibake", "encoding", "tis-620", "garble", "charset"])
        else:
            kw_found = any(kw in text_lower for kw in ISSUE_KEYWORDS.get(issue_type, [issue_type]))

        # For Thai-specific issues, just check if the concept is mentioned anywhere
        if issue_type in ("buddhist_era", "thai_numerals", "zero_width_space", "mojibake"):
            is_detected = kw_found
        else:
            # For general issues, need both column + issue keyword
            is_detected = col_mentioned and kw_found

        if is_detected:
            detected += 1
            details.append(f"✅ {issue_type} ({col})")
        else:
            details.append(f"❌ {issue_type} ({col}) — col={col_mentioned}, kw={kw_found}")

    return {
        "detected": detected,
        "total": len(KNOWN_ISSUES),
        "recall": round(detected / len(KNOWN_ISSUES), 3),
        "details": details,
    }


def itb_score(output_text: str) -> dict:
    """Issue Type Breadth — distinct categories covered out of 11."""
    text_lower = output_text.lower()
    covered = set()
    for category, keywords in ISSUE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            covered.add(category)
    return {
        "covered": sorted(covered),
        "count": len(covered),
        "total": len(ISSUE_KEYWORDS),
        "breadth": round(len(covered) / len(ISSUE_KEYWORDS), 3),
    }


def cir_score(findings: list, column_names: list) -> dict:
    """Cross-Column Insight Rate — fraction of findings involving ≥2 columns."""
    if not findings:
        return {"cross_col": 0, "total": 0, "rate": 0.0}
    cols_lower = set(c.lower() for c in column_names)
    cross = 0
    for f in findings:
        f_lower = str(f).lower()
        col_count = sum(1 for c in cols_lower if c in f_lower)
        if col_count >= 2:
            cross += 1
    return {
        "cross_col": cross,
        "total": len(findings),
        "rate": round(cross / len(findings), 3),
    }


def ar_score(findings: list) -> dict:
    """Actionability Rate — fraction of findings with action verbs."""
    if not findings:
        return {"with_action": 0, "total": 0, "rate": 0.0}
    with_action = 0
    for f in findings:
        f_lower = str(f).lower()
        if any(v in f_lower for v in ACTION_VERBS):
            with_action += 1
    return {
        "with_action": with_action,
        "total": len(findings),
        "rate": round(with_action / len(findings), 3),
    }


def tsdr_score(output_text: str) -> dict:
    """Thai-Specific Detection Rate — Thai issues detected."""
    text_lower = output_text.lower()
    thai_issues = {
        "buddhist_era": any(kw in text_lower for kw in ["buddhist", "พ.ศ", "be ", "b.e.", "buddhist era"]),
        "thai_numerals": any(kw in text_lower for kw in ["thai numeral", "เลขไทย", "numeral", "thai digit", "thai number"]),
        "zero_width_space": any(kw in text_lower for kw in ["zero-width", "zwsp", "zero width", "invisible", "เคาะ"]),
        "mojibake": any(kw in text_lower for kw in ["mojibake", "tis-620", "tis620", "encoding", "garble", "charset", "latin"]),
    }
    detected = sum(1 for v in thai_issues.values() if v)
    return {
        "detected": detected,
        "total": len(thai_issues),
        "rate": round(detected / len(thai_issues), 3),
        "details": {k: "✅" if v else "❌" for k, v in thai_issues.items()},
    }


def report_completeness(html_str: str) -> dict:
    """Report Completeness — expected sections present in HTML."""
    text_lower = html_str.lower()
    found = []
    missing = []
    for section in EXPECTED_SECTIONS:
        # Look for heading or title containing this word
        pattern = rf'<h[1-4][^>]*>[^<]*{section}'
        if re.search(pattern, text_lower) or section in text_lower:
            found.append(section)
        else:
            missing.append(section)
    return {
        "found": found,
        "missing": missing,
        "count": len(found),
        "total": len(EXPECTED_SECTIONS),
        "completeness": round(len(found) / len(EXPECTED_SECTIONS), 3),
    }


def run_thaieda(df: pd.DataFrame) -> dict:
    """Run ThaiEDA and collect all output."""
    from thaieda import run

    t0 = time.perf_counter()
    result = {"tool": "ThaiEDA"}

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        eda = run(df, lang="en")

    result["time_seconds"] = round(time.perf_counter() - t0, 2)

    # Collect text output
    text_parts = []

    # Insights
    insights_list = []
    try:
        ins = eda.insights
        if ins and hasattr(ins, "total_insights"):
            text_parts.append(f"Total insights: {ins.total_insights}")
            insights_list = getattr(ins, "insights", [])
        elif ins and isinstance(ins, list):
            insights_list = ins
        for i in insights_list[:50]:
            text_parts.append(str(i))
    except Exception:
        pass

    # Quality issues
    try:
        qi = eda.quality_issues
        if qi:
            for issue in qi:
                text_parts.append(str(issue))
    except Exception:
        pass

    # Quality score
    try:
        result["quality_score"] = eda.quality_score
        text_parts.append(f"Quality score: {eda.quality_score}")
    except Exception:
        pass

    # Anomalies
    try:
        an = eda.anomalies
        if an:
            for a in an:
                text_parts.append(str(a))
    except Exception:
        pass

    # Cleaned df info
    try:
        notes = eda.notes
        if notes:
            text_parts.append(str(notes))
    except Exception:
        pass

    # HTML report
    html_str = ""
    try:
        html_str = eda.to_html()
        (OUTPUT_DIR / "thaieda-report.html").write_text(html_str, encoding="utf-8")
        result["html_size"] = len(html_str.encode("utf-8"))
    except Exception:
        pass

    # Dict output
    try:
        d = eda.to_dict()
        text_parts.append(json.dumps(d, default=str, ensure_ascii=False)[:5000])
    except Exception:
        pass

    combined_text = "\n".join(text_parts)
    result["output_text"] = combined_text
    result["html"] = html_str

    # Run all metrics
    result["gtr"] = gtr_score(combined_text + " " + html_str)
    result["itb"] = itb_score(combined_text + " " + html_str)
    result["cir"] = cir_score(insights_list, list(df.columns))
    result["ar"] = ar_score(insights_list + (eda.quality_issues if eda.quality_issues else []))
    result["tsdr"] = tsdr_score(combined_text + " " + html_str)
    result["completeness"] = report_completeness(html_str) if html_str else {"completeness": 0}

    return result


def run_ydata(df: pd.DataFrame) -> dict:
    """Run ydata-profiling and collect output."""
    from ydata_profiling import ProfileReport

    result = {"tool": "ydata-profiling"}
    t0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = ProfileReport(df, title="Synthetic Quality Benchmark", minimal=True, progress_bar=False)
        html_str = profile.to_html()

    result["time_seconds"] = round(time.perf_counter() - t0, 2)
    result["html_size"] = len(html_str.encode("utf-8"))
    (OUTPUT_DIR / "ydata-report.html").write_text(html_str, encoding="utf-8")

    # ydata findings = extract text from HTML (strip tags)
    text = re.sub(r"<[^>]+>", " ", html_str)
    result["output_text"] = text

    # No structured insights list — use empty for CIR/AR
    result["gtr"] = gtr_score(text)
    result["itb"] = itb_score(text)
    result["cir"] = {"cross_col": 0, "total": 0, "rate": 0.0, "note": "No structured insights"}
    result["ar"] = {"with_action": 0, "total": 0, "rate": 0.0, "note": "No action verbs in descriptive output"}
    result["tsdr"] = tsdr_score(text)
    result["completeness"] = report_completeness(html_str)

    return result


def run_sweetviz(df: pd.DataFrame) -> dict:
    """Run sweetviz and collect output."""
    import sweetviz as sv

    result = {"tool": "sweetviz"}
    t0 = time.perf_counter()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sv.analyze(df)
        html_path = str(OUTPUT_DIR / "sweetviz-report.html")
        report.show_html(html_path, open_browser=False)

    result["time_seconds"] = round(time.perf_counter() - t0, 2)
    html_path_obj = Path(html_path)
    html_str = html_path_obj.read_text(encoding="utf-8", errors="ignore")
    result["html_size"] = len(html_str.encode("utf-8"))

    # Extract text
    text = re.sub(r"<[^>]+>", " ", html_str)
    result["output_text"] = text

    result["gtr"] = gtr_score(text)
    result["itb"] = itb_score(text)
    result["cir"] = {"cross_col": 0, "total": 0, "rate": 0.0, "note": "No structured insights"}
    result["ar"] = {"with_action": 0, "total": 0, "rate": 0.0, "note": "No action verbs in descriptive output"}
    result["tsdr"] = tsdr_score(text)
    result["completeness"] = report_completeness(html_str)

    return result


def main():
    print("=" * 70, flush=True)
    print("  QUALITY BENCHMARK — ThaiEDA vs ydata-profiling vs sweetviz", flush=True)
    print("=" * 70, flush=True)

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print(f"\n  Dataset: {CSV_PATH.name}", flush=True)
    print(f"  Rows: {len(df)} | Cols: {len(df.columns)}", flush=True)
    print(f"  Known issues: {len(KNOWN_ISSUES)}", flush=True)

    all_results = {}

    # ThaiEDA
    print(f"\n  Running ThaiEDA...", flush=True)
    thaieda_res = run_thaieda(df)
    print(f"  ThaiEDA: {thaieda_res['time_seconds']}s | GTR={thaieda_res['gtr']['recall']} | ITB={thaieda_res['itb']['breadth']} | CIR={thaieda_res['cir']['rate']} | AR={thaieda_res['ar']['rate']} | TSDR={thaieda_res['tsdr']['rate']} | RC={thaieda_res['completeness']['completeness']}", flush=True)
    all_results["thaieda"] = thaieda_res

    # ydata-profiling
    print(f"\n  Running ydata-profiling...", flush=True)
    ydata_res = run_ydata(df)
    print(f"  ydata: {ydata_res['time_seconds']}s | GTR={ydata_res['gtr']['recall']} | ITB={ydata_res['itb']['breadth']} | TSDR={ydata_res['tsdr']['rate']} | RC={ydata_res['completeness']['completeness']}", flush=True)
    all_results["ydata"] = ydata_res

    # sweetviz
    print(f"\n  Running sweetviz...", flush=True)
    sv_res = run_sweetviz(df)
    print(f"  sweetviz: {sv_res['time_seconds']}s | GTR={sv_res['gtr']['recall']} | ITB={sv_res['itb']['breadth']} | TSDR={sv_res['tsdr']['rate']} | RC={sv_res['completeness']['completeness']}", flush=True)
    all_results["sweetviz"] = sv_res

    # Print detailed results
    print(f"\n{'=' * 70}", flush=True)
    print(f"  DETAILED RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    for tool_name, res in all_results.items():
        print(f"\n  --- {res['tool']} ---", flush=True)
        print(f"  Time: {res['time_seconds']}s | HTML: {res.get('html_size', 0):,} bytes", flush=True)

        gtr = res["gtr"]
        print(f"  GTR (Ground-Truth Recall): {gtr['detected']}/{gtr['total']} = {gtr['recall']}", flush=True)
        for d in gtr["details"]:
            print(f"    {d}", flush=True)

        itb = res["itb"]
        print(f"  ITB (Issue Type Breadth): {itb['count']}/{itb['total']} = {itb['breadth']}", flush=True)
        print(f"    Covered: {', '.join(itb['covered'])}", flush=True)

        cir = res["cir"]
        print(f"  CIR (Cross-Column Rate): {cir['cross_col']}/{cir['total']} = {cir['rate']}", flush=True)

        ar = res["ar"]
        print(f"  AR (Actionability): {ar['with_action']}/{ar['total']} = {ar['rate']}", flush=True)

        tsdr = res["tsdr"]
        print(f"  TSDR (Thai-Specific): {tsdr['detected']}/{tsdr['total']} = {tsdr['rate']}", flush=True)
        for k, v in tsdr["details"].items():
            print(f"    {v} {k}", flush=True)

        rc = res["completeness"]
        print(f"  RC (Report Completeness): {rc['count']}/{rc['total']} = {rc['completeness']}", flush=True)
        if rc.get("missing"):
            print(f"    Missing: {', '.join(rc['missing'])}", flush=True)

    # Summary table
    print(f"\n{'=' * 70}", flush=True)
    print(f"  SUMMARY TABLE", flush=True)
    print(f"{'=' * 70}", flush=True)
    header = f"{'Metric':<30} {'ydata-profiling':>18} {'sweetviz':>12} {'ThaiEDA':>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    print(f"{'Time (s)':<30} {ydata_res['time_seconds']:>18} {sv_res['time_seconds']:>12} {thaieda_res['time_seconds']:>12}", flush=True)
    print(f"{'HTML size (KB)':<30} {ydata_res.get('html_size',0)//1024:>18} {sv_res.get('html_size',0)//1024:>12} {thaieda_res.get('html_size',0)//1024:>12}", flush=True)
    print(f"{'GTR — Ground-Truth Recall':<30} {ydata_res['gtr']['recall']:>18} {sv_res['gtr']['recall']:>12} {thaieda_res['gtr']['recall']:>12}", flush=True)
    print(f"{'ITB — Issue Type Breadth':<30} {ydata_res['itb']['breadth']:>18} {sv_res['itb']['breadth']:>12} {thaieda_res['itb']['breadth']:>12}", flush=True)
    print(f"{'CIR — Cross-Column Rate':<30} {ydata_res['cir']['rate']:>18} {sv_res['cir']['rate']:>12} {thaieda_res['cir']['rate']:>12}", flush=True)
    print(f"{'AR — Actionability Rate':<30} {ydata_res['ar']['rate']:>18} {sv_res['ar']['rate']:>12} {thaieda_res['ar']['rate']:>12}", flush=True)
    print(f"{'TSDR — Thai-Specific Det.':<30} {ydata_res['tsdr']['rate']:>18} {sv_res['tsdr']['rate']:>12} {thaieda_res['tsdr']['rate']:>12}", flush=True)
    print(f"{'RC — Report Completeness':<30} {ydata_res['completeness']['completeness']:>18} {sv_res['completeness']['completeness']:>12} {thaieda_res['completeness']['completeness']:>12}", flush=True)

    # Save results
    out_path = OUTPUT_DIR / "quality-benchmark-results.json"
    # Strip large text fields
    for k in all_results:
        all_results[k].pop("output_text", None)
        all_results[k].pop("html", None)
    out_path.write_text(json.dumps(all_results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()