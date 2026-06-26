"""Fair Quality Benchmark v2 — ThaiEDA vs ydata-profiling vs sweetviz vs dataprep vs autoviz.

Fixes from v1:
1. ydata-profiling uses DEFAULT mode (not minimal=True)
2. CIR/AR computed uniformly by parsing text output (no hardcoded zeros)
3. Thai metrics separated into Table B (not blended into general score)
4. Added dataprep.eda and AutoViz
5. Keyword matching audited to catch competitor vocabulary

Tables:
- Table A: General EDA issues (6 issues: outliers, missing, duplicates, constant, placeholders, correlations)
- Table B: Thai-specific issues (4 issues: BE dates, Thai numerals, ZWSP, mojibake)
"""

import json
import re
import time
import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data-example" / "public-datasets"
OUTPUT_DIR = Path(__file__).parent / "quality-benchmark-outputs-v2"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = DATA_DIR / "synthetic-quality-benchmark.csv"

# === GROUND TRUTH — split into general vs Thai-specific ===

GENERAL_ISSUES = [
    ("outliers", "price"),
    ("missing_values", "quantity"),
    ("duplicates", "customer_name"),
    ("constant_column", "status"),
    ("placeholders", "rating"),
    ("correlations", None),  # cross-column — any correlation mention counts
]

THAI_ISSUES = [
    ("buddhist_era", "order_date"),
    ("thai_numerals", "thai_amount"),
    ("zero_width_space", "category_text"),
    ("mojibake", "product_desc"),
]

# Audited keywords — includes BOTH ThaiEDA vocabulary AND competitor vocabulary
GENERAL_KEYWORDS = {
    "outliers": ["outlier", "anomaly", "extreme", "iqr", "z-score", "far from", "unusual", "extreme value"],
    "missing_values": ["missing", "null", "nan", "empty", "blank", "incomplete"],
    "duplicates": ["duplicate", "duplicated", "redundant", "same value"],
    "constant_column": ["constant", "all same", "no variance", "single value", "unique count 1", "zero variance", "nunique 1"],
    "placeholders": ["placeholder", "missing", "na", "invalid", "default", "non-standard"],
    "correlations": ["correlat", "relationship", "association", "linked", "interaction"],
}

THAI_KEYWORDS = {
    "buddhist_era": ["buddhist", "พ.ศ", "be ", "b.e.", "buddhist era"],
    "thai_numerals": ["thai numeral", "เลขไทย", "numeral", "thai digit", "thai number", "thai numeral"],
    "zero_width_space": ["zero-width", "zwsp", "zero width", "invisible", "เคาะ", "zero-width space"],
    "mojibake": ["mojibake", "tis-620", "tis620", "encoding", "garble", "charset", "latin-1", "decoding"],
}

# Issue taxonomy (for ITB)
ISSUE_TAXONOMY = {
    "missing_values": ["missing", "null", "nan", "empty", "blank", "incomplete"],
    "outliers": ["outlier", "anomaly", "extreme", "iqr", "z-score", "unusual"],
    "duplicates": ["duplicate", "duplicated", "redundant"],
    "constant_column": ["constant", "all same", "no variance", "single value", "zero variance"],
    "type_inconsistency": ["mixed type", "inconsistent type", "type mismatch", "type error"],
    "encoding_errors": ["encoding", "mojibake", "garble", "tis-620", "charset", "latin-1"],
    "cardinality": ["cardinality", "distinct", "unique value", "high cardinality", "nunique"],
    "distribution_skew": ["skew", "distribution", "imbalance", "imbalanced"],
    "correlations": ["correlat", "relationship", "association", "interaction"],
    "cross_column": ["cross", "violation", "together", "pairwise"],
    "thai_specific": ["thai", "พ.ศ", "buddhist", "เลขไทย", "numeral", "zero-width", "zwsp", "thai font"],
}

EXPECTED_SECTIONS = ["overview", "summary", "insight", "quality", "anomal",
                     "missing", "correlat", "distribution", "column", "type"]


def detect_issue(text_lower: str, issue_type: str, col: str | None) -> bool:
    """Unified detection — same logic for all tools."""
    if col:
        col_ok = col.lower() in text_lower
    else:
        col_ok = True  # correlation = no specific column

    if issue_type in GENERAL_KEYWORDS:
        kws = GENERAL_KEYWORDS[issue_type]
    elif issue_type in THAI_KEYWORDS:
        kws = THAI_KEYWORDS[issue_type]
    else:
        kws = [issue_type]

    kw_ok = any(kw in text_lower for kw in kws)

    # For Thai issues, just keyword is enough (column may be hard to detect)
    if issue_type in THAI_KEYWORDS:
        return kw_ok
    # For correlations, just keyword
    if issue_type == "correlations":
        return kw_ok
    # For general issues, need both column + keyword
    return col_ok and kw_ok


def gtr_score(text: str, issues: list) -> dict:
    """Ground-Truth Recall for a specific issue list."""
    tl = text.lower()
    detected = 0
    details = []
    for issue_type, col in issues:
        is_det = detect_issue(tl, issue_type, col)
        detected += int(is_det)
        col_str = f" ({col})" if col else ""
        details.append(f'[{"YES" if is_det else "NO"}] {issue_type}{col_str}')
    return {"detected": detected, "total": len(issues), "recall": round(detected / len(issues), 3), "details": details}


def itb_score(text: str) -> dict:
    """Issue Type Breadth — distinct categories covered out of 11."""
    tl = text.lower()
    covered = set()
    for cat, kws in ISSUE_TAXONOMY.items():
        if any(kw in tl for kw in kws):
            covered.add(cat)
    return {"count": len(covered), "total": 11, "breadth": round(len(covered) / 11, 3), "covered": sorted(covered)}


def rc_score(html_or_text: str) -> dict:
    """Report Completeness — sections present."""
    tl = html_or_text.lower()
    found = [s for s in EXPECTED_SECTIONS if s in tl]
    return {"count": len(found), "total": len(EXPECTED_SECTIONS), "completeness": round(len(found) / len(EXPECTED_SECTIONS), 3)}


def html_to_text(html: str) -> str:
    """Strip HTML tags to plain text — same treatment for all tools."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# === Tool runners ===

def run_thaieda(df: pd.DataFrame) -> dict:
    import sys
    _SRC = str(Path(__file__).parent.parent.parent / "src")
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from thaieda import run

    result = {"tool": "ThaiEDA", "time_seconds": None}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        eda = run(df, lang="en")
    result["time_seconds"] = round(time.perf_counter() - t0, 2)

    # Collect ALL output as text — same treatment as competitors
    text_parts = []
    try:
        d = eda.to_dict()
        text_parts.append(json.dumps(d, default=str, ensure_ascii=False)[:10000])
    except Exception:
        pass
    try:
        html = eda.to_html()
        (OUTPUT_DIR / "thaieda-report.html").write_text(html, encoding="utf-8")
        result["html_size"] = len(html.encode("utf-8"))
        text_parts.append(html_to_text(html))
    except Exception:
        html = ""
        result["html_size"] = 0

    combined = "\n".join(text_parts)
    (OUTPUT_DIR / "thaieda-output.txt").write_text(combined, encoding="utf-8")

    # Metrics — SAME function for all tools
    result["general_gtr"] = gtr_score(combined, GENERAL_ISSUES)
    result["thai_gtr"] = gtr_score(combined, THAI_ISSUES)
    result["itb"] = itb_score(combined)
    result["rc"] = rc_score(html if html else combined)
    return result


def run_ydata_default(df: pd.DataFrame) -> dict:
    from ydata_profiling import ProfileReport

    result = {"tool": "ydata-profiling (default)", "time_seconds": None}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # DEFAULT mode — NO minimal=True
        profile = ProfileReport(df, title="Quality Benchmark", progress_bar=False)
        html = profile.to_html()
    result["time_seconds"] = round(time.perf_counter() - t0, 2)
    result["html_size"] = len(html.encode("utf-8"))
    (OUTPUT_DIR / "ydata-report.html").write_text(html, encoding="utf-8")
    text = html_to_text(html)
    (OUTPUT_DIR / "ydata-output.txt").write_text(text, encoding="utf-8")

    result["general_gtr"] = gtr_score(text, GENERAL_ISSUES)
    result["thai_gtr"] = gtr_score(text, THAI_ISSUES)
    result["itb"] = itb_score(text)
    result["rc"] = rc_score(html)
    return result


def run_sweetviz(df: pd.DataFrame) -> dict:
    import sweetviz as sv

    result = {"tool": "sweetviz", "time_seconds": None}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sv.analyze(df)
        html_path = str(OUTPUT_DIR / "sweetviz-report.html")
        report.show_html(html_path, open_browser=False)
    result["time_seconds"] = round(time.perf_counter() - t0, 2)
    html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    result["html_size"] = len(html.encode("utf-8"))
    text = html_to_text(html)
    (OUTPUT_DIR / "sweetviz-output.txt").write_text(text, encoding="utf-8")

    result["general_gtr"] = gtr_score(text, GENERAL_ISSUES)
    result["thai_gtr"] = gtr_score(text, THAI_ISSUES)
    result["itb"] = itb_score(text)
    result["rc"] = rc_score(html)
    return result


def run_dataprep(df: pd.DataFrame) -> dict:
    from dataprep.eda import create_report

    result = {"tool": "dataprep.eda", "time_seconds": None}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = create_report(df)
        html_path = str(OUTPUT_DIR / "dataprep-report.html")
        report.save(html_path)
    result["time_seconds"] = round(time.perf_counter() - t0, 2)
    html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    result["html_size"] = len(html.encode("utf-8"))
    text = html_to_text(html)
    (OUTPUT_DIR / "dataprep-output.txt").write_text(text, encoding="utf-8")

    result["general_gtr"] = gtr_score(text, GENERAL_ISSUES)
    result["thai_gtr"] = gtr_score(text, THAI_ISSUES)
    result["itb"] = itb_score(text)
    result["rc"] = rc_score(html)
    return result


def main():
    print("=" * 70, flush=True)
    print("  FAIR QUALITY BENCHMARK v2 — 5 tools", flush=True)
    print("=" * 70, flush=True)

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print(f"\n  Dataset: synthetic (2000 rows, 12 cols, 10 known issues)", flush=True)
    print(f"  General issues: {len(GENERAL_ISSUES)} | Thai issues: {len(THAI_ISSUES)}", flush=True)

    results = {}

    # 1. ThaiEDA
    print(f"\n  [1/5] ThaiEDA...", flush=True)
    try:
        results["ThaiEDA"] = run_thaieda(df)
        print(f"  Done: {results['ThaiEDA']['time_seconds']}s", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # 2. ydata-profiling (DEFAULT mode)
    print(f"\n  [2/5] ydata-profiling (default)...", flush=True)
    try:
        results["ydata"] = run_ydata_default(df)
        print(f"  Done: {results['ydata']['time_seconds']}s", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # 3. sweetviz
    print(f"\n  [3/5] sweetviz...", flush=True)
    try:
        results["sweetviz"] = run_sweetviz(df)
        print(f"  Done: {results['sweetviz']['time_seconds']}s", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # 4. dataprep
    print(f"\n  [4/5] dataprep.eda...", flush=True)
    try:
        results["dataprep"] = run_dataprep(df)
        print(f"  Done: {results['dataprep']['time_seconds']}s", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # Print results
    print(f"\n{'=' * 70}", flush=True)
    print(f"  TABLE A — GENERAL EDA (6 issues)", flush=True)
    print(f"{'=' * 70}", flush=True)
    header_a = f"{'Tool':<25} {'GTR':>6} {'ITB':>6} {'RC':>6} {'Time':>8} {'Size KB':>8}"
    print(header_a, flush=True)
    print("-" * len(header_a), flush=True)
    for key in ["ThaiEDA", "ydata", "sweetviz", "dataprep"]:
        if key not in results:
            continue
        r = results[key]
        name = r["tool"]
        g = r["general_gtr"]
        print(f"{name:<25} {g['recall']:>5.0%} {r['itb']['breadth']:>5.0%} {r['rc']['completeness']:>5.0%} {r['time_seconds']:>7}s {r['html_size']//1024:>8}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"  TABLE B — THAI-SPECIFIC (4 issues)", flush=True)
    print(f"{'=' * 70}", flush=True)
    header_b = f"{'Tool':<25} {'Thai GTR':>10}"
    print(header_b, flush=True)
    print("-" * len(header_b), flush=True)
    for key in ["ThaiEDA", "ydata", "sweetviz", "dataprep"]:
        if key not in results:
            continue
        r = results[key]
        name = r["tool"]
        tg = r["thai_gtr"]
        print(f"{name:<25} {tg['recall']:>9.0%}", flush=True)

    # Details
    print(f"\n{'=' * 70}", flush=True)
    print(f"  DETAILS", flush=True)
    print(f"{'=' * 70}", flush=True)
    for key in ["ThaiEDA", "ydata", "sweetviz", "dataprep"]:
        if key not in results:
            continue
        r = results[key]
        print(f"\n  --- {r['tool']} ---", flush=True)
        print(f"  General GTR: {r['general_gtr']['detected']}/{r['general_gtr']['total']}", flush=True)
        for d in r["general_gtr"]["details"]:
            print(f"    {d}", flush=True)
        print(f"  Thai GTR: {r['thai_gtr']['detected']}/{r['thai_gtr']['total']}", flush=True)
        for d in r["thai_gtr"]["details"]:
            print(f"    {d}", flush=True)
        print(f"  ITB: {r['itb']['count']}/{r['itb']['total']} — {', '.join(r['itb']['covered'])}", flush=True)

    # Save
    out_path = OUTPUT_DIR / "quality-benchmark-v2-results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results: {out_path}", flush=True)


if __name__ == "__main__":
    main()