"""Compute fair quality benchmark metrics v2 — from saved outputs.

Fairness fixes:
1. Same detection logic for ALL tools (no privileged structured access)
2. All tools: HTML → text (strip tags), same treatment
3. Thai metrics separated (Table B)
4. ydata uses DEFAULT mode (not minimal)
"""
import json
import re
from pathlib import Path

OUT = Path(__file__).parent / "quality-benchmark-outputs-v2"

# Load all outputs — all processed the same way (HTML stripped to text)
thaieda_text = (OUT / "thaieda-output.txt").read_text(encoding="utf-8")
thaieda_html = (OUT / "thaieda-report.html").read_text(encoding="utf-8")
ydata_text = (OUT / "ydata-output.txt").read_text(encoding="utf-8")
ydata_html = (OUT / "ydata-report.html").read_text(encoding="utf-8")
sv_text = (OUT / "sweetviz-output.txt").read_text(encoding="utf-8")
sv_html = (OUT / "sweetviz-report.html").read_text(encoding="utf-8")

# === GROUND TRUTH — split into general vs Thai-specific ===

GENERAL_ISSUES = [
    ("outliers", "price"),
    ("missing_values", "quantity"),
    ("duplicates", "customer_name"),
    ("constant_column", "status"),
    ("placeholders", "rating"),
    ("correlations", None),
]

THAI_ISSUES = [
    ("buddhist_era", "order_date"),
    ("thai_numerals", "thai_amount"),
    ("zero_width_space", "category_text"),
    ("mojibake", "product_desc"),
]

# Audited keywords — includes competitor vocabulary (ydata "Constant", "High cardinality", etc.)
GENERAL_KEYWORDS = {
    "outliers": ["outlier", "anomaly", "extreme", "iqr", "z-score", "unusual", "far from"],
    "missing_values": ["missing", "null", "nan", "empty", "blank", "incomplete"],
    "duplicates": ["duplicate", "duplicated", "redundant"],
    "constant_column": ["constant", "all same", "no variance", "single value", "zero variance", "nunique 1", "unique count 1"],
    "placeholders": ["placeholder", "missing", "na", "invalid", "default", "non-standard"],
    "correlations": ["correlat", "relationship", "association", "interaction", "linked"],
}

THAI_KEYWORDS = {
    "buddhist_era": ["buddhist", "พ.ศ", "be ", "b.e.", "buddhist era"],
    "thai_numerals": ["thai numeral", "เลขไทย", "numeral", "thai digit", "thai number"],
    "zero_width_space": ["zero-width", "zwsp", "zero width", "invisible", "เคาะ"],
    "mojibake": ["mojibake", "tis-620", "tis620", "encoding", "garble", "charset", "latin-1"],
}

ISSUE_TAXONOMY = {
    "missing_values": ["missing", "null", "nan", "empty", "blank", "incomplete"],
    "outliers": ["outlier", "anomaly", "extreme", "iqr", "z-score", "unusual"],
    "duplicates": ["duplicate", "duplicated", "redundant"],
    "constant_column": ["constant", "all same", "no variance", "single value", "zero variance"],
    "type_inconsistency": ["mixed type", "inconsistent type", "type mismatch"],
    "encoding_errors": ["encoding", "mojibake", "garble", "tis-620", "charset", "latin-1"],
    "cardinality": ["cardinality", "distinct", "unique value", "high cardinality", "nunique"],
    "distribution_skew": ["skew", "distribution", "imbalance", "imbalanced"],
    "correlations": ["correlat", "relationship", "association", "interaction"],
    "cross_column": ["cross", "violation", "together", "pairwise"],
    "thai_specific": ["thai", "พ.ศ", "buddhist", "เลขไทย", "numeral", "zero-width", "zwsp"],
}

EXPECTED_SECTIONS = ["overview", "summary", "insight", "quality", "anomal",
                     "missing", "correlat", "distribution", "column", "type"]


def detect_issue(tl, issue_type, col):
    """Unified detection — IDENTICAL logic for all tools."""
    if col:
        col_ok = col.lower() in tl
    else:
        col_ok = True

    if issue_type in GENERAL_KEYWORDS:
        kws = GENERAL_KEYWORDS[issue_type]
    elif issue_type in THAI_KEYWORDS:
        kws = THAI_KEYWORDS[issue_type]
    else:
        kws = [issue_type]

    kw_ok = any(kw in tl for kw in kws)

    if issue_type in THAI_KEYWORDS or issue_type == "correlations":
        return kw_ok
    return col_ok and kw_ok


def gtr_score(text, issues):
    tl = text.lower()
    detected = 0
    details = []
    for issue_type, col in issues:
        is_det = detect_issue(tl, issue_type, col)
        detected += int(is_det)
        col_str = f" ({col})" if col else ""
        details.append(f'[{"YES" if is_det else "NO"}] {issue_type}{col_str}')
    return {"detected": detected, "total": len(issues), "recall": round(detected / len(issues), 3), "details": details}


def itb_score(text):
    tl = text.lower()
    covered = set()
    for cat, kws in ISSUE_TAXONOMY.items():
        if any(kw in tl for kw in kws):
            covered.add(cat)
    return {"count": len(covered), "total": 11, "breadth": round(len(covered) / 11, 3), "covered": sorted(covered)}


def rc_score(html_or_text):
    tl = html_or_text.lower()
    found = [s for s in EXPECTED_SECTIONS if s in tl]
    return {"count": len(found), "total": len(EXPECTED_SECTIONS), "completeness": round(len(found) / len(EXPECTED_SECTIONS), 3)}


# Load meta for time/size
meta = json.loads((OUT / "baseline-meta.json").read_text(encoding="utf-8"))

# Compute for each tool — ALL use same text, same functions
tools_data = {
    "ThaiEDA": {"text": thaieda_text, "html": thaieda_html},
    "ydata-profiling (default)": {"text": ydata_text, "html": ydata_html},
    "sweetviz": {"text": sv_text, "html": sv_html},
}

results = {}
for name, data in tools_data.items():
    results[name] = {
        "general_gtr": gtr_score(data["text"], GENERAL_ISSUES),
        "thai_gtr": gtr_score(data["text"], THAI_ISSUES),
        "itb": itb_score(data["text"]),
        "rc": rc_score(data["html"]),
    }

# Add time/size from meta
results["ThaiEDA"]["time"] = 15.75  # from earlier run
results["ThaiEDA"]["html_kb"] = len(thaieda_html.encode("utf-8")) // 1024
results["ydata-profiling (default)"]["time"] = meta["ydata"]["time"]
results["ydata-profiling (default)"]["html_kb"] = meta["ydata"]["html_size"] // 1024
results["sweetviz"]["time"] = meta["sweetviz"]["time"]
results["sweetviz"]["html_kb"] = meta["sweetviz"]["html_size"] // 1024

# Print
print("=" * 70)
print("  FAIR QUALITY BENCHMARK v2 — 3 tools")
print("  Fixes: ydata DEFAULT mode, uniform text processing, Thai separated")
print("=" * 70)

# TABLE A — General
print("\n  TABLE A — GENERAL EDA QUALITY (6 issues, all tools)")
print(f"  {'Tool':<30} {'GTR':>6} {'ITB':>6} {'RC':>6} {'Time':>8} {'KB':>8}")
print(f"  {'-' * 68}")
for name in ["ThaiEDA", "ydata-profiling (default)", "sweetviz"]:
    r = results[name]
    print(f"  {name:<30} {r['general_gtr']['recall']:>5.0%} {r['itb']['breadth']:>5.0%} {r['rc']['completeness']:>5.0%} {r['time']:>7}s {r['html_kb']:>8}")

print("\n  General GTR details:")
for name in ["ThaiEDA", "ydata-profiling (default)", "sweetviz"]:
    r = results[name]
    print(f"\n  {name}:")
    for d in r["general_gtr"]["details"]:
        tag = "OK" if "YES" in d else "MISS"
        print(f"    [{tag}] {d.replace('[YES] ', '').replace('[NO] ', '')}")

# TABLE B — Thai
print(f"\n{'=' * 70}")
print("  TABLE B — THAI-SPECIFIC DETECTION (4 issues)")
print("  Note: competitors don't claim Thai support — 0 is expected, not a failure")
print(f"{'=' * 70}")
print(f"  {'Tool':<30} {'Thai GTR':>10}")
print(f"  {'-' * 42}")
for name in ["ThaiEDA", "ydata-profiling (default)", "sweetviz"]:
    r = results[name]
    print(f"  {name:<30} {r['thai_gtr']['recall']:>9.0%}")

print("\n  Thai GTR details:")
for name in ["ThaiEDA", "ydata-profiling (default)", "sweetviz"]:
    r = results[name]
    print(f"\n  {name}:")
    for d in r["thai_gtr"]["details"]:
        tag = "OK" if "YES" in d else "MISS"
        print(f"    [{tag}] {d.replace('[YES] ', '').replace('[NO] ', '')}")

# Save
out = OUT / "quality-benchmark-v2-final.json"
out.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
print(f"\n  Results: {out}")