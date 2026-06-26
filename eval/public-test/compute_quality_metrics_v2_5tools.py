"""Compute fair quality benchmark metrics v2 — 5 tools.

Tools: ThaiEDA, ydata-profiling (default), sweetviz, Evidently, PyGWalker
Fairness: uniform text processing, Thai metrics separated, ydata default mode.
"""
import json
import re
from pathlib import Path

OUT = Path(__file__).parent / "quality-benchmark-outputs-v2"

# Load all outputs
files = {
    "ThaiEDA": ("thaieda-output.txt", "thaieda-report.html"),
    "ydata-profiling (default)": ("ydata-output.txt", "ydata-report.html"),
    "sweetviz": ("sweetviz-output.txt", "sweetviz-report.html"),
    "Evidently": ("evidently-output.txt", "evidently-report.html"),
    "PyGWalker": ("pygwalker-output.txt", "pygwalker-report.html"),
}

outputs = {}
for name, (txt_file, html_file) in files.items():
    txt_path = OUT / txt_file
    html_path = OUT / html_file
    if txt_path.exists():
        outputs[name] = {
            "text": txt_path.read_text(encoding="utf-8"),
            "html": html_path.read_text(encoding="utf-8") if html_path.exists() else "",
        }
    else:
        print(f"  SKIP: {name} (file not found)")

# Ground truth
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


# Load meta
meta = json.loads((OUT / "baseline-meta.json").read_text(encoding="utf-8"))

# Compute for each tool
results = {}
for name, data in outputs.items():
    results[name] = {
        "general_gtr": gtr_score(data["text"], GENERAL_ISSUES),
        "thai_gtr": gtr_score(data["text"], THAI_ISSUES),
        "itb": itb_score(data["text"]),
        "rc": rc_score(data["html"] if data["html"] else data["text"]),
        "time": meta.get(name.split(" ")[0].lower(), meta.get(name.lower(), {})).get("time", None),
        "html_kb": len(data["html"].encode("utf-8")) // 1024 if data["html"] else 0,
    }

# ThaiEDA time from earlier run
if "ThaiEDA" in results:
    results["ThaiEDA"]["time"] = 15.75
    results["ThaiEDA"]["html_kb"] = len(outputs["ThaiEDA"]["html"].encode("utf-8")) // 1024

# Map meta keys
key_map = {
    "ydata-profiling (default)": "ydata",
    "sweetviz": "sweetviz",
    "Evidently": "evidently",
    "PyGWalker": "pygwalker",
}
for name, meta_key in key_map.items():
    if name in results and meta_key in meta:
        results[name]["time"] = meta[meta_key]["time"]
        results[name]["html_kb"] = meta[meta_key]["html_size"] // 1024

# Print
print("=" * 75)
print("  FAIR QUALITY BENCHMARK v2 — 5 tools")
print("  Fixes: ydata DEFAULT mode, uniform text, Thai separated")
print("=" * 75)

# TABLE A
print("\n  TABLE A — GENERAL EDA QUALITY (6 issues)")
print(f"  {'Tool':<30} {'GTR':>6} {'ITB':>6} {'RC':>6} {'Time':>8} {'KB':>8}")
print(f"  {'-' * 70}")
for name in ["ThaiEDA", "ydata-profiling (default)", "sweetviz", "Evidently", "PyGWalker"]:
    if name not in results:
        continue
    r = results[name]
    time_str = f'{r["time"]}s' if r["time"] else "N/A"
    print(f"  {name:<30} {r['general_gtr']['recall']:>5.0%} {r['itb']['breadth']:>5.0%} {r['rc']['completeness']:>5.0%} {time_str:>8} {r['html_kb']:>8}")

# TABLE B
print(f"\n{'=' * 75}")
print("  TABLE B — THAI-SPECIFIC DETECTION (4 issues)")
print("  Note: competitors don't claim Thai support — 0 is expected, not a failure")
print(f"{'=' * 75}")
print(f"  {'Tool':<30} {'Thai GTR':>10}")
print(f"  {'-' * 42}")
for name in ["ThaiEDA", "ydata-profiling (default)", "sweetviz", "Evidently", "PyGWalker"]:
    if name not in results:
        continue
    r = results[name]
    print(f"  {name:<30} {r['thai_gtr']['recall']:>9.0%}")

# Details
print(f"\n{'=' * 75}")
print("  DETAILS")
print(f"{'=' * 75}")
for name in ["ThaiEDA", "ydata-profiling (default)", "sweetviz", "Evidently", "PyGWalker"]:
    if name not in results:
        continue
    r = results[name]
    print(f"\n  --- {name} ---")
    print(f"  General GTR: {r['general_gtr']['detected']}/{r['general_gtr']['total']}")
    for d in r["general_gtr"]["details"]:
        print(f"    {d}")
    print(f"  Thai GTR: {r['thai_gtr']['detected']}/{r['thai_gtr']['total']}")
    for d in r["thai_gtr"]["details"]:
        print(f"    {d}")
    print(f"  ITB: {r['itb']['count']}/{r['itb']['total']} — {', '.join(r['itb']['covered'])}")

# Save
out = OUT / "quality-benchmark-v2-5tools.json"
out.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
print(f"\n  Results: {out}")