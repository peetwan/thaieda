"""Compute quality benchmark metrics from saved outputs."""
import json
import re
from pathlib import Path

OUT = Path(__file__).parent / "quality-benchmark-outputs"

# Load all outputs
thaieda_text = (OUT / "thaieda-output.txt").read_text(encoding="utf-8")
thaieda_html = (OUT / "thaieda-report.html").read_text(encoding="utf-8")
ydata_text = (OUT / "ydata-output.txt").read_text(encoding="utf-8")
ydata_html = (OUT / "ydata-report.html").read_text(encoding="utf-8")
sv_text = (OUT / "sweetviz-output.txt").read_text(encoding="utf-8")
sv_html = (OUT / "sweetviz-report.html").read_text(encoding="utf-8")

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

ISSUE_KEYWORDS = {
    "missing_values": ["missing", "null", "nan", "empty", "blank"],
    "outliers": ["outlier", "anomaly", "extreme", "iqr", "z-score", "far from"],
    "duplicates": ["duplicate", "duplicated", "redundant"],
    "constant_column": ["constant", "all same", "no variance", "single value", "unique count 1"],
    "type_inconsistency": ["mixed type", "inconsistent type", "type mismatch"],
    "encoding_errors": ["encoding", "mojibake", "garble", "tis-620", "charset", "latin"],
    "cardinality": ["cardinality", "distinct", "unique value"],
    "distribution_skew": ["skew", "distribution", "imbalance"],
    "correlations": ["correlat", "relationship", "association"],
    "cross_column_violation": ["cross", "violation", "together"],
    "thai_specific": ["thai", "พ.ศ", "buddhist", "เลขไทย", "numeral", "zero-width", "zwsp"],
}

EXPECTED_SECTIONS = ["overview", "summary", "insight", "quality", "anomal",
                     "missing", "correlat", "distribution", "column", "type"]


def gtr(text):
    tl = text.lower()
    detected = 0
    details = []
    for issue_type, col in KNOWN_ISSUES:
        col_ok = col.lower() in tl
        if issue_type == "buddhist_era":
            kw_ok = any(kw in tl for kw in ["buddhist", "พ.ศ", "be ", "b.e.", "buddhist era"])
        elif issue_type == "thai_numerals":
            kw_ok = any(kw in tl for kw in ["thai numeral", "เลขไทย", "numeral", "thai digit", "thai number"])
        elif issue_type == "zero_width_space":
            kw_ok = any(kw in tl for kw in ["zero-width", "zwsp", "zero width", "invisible", "เคาะ"])
        elif issue_type == "mojibake":
            kw_ok = any(kw in tl for kw in ["mojibake", "tis-620", "tis620", "encoding", "garble", "charset", "latin"])
        elif issue_type == "placeholders":
            kw_ok = any(kw in tl for kw in ["placeholder", "missing", "na", "invalid", "default"])
        elif issue_type == "constant_column":
            kw_ok = any(kw in tl for kw in ["constant", "all same", "no variance", "single value", "unique count 1"])
        else:
            kws = ISSUE_KEYWORDS.get(issue_type, [issue_type])
            kw_ok = any(kw in tl for kw in kws)

        if issue_type in ("buddhist_era", "thai_numerals", "zero_width_space", "mojibake"):
            is_det = kw_ok
        else:
            is_det = col_ok and kw_ok

        detected += int(is_det)
        mark = "YES" if is_det else "NO"
        details.append(f"[{mark}] {issue_type} ({col})")
    return {"detected": detected, "total": 10, "recall": round(detected / 10, 3), "details": details}


def itb(text):
    tl = text.lower()
    covered = set()
    for cat, kws in ISSUE_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            covered.add(cat)
    return {"count": len(covered), "total": 11, "breadth": round(len(covered) / 11, 3), "covered": sorted(covered)}


def tsdr(text):
    tl = text.lower()
    checks = {
        "buddhist_era": any(kw in tl for kw in ["buddhist", "พ.ศ", "be ", "buddhist era"]),
        "thai_numerals": any(kw in tl for kw in ["thai numeral", "เลขไทย", "numeral", "thai number"]),
        "zero_width_space": any(kw in tl for kw in ["zero-width", "zwsp", "zero width", "invisible"]),
        "mojibake": any(kw in tl for kw in ["mojibake", "tis-620", "encoding", "garble", "latin"]),
    }
    det = sum(1 for v in checks.values() if v)
    return {"detected": det, "total": 4, "rate": round(det / 4, 3), "details": {k: "YES" if v else "NO" for k, v in checks.items()}}


def rc(html_or_text):
    tl = html_or_text.lower()
    found = [s for s in EXPECTED_SECTIONS if s in tl]
    return {"count": len(found), "total": 10, "completeness": round(len(found) / 10, 3), "found": found}


# Compute for each tool
tools_data = {
    "ThaiEDA": {"text": thaieda_text + " " + thaieda_html, "html": thaieda_html},
    "ydata-profiling": {"text": ydata_text, "html": ydata_html},
    "sweetviz": {"text": sv_text, "html": sv_html},
}

results = {}
for name, data in tools_data.items():
    results[name] = {
        "gtr": gtr(data["text"]),
        "itb": itb(data["text"]),
        "tsdr": tsdr(data["text"]),
        "rc": rc(data["html"]),
    }

# Print results
print("=" * 70)
print("  QUALITY BENCHMARK RESULTS")
print("=" * 70)

# GTR
print("\n  M1 - Ground-Truth Recall (fraction of 10 known issues detected)")
print(f"  {'Tool':<20} {'Detected':>10} {'Recall':>8}")
print(f"  {'-' * 40}")
for name, r in results.items():
    det_str = f'{r["gtr"]["detected"]}/{r["gtr"]["total"]}'
    print(f"  {name:<20} {det_str:>10} {r['gtr']['recall']:>8.0%}")

print("\n  Details:")
for name, r in results.items():
    print(f"\n  {name}:")
    for d in r["gtr"]["details"]:
        tag = "OK" if "YES" in d else "MISS"
        print(f"    [{tag}] {d.replace('[YES] ', '').replace('[NO] ', '')}")

# ITB
print(f"\n  M2 - Issue Type Breadth (distinct categories / 11)")
print(f"  {'Tool':<20} {'Covered':>10} {'Breadth':>8}")
print(f"  {'-' * 40}")
for name, r in results.items():
    cov_str = f'{r["itb"]["count"]}/{r["itb"]["total"]}'
    print(f"  {name:<20} {cov_str:>10} {r['itb']['breadth']:>8.0%}")
for name, r in results.items():
    print(f"    {name}: {', '.join(r['itb']['covered'])}")

# TSDR
print(f"\n  M3 - Thai-Specific Detection Rate (4 Thai issues)")
print(f"  {'Tool':<20} {'Detected':>10} {'Rate':>8}")
print(f"  {'-' * 40}")
for name, r in results.items():
    det_str = f'{r["tsdr"]["detected"]}/{r["tsdr"]["total"]}'
    print(f"  {name:<20} {det_str:>10} {r['tsdr']['rate']:>8.0%}")

# RC
print(f"\n  M4 - Report Completeness (sections / 10)")
print(f"  {'Tool':<20} {'Found':>10} {'Score':>8}")
print(f"  {'-' * 40}")
for name, r in results.items():
    f_str = f'{r["rc"]["count"]}/{r["rc"]["total"]}'
    print(f"  {name:<20} {f_str:>10} {r['rc']['completeness']:>8.0%}")

# Summary table
print(f"\n{'=' * 70}")
print(f"  SUMMARY TABLE")
print(f"{'=' * 70}")
header = f"{'Metric':<35} {'ydata':>10} {'sweetviz':>10} {'ThaiEDA':>10}"
print(header)
print("-" * len(header))
metrics = [
    ("GTR - Ground-Truth Recall", "gtr", "recall"),
    ("ITB - Issue Type Breadth", "itb", "breadth"),
    ("TSDR - Thai-Specific Detection", "tsdr", "rate"),
    ("RC - Report Completeness", "rc", "completeness"),
]
for label, key, subkey in metrics:
    y_val = f'{results["ydata-profiling"][key][subkey]:.0%}'
    s_val = f'{results["sweetviz"][key][subkey]:.0%}'
    t_val = f'{results["ThaiEDA"][key][subkey]:.0%}'
    print(f"  {label:<33} {y_val:>10} {s_val:>10} {t_val:>10}")

# Save
out = OUT / "quality-benchmark-final.json"
out.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
print(f"\n  Results saved: {out}")