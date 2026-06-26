"""Deep QA analysis for Thai dataset pipeline v4 results."""
import json
import re
from pathlib import Path

base = Path(__file__).parent / "thai-qa-outputs"
summary = json.loads((base / "qa-summary.json").read_text(encoding="utf-8"))

print("=== QA DEEP ANALYSIS ===\n")

for ds in summary:
    name = ds["name"]
    scan = ds["html_scan"]
    html_path = base / f"{name}-report.html"
    html = html_path.read_text(encoding="utf-8")

    print(f"--- {name} ---")
    print(f"  Rows: {ds['rows']} | Cols: {ds['cols']} | Time: {ds['time_seconds']:.1f}s")
    print(f"  Insights: {ds['insights_total']} | Quality: {ds['quality_issues_count']} | Anomalies: {ds['anomalies_count']}")
    print(f"  Charts: {scan['chart_count']} | Tables: {scan['table_count']} | Sections: {len(scan['sections'])}")
    print(f"  HTML size: {scan['html_size']:,} bytes")
    print(f"  Unrendered jinja: {scan['unrendered_jinja']} | Raw NaN: {scan['raw_nan']} | Broken images: {scan['broken_images']}")

    # Thai garbled — look for literal backslash-u sequences
    thai_garbled = len(re.findall(r"\\u[0-9a-f]{4}", html))
    print(f"  Thai garbled (literal): {thai_garbled}")

    empty_sections = [s for s in scan["sections"] if not s.strip() or len(s.strip()) < 3]
    print(f"  Empty/tiny sections: {len(empty_sections)}")

    img_pattern = r"data:image/png;base64,([A-Za-z0-9+/=]{100,})"
    imgs = re.findall(img_pattern, html)
    tiny_imgs = [i for i in imgs if len(i) < 1000]
    print(f"  Total images: {len(imgs)} | Tiny (<1KB base64): {len(tiny_imgs)}")

    table_rows = len(re.findall(r"<tr", html))
    print(f"  Total <tr> rows: {table_rows}")

    placeholder = len(re.findall(r"no data|N/A|empty|none found", html, re.IGNORECASE))
    print(f"  Placeholder text: {placeholder}")

    # Check for mojibake — common patterns
    mojibake = len(re.findall(r"Ã[^a-zA-Z]|â€|Â°|Ã©|Ã¨", html))
    print(f"  Mojibake patterns: {mojibake}")

    # Check for tofu/boxes
    tofu = len(re.findall(r"\uFFFD", html))
    print(f"  Replacement chars (tofu): {tofu}")

    print()

# Compare with english datasets
print("\n=== COMPARISON: English datasets (v3) ===")
en_base = Path(__file__).parent / "qa-outputs"
en_summary_path = en_base / "qa-summary.json"
if en_summary_path.exists():
    en_summary = json.loads(en_summary_path.read_text(encoding="utf-8"))
    for ds in en_summary:
        scan = ds.get("html_scan", {})
        print(f"  {ds['name']}: Insights={ds.get('insights_total',0)} | Charts={scan.get('chart_count',0)} | Tables={scan.get('table_count',0)} | Time={ds.get('time_seconds',0):.1f}s")
else:
    print("  (v3 summary not found)")

# Key issues summary
print("\n\n=== KEY ISSUES IDENTIFIED ===")
print()
print("1. PERFORMANCE: wongnai-reviews-40k took 724s (12 min) for only 2 cols × 40k rows")
print("   - wongnai-train-50k: 508s (8.5 min) for 2 cols × 40k rows")
print("   - thai-restaurant-hybrid-20k: 474s (8 min) for 7 cols × 20k rows")
print("   - Compare: wisesight 27k rows = 20s, thai-ecommerce 15k × 11 cols = 20s")
print("   => Text-heavy datasets with long Thai reviews are extremely slow")
print()
print("2. INSIGHTS LOCKED AT 7: Every single dataset has exactly 7 insights")
print("   - English datasets had 19-679 insights (wide variation)")
print("   - Looks like a hard cap or a bug in insight generation for Thai text")
print()
print("3. CHARTS/TABLES LOW for text datasets:")
print("   - wongnai: 6 charts, 4 tables (2-col text dataset)")
print("   - wisesight: 6 charts, 6 tables")
print("   - vs thai-ecommerce (11 cols): 15 charts, 22 tables")
print("   - Text datasets may need text-specific visualizations (word freq, sentiment dist, etc.)")