"""ThaiEDA Full Pipeline QA v2 — run every dataset, scan every HTML section for defects.

สแกนละเอียดทุก section ของ HTML report:
- Executive Summary, Priority Actions, Overview, Key Insights, Cross-Column Insights
- Data Quality Issues, Anomalies, Data Cleaning, Timeseries, Distributions, Column Details, NER
- ตรวจ: empty sections, missing labels, broken charts, placeholder text, encoding issues
- จับ: warnings, errors, execution time, insights count (แบบถูกต้อง)
"""

import json
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

# ใช้ source จาก src/ — insert หลัง site-packages
_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.append(_SRC)

from thaieda import run  # noqa: E402

DATASETS_DIR = Path(__file__).parent.parent.parent / "data-example" / "public-datasets"
OUTPUT_DIR = Path(__file__).parent / "qa-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = [
    {"name": "titanic", "file": "titanic.csv", "domain": "passenger survival", "has_dates": False, "has_text": True, "has_thai": False},
    {"name": "telco-churn", "file": "telco-churn.csv", "domain": "telecom churn", "has_dates": False, "has_text": True, "has_thai": False},
    {"name": "wine-quality", "file": "winequality-red.csv", "domain": "wine chemistry", "has_dates": False, "has_text": False, "has_thai": False},
    {"name": "california-housing", "file": "california-housing.csv", "domain": "housing census", "has_dates": False, "has_text": True, "has_thai": False},
    {"name": "superstore", "file": "superstore.csv", "domain": "retail sales", "has_dates": True, "has_text": True, "has_thai": False},
    {"name": "adult", "file": "adult.csv", "domain": "census demographics", "has_dates": False, "has_text": True, "has_thai": False},
    {"name": "bank-marketing", "file": "bank-marketing.csv", "domain": "bank marketing", "has_dates": False, "has_text": True, "has_thai": False, "sep": ";"},
    {"name": "online-retail", "file": "online-retail.csv", "domain": "e-commerce transactions", "has_dates": True, "has_text": True, "has_thai": False},
    {"name": "dirty-thai-retail", "file": "../dirty-thai-retail.csv", "domain": "thai retail (dirty)", "has_dates": True, "has_text": True, "has_thai": True},
]

# Sections ที่ต้องตรวจ (เก็บ heading text)
SECTIONS_TO_SCAN = [
    "Executive Summary", "Priority Actions", "Overview", "Column Types",
    "Columns to Watch", "Key Insights", "Cross-Column Insights",
    "Data Quality Issues", "Anomalies", "Data Cleaning Applied",
    "Timeseries Analysis", "Distributions & Correlations", "Column Details",
    "Named Entities",
]

# alias: ชื่อ section ใน SECTIONS_TO_SCAN → ชื่อจริงที่อยู่ใน HTML (เพราะ template เปลี่ยนชื่อ)
SECTION_ALIASES = {
    "Key Insights": ["Key Insights", "2. Most Important", "Most Important"],
    "Data Quality Issues": ["Data Quality Issues", "Quality Issues"],
    "Data Cleaning Applied": ["Data Cleaning Applied", "Data Cleaning", "Cleaning Applied"],
    "Timeseries Analysis": ["Timeseries Analysis", "Timeseries", "Time Series"],
    "Distributions & Correlations": ["Distributions & Correlations", "Distributions &amp; Correlations"],
    "Named Entities": ["Named Entities", "Named Entities (NER)", "NER"],
    "Cross-Column Insights": ["Cross-Column Insights", "Cross-Column"],
}

# Patterns ที่บ่งบอก defect
DEFECT_PATTERNS = {
    "empty_section": r"<h2[^>]*>\s*([^<]+)</h2>\s*(?:<p[^>]*>\s*</p>\s*){0,3}(?=<h2|$)",
    "breakdown_empty": r"Breakdown:\s*</span>\s*<span[^>]*>\s*·\s*Measure:\s*</span>",
    "no_data_placeholder": r"(?:No data|ไม่มีข้อมูล|ไม่พบข้อมูล|N/A|undefined)",
    "broken_base64": r'src="data:image/png;base64,\s*"',
    "placeholder_label": r"\{\{[^}]+\}\}",  # Jinja2 template not rendered
    "raw_nan_in_text": r">NaN<|>nan<|>None<",
    "empty_table": r"<table[^>]*>\s*</table>",
    "missing_chart_img": r'<img[^>]*alt=""[^>]*>',
}


def scan_html_for_defects(html_str: str) -> dict:
    """สแกน HTML หา defects ในทุก section."""
    defects = {
        "sections_found": [],
        "sections_missing": [],
        "section_item_counts": {},
        "defect_details": [],
        "chart_count": 0,
        "table_count": 0,
        "total_size": len(html_str),
    }

    # ตรวจแต่ละ section
    for section in SECTIONS_TO_SCAN:
        # หา heading — ลองทั้งชื่อ section และ aliases
        candidates = SECTION_ALIASES.get(section, [section])
        match = None
        for alias in candidates:
            # แปลง & → &amp; ใน regex (HTML entity)
            alias_escaped = re.escape(alias).replace(re.escape("&"), "(?:&|&amp;)")
            # h2 อาจมี span ข้างใน เช่น <h2 id="quality">Data Quality Issues <span>(13)</span></h2>
            pattern = rf'<h[12][^>]*>\s*({alias_escaped}.*?)(?:<span[^>]*>[^<]*</span>)?\s*</h[12]>'
            match = re.search(pattern, html_str, re.IGNORECASE | re.DOTALL)
            if match:
                break
        if match:
            # strip HTML tags จาก heading text
            heading_text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
            defects["sections_found"].append(heading_text)

            # นับ items ใน section (count ในวงเล็บ เช่น "Key Insights 38 Warning 19 Info")
            count_match = re.search(r'(\d+)', heading_text)
            if count_match:
                defects["section_item_counts"][section] = int(count_match.group(1))

            # ตรวจว่า section ว่างไหม (ไม่มี content ระหว่าง heading นี้และ heading ถัดไป)
            start = match.end()
            next_heading = re.search(r'<h[12]', html_str[start:])
            if next_heading:
                section_content = html_str[start:start + next_heading.start()]
            else:
                section_content = html_str[start:]

            # ล้าง whitespace
            content_stripped = re.sub(r'\s+', '', section_content)
            # ตัด HTML tags ออก
            text_only = re.sub(r'<[^>]+>', '', content_stripped).strip()

            if len(text_only) < 10 and "<img" not in section_content and "<table" not in section_content:
                defects["defect_details"].append({
                    "type": "empty_section",
                    "section": section,
                    "text_length": len(text_only),
                    "detail": f"Section '{section}' มี text น้อยกว่า 10 chars — อาจว่าง"
                })
        else:
            # ไม่พบ heading — อาจเป็นเพราะไม่มีข้อมูล (normal สำหรับบาง section)
            defects["sections_missing"].append(section)

    # นับ charts + tables
    defects["chart_count"] = len(re.findall(r'data:image/png;base64', html_str))
    defects["table_count"] = len(re.findall(r'<table', html_str))

    # ตรวจ defect patterns
    for defect_name, pattern in DEFECT_PATTERNS.items():
        matches = re.findall(pattern, html_str, re.IGNORECASE)
        if matches:
            # กรอง false positives — "NaN" ในคำอธิบาย quality issue ไม่ใช่ defect
            if defect_name == "raw_nan_in_text":
                # ตรวจว่าเป็น NaN ใน HTML content จริง (ไม่ใช่ในคำอธิบาย)
                real_matches = [m for m in matches if m in ['>NaN<', '>nan<', '>None<']]
                if real_matches:
                    defects["defect_details"].append({
                        "type": defect_name,
                        "count": len(real_matches),
                        "detail": f"พบ raw NaN/nan/None ใน HTML content: {len(real_matches)} ครั้ง"
                    })
            elif defect_name == "breakdown_empty":
                defects["defect_details"].append({
                    "type": defect_name,
                    "count": len(matches),
                    "detail": f"Cross-Column insight มี Breakdown/Measure label ว่าง: {len(matches)} ครั้ง"
                })
            elif defect_name == "no_data_placeholder":
                # กรอง false positive: "✓ No data quality issues detected" เป็นข้อความปกติ
                # จับเฉพาะ "No data" ที่ไม่ได้อยู่ในบริบท ✓ ... หรือ <p class="empty">
                real = []
                for m in matches:
                    if m not in ['No data', 'ไม่มีข้อมูล', 'ไม่พบข้อมูล']:
                        continue
                    # หา context รอบ match ว่าเป็นข้อความปกติหรือไม่
                    for match_obj in re.finditer(pattern, html_str, re.IGNORECASE):
                        if match_obj.group() != m:
                            continue
                        start = max(0, match_obj.start() - 60)
                        context = html_str[start:match_obj.end() + 60]
                        # ข้ามถ้าอยู่ในบริบทปกติ: ✓, class="empty", detected, needed, operations, missing
                        # รวม Thai insight messages: "ไม่มีข้อมูลที่ช่วยแยกความแตกต่าง"
                        if any(kw in context for kw in ['✓', 'class="empty"', 'detected', 'needed', 'operations', 'missing', 'cleaning', 'แยกความแตกต่าง', 'เหมือนกัน']):
                            continue
                        real.append(m)
                if real:
                    defects["defect_details"].append({
                        "type": defect_name,
                        "count": len(real),
                        "detail": f"พบ 'No data' placeholder: {len(real)} ครั้ง"
                    })
            elif defect_name in ("placeholder_label", "broken_base64", "empty_table", "missing_chart_img"):
                defects["defect_details"].append({
                    "type": defect_name,
                    "count": len(matches),
                    "detail": f"พบ {defect_name}: {len(matches)} ครั้ง"
                })

    return defects


def run_one_dataset(ds_meta: dict) -> dict:
    """Run ThaiEDA pipeline บน dataset เดียว แล้วสแกน HTML ละเอียด."""
    name = ds_meta["name"]
    csv_path = DATASETS_DIR / ds_meta["file"]
    sep = ds_meta.get("sep", ",")

    result = {
        "name": name,
        "domain": ds_meta["domain"],
        "status": "pending",
        "error": None,
        "traceback": None,
        "rows": None,
        "cols": None,
        "time_seconds": None,
        "insights_total": None,
        "insights_warning": None,
        "insights_info": None,
        "cross_insights_count": None,
        "quality_issues_count": None,
        "anomalies_count": None,
        "html_size_bytes": None,
        "html_scan": None,
        "warnings_captured": [],
        "warning_count": None,
    }

    print(f"\n{'='*60}", flush=True)
    print(f"  RUNNING: {name} ({ds_meta['domain']})", flush=True)
    print(f"{'='*60}", flush=True)

    # skip ถ้ามี HTML อยู่แล้วและ size > 10KB
    html_check = OUTPUT_DIR / f"{name}-report.html"
    if html_check.exists() and html_check.stat().st_size > 10000:
        print(f"  SKIP: HTML exists ({html_check.stat().st_size} bytes) — scanning only", flush=True)
        html_str = html_check.read_text(encoding="utf-8")
        result["status"] = "ok_scanned"
        result["html_size_bytes"] = len(html_str.encode("utf-8"))
        result["html_scan"] = scan_html_for_defects(html_str)
        return result

    # อ่านข้อมูล
    try:
        df = pd.read_csv(csv_path, sep=sep, encoding="utf-8")
        result["rows"] = len(df)
        result["cols"] = len(df.columns)
        print(f"  Loaded: {len(df)} rows × {len(df.columns)} cols", flush=True)
    except Exception as exc:
        result["status"] = "load_error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        return result

    # วิ่ง pipeline
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        t0 = time.time()
        try:
            eda_result = run(df, lang="en")
            t1 = time.time()
            result["time_seconds"] = round(t1 - t0, 2)

            # อ่าน insights แบบถูกต้อง
            try:
                insights = eda_result.insights
                if insights and hasattr(insights, "total_insights"):
                    result["insights_total"] = insights.total_insights
                    result["insights_warning"] = getattr(insights, "warning_count", None)
                    result["insights_info"] = getattr(insights, "info_count", None)
                elif insights and isinstance(insights, list):
                    result["insights_total"] = len(insights)
            except Exception:
                pass

            # cross-column insights
            try:
                report = eda_result.report
                if hasattr(report, "cross_column_insights"):
                    cross = report.cross_column_insights
                    result["cross_insights_count"] = len(cross) if cross else 0
                elif hasattr(report, "insights") and report.insights:
                    # ลองจาก notes หรือ report
                    pass
            except Exception:
                pass

            # quality issues
            try:
                qi = eda_result.quality_issues
                result["quality_issues_count"] = len(qi) if qi else 0
            except Exception:
                pass

            # anomalies
            try:
                an = eda_result.anomalies
                result["anomalies_count"] = len(an) if an else 0
            except Exception:
                pass

            # สร้าง HTML
            html_path = OUTPUT_DIR / f"{name}-report.html"
            try:
                html_str = eda_result.to_html()
                html_path.write_text(html_str, encoding="utf-8")
                result["html_size_bytes"] = len(html_str.encode("utf-8"))
                result["html_scan"] = scan_html_for_defects(html_str)
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

        # เก็บ warnings (dedupe)
        unique_warnings = set()
        for w in w_list:
            msg = str(w.message)[:200]
            unique_warnings.add(f"{w.category.__name__}: {msg}")
        result["warnings_captured"] = list(unique_warnings)
        result["warning_count"] = len(unique_warnings)

    print(f"  Status: {result['status']} | Time: {result['time_seconds']}s", flush=True)
    print(f"  Insights: {result['insights_total']} | Quality: {result['quality_issues_count']} | Anomalies: {result['anomalies_count']}", flush=True)
    if result.get("html_scan"):
        scan = result["html_scan"]
        print(f"  Sections found: {len(scan['sections_found'])} | Charts: {scan['chart_count']} | Tables: {scan['table_count']}", flush=True)
        if scan["defect_details"]:
            print(f"  Defects: {len(scan['defect_details'])}", flush=True)
            for d in scan["defect_details"][:3]:
                print(f"    - {d['type']}: {d['detail']}", flush=True)
    print(f"  Warnings: {result['warning_count']}", flush=True)

    return result


def main():
    results = []

    for ds in DATASETS:
        try:
            r = run_one_dataset(ds)
            results.append(r)
        except Exception as exc:
            results.append({
                "name": ds["name"],
                "status": "harness_error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    # สรุป
    summary_path = OUTPUT_DIR / "qa-summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print("  QA SUMMARY v2", flush=True)
    print("=" * 60, flush=True)

    ok = sum(1 for r in results if r.get("status") in ("ok", "ok_scanned"))
    err = sum(1 for r in results if r.get("status") not in ("ok", "ok_scanned"))
    print(f"  OK: {ok}/{len(results)} | Errors: {err}", flush=True)

    # แสดง errors (กัน NoneType crash)
    for r in results:
        if r.get("status") not in ("ok", "ok_scanned"):
            err_msg = r.get("error") or "no error message"
            if isinstance(err_msg, str):
                err_msg = err_msg[:120]
            print(f"\n  ❌ {r['name']}: {r.get('status')} — {err_msg}", flush=True)

    # สรุป defects ทั้งหมด
    total_defects = 0
    all_defects = []
    for r in results:
        scan = r.get("html_scan")
        if scan and scan.get("defect_details"):
            for d in scan["defect_details"]:
                total_defects += 1
                all_defects.append({"dataset": r["name"], **d})

    print(f"\n  Total defects found: {total_defects}", flush=True)
    if all_defects:
        # group by type
        by_type = {}
        for d in all_defects:
            t = d["type"]
            by_type.setdefault(t, []).append(d)
        for dtype, items in sorted(by_type.items()):
            datasets_affected = [i["dataset"] for i in items]
            print(f"    {dtype}: {len(items)} ({', '.join(datasets_affected)})", flush=True)

    # เขียน defect list แยก
    defect_path = OUTPUT_DIR / "defect-list.json"
    defect_path.write_text(json.dumps(all_defects, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Summary: {summary_path}", flush=True)
    print(f"  Defects: {defect_path}", flush=True)

    return results


if __name__ == "__main__":
    main()