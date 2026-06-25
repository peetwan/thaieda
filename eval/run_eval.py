"""รัน eval ทั้งหมด → เขียน results/REPORT.md + results/results.json.

ใช้งาน (จาก worktree root, ผ่าน venv ของโปรเจกต์):
    PYTHONPATH="src" .../.venv/Scripts/python.exe eval/run_eval.py

ออกแบบให้ "รายงานตามจริง" — ตัวเลขที่ได้คือพฤติกรรมจริงของไลบรารีบน stack ปัจจุบัน
(เช่น pandas 3.x) ไม่ใช่ตัวเลขที่ปรับให้สวย; ช่องว่างที่พบถูกบันทึกไว้เป็น "ข้อค้นพบ"
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import thaieda

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from scenarios.s1_thai_quality import run_s1  # noqa: E402
from scenarios.s2_relationships import run_s2  # noqa: E402
from scenarios.s3_insight_honesty import run_s3  # noqa: E402

FIXTURES = EVAL_DIR / "fixtures"
RESULTS_DIR = EVAL_DIR / "results"


# ----------------------------------------------------------------------------
# helper: ตัดสินผ่าน/ไม่ผ่านเทียบเป้าหมาย
# ----------------------------------------------------------------------------
def _icon(ok: bool) -> str:
    return "✅" if ok else "⚠️"


def _fmt(v: float) -> str:
    return f"{v:.2f}"


# ----------------------------------------------------------------------------
# REPORT.md
# ----------------------------------------------------------------------------
def build_report(results: dict, meta: dict) -> str:
    s1 = results["s1_thai_quality"]
    s2 = results["s2_relationships"]
    s3 = results["s3_insight_honesty"]

    # --- เกณฑ์ผ่าน ---
    s1_recall_ok = s1["recall"] >= 1.0
    s1_prec_ok = s1["precision"] >= 1.0
    s1_sev_ok = s1["severity_accuracy"] >= 0.8
    s2_prec_ok = s2["precision"] >= 1.0
    s2_recall_ok = s2["recall"] >= 0.9
    s2_f1_ok = s2["f1"] >= 0.9
    s2_hn_ok = len(s2["hard_negative_violations"]) == 0
    s3_noise_ok = s3["noise_insight_count"] <= 2
    s3_det_ok = bool(s3["deterministic"])
    s3_taut_ok = s3["tautology_count"] == 0
    s3_plant_ok = bool(s3["planted_signal_found"])

    sc = s1["silent_corruption"]
    lines: list[str] = []
    add = lines.append

    add("# ThaiEDA Eval Report")
    add("")
    add(f"วันที่รัน (run date): {meta['generated_at']}  ")
    add(f"เวอร์ชัน (version): thaieda {meta['thaieda_version']} · pandas {meta['pandas_version']}")
    add("")
    add(
        "> รายงานนี้สร้างอัตโนมัติจาก `eval/run_eval.py` และวัดพฤติกรรม **จริง** ของไลบรารี "
        "บน stack ปัจจุบัน — ตัวเลขไม่ได้ถูกปรับให้สวย ช่องว่างที่พบถูกบันทึกเป็น *ข้อค้นพบ* ด้านล่าง"
    )
    add("")

    # ---------------- S1 ----------------
    add("## S1: การตรวจจับปัญหาข้อมูลไทย (Thai quality detection)")
    add("")
    add("| Metric | ผล (result) | เป้าหมาย (target) | สถานะ |")
    add("|--------|------|---------|------|")
    add(f"| Detection Recall | {_fmt(s1['recall'])} | 1.00 | {_icon(s1_recall_ok)} |")
    add(f"| Detection Precision | {_fmt(s1['precision'])} | 1.00 | {_icon(s1_prec_ok)} |")
    add(f"| Severity Accuracy | {_fmt(s1['severity_accuracy'])} | ≥0.80 | {_icon(s1_sev_ok)} |")
    clean_txt = "ผ่าน (0 false positives)" if s1["clean_control_passed"] else "ไม่ผ่าน"
    clean_ic = _icon(s1["clean_control_passed"])
    add(f"| Clean Control | {clean_txt} | 0 false positives | {clean_ic} |")
    add("")
    add(f"- ตรวจพบถูกต้อง (TP): {', '.join(s1['true_positives'])}")
    if s1["false_negatives"]:
        add(f"- **พลาด (FN)**: {', '.join(s1['false_negatives'])}")
    if s1["false_positives"]:
        add(f"- เตือนเกิน (FP): {', '.join(s1['false_positives'])}")
    add("")
    add("### Silent Corruption Demo")
    add(
        f"- `{sc['column']}`: {sc['unique_before']} unique → {sc['unique_after']} unique "
        f"หลัง clean (ยุบ {sc['collapsed']} กลุ่มที่ตาเห็นเหมือนกันแต่ถูกแยกด้วยอักขระล่องหน)"
    )
    add("")

    # ---------------- S2 ----------------
    add("## S2: การค้นหาความสัมพันธ์ระหว่างตาราง (Relationship discovery)")
    add("")
    add("| Metric | ผล (result) | เป้าหมาย (target) | สถานะ |")
    add("|--------|------|---------|------|")
    add(f"| Precision | {_fmt(s2['precision'])} | 1.00 | {_icon(s2_prec_ok)} |")
    add(f"| Recall | {_fmt(s2['recall'])} | ≥0.90 | {_icon(s2_recall_ok)} |")
    add(f"| F1 | {_fmt(s2['f1'])} | ≥0.90 | {_icon(s2_f1_ok)} |")
    add(
        f"| False Positives (hard neg) | {len(s2['hard_negative_violations'])} | 0 | "
        f"{_icon(s2_hn_ok)} |"
    )
    add("")
    add(
        f"- ค้นพบ {s2['discovered_count']} เส้น จากเส้นจริง {s2['true_edge_count']} เส้น "
        f"(TP={s2['true_positives']}, FP={s2['false_positives']}, FN={s2['false_negatives']})"
    )
    if s2["known_blind_spot_missed"]:
        add(
            "- หมายเหตุ (blind spot): ไม่พบ "
            + ", ".join(s2["known_blind_spot_missed"])
            + " (ชื่อคอลัมน์ต่างกัน — name match พลาด, รอแก้ v0.7; ไม่นับใน recall)"
        )
    add("")

    # ---------------- S3 ----------------
    add("## S3: ความน่าเชื่อถือของ Insight (Insight honesty)")
    add("")
    add("| Metric | ผล (result) | เป้าหมาย (target) | สถานะ |")
    add("|--------|------|---------|------|")
    add(
        f"| Noise FDR (insights บนข้อมูลสุ่ม) | {s3['noise_insight_count']} | ≤2 | "
        f"{_icon(s3_noise_ok)} |"
    )
    det_txt = "เหมือนกันทุกครั้ง" if s3["deterministic"] else "ไม่คงที่"
    add(f"| Determinism (รัน 2 ครั้ง) | {det_txt} | เหมือนกัน | {_icon(s3_det_ok)} |")
    add(f"| Tautology (ID/รหัส เป็น measure) | {s3['tautology_count']} | 0 | {_icon(s3_taut_ok)} |")
    plant_txt = "พบ" if s3["planted_signal_found"] else "ไม่พบ"
    add(f"| Planted signal found | {plant_txt} | พบ | {_icon(s3_plant_ok)} |")
    add("")
    if s3["tautology_cards"]:
        add(f"- **Tautology ที่พบ**: {', '.join(s3['tautology_cards'])}")
    add("")

    # ---------------- ข้อค้นพบ / Findings ----------------
    add("## ข้อค้นพบและข้อจำกัด (Findings & limitations)")
    add("")
    findings: list[str] = []
    if not s1_recall_ok and s1["false_negatives"]:
        findings.append(
            "**[S1] phone ไม่ถูกตรวจ `thai_numerals`** — ภายใต้ pandas 3.x คอลัมน์สตริงมี dtype "
            "`str` (ไม่ใช่ `object`) และคอลัมน์ถูกจัดเป็น `PHONE_NUMBER`; ตัวรัน "
            "`run_quality_checks` เปิดเช็ค thai_numerals เฉพาะเมื่อ `ctype ∈ TEXT_TYPES "
            "or dtype == object` จึงข้ามไป (ตัวเช็ค `check_thai_numerals` เองทำงานถูกต้อง — "
            "เป็นช่องว่างที่ชั้นเชื่อม). เสนอแก้ v0.7: ใช้ `pd.api.types.is_string_dtype` หรือ "
            "เปิด thai_numerals กับ PHONE_NUMBER ด้วย"
        )
    if not s3_taut_ok and s3["tautology_cards"]:
        findings.append(
            "**[S3] `Postal Code` ถูกใช้เป็น measure** — engine กรองคอลัมน์ที่ชื่อบ่งบอก ID "
            "(`Row ID` ถูกกรองสำเร็จ) แต่ไม่กรอง 'รหัสเชิงตัวเลข' เช่น Postal Code ที่ sum/mean "
            "ไม่มีความหมาย. เสนอแก้ v0.7: ตรวจคอลัมน์รหัส (ชื่อลงท้าย code/zip/postal หรือ "
            "integer cardinality สูงที่ค่าไม่ต่อเนื่อง)"
        )
    findings.append(
        "**[S2] blind spot `preferred_store_id → store_id`** — การจับคู่อาศัยชื่อคอลัมน์ตรงกัน "
        "จึงพลาดคู่ที่ชื่อต่าง (รอแก้ด้วย fuzzy/semantic matching ใน v0.7)"
    )
    for i, f in enumerate(findings, 1):
        add(f"{i}. {f}")
    add("")
    add("## วิธีทำซ้ำ (Reproduce)")
    add("")
    add("```bash")
    add('PYTHONPATH="src" .venv/Scripts/python.exe eval/run_eval.py')
    add("```")
    add("")
    add(
        "> ข้อมูล Coffee-Chain ใน `eval/fixtures/coffee-chain/` เป็นตัวอย่าง **ย่อขนาดแบบรักษา "
        "ความสัมพันธ์** (relational downsample) จากชุดจริง (ORDER/TRANSACTION/INVENTORY รวม ~200MB) "
        "— ทุกเส้น FK→PK ถูกรักษาไว้ ผล P/R/F1 จึงเท่ากับบนข้อมูลเต็ม ดู `eval/fixtures/build_fixtures.py`"
    )
    add("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ThaiEDA Eval ===")
    print("S1: Thai quality detection ...")
    s1 = run_s1(FIXTURES)
    print("S2: Relationship discovery ...")
    s2 = run_s2(FIXTURES)
    print("S3: Insight honesty ...")
    s3 = run_s3(FIXTURES)

    results = {
        "s1_thai_quality": s1,
        "s2_relationships": s2,
        "s3_insight_honesty": s3,
    }
    meta = {
        "thaieda_version": getattr(thaieda, "__version__", "?"),
        "pandas_version": pd.__version__,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # results.json — เก็บเฉพาะ metric (ไม่ใส่ timestamp) เพื่อให้ diff เสถียรสำหรับ CI regression
    json_payload = {
        "meta": {
            "thaieda_version": meta["thaieda_version"],
            "pandas_version": meta["pandas_version"],
        },
        **results,
    }
    (RESULTS_DIR / "results.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = build_report(results, meta)
    (RESULTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")

    # --- สรุปบน console ---
    print()
    print("--- สรุป (summary) ---")
    clean_state = "ok" if s1["clean_control_passed"] else "FAIL"
    print(
        f"S1  recall={s1['recall']:.2f} precision={s1['precision']:.2f} "
        f"severity={s1['severity_accuracy']:.2f} clean_control={clean_state}"
    )
    if s1["false_negatives"]:
        print(f"     FN: {', '.join(s1['false_negatives'])}")
    print(
        f"S2  precision={s2['precision']:.2f} recall={s2['recall']:.2f} f1={s2['f1']:.2f} "
        f"hard_neg_violations={len(s2['hard_negative_violations'])}"
    )
    print(
        f"S3  noise={s3['noise_insight_count']} deterministic={s3['deterministic']} "
        f"tautology={s3['tautology_count']} planted_found={s3['planted_signal_found']}"
    )
    print()
    print(f"เขียนแล้ว: {RESULTS_DIR / 'results.json'}")
    print(f"เขียนแล้ว: {RESULTS_DIR / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
