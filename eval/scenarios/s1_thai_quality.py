"""Scenario 1 — การตรวจจับปัญหาคุณภาพข้อมูลไทย (P/R/F1 เทียบ manifest).

วัด run_quality_checks() แบบ end-to-end (สิ่งที่ผู้ใช้ได้รับจริง) ว่าตรวจพบปัญหาที่
"ฝังไว้และรู้คำตอบ" ใน dirty-thai-labeled.csv ครบถ้วน/แม่นยำแค่ไหน + พิสูจน์ silent corruption
จากอักขระล่องหน + ยืนยันว่าไม่เตือนพร่ำเพรื่อบนข้อมูลสะอาด (false-positive control)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from thaieda.clean import clean_thai_text
from thaieda.detect import detect_all
from thaieda.quality import run_quality_checks


def _detected_issue_map(df: pd.DataFrame) -> dict[str, list]:
    """รัน pipeline จริง → คืน mapping column -> list[(check_name, severity)]."""
    column_types = detect_all(df)
    issues = run_quality_checks(df, column_types)
    out: dict[str, list] = {}
    for iss in issues:
        out.setdefault(iss.column, []).append((iss.check_name, iss.severity))
    return out


def run_s1(fixture_dir: Path, manifest_dir: Path | None = None) -> dict:
    """Scenario 1: Thai quality detection P/R/F1.

    คืน dict: recall, precision, f1, severity_accuracy, silent_corruption,
    clean_control_passed, รวมทั้งรายละเอียด TP/FP/FN และ false negatives ที่พบ
    """
    fixture_dir = Path(fixture_dir)
    manifest_dir = manifest_dir or (fixture_dir.parent / "manifests")

    # --- 1. โหลด fixture + รันการตรวจจริง ---
    df = pd.read_csv(fixture_dir / "dirty-thai-labeled.csv")
    detected = _detected_issue_map(df)

    # --- 2. โหลด manifest (ground truth) ---
    with open(manifest_dir / "dirty-thai-labeled.expected.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    columns_spec: dict = manifest["columns"]

    # --- 3. สร้างเซ็ต (column, check) ของที่ฝังไว้ vs ที่ตรวจพบ ---
    injected: set[tuple[str, str]] = set()
    expected_severity: dict[tuple[str, str], str] = {}
    for col, spec in columns_spec.items():
        for check in spec.get("issues") or []:
            injected.add((col, check))
            if spec.get("expected_severity"):
                expected_severity[(col, check)] = spec["expected_severity"]

    detected_set: set[tuple[str, str]] = set()
    detected_severity: dict[tuple[str, str], str] = {}
    for col, lst in detected.items():
        for check, sev in lst:
            detected_set.add((col, check))
            detected_severity[(col, check)] = sev

    tp = injected & detected_set
    fn = injected - detected_set
    fp = detected_set - injected

    recall = len(tp) / len(injected) if injected else 1.0
    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # --- 4. severity accuracy (เฉพาะ TP ที่ manifest ระบุ expected_severity) ---
    sev_checked = [k for k in tp if k in expected_severity]
    sev_match = [k for k in sev_checked if detected_severity.get(k) == expected_severity[k]]
    severity_accuracy = (len(sev_match) / len(sev_checked)) if sev_checked else 1.0

    # --- 5. silent corruption demo: city unique หดเมื่อลบ zero-width ---
    before = int(df["city"].nunique())
    cleaned_city, _ = clean_thai_text(df["city"], ["zwspace"])
    after = int(cleaned_city.nunique())
    silent_corruption = {
        "column": "city",
        "unique_before": before,
        "unique_after": after,
        "collapsed": before - after,
        "note_th": (
            f"city: {before} ค่าไม่ซ้ำ → {after} หลังลบ zero-width "
            "(กรุงเทพ​ฯ กับ กรุงเทพฯ ถูกนับเป็นคนละกลุ่มทั้งที่ตาเห็นเหมือนกัน)"
        ),
    }

    # --- 6. clean control: ข้อมูลสะอาดต้องไม่มี critical/warning ---
    clean_df = pd.read_csv(fixture_dir / "clean-thai.csv")
    clean_types = detect_all(clean_df)
    clean_issues = run_quality_checks(clean_df, clean_types)
    clean_bad = [i for i in clean_issues if i.severity in ("critical", "warning")]
    clean_control_passed = len(clean_bad) == 0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "severity_accuracy": round(severity_accuracy, 4),
        "true_positives": sorted(f"{c}:{k}" for c, k in tp),
        "false_negatives": sorted(f"{c}:{k}" for c, k in fn),
        "false_positives": sorted(f"{c}:{k}" for c, k in fp),
        "silent_corruption": silent_corruption,
        "clean_control_passed": clean_control_passed,
        "clean_control_bad_issues": [f"{i.column}:{i.check_name}({i.severity})" for i in clean_bad],
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    print(json.dumps(run_s1(here / "fixtures"), ensure_ascii=False, indent=2))
