"""Scenario 2 — การค้นหาความสัมพันธ์ระหว่างตาราง (P/R/F1 เทียบ known edges).

วัด profile_dataset() บนชุด Coffee-Chain (ย่อขนาดแบบรักษาความสัมพันธ์) ว่าค้นพบเส้น FK→PK
ที่ถูกต้องครบไหม, สร้าง false positive ไหม (โดยเฉพาะ hard negatives ที่ชื่อพ้องแต่ไม่เกี่ยว),
และพลาด known blind spot ตามคาดหรือไม่
"""

from __future__ import annotations

import json
from pathlib import Path

from thaieda.schema import profile_dataset


def _edge(pair: list[str]) -> tuple[str, str]:
    """normalize edge จาก manifest ['A.col','B.col'] -> ('A.col','B.col')."""
    return (pair[0].strip(), pair[1].strip())


def run_s2(fixture_dir: Path, manifest_dir: Path | None = None) -> dict:
    """Scenario 2: Relationship discovery P/R/F1.

    คืน dict: precision, recall, f1, true_positives, false_positives, false_negatives,
    hard_negative_violations (target 0), known_blind_spot_missed (เป็นข้อมูลประกอบ)
    """
    fixture_dir = Path(fixture_dir)
    manifest_dir = manifest_dir or (fixture_dir.parent / "manifests")

    with open(manifest_dir / "coffee-chain-schema.expected.json", encoding="utf-8") as fh:
        manifest = json.load(fh)

    true_edges = {_edge(e) for e in manifest["true_edges"]}
    hard_negatives = {_edge(e) for e in manifest.get("hard_negatives", [])}
    blind_spots = {_edge(e) for e in manifest.get("known_blind_spot", [])}

    # --- ค้นหาความสัมพันธ์จริง ---
    profile = profile_dataset(fixture_dir / "coffee-chain")
    discovered: set[tuple[str, str]] = {
        (f"{r.from_table}.{r.from_column}", f"{r.to_table}.{r.to_column}")
        for r in profile.relationships
    }

    tp = discovered & true_edges
    fp = discovered - true_edges
    fn = true_edges - discovered

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 1.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # hard negatives ที่ "ถูกค้นพบผิด" — ต้องเป็น 0 (ค่า edge อาจสลับทิศ จึงเช็คทั้งสองทาง)
    hn_violations = []
    for a, b in hard_negatives:
        if (a, b) in discovered or (b, a) in discovered:
            hn_violations.append(f"{a} <-> {b}")

    # known blind spots ที่พลาดจริง (คาดว่าพลาด — เป็นข้อมูลยืนยันข้อจำกัด ไม่หักคะแนน)
    bs_missed = [f"{a} -> {b}" for a, b in blind_spots if (a, b) not in discovered]

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "discovered_count": len(discovered),
        "true_edge_count": len(true_edges),
        "hard_negative_violations": hn_violations,
        "false_positive_edges": sorted(f"{a} -> {b}" for a, b in fp),
        "false_negative_edges": sorted(f"{a} -> {b}" for a, b in fn),
        "known_blind_spot_missed": bs_missed,
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    print(json.dumps(run_s2(here / "fixtures"), ensure_ascii=False, indent=2))
