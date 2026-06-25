"""Scenario 3 — ความน่าเชื่อถือของ insight engine (FDR / determinism / tautology / recall).

insight engine ที่ "ขุดข้อค้นพบจากการผสมคอลัมน์" เสี่ยงสร้างข้อค้นพบลวง (false discovery)
สูงมากเพราะทดสอบหลายร้อยมุมมอง. scenario นี้กดดัน 4 ด้าน:
  1. FDR      — ข้อมูลสุ่มล้วนต้อง "เกือบไม่มี" insight (Benjamini-Hochberg ต้องกรองออก)
  2. determinism — รันซ้ำต้องได้ผลเหมือนเดิมเป๊ะ (ทำซ้ำได้ = เชื่อถือได้)
  3. tautology — ห้ามใช้คอลัมน์ ID/รหัส (Row ID, Postal Code) เป็น measure (sum/mean ไร้ความหมาย)
  4. recall   — สัญญาณจริงที่ชัดเจน (กลุ่มเดียว 3 เท่า) ต้องโผล่ใน top-5
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from thaieda.detect import detect_all
from thaieda.insight_engine import discover_insights


def _make_noise(seed: int = 42, n: int = 500) -> pd.DataFrame:
    """ข้อมูลสุ่มล้วน i.i.d. — 5 categorical × 5 numeric ไม่มีความสัมพันธ์จริงใด ๆ."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for j in range(5):
        k = 3 + j  # 3..7 กลุ่ม
        data[f"cat_{j}"] = rng.choice([f"g{i}" for i in range(k)], size=n)
    for j in range(5):
        data[f"num_{j}"] = rng.normal(50.0, 10.0, size=n)
    return pd.DataFrame(data)


def _make_planted(seed: int = 7, n: int = 600) -> pd.DataFrame:
    """ข้อมูลที่ฝังสัญญาณชัด — กลุ่ม 'north' มี sales ~3 เท่าของกลุ่มอื่น."""
    rng = np.random.default_rng(seed)
    region = rng.choice(["north", "south", "east", "west"], size=n)
    sales = rng.normal(100.0, 15.0, size=n) + (region == "north") * 200.0
    noise_cat = rng.choice(["a", "b", "c"], size=n)
    return pd.DataFrame({"region": region, "sales": sales, "noise_cat": noise_cat})


def _cards_json(cards) -> str:
    """serialize รายการ card แบบ deterministic เพื่อเทียบความเหมือน."""
    return json.dumps([c.to_dict() for c in cards], ensure_ascii=False, sort_keys=True)


def run_s3(fixture_dir: Path | None = None) -> dict:
    """Scenario 3: Insight honesty.

    คืน dict: noise_insight_count, noise_comparison_trend, deterministic,
    tautology_count, tautology_cards, planted_signal_found
    """
    # --- 1. FDR บนข้อมูลสุ่ม ---
    noise = _make_noise()
    noise_types = detect_all(noise)
    noise_res = discover_insights(noise, noise_types, top_n=8)
    noise_comp_trend = sum(1 for c in noise_res.cards if c.pattern in ("comparison", "trend"))

    # --- 2. determinism (รันซ้ำบนข้อมูลที่มีสัญญาณจริง) ---
    planted = _make_planted()
    planted_types = detect_all(planted)
    run_a = discover_insights(planted, planted_types, top_n=8)
    run_b = discover_insights(planted, planted_types, top_n=8)
    deterministic = _cards_json(run_a.cards) == _cards_json(run_b.cards)

    # --- 3. tautology บน Superstore (มี Row ID, Postal Code) ---
    tautology_count = -1
    tautology_cards: list[str] = []
    tautology_note = "ข้าม (ไม่พบ superstore.csv)"
    if fixture_dir is not None:
        ss_path = Path(fixture_dir) / "superstore.csv"
        if ss_path.is_file():
            ss = pd.read_csv(ss_path)
            ss_types = detect_all(ss)
            ss_res = discover_insights(ss, ss_types, top_n=8)
            id_like = {"Row ID", "Postal Code"}
            taut = [c for c in ss_res.cards if c.perspective.measure in id_like]
            tautology_count = len(taut)
            tautology_cards = [
                f"{c.perspective.measure} ตาม {c.perspective.breakdown} ({c.pattern})" for c in taut
            ]
            tautology_note = ""

    # --- 4. sanity recall: สัญญาณ planted ต้องอยู่ใน top-5 ---
    planted_res = discover_insights(planted, planted_types, top_n=5)
    planted_signal_found = any(
        c.perspective.breakdown == "region" and c.evidence.get("top_segment") == "north"
        for c in planted_res.cards
    )

    return {
        "noise_insight_count": noise_res.total,
        "noise_comparison_trend": noise_comp_trend,
        "deterministic": deterministic,
        "tautology_count": tautology_count,
        "tautology_cards": tautology_cards,
        "tautology_note": tautology_note,
        "planted_signal_found": planted_signal_found,
    }


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    print(json.dumps(run_s3(here / "fixtures"), ensure_ascii=False, indent=2))
