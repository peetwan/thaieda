"""Simpson's paradox detection — ความสัมพันธ์กลับทิศเมื่อแยก segment — v2.0.

Simpson's paradox: ทิศทางความสัมพันธ์ระหว่าง ``group_col`` กับ ``target_col`` ในภาพรวม
"กลับทิศ" เมื่อแบ่งข้อมูลตาม ``subgroup_col`` (ตัวแปรกวน/confounder).

รองรับ 2 กรณี:
  * group_col เป็นตัวเลข → ใช้สัญญาณของ correlation (overall vs ในแต่ละ subgroup)
  * group_col เป็นหมวดหมู่ (เทียบ 2 กลุ่มเด่น) → ใช้สัญญาณของผลต่างค่าเฉลี่ย target

หลักการ: numpy/pandas ล้วน ไม่ต้องมี scipy; deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# เกณฑ์ขั้นต่ำ
_MIN_OVERALL_N = 20  # แถวขั้นต่ำสำหรับคำนวณทิศทางภาพรวม
_MIN_SUBGROUP_N = 10  # แถวขั้นต่ำต่อ subgroup level
_CORR_EPS = 0.05  # |r| ต่ำกว่านี้ถือว่า "ไม่มีทิศทางชัด"
_MAX_SUBGROUP_LEVELS = 20  # subgroup ที่มีระดับมากเกินไป (≈ ID) จะข้าม


def detect_simpsons_paradox(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    subgroup_col: str | None = None,
) -> list[dict]:
    """ตรวจจับ Simpson's paradox — ความสัมพันธ์กลับทิศเมื่อแยก segment — v2.0.

    Args:
        df: ข้อมูลที่วิเคราะห์.
        target_col: คอลัมน์ผลลัพธ์ (ตัวเลข).
        group_col: คอลัมน์ตัวแปรต้น (ตัวเลข หรือ หมวดหมู่ที่มี ≥2 กลุ่ม).
        subgroup_col: คอลัมน์ตัวแปรกวนที่ใช้แบ่ง segment — None = ค้นหาอัตโนมัติจากคอลัมน์หมวดหมู่.

    Returns:
        list[dict] ของ paradox ที่พบ (ว่างถ้าไม่พบ) — แต่ละ dict มี target/group/subgroup,
        overall_direction, within_directions, description_th, description_en.

    Raises:
        KeyError: ถ้า target_col หรือ group_col ไม่มีใน df.
    """
    for col in (target_col, group_col):
        if col not in df.columns:
            raise KeyError(f"ไม่พบคอลัมน์ '{col}' ใน DataFrame")
    if subgroup_col is not None and subgroup_col not in df.columns:
        raise KeyError(f"ไม่พบคอลัมน์ subgroup '{subgroup_col}' ใน DataFrame")

    target = pd.to_numeric(df[target_col], errors="coerce")
    if target.notna().sum() < _MIN_OVERALL_N:
        return []

    group_is_numeric = pd.api.types.is_numeric_dtype(df[group_col])
    cats: list | None = None
    if not group_is_numeric:
        vc = df[group_col].dropna().value_counts()
        if len(vc) < 2:
            return []
        cats = list(vc.index[:2])  # เทียบ 2 กลุ่มเด่นสุด

    overall_dir, overall_stat = _direction(df, group_col, target_col, group_is_numeric, cats)
    if overall_dir == 0:
        # ไม่มีทิศทางภาพรวมชัด → ไม่สามารถ "กลับทิศ" ได้
        return []

    if subgroup_col is not None:
        candidates = [subgroup_col]
    else:
        candidates = _candidate_subgroups(df, target_col, group_col)

    findings: list[dict] = []
    for sg in candidates:
        finding = _check_subgroup(
            df, target_col, group_col, sg, group_is_numeric, cats, overall_dir, overall_stat
        )
        if finding is not None:
            findings.append(finding)
    return findings


def _check_subgroup(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    subgroup_col: str,
    group_is_numeric: bool,
    cats: list | None,
    overall_dir: int,
    overall_stat: float | None,
) -> dict | None:
    """ตรวจ subgroup เดียว — คืน finding dict ถ้าพบ paradox มิฉะนั้น None."""
    levels = df[subgroup_col].dropna().unique()
    if len(levels) < 2 or len(levels) > _MAX_SUBGROUP_LEVELS:
        return None

    within: list[dict] = []
    for lv in levels:
        sub = df[df[subgroup_col] == lv]
        if len(sub) < _MIN_SUBGROUP_N:
            continue
        d, stat = _direction(sub, group_col, target_col, group_is_numeric, cats)
        if d != 0:
            within.append({"level": _to_native(lv), "direction": d, "stat": _round(stat)})

    if len(within) < 2:
        return None

    reversed_levels = [w for w in within if w["direction"] == -overall_dir]
    # paradox: subgroup ส่วนใหญ่ (เกินครึ่ง) กลับทิศจากภาพรวม และมีอย่างน้อย 2 segment
    if len(reversed_levels) >= 2 and len(reversed_levels) > len(within) / 2:
        return _build_finding(
            target_col,
            group_col,
            subgroup_col,
            overall_dir,
            overall_stat,
            within,
            reversed_levels,
            group_is_numeric,
            cats,
        )
    return None


def _direction(
    frame: pd.DataFrame,
    group_col: str,
    target_col: str,
    group_is_numeric: bool,
    cats: list | None,
) -> tuple[int, float | None]:
    """ทิศทางความสัมพันธ์ group→target ใน frame — คืน (-1/0/+1, stat)."""
    target = pd.to_numeric(frame[target_col], errors="coerce")
    if group_is_numeric:
        x = pd.to_numeric(frame[group_col], errors="coerce")
        valid = x.notna() & target.notna()
        if int(valid.sum()) < _MIN_SUBGROUP_N:
            return 0, None
        xv, tv = x[valid], target[valid]
        if xv.std() == 0 or tv.std() == 0:
            return 0, None
        r = float(xv.corr(tv))
        if not np.isfinite(r) or abs(r) < _CORR_EPS:
            return 0, r if np.isfinite(r) else None
        return (1 if r > 0 else -1), r
    # categorical: เทียบค่าเฉลี่ย target ของ 2 กลุ่มเด่น (cats[0] vs cats[1])
    assert cats is not None
    a, b = cats[0], cats[1]
    ta = target[frame[group_col] == a]
    tb = target[frame[group_col] == b]
    if ta.notna().sum() < 1 or tb.notna().sum() < 1:
        return 0, None
    ma, mb = ta.mean(), tb.mean()
    if pd.isna(ma) or pd.isna(mb):
        return 0, None
    diff = float(ma - mb)
    if abs(diff) < 1e-12:
        return 0, diff
    return (1 if diff > 0 else -1), diff


def _candidate_subgroups(df: pd.DataFrame, target_col: str, group_col: str) -> list[str]:
    """หาคอลัมน์หมวดหมู่ที่ใช้เป็น subgroup ได้ (2-20 ระดับ, ไม่ใช่ target/group)."""
    candidates: list[str] = []
    for col in df.columns:
        if str(col) in (str(target_col), str(group_col)):
            continue
        series = df[col]
        # หมวดหมู่: object/category หรือ numeric ที่ค่าไม่ซ้ำน้อย (เช่น ปี/ระดับ)
        nun = series.nunique(dropna=True)
        if 2 <= nun <= _MAX_SUBGROUP_LEVELS:
            candidates.append(str(col))
    return candidates


def _build_finding(
    target_col: str,
    group_col: str,
    subgroup_col: str,
    overall_dir: int,
    overall_stat: float | None,
    within: list[dict],
    reversed_levels: list[dict],
    group_is_numeric: bool,
    cats: list | None,
) -> dict:
    """ประกอบ finding dict + คำอธิบาย 2 ภาษา."""
    up_th = "เพิ่มขึ้น" if overall_dir > 0 else "ลดลง"
    up_en = "positive" if overall_dir > 0 else "negative"
    rev_th = "ลดลง" if overall_dir > 0 else "เพิ่มขึ้น"
    rev_en = "negative" if overall_dir > 0 else "positive"
    rel = "ความสัมพันธ์" if group_is_numeric else f"ผลต่างระหว่างกลุ่ม ({cats[0]} vs {cats[1]})"

    desc_th = (
        f"พบ Simpson's paradox: ภาพรวม {rel}ของ '{group_col}' "
        f"กับ '{target_col}' เป็นแบบ{up_th} "
        f"แต่เมื่อแยกตาม '{subgroup_col}' กลับเป็นแบบ{rev_th} "
        f"ใน {len(reversed_levels)}/{len(within)} segment "
        f"— ตัวแปร '{subgroup_col}' อาจเป็นตัวแปรกวน (confounder)"
    )
    desc_en = (
        f"Simpson's paradox: the overall relationship between '{group_col}' and '{target_col}' "
        f"is {up_en}, but it flips to {rev_en} within {len(reversed_levels)}/{len(within)} "
        f"segments of '{subgroup_col}' — '{subgroup_col}' may be a confounder."
    )
    return {
        "paradox": True,
        "target": str(target_col),
        "group": str(group_col),
        "subgroup": str(subgroup_col),
        "overall_direction": "positive" if overall_dir > 0 else "negative",
        "overall_stat": _round(overall_stat),
        "within_directions": within,
        "n_reversed": len(reversed_levels),
        "n_subgroups": len(within),
        "description_th": desc_th,
        "description_en": desc_en,
    }


def _round(v: float | None) -> float | None:
    return round(float(v), 4) if v is not None and np.isfinite(v) else None


def _to_native(v):
    """แปลงค่า numpy → python native เพื่อให้ JSON-serializable."""
    if isinstance(v, np.generic):
        return v.item()
    return v


__all__ = ["detect_simpsons_paradox"]
