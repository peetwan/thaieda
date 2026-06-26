"""Target leakage detection — feature ที่ทำนาย target แม่นเกินจริง — v2.0.

Target leakage: คอลัมน์ feature ที่ "รู้คำตอบ" ของ target (มักเป็นข้อมูลที่เกิด*หลัง* target
หรือเป็นสำเนา/อนุพันธ์ของ target) ทำให้โมเดลดูแม่นเกินจริงและพังเมื่อใช้งานจริง.

heuristics ที่ตรวจ:
  1. duplicate            — ค่าตรงกับ target ทุกแถว (สำเนาตรง ๆ)
  2. high_correlation     — feature ตัวเลข |corr| กับ target ตัวเลข ≥ 0.98
  3. deterministic_mapping— ค่า feature แต่ละค่า map ไป target ค่าเดียว (≈ ฟังก์ชัน) เกือบทุกแถว
  4. near_perfect_separation — feature ตัวเลขแยก class ของ target หมวดหมู่ได้เกือบสมบูรณ์ (eta ≥ 0.98)

หลักการ: numpy/pandas ล้วน ไม่ต้องมี scipy; deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# เกณฑ์
_MIN_ROWS = 20
_CORR_LEAK = 0.98  # |pearson| ≥ นี้ → สงสัย leakage
_DETERMINISM = 0.99  # สัดส่วนแถวที่ feature→target เป็นฟังก์ชัน
_ID_CARDINALITY = 0.95  # feature ที่ค่าไม่ซ้ำเกินสัดส่วนนี้ของแถว ≈ ID (ข้าม deterministic check)
_ETA_LEAK = 0.98  # correlation ratio (eta) ≥ นี้ → แยก class ได้เกือบสมบูรณ์


def detect_target_leakage(df: pd.DataFrame, target_col: str) -> list[dict]:
    """ตรวจจับ target leakage — feature ทำนาย target แม่นเกินจริง — v2.0.

    Args:
        df: ข้อมูลที่วิเคราะห์.
        target_col: คอลัมน์เป้าหมาย.

    Returns:
        list[dict] ของ feature ที่สงสัยว่ารั่ว (ว่างถ้าไม่พบ) — เรียงตาม score มาก→น้อย.
        แต่ละ dict มี feature/target/kind/score/description_th/description_en.

    Raises:
        KeyError: ถ้า target_col ไม่มีใน df.
    """
    if target_col not in df.columns:
        raise KeyError(f"ไม่พบคอลัมน์เป้าหมาย '{target_col}' ใน DataFrame")

    n = len(df)
    if n < _MIN_ROWS:
        return []

    target = df[target_col]
    target_numeric = pd.api.types.is_numeric_dtype(target)
    target_num_vals = pd.to_numeric(target, errors="coerce") if target_numeric else None

    findings: list[dict] = []
    for col in df.columns:
        if str(col) == str(target_col):
            continue
        feature = df[col]
        finding = _check_feature(
            col, feature, target, target_col, target_numeric, target_num_vals, n
        )
        if finding is not None:
            findings.append(finding)

    findings.sort(key=lambda d: d["score"], reverse=True)
    return findings


def _check_feature(
    col: str,
    feature: pd.Series,
    target: pd.Series,
    target_col: str,
    target_numeric: bool,
    target_num_vals: pd.Series | None,
    n: int,
) -> dict | None:
    """ตรวจ feature เดียวเทียบกับ target — คืน finding dict (kind ที่มั่นใจสุด) หรือ None."""
    # 1. duplicate ของ target (ค่าตรงกันทุกแถวที่ไม่ว่าง)
    both = feature.notna() & target.notna()
    if int(both.sum()) >= _MIN_ROWS:
        eq = feature[both].astype(str).to_numpy() == target[both].astype(str).to_numpy()
        if eq.all():
            return _finding(
                col,
                target_col,
                "duplicate",
                1.0,
                th=(
                    f"คอลัมน์ '{col}' มีค่าตรงกับ target "
                    f"'{target_col}' ทุกแถว — เป็นสำเนาของ target "
                    f"(leakage ชัดเจน)"
                ),
                en=(
                    f"Column '{col}' is identical to target "
                    f"'{target_col}' — a direct copy (clear leakage)."
                ),
            )

    feature_numeric = pd.api.types.is_numeric_dtype(feature)

    # 2. high correlation (numeric feature × numeric target)
    if feature_numeric and target_numeric and target_num_vals is not None:
        x = pd.to_numeric(feature, errors="coerce")
        valid = x.notna() & target_num_vals.notna()
        if int(valid.sum()) >= _MIN_ROWS:
            xv, tv = x[valid], target_num_vals[valid]
            if xv.std() > 0 and tv.std() > 0:
                r = float(xv.corr(tv))
                if np.isfinite(r) and abs(r) >= _CORR_LEAK:
                    return _finding(
                        col,
                        target_col,
                        "high_correlation",
                        min(1.0, abs(r)),
                        th=(
                            f"คอลัมน์ '{col}' มี correlation กับ target "
                            f"สูงมาก (r={r:.3f}) — อาจเป็นอนุพันธ์ "
                            f"ของ target (leakage)"
                        ),
                        en=(
                            f"Column '{col}' correlates almost perfectly "
                            f"with the target (r={r:.3f}) — likely derived "
                            f"from it (leakage)."
                        ),
                    )

    # 4. near-perfect separation (numeric feature แยก class ของ categorical target)
    if feature_numeric and not target_numeric:
        eta = _correlation_ratio(target, pd.to_numeric(feature, errors="coerce"))
        if eta >= _ETA_LEAK:
            return _finding(
                col,
                target_col,
                "near_perfect_separation",
                min(1.0, eta),
                th=(
                    f"คอลัมน์ '{col}' (ตัวเลข) แยกกลุ่มของ "
                    f"target ได้เกือบสมบูรณ์ (eta={eta:.3f}) "
                    f"— สงสัย leakage"
                ),
                en=(
                    f"Numeric column '{col}' separates target "
                    f"classes almost perfectly (eta={eta:.3f}) "
                    f"— suspected leakage."
                ),
            )

    # 3. deterministic mapping (feature → target เป็นฟังก์ชัน) — ข้ามคอลัมน์ที่ ≈ ID
    nunique = int(feature.nunique(dropna=True))
    if nunique >= 2 and nunique / n < _ID_CARDINALITY:
        det = _determinism_ratio(feature, target)
        if det >= _DETERMINISM:
            return _finding(
                col,
                target_col,
                "deterministic_mapping",
                det,
                th=(
                    f"คอลัมน์ '{col}' กำหนดค่า target ได้แทบจะ "
                    f"แน่นอน ({det * 100:.0f}% ของแถว) — สงสัย "
                    f"leakage หรือเป็น proxy ของ target"
                ),
                en=(
                    f"Column '{col}' almost perfectly determines "
                    f"the target ({det * 100:.0f}% of rows) — "
                    f"suspected leakage or a target proxy."
                ),
            )

    return None


def _determinism_ratio(feature: pd.Series, target: pd.Series) -> float:
    """สัดส่วนแถวที่ feature-value แต่ละค่า map ไป target ค่าเดียว (1.0 = เป็นฟังก์ชันสมบูรณ์)."""
    both = feature.notna() & target.notna()
    if int(both.sum()) < _MIN_ROWS:
        return 0.0
    f = feature[both].astype(str)
    t = target[both].astype(str)
    tmp = pd.DataFrame({"f": f.to_numpy(), "t": t.to_numpy()})
    # สำหรับแต่ละค่า f นับ target ที่พบบ่อยสุด — แถวที่ตรง mode ถือว่า "ทำนายถูก"
    grouped = tmp.groupby("f")["t"]
    pure = grouped.transform(lambda s: s.value_counts().iloc[0])
    return float(pure.sum()) / float(len(tmp))


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """correlation ratio (eta) ระหว่างกลุ่มหมวดหมู่กับค่าตัวเลข — 0..1."""
    valid = categories.notna() & values.notna()
    cats = categories[valid]
    vals = values[valid].astype(float)
    if len(vals) < 3:
        return 0.0
    overall_mean = vals.mean()
    ss_total = float(((vals - overall_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for c in cats.unique():
        grp = vals[cats == c]
        if len(grp) == 0:
            continue
        ss_between += len(grp) * (grp.mean() - overall_mean) ** 2
    ratio = ss_between / ss_total
    return float(np.sqrt(max(0.0, min(1.0, ratio))))


def _finding(col, target_col, kind, score, *, th, en) -> dict:
    return {
        "feature": str(col),
        "target": str(target_col),
        "kind": kind,
        "score": round(float(score), 4),
        "description_th": th,
        "description_en": en,
    }


__all__ = ["detect_target_leakage"]
