"""Target leakage detection — feature ที่ทำนาย target แม่นเกินจริง — v2.0.

Target leakage: คอลัมน์ feature ที่ "รู้คำตอบ" ของ target (มักเป็นข้อมูลที่เกิด*หลัง* target
หรือเป็นสำเนา/อนุพันธ์ของ target) ทำให้โมเดลดูแม่นเกินจริงและพังเมื่อใช้งานจริง.

heuristics ที่ตรวจ:
  Tier A (critical):
    1. duplicate            — ค่าตรงกับ target ทุกแถว (สำเนาตรง ๆ)
    2. high_correlation     — feature ตัวเลข |corr| กับ target ตัวเลข ≥ 0.98
    3. deterministic_mapping— ค่า feature แต่ละค่า map ไป target ค่าเดียว (≈ ฟังก์ชัน) เกือบทุกแถว
    4. near_perfect_separation — feature ตัวเลขแยก class ของ target หมวดหมู่ได้เกือบสมบูรณ์ (eta ≥ 0.98)
  Tier B (warning — suspected proxy):
    5. suspected_proxy      — ชื่อคอลัมน์บ่ง temporal/aggregate hint + association ปานกลาง-สูง

หลักการ: numpy/pandas ล้วน ไม่ต้องมี scipy; deterministic.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# เกณฑ์ Tier A
_MIN_ROWS = 20
_CORR_LEAK = 0.98  # |pearson| ≥ นี้ → สงสัย leakage
_DETERMINISM = 0.99  # สัดส่วนแถวที่ feature→target เป็นฟังก์ชัน
_ID_CARDINALITY = 0.95  # feature ที่ค่าไม่ซ้ำเกินสัดส่วนนี้ของแถว ≈ ID (ข้าม deterministic check)
_ETA_LEAK = 0.98  # correlation ratio (eta) ≥ นี้ → แยก class ได้เกือบสมบูรณ์

# เกณฑ์ Tier B — proxy/leakage ที่ต้อง review ก่อน modeling
_PROXY_EFFECT_MIN = 0.50
_PROXY_NAME_RE = re.compile(
    r"(?:^|_)historical_|(?:^|_)future_|_after_|_lag0$|_lag0_|"
    r"(?:^|_).*_ctr(?:$|_)|(?:^|_).*_rate(?:$|_)|"
    r"total.*charg",
    re.IGNORECASE,
)


def detect_target_leakage(df: pd.DataFrame, target_col: str) -> list[dict]:
    """ตรวจจับ target leakage — feature ทำนาย target แม่นเกินจริง — v2.0.

    Args:
        df: ข้อมูลที่วิเคราะห์.
        target_col: คอลัมน์เป้าหมาย.

    Returns:
        list[dict] ของ feature ที่สงสัยว่ารั่ว (ว่างถ้าไม่พบ) — เรียงตาม score มาก→น้อย.
        แต่ละ dict มี feature/target/kind/tier/severity/score/description_th/description_en.

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
    target_binary = _is_binary_target(target)

    findings: list[dict] = []
    for col in df.columns:
        if str(col) == str(target_col):
            continue
        feature = df[col]
        finding = _check_feature(
            col,
            feature,
            target,
            target_col,
            target_numeric,
            target_num_vals,
            target_binary,
            n,
        )
        if finding is not None:
            findings.append(finding)

    findings.sort(key=lambda d: d["score"], reverse=True)
    return findings


def _has_proxy_name_hint(col: str) -> bool:
    return bool(_PROXY_NAME_RE.search(str(col)))


def _proxy_effect_threshold(col: str) -> float:
    """เกณฑ์ effect ต่ำสุดตามความแรงของชื่อ — temporal hint แรงกว่า generic rate/ctr."""
    name = str(col).lower()
    if re.search(r"(?:^|_)historical_|(?:^|_)future_", name):
        return 0.35
    return _PROXY_EFFECT_MIN


def _is_binary_target(target: pd.Series) -> bool:
    t = target.dropna()
    if t.empty:
        return False
    if pd.api.types.is_bool_dtype(target):
        return True
    if pd.api.types.is_numeric_dtype(target):
        return int(t.nunique()) == 2
    return int(t.nunique()) == 2


def _check_feature(
    col: str,
    feature: pd.Series,
    target: pd.Series,
    target_col: str,
    target_numeric: bool,
    target_num_vals: pd.Series | None,
    target_binary: bool,
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
                tier="critical",
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
                        tier="critical",
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
                tier="critical",
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
                tier="critical",
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

    # Tier B: suspected proxy — ชื่อบ่ง hint + หลักฐานเชิงสถิติ (ไม่ใช่ชื่ออย่างเดียว)
    if _has_proxy_name_hint(col):
        proxy = _check_suspected_proxy(
            col,
            feature,
            target,
            target_col,
            feature_numeric,
            target_numeric,
            target_num_vals,
            target_binary,
        )
        if proxy is not None:
            return proxy

    return None


def _check_suspected_proxy(
    col: str,
    feature: pd.Series,
    target: pd.Series,
    target_col: str,
    feature_numeric: bool,
    target_numeric: bool,
    target_num_vals: pd.Series | None,
    target_binary: bool,
) -> dict | None:
    """Tier B: proxy feature — name hint + moderate-high association."""
    threshold = _proxy_effect_threshold(col)
    if feature_numeric and target_binary and target_num_vals is not None:
        x = pd.to_numeric(feature, errors="coerce")
        valid = x.notna() & target_num_vals.notna()
        if int(valid.sum()) >= _MIN_ROWS:
            xv, tv = x[valid], target_num_vals[valid]
            if xv.std() > 0 and tv.std() > 0:
                r = float(xv.corr(tv))
                eta = _correlation_ratio(tv.astype(str), xv)
                effect = max(abs(r) if np.isfinite(r) else 0.0, eta)
                if (
                    effect >= threshold
                    and effect < _CORR_LEAK
                    and (_is_monotonic_with_binary(xv, tv) or eta >= threshold)
                ):
                    return _finding(
                        col,
                        target_col,
                        "suspected_proxy",
                        effect,
                        tier="warning",
                        th=(
                            f"คอลัมน์ '{col}' มีชื่อบ่งชี้ข้อมูลย้อนหลัง/สรุป "
                            f"และสัมพันธ์กับ target สูง (r={r:.3f}, eta={eta:.3f}) — "
                            f"ควรตรวจก่อนใช้เป็น feature"
                        ),
                        en=(
                            f"Column '{col}' has a temporal/aggregate name hint "
                            f"and moderate-high association with the target "
                            f"(r={r:.3f}, eta={eta:.3f}) — review before modeling."
                        ),
                    )

    if not feature_numeric and (target_binary or not target_numeric):
        effect = _categorical_association_effect(feature, target)
        if effect >= threshold:
            return _finding(
                col,
                target_col,
                "suspected_proxy",
                effect,
                tier="warning",
                th=(
                    f"คอลัมน์ '{col}' มีชื่อบ่งชี้ข้อมูลย้อนหลัง/สรุป "
                    f"และความสัมพันธ์กับ target สูง (effect={effect:.3f}) — "
                    f"ควรตรวจก่อนใช้เป็น feature"
                ),
                en=(
                    f"Column '{col}' has a temporal/aggregate name hint "
                    f"and strong association with the target "
                    f"(effect={effect:.3f}) — review before modeling."
                ),
            )

    if feature_numeric and target_numeric and target_num_vals is not None:
        x = pd.to_numeric(feature, errors="coerce")
        valid = x.notna() & target_num_vals.notna()
        if int(valid.sum()) >= _MIN_ROWS:
            xv, tv = x[valid], target_num_vals[valid]
            if xv.std() > 0 and tv.std() > 0:
                r = float(xv.corr(tv))
                if np.isfinite(r) and threshold <= abs(r) < _CORR_LEAK:
                    return _finding(
                        col,
                        target_col,
                        "suspected_proxy",
                        abs(r),
                        tier="warning",
                        th=(
                            f"คอลัมน์ '{col}' มีชื่อบ่งชี้ข้อมูลย้อนหลัง/สรุป "
                            f"และสัมพันธ์กับ target สูง (r={r:.3f}) — "
                            f"ควรตรวจก่อนใช้เป็น feature"
                        ),
                        en=(
                            f"Column '{col}' has a temporal/aggregate name hint "
                            f"and moderate-high association with the target "
                            f"(r={r:.3f}) — review before modeling."
                        ),
                    )

    return None


def _is_monotonic_with_binary(x: pd.Series, y: pd.Series) -> bool:
    """True ถ้า mean(x) เรียงตาม class ของ target binary (monotonic trend)."""
    means = x.groupby(y).mean()
    if len(means) < 2:
        return False
    vals = means.sort_index().to_numpy()
    return vals[0] != vals[1]


def _categorical_association_effect(feature: pd.Series, target: pd.Series) -> float:
    """Cramér's V แบบง่ายระหว่าง feature กับ target (0..1)."""
    both = feature.notna() & target.notna()
    if int(both.sum()) < _MIN_ROWS:
        return 0.0
    f = feature[both].astype(str)
    t = target[both].astype(str)
    contingency = pd.crosstab(f, t)
    if contingency.size == 0:
        return 0.0
    observed = contingency.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0:
        return 0.0
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(((observed - expected) ** 2 / expected).sum())
    if chi2 <= 0:
        return 0.0
    k = min(observed.shape[0] - 1, observed.shape[1] - 1)
    if k <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def _determinism_ratio(feature: pd.Series, target: pd.Series) -> float:
    """สัดส่วนแถวที่ feature-value แต่ละค่า map ไป target ค่าเดียว (1.0 = เป็นฟังก์ชันสมบูรณ์)."""
    both = feature.notna() & target.notna()
    if int(both.sum()) < _MIN_ROWS:
        return 0.0
    f = feature[both].astype(str)
    t = target[both].astype(str)
    tmp = pd.DataFrame({"f": f.to_numpy(), "t": t.to_numpy()})
    # สำหรับแต่ละค่า f นับ target ที่พบบ่อยสุด — แถวที่ตรง mode ถือว่า "ทำนายถูก" (vectorized)
    counts = tmp.groupby(["f", "t"], sort=False).size()
    max_counts = counts.groupby(level="f").max()
    return float(max_counts.sum()) / float(len(tmp))


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


def _finding(col, target_col, kind, score, *, tier: str, th, en) -> dict:
    severity = "critical" if tier == "critical" else "warning"
    return {
        "feature": str(col),
        "target": str(target_col),
        "kind": kind,
        "tier": tier,
        "severity": severity,
        "score": round(float(score), 4),
        "description_th": th,
        "description_en": en,
    }


__all__ = ["detect_target_leakage"]
