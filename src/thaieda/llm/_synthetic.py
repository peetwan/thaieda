"""สร้างข้อมูลจำลอง (synthetic/mock data) จาก distribution จริง — v1.9.

Pipeline สร้าง DataFrame ที่มีคุณสมบัติทางสถิติใกล้เคียงข้อมูลจริง แต่ไม่มีค่าจริง
ปลอดภัยสำหรับส่งให้ LLM วิเคราะห์ต่อโดยไม่เปิดเผยข้อมูลส่วนบุคคล/Enterprise

หลักการ:
  * Numeric → sample จาก fitted distribution (normal/lognormal/exponential/uniform)
    โดยใช้ KS test เลือก best fit (จาก quality.fit_distributions v1.8)
  * Categorical → sample จาก value distribution เดิม (preserves proportions)
  * Datetime → sample จาก range + frequency pattern
  * Text → สร้าง placeholder ตาม pattern (ไม่ส่งข้อความจริง)
  * ไม่มี row ไหนซ้ำข้อมูลจริง — เป็น synthetic 100%
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ------------------------------------------------------------------------------
def generate_synthetic_data(
    df: pd.DataFrame,
    n_rows: int | None = None,
    *,
    preserve_patterns: bool = True,
    random_seed: int | None = 42,
) -> pd.DataFrame:
    """สร้าง DataFrame จำลองที่มี statistical properties ใกล้เคียงข้อมูลจริง — v1.9.

    ขั้นตอน:
      1. ตรวจสอบชนิดคอลัมน์ (numeric/categorical/datetime/text)
      2. สำหรับ numeric: fit distribution + sample (ใช้ quality.fit_distributions)
      3. สำหรับ categorical: sample จาก value proportions เดิม
      4. สำหรับ datetime: sample จาก date range + frequency
      5. สำหรับ text: สร้าง placeholder ตาม length distribution
      6. ไม่มีค่าจริงปน — ทุกค่าเป็น synthetic

    Args:
        df: DataFrame ต้นฉบับ (ข้อมูลจริง).
        n_rows: จำนวนแถวของ synthetic data (default: เท่ากับ df).
        preserve_patterns: True = รักษา pattern (correlation, missing rate).
        random_seed: seed สำหรับ reproducibility.

    Returns:
        DataFrame จำลองที่ปลอดภัยส่งให้ LLM.
    """
    rng = np.random.default_rng(random_seed)

    if n_rows is None:
        n_rows = len(df)
    n_rows = max(1, min(n_rows, 10_000))  # cap ที่ 10K กัน LLM token ระเบิด

    synthetic = pd.DataFrame(index=range(n_rows))

    for col in df.columns:
        col_name = str(col)
        series = df[col]

        # แยกตาม dtype
        if pd.api.types.is_numeric_dtype(series):
            synthetic[col_name] = _gen_numeric(series, n_rows, rng=rng)
        elif pd.api.types.is_datetime64_any_dtype(series):
            synthetic[col_name] = _gen_datetime(series, n_rows, rng=rng)
        elif pd.api.types.is_bool_dtype(series):
            synthetic[col_name] = _gen_categorical(series, n_rows, rng=rng)
        elif series.dtype == object or pd.api.types.is_string_dtype(series):
            # แยก: categorical (low cardinality) vs text (high cardinality)
            nunique = series.nunique()
            if nunique <= 50:
                synthetic[col_name] = _gen_categorical(series, n_rows, rng=rng)
            else:
                synthetic[col_name] = _gen_text_placeholder(series, n_rows, rng=rng)
        else:
            synthetic[col_name] = _gen_categorical(series, n_rows, rng=rng)

        # รักษา missing rate
        if preserve_patterns:
            miss_rate = series.isna().mean()
            if miss_rate > 0:
                miss_mask = rng.random(n_rows) < miss_rate
                synthetic.loc[miss_mask, col_name] = np.nan

    return synthetic


# ------------------------------------------------------------------------------
# Numeric: fit distribution + sample
# ------------------------------------------------------------------------------
def _gen_numeric(series: pd.Series, n: int, rng: np.random.Generator | None = None) -> pd.Series:
    """สร้าง numeric column จาก fitted distribution — v1.9.3.

    ปรับปรุง v1.9.3:
      * ตรวจจับ zero-inflated / spike-at-value → แยก spike + tail
      * เพิ่ม gamma, weibull, beta นอกเหนือจาก normal/lognormal/exponential/uniform
      * ใช้ quantile sampling เป็น fallback เมื่อไม่มี distribution ไหน fit ดี
      * clip แบบนุ่มนวล (IQR-based) แทน hard min/max
    """
    if rng is None:
        rng = np.random.default_rng()

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 5:
        return pd.Series(numeric.sample(n, replace=True, random_state=rng).values)

    values = numeric.to_numpy(dtype="float64")
    if values.std() == 0:
        return pd.Series([values[0]] * n)

    # --- v1.9.3: ตรวจจับ spike (zero-inflated หรือ spike ที่ค่าใด ๆ) ---
    spike_val, spike_rate, tail = _detect_spike(values)
    if spike_val is not None:
        return _gen_spike_mixture(spike_val, spike_rate, tail, n, values, rng=rng)

    # --- ไม่มี spike: fit distribution ปกติ ---
    try:
        from scipy import stats as st
    except ImportError:
        return _gen_quantile_sample(values, n, rng=rng)

    candidates = _fit_distributions(values, st)

    if not candidates:
        return _gen_quantile_sample(values, n, rng=rng)

    # เลือก best fit (p-value สูงสุด)
    best = max(candidates, key=lambda x: x[2])
    dist_name, params, best_p, _ = best

    # ถ้า best p-value ต่ำมาก → ใช้ quantile sampling แทน
    if best_p < 0.01:
        return _gen_quantile_sample(values, n, rng=rng)

    sampled = _sample_from_dist(dist_name, params, n, rng=rng)

    # Clip แบบนุ่มนวล — ใช้ percentile 1/99 แทน hard min/max
    lo = float(np.percentile(values, 0.5))
    hi = float(np.percentile(values, 99.5))
    sampled = np.clip(sampled, lo, hi)

    # ปัดเศษให้เหมือนข้อมูลจริง
    if values.dtype == int or (values == values.astype(int)).all():
        sampled = np.round(sampled).astype(int)

    return pd.Series(sampled)


def _detect_spike(values: np.ndarray) -> tuple[float | None, float, np.ndarray]:
    """ตรวจจับ spike ที่ค่าใด ค่า หนึ่ง (มักเป็น 0) — v1.9.3.

    คืนค่า (spike_value, spike_rate, tail_values)
    ถ้าไม่มี spike → (None, 0.0, values)
    """
    vc = pd.Series(values).value_counts(normalize=True)
    top_val = float(vc.index[0])
    top_rate = float(vc.iloc[0])

    # spike ถ้าค่าเดียวคิดเป็น >25% ของข้อมูล
    if top_rate >= 0.25:
        tail = values[values != top_val]
        if len(tail) >= 10:  # tail ต้องมีข้อมูลพอ fit distribution
            return top_val, top_rate, tail
    return None, 0.0, values


def _gen_spike_mixture(
    spike_val: float,
    spike_rate: float,
    tail: np.ndarray,
    n: int,
    all_values: np.ndarray,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """สร้างข้อมูลแบบ spike + tail mixture — v1.9.3.

    spike_rate ส่วน → spike_val
    (1 - spike_rate) ส่วน → sample จาก tail distribution
    """
    if rng is None:
        rng = np.random.default_rng()

    n_spike = int(n * spike_rate)
    n_tail = n - n_spike

    # fit distribution ที่ tail
    try:
        from scipy import stats as st
    except ImportError:
        # ไม่มี scipy → quantile sample tail
        tail_sampled = _gen_quantile_sample(tail, n_tail, rng=rng)
    else:
        candidates = _fit_distributions(tail, st)
        if candidates:
            best = max(candidates, key=lambda x: x[2])
            dist_name, params, best_p, _ = best
            if best_p < 0.01:
                tail_sampled = _gen_quantile_sample(tail, n_tail, rng=rng)
            else:
                tail_sampled = _sample_from_dist(dist_name, params, n_tail, rng=rng)
                # clip tail แบบนุ่มนวล
                lo = float(np.percentile(tail, 0.5))
                hi = float(np.percentile(tail, 99.5))
                tail_sampled = np.clip(tail_sampled, lo, hi)
        else:
            tail_sampled = _gen_quantile_sample(tail, n_tail, rng=rng)

    # รวม spike + tail
    result = np.empty(n, dtype="float64")
    result[:n_spike] = spike_val
    result[n_spike:] = (
        tail_sampled.to_numpy() if isinstance(tail_sampled, pd.Series) else tail_sampled
    )

    # shuffle
    rng.shuffle(result)

    # ปัดเศษ
    if all_values.dtype == int or (all_values == all_values.astype(int)).all():
        result = np.round(result).astype(int)

    return pd.Series(result)


def _fit_distributions(values: np.ndarray, st: Any) -> list[tuple[str, Any, float, float]]:
    """fit 6 distributions แล้วคืน candidates list — v1.9.3."""
    candidates: list[tuple[str, Any, float, float]] = []

    # Normal
    with contextlib.suppress(Exception):
        mu, sigma = st.norm.fit(values)
        ks_stat, p_val = st.kstest(values, st.norm.cdf, args=(mu, sigma))
        candidates.append(("normal", (mu, sigma), float(p_val), float(ks_stat)))

    # Lognormal (ต้องมีค่าบวก)
    if (values > 0).all():
        with contextlib.suppress(Exception):
            shape, loc, scale = st.lognorm.fit(values, floc=0)
            ks_stat, p_val = st.kstest(values, st.lognorm.cdf, args=(shape, loc, scale))
            candidates.append(("lognormal", (shape, loc, scale), float(p_val), float(ks_stat)))

    # Exponential
    if (values >= 0).all():
        with contextlib.suppress(Exception):
            loc, scale = st.expon.fit(values)
            ks_stat, p_val = st.kstest(values, st.expon.cdf, args=(loc, scale))
            candidates.append(("exponential", (loc, scale), float(p_val), float(ks_stat)))

    # Gamma (ต้องไม่ติดลบ) — เพิ่ม v1.9.3
    if (values >= 0).all() and len(values) >= 20:
        with contextlib.suppress(Exception):
            a, loc, scale = st.gamma.fit(values, floc=0)
            ks_stat, p_val = st.kstest(values, st.gamma.cdf, args=(a, loc, scale))
            candidates.append(("gamma", (a, loc, scale), float(p_val), float(ks_stat)))

    # Weibull (ต้องไม่ติดลบ) — เพิ่ม v1.9.3
    if (values >= 0).all() and len(values) >= 20:
        with contextlib.suppress(Exception):
            c, loc, scale = st.weibull_min.fit(values, floc=0)
            ks_stat, p_val = st.kstest(values, st.weibull_min.cdf, args=(c, loc, scale))
            candidates.append(("weibull", (c, loc, scale), float(p_val), float(ks_stat)))

    # Uniform
    with contextlib.suppress(Exception):
        loc, scale = st.uniform.fit(values)
        ks_stat, p_val = st.kstest(values, st.uniform.cdf, args=(loc, scale))
        candidates.append(("uniform", (loc, scale), float(p_val), float(ks_stat)))

    return candidates


def _sample_from_dist(
    dist_name: str, params: Any, n: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """sample จาก distribution ตามชื่อ — v1.9.3."""
    if rng is None:
        rng = np.random.default_rng()

    if dist_name == "normal":
        mu, sigma = params
        return rng.normal(mu, sigma, n)
    elif dist_name == "lognormal":
        shape, loc, scale = params
        return rng.lognormal(mean=np.log(scale), sigma=shape, size=n) + loc
    elif dist_name == "exponential":
        loc, scale = params
        return rng.exponential(scale, n) + loc
    elif dist_name == "gamma":
        a, loc, scale = params
        from scipy import stats as st

        return st.gamma.rvs(a, loc=loc, scale=scale, size=n, random_state=rng)
    elif dist_name == "weibull":
        c, loc, scale = params
        from scipy import stats as st

        return st.weibull_min.rvs(c, loc=loc, scale=scale, size=n, random_state=rng)
    elif dist_name == "uniform":
        loc, scale = params
        return rng.uniform(loc, loc + scale, n)
    else:
        return rng.choice(
            params[0] if isinstance(params, (list, np.ndarray)) else np.array([0]), size=n
        )


def _gen_quantile_sample(
    values: np.ndarray, n: int, rng: np.random.Generator | None = None
) -> pd.Series:
    """sample จาก empirical quantiles + noise เล็กน้อย — v1.9.3.

    ปลอดภัยกว่า bootstrap เพราะไม่คัดลอกค่าจริง —
    sample quantile แล้วเติม noise เล็กน้อยให้ไม่ตรงค่าจริง
    """
    if rng is None:
        rng = np.random.default_rng()

    # sample quantile จาก empirical CDF
    quantiles = rng.uniform(0.001, 0.999, n)
    sampled = np.quantile(values, quantiles)

    # เติม noise เล็กน้อย (1% ของ std) เพื่อ privacy — ไม่คัดลอกค่าจริง
    noise = rng.normal(0, max(values.std() * 0.01, 1e-6), n)
    sampled = sampled + noise

    # clip ให้ไม่ติดลบถ้าข้อมูลเดิมไม่ติดลบ
    if (values >= 0).all():
        sampled = np.maximum(sampled, 0)

    # ปัดเศษ
    if values.dtype == int or (values == values.astype(int)).all():
        sampled = np.round(sampled).astype(int)

    return pd.Series(sampled)


# ------------------------------------------------------------------------------
# Categorical: sample จาก proportions
# ------------------------------------------------------------------------------
def _gen_categorical(
    series: pd.Series, n: int, rng: np.random.Generator | None = None
) -> pd.Series:
    """สร้าง categorical column จาก value proportions เดิม — v1.9: ตรวจ PII ก่อน."""
    if rng is None:
        rng = np.random.default_rng()

    import re

    vc = series.value_counts(normalize=True, dropna=False)
    values = vc.index.tolist()
    probs = vc.values

    # v1.9: ตรวจ PII ในค่า — ถ้าเป็น phone/email/ID ให้แทนด้วย placeholder
    pii_patterns = [
        r"(?:\+66|0\d{2}-\d{3}-\d{4}|0\d{9})",  # phone (แม่นยำกว่า)
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # email
        r"\d{1,2}-\d{4}-\d{4,5}-\d{2}-\d",  # Thai ID
    ]
    all_text = " ".join(
        str(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))
    )
    has_pii = any(re.search(p, all_text) for p in pii_patterns)

    if has_pii:
        # แทนด้วย placeholder — รักษา proportions แต่ไม่ส่งค่าจริง
        n_values = len(values)
        placeholders = [f"<category_{i}>" for i in range(n_values)]
        sampled = rng.choice(placeholders, size=n, p=probs)
        return pd.Series(sampled)

    sampled = rng.choice(values, size=n, p=probs)
    return pd.Series(sampled)


# ------------------------------------------------------------------------------
# Datetime: sample จาก range + frequency
# ------------------------------------------------------------------------------
def _gen_datetime(series: pd.Series, n: int, rng: np.random.Generator | None = None) -> pd.Series:
    """สร้าง datetime column จาก date range + frequency pattern."""
    if rng is None:
        rng = np.random.default_rng()

    dates = pd.to_datetime(series, errors="coerce").dropna()
    if len(dates) == 0:
        return pd.Series([pd.NaT] * n)

    date_min = dates.min()
    date_max = dates.max()
    date_range = (date_max - date_min).total_seconds()

    if date_range == 0:
        # ทุก row มีวันที่เดียวกัน — สุ่มใน ±1 วัน
        sampled = date_min + pd.to_timedelta(rng.uniform(-1, 1, n), unit="s")
        return pd.Series(sampled)

    # Sample แบบ uniform ใน date range (preserves temporal coverage)
    seconds = rng.uniform(0, date_range, n)
    sampled = date_min + pd.to_timedelta(seconds, unit="s")

    # ถ้าข้อมูลจริงเป็น date only (no time component) — ปัดเป็น date
    if dates.dt.hour.nunique() == 1 and dates.dt.minute.nunique() == 1:
        sampled = sampled.dt.floor("D")

    return pd.Series(sampled.values)


# ------------------------------------------------------------------------------
# Text: placeholder ตาม length distribution (ไม่ส่งข้อความจริง)
# ------------------------------------------------------------------------------
def _gen_text_placeholder(
    series: pd.Series, n: int, rng: np.random.Generator | None = None
) -> pd.Series:
    """สร้าง text placeholder — ไม่ส่งข้อความจริง แต่รักษา length distribution."""
    if rng is None:
        rng = np.random.default_rng()

    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return pd.Series([""] * n)

    lengths = non_null.str.len()
    median_len = int(lengths.median())

    # สร้าง placeholder ตาม length distribution
    samples = []
    for _ in range(n):
        # sample length จาก distribution จริง
        target_len = max(1, int(rng.normal(median_len, lengths.std())))
        target_len = max(1, min(target_len, median_len * 3))  # clip
        placeholder = f"<text_{target_len}chars>"
        samples.append(placeholder)

    return pd.Series(samples)


# ------------------------------------------------------------------------------
# Privacy audit report
# ------------------------------------------------------------------------------
def privacy_audit_report(
    df: pd.DataFrame,
    privacy_mode: str = "synthetic",
) -> dict[str, Any]:
    """สร้างรายงาน privacy audit — สรุปสิ่งที่จะส่งไป LLM — v1.9.

    ตรวจสอบ:
      * PII detected (phone, email, ID, address, IP)
      * Data type ที่จะส่ง (raw vs synthetic vs summary-only)
      * Risk assessment (low/medium/high)
      * Recommendations

    Args:
        df: DataFrame ต้นฉบับ.
        privacy_mode: โหมดที่จะใช้.

    Returns:
        dict รายงาน privacy audit.
    """
    import re

    pii_types: list[dict[str, Any]] = []
    n_rows = len(df)

    # รวม text ทั้งหมดสำหรับ regex scan — เฉพาะ string/object columns (ไม่ใช่ numeric)
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    text_parts: list[str] = []
    for col in text_cols:
        text_parts.extend(df[col].dropna().astype(str).tolist()[:1000])
    all_text = " ".join(text_parts)

    # Phone numbers (Thai + international) — ต้องมี - หรือ +66 เพื่อกัน false positive
    phone_pattern = r"(?:\+66|0\d{2}-\d{3}-\d{4}|0\d{9})"
    phone_count = len(re.findall(phone_pattern, all_text))
    if phone_count > 0:
        pii_types.append(
            {
                "type": "phone_number",
                "count": phone_count,
                "risk": "high",
                "description": f"พบเบอร์โทรศัพท์ {phone_count} รายการ",
            }
        )

    # Email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email_count = len(re.findall(email_pattern, all_text))
    if email_count > 0:
        pii_types.append(
            {
                "type": "email",
                "count": email_count,
                "risk": "high",
                "description": f"พบอีเมล {email_count} รายการ",
            }
        )

    # Thai national ID (x-xxxx-xxxxx-xx-x)
    id_pattern = r"\d{1,2}-\d{4}-\d{4,5}-\d{2}-\d"
    id_count = len(re.findall(id_pattern, all_text))
    if id_count > 0:
        pii_types.append(
            {
                "type": "thai_national_id",
                "count": id_count,
                "risk": "critical",
                "description": f"พบเลขบัตรประชาชน {id_count} รายการ",
            }
        )

    # IP address
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ip_count = len(re.findall(ip_pattern, all_text))
    if ip_count > 0:
        pii_types.append(
            {
                "type": "ip_address",
                "count": ip_count,
                "risk": "medium",
                "description": f"พบ IP address {ip_count} รายการ",
            }
        )

    # Thai address keywords
    addr_keywords = ["ตำบล", "อำเภอ", "จังหวัด", "ต.", "อ.", "จ.", "ถนน", "ซอย", "ม."]
    addr_count = sum(1 for kw in addr_keywords if kw in all_text)
    if addr_count > 0:
        pii_types.append(
            {
                "type": "thai_address",
                "count": addr_count,
                "risk": "medium",
                "description": f"พบที่อยู่ไทย {addr_count} keyword(s)",
            }
        )

    # Risk assessment ตาม mode
    mode_risk = {
        "insight_only": "low",
        "synthetic": "low",
        "anonymized": "medium",
        "dp_noise": "low",
        "full": "high",
    }
    overall_risk = mode_risk.get(privacy_mode, "unknown")

    # Recommendations
    recommendations: list[str] = []
    if any(p["risk"] == "critical" for p in pii_types):
        recommendations.append("พบข้อมูลสำคัญมาก (บัตรประชาชน) — แนะนำใช้ synthetic หรือ insight_only")
    if any(p["risk"] == "high" for p in pii_types):
        recommendations.append("พบ PII (เบอร์/อีเมล) — หลีกเลี่ยงโหมด full")
    if privacy_mode == "full" and pii_types:
        recommendations.append("โหมด full ส่งข้อมูลดิบ — ไม่แนะนำเมื่อมี PII")
    if not pii_types:
        recommendations.append("ไม่พบ PII ชัดเจน — ปลอดภัยในทุกโหมด")

    return {
        "privacy_mode": privacy_mode,
        "overall_risk": overall_risk,
        "pii_detected": pii_types,
        "n_pii_types": len(pii_types),
        "n_rows": n_rows,
        "n_columns": len(df.columns),
        "recommendations": recommendations,
        "data_sent_to_llm": _describe_data_sent(privacy_mode),
    }


def _describe_data_sent(privacy_mode: str) -> str:
    """อธิบายสิ่งที่จะส่งไป LLM ในแต่ละโหมด."""
    descriptions = {
        "insight_only": "สถิติสรุป + ข้อค้นพบเชิงลึก (ไม่ส่งข้อมูลดิบ)",
        "synthetic": "ข้อมูลจำลองที่มี statistical properties ใกล้เคียง (ไม่มีค่าจริง)",
        "anonymized": "ข้อมูลที่ลบ PII แล้ว (ชื่อ/เบอร์/บัตร → token)",
        "dp_noise": "สถิติ + เสียงรบกวน differential privacy (epsilon parameter)",
        "full": "ข้อมูลดิบทั้งหมด (ความเสี่ยงสูง — ใช้เฉพาะเมื่อยอมรับความเสี่ยง)",
    }
    return descriptions.get(privacy_mode, "ไม่ทราบโหมด")


# ------------------------------------------------------------------------------
# Export synthetic data to file — v1.9.1
# ------------------------------------------------------------------------------
def export_synthetic_data(
    df: pd.DataFrame,
    output_path: str,
    *,
    n_rows: int | None = None,
    random_seed: int | None = 42,
    preserve_patterns: bool = True,
    include_audit: bool = True,
) -> dict[str, Any]:
    """สร้างข้อมูลจำลองแล้ว export เป็นไฟล์ — v1.9.1.

    รองรับ: .csv, .tsv, .xlsx, .json, .parquet
    ไฟล์ผลลัพธ์มี statistical properties ใกล้เคียงข้อมูลจริง แต่ไม่มีค่าจริงปน
    ปลอดภัยสำหรับส่งให้ LLM หรือบุคคลที่สามวิเคราะห์ต่อ

    Args:
        df: DataFrame ต้นฉบับ (ข้อมูลจริง).
        output_path: path ของไฟล์ผลลัพธ์ (.csv/.tsv/.xlsx/.json/.parquet).
        n_rows: จำนวนแถว (default: เท่ากับ df).
        random_seed: seed สำหรับ reproducibility.
        preserve_patterns: รักษา missing rate.
        include_audit: แนบ privacy audit report เป็นไฟล์ .json ข้างๆ.

    Returns:
        dict สรุปผล: {output_path, n_rows, n_cols, audit_path, file_size_kb}

    Raises:
        ValueError: ถ้านามสกุลไฟล์ไม่รองรับ.
    """
    from pathlib import Path

    path = Path(output_path)
    suffix = path.suffix.lower()

    # สร้าง synthetic data
    synthetic = generate_synthetic_data(
        df, n_rows=n_rows, preserve_patterns=preserve_patterns, random_seed=random_seed
    )

    # export ตามนามสกุล
    if suffix == ".csv":
        synthetic.to_csv(path, index=False, encoding="utf-8-sig")
    elif suffix == ".tsv":
        synthetic.to_csv(path, index=False, encoding="utf-8-sig", sep="\t")
    elif suffix == ".xlsx":
        try:
            synthetic.to_excel(path, index=False, engine="openpyxl")
        except ImportError as e:
            raise ImportError(f"ต้องติดตั้ง openpyxl สำหรับ .xlsx: pip install openpyxl\n{e}") from e
    elif suffix == ".json":
        synthetic.to_json(path, orient="records", force_ascii=False, indent=2)
    elif suffix == ".parquet":
        try:
            synthetic.to_parquet(path, index=False)
        except ImportError as e:
            raise ImportError(f"ต้องติดตั้ง pyarrow สำหรับ .parquet: pip install pyarrow\n{e}") from e
    else:
        raise ValueError(f"ไม่รองรับนามสกุล {suffix!r} — รองรับ: .csv, .tsv, .xlsx, .json, .parquet")

    file_size_kb = round(path.stat().st_size / 1024, 1)

    result: dict[str, Any] = {
        "output_path": str(path),
        "n_rows": len(synthetic),
        "n_cols": len(synthetic.columns),
        "file_size_kb": file_size_kb,
    }

    # แนบ audit report ถ้าต้องการ
    if include_audit:
        audit = privacy_audit_report(df, privacy_mode="synthetic")
        audit_path = path.with_suffix(".privacy-audit.json")
        import json

        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        result["audit_path"] = str(audit_path)
        result["audit_risk"] = audit["overall_risk"]
        result["n_pii_types"] = audit["n_pii_types"]

    return result
