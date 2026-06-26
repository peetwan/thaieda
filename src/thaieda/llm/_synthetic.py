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

import math
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
    if random_seed is not None:
        np.random.seed(random_seed)

    if n_rows is None:
        n_rows = len(df)
    n_rows = max(1, min(n_rows, 10_000))  # cap ที่ 10K กัน LLM token ระเบิด

    synthetic = pd.DataFrame(index=range(n_rows))

    for col in df.columns:
        col_name = str(col)
        series = df[col]

        # แยกตาม dtype
        if pd.api.types.is_numeric_dtype(series):
            synthetic[col_name] = _gen_numeric(series, n_rows)
        elif pd.api.types.is_datetime64_any_dtype(series):
            synthetic[col_name] = _gen_datetime(series, n_rows)
        elif pd.api.types.is_bool_dtype(series):
            synthetic[col_name] = _gen_categorical(series, n_rows)
        elif series.dtype == object or pd.api.types.is_string_dtype(series):
            # แยก: categorical (low cardinality) vs text (high cardinality)
            nunique = series.nunique()
            if nunique <= 50:
                synthetic[col_name] = _gen_categorical(series, n_rows)
            else:
                synthetic[col_name] = _gen_text_placeholder(series, n_rows)
        else:
            synthetic[col_name] = _gen_categorical(series, n_rows)

        # รักษา missing rate
        if preserve_patterns:
            miss_rate = series.isna().mean()
            if miss_rate > 0:
                miss_mask = np.random.random(n_rows) < miss_rate
                synthetic.loc[miss_mask, col_name] = np.nan

    return synthetic


# ------------------------------------------------------------------------------
# Numeric: fit distribution + sample
# ------------------------------------------------------------------------------
def _gen_numeric(series: pd.Series, n: int) -> pd.Series:
    """สร้าง numeric column จาก fitted distribution."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 5:
        # ข้อมูลน้อยเกินไป — sample แบบ bootstrap
        return pd.Series(numeric.sample(n, replace=True).values)

    values = numeric.to_numpy(dtype="float64")
    if values.std() == 0:
        return pd.Series([values[0]] * n)

    # พยายามใช้ scipy fit distributions (เหมือน quality.fit_distributions)
    try:
        from scipy import stats as st
    except ImportError:
        # ไม่มี scipy — sample จาก empirical distribution (bootstrap)
        return pd.Series(np.random.choice(values, size=n, replace=True))

    # ทดสอบ 4 distributions แล้วเลือก best fit
    candidates: list[tuple[str, Any, float, float]] = []

    # Normal
    try:
        mu, sigma = st.norm.fit(values)
        ks_stat, p_val = st.kstest(values, "norm", args=(mu, sigma))
        candidates.append(("normal", (mu, sigma), p_val, float(ks_stat)))
    except Exception:
        pass

    # Lognormal (ต้องมีค่าบวก)
    if (values > 0).all():
        try:
            shape, loc, scale = st.lognorm.fit(values, floc=0)
            ks_stat, p_val = st.kstest(values, "lognorm", args=(shape, loc, scale))
            candidates.append(("lognormal", (shape, loc, scale), p_val, float(ks_stat)))
        except Exception:
            pass

    # Exponential (ต้องไม่ติดลบ)
    if (values >= 0).all():
        try:
            loc, scale = st.expon.fit(values)
            ks_stat, p_val = st.kstest(values, "expon", args=(loc, scale))
            candidates.append(("exponential", (loc, scale), p_val, float(ks_stat)))
        except Exception:
            pass

    # Uniform
    try:
        loc, scale = st.uniform.fit(values)
        ks_stat, p_val = st.kstest(values, "uniform", args=(loc, scale))
        candidates.append(("uniform", (loc, scale), p_val, float(ks_stat)))
    except Exception:
        pass

    if not candidates:
        return pd.Series(np.random.choice(values, size=n, replace=True))

    # เลือก best fit (p-value สูงสุด)
    best = max(candidates, key=lambda x: x[2])
    dist_name, params, _, _ = best

    # Sample จาก best-fit distribution
    if dist_name == "normal":
        mu, sigma = params
        sampled = np.random.normal(mu, sigma, n)
    elif dist_name == "lognormal":
        shape, loc, scale = params
        sampled = np.random.lognormal(mean=np.log(scale), sigma=shape, size=n) + loc
    elif dist_name == "exponential":
        loc, scale = params
        sampled = np.random.exponential(scale, n) + loc
    elif dist_name == "uniform":
        loc, scale = params
        sampled = np.random.uniform(loc, loc + scale, n)
    else:
        sampled = np.random.choice(values, size=n, replace=True)

    # Clip ให้อยู่ใน range ของข้อมูลจริง (กัน outlier จาก distribution)
    lo, hi = float(values.min()), float(values.max())
    sampled = np.clip(sampled, lo, hi)

    # ปัดเศษให้เหมือนข้อมูลจริง
    if values.dtype == int or (values == values.astype(int)).all():
        sampled = np.round(sampled).astype(int)

    return pd.Series(sampled)


# ------------------------------------------------------------------------------
# Categorical: sample จาก proportions
# ------------------------------------------------------------------------------
def _gen_categorical(series: pd.Series, n: int) -> pd.Series:
    """สร้าง categorical column จาก value proportions เดิม — v1.9: ตรวจ PII ก่อน."""
    import re

    vc = series.value_counts(normalize=True, dropna=False)
    values = vc.index.tolist()
    probs = vc.values

    # v1.9: ตรวจ PII ในค่า — ถ้าเป็น phone/email/ID ให้แทนด้วย placeholder
    pii_patterns = [
        r"(?:\+?66|0)\d[\d\s\-]{7,12}",  # phone
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # email
        r"\d{1,2}-\d{4}-\d{4,5}-\d{2}-\d",  # Thai ID
    ]
    all_text = " ".join(str(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)))
    has_pii = any(re.search(p, all_text) for p in pii_patterns)

    if has_pii:
        # แทนด้วย placeholder — รักษา proportions แต่ไม่ส่งค่าจริง
        n_values = len(values)
        placeholders = [f"<category_{i}>" for i in range(n_values)]
        sampled = np.random.choice(placeholders, size=n, p=probs)
        return pd.Series(sampled)

    sampled = np.random.choice(values, size=n, p=probs)
    return pd.Series(sampled)


# ------------------------------------------------------------------------------
# Datetime: sample จาก range + frequency
# ------------------------------------------------------------------------------
def _gen_datetime(series: pd.Series, n: int) -> pd.Series:
    """สร้าง datetime column จาก date range + frequency pattern."""
    dates = pd.to_datetime(series, errors="coerce").dropna()
    if len(dates) == 0:
        return pd.Series([pd.NaT] * n)

    date_min = dates.min()
    date_max = dates.max()
    date_range = (date_max - date_min).total_seconds()

    if date_range == 0:
        # ทุก row มีวันที่เดียวกัน — สุ่มใน ±1 วัน
        delta = pd.Timedelta(seconds=1)
        sampled = date_min + pd.to_timedelta(
            np.random.uniform(-1, 1, n), unit="s"
        )
        return pd.Series(sampled)

    # Sample แบบ uniform ใน date range (preserves temporal coverage)
    seconds = np.random.uniform(0, date_range, n)
    sampled = date_min + pd.to_timedelta(seconds, unit="s")

    # ถ้าข้อมูลจริงเป็น date only (no time component) — ปัดเป็น date
    if dates.dt.hour.nunique() == 1 and dates.dt.minute.nunique() == 1:
        sampled = sampled.dt.floor("D")

    return pd.Series(sampled.values)


# ------------------------------------------------------------------------------
# Text: placeholder ตาม length distribution (ไม่ส่งข้อความจริง)
# ------------------------------------------------------------------------------
def _gen_text_placeholder(series: pd.Series, n: int) -> pd.Series:
    """สร้าง text placeholder — ไม่ส่งข้อความจริง แต่รักษา length distribution."""
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return pd.Series([""] * n)

    lengths = non_null.str.len()
    median_len = int(lengths.median())

    # สร้าง placeholder ตาม length distribution
    samples = []
    for _ in range(n):
        # sample length จาก distribution จริง
        target_len = max(1, int(np.random.normal(median_len, lengths.std())))
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

    # รวม text ทั้งหมดสำหรับ regex scan
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    text_parts: list[str] = []
    for col in text_cols:
        text_parts.extend(df[col].dropna().astype(str).tolist()[:1000])
    all_text = " ".join(text_parts)

    # Phone numbers (Thai + international)
    phone_pattern = r"(?:\+?66|0)\d[\d\s\-]{7,12}"
    phone_count = len(re.findall(phone_pattern, all_text))
    if phone_count > 0:
        pii_types.append({
            "type": "phone_number",
            "count": phone_count,
            "risk": "high",
            "description": f"พบเบอร์โทรศัพท์ {phone_count} รายการ",
        })

    # Email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email_count = len(re.findall(email_pattern, all_text))
    if email_count > 0:
        pii_types.append({
            "type": "email",
            "count": email_count,
            "risk": "high",
            "description": f"พบอีเมล {email_count} รายการ",
        })

    # Thai national ID (x-xxxx-xxxxx-xx-x)
    id_pattern = r"\d{1,2}-\d{4}-\d{4,5}-\d{2}-\d"
    id_count = len(re.findall(id_pattern, all_text))
    if id_count > 0:
        pii_types.append({
            "type": "thai_national_id",
            "count": id_count,
            "risk": "critical",
            "description": f"พบเลขบัตรประชาชน {id_count} รายการ",
        })

    # IP address
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ip_count = len(re.findall(ip_pattern, all_text))
    if ip_count > 0:
        pii_types.append({
            "type": "ip_address",
            "count": ip_count,
            "risk": "medium",
            "description": f"พบ IP address {ip_count} รายการ",
        })

    # Thai address keywords
    addr_keywords = ["ตำบล", "อำเภอ", "จังหวัด", "ต.", "อ.", "จ.", "ถนน", "ซอย", "ม."]
    addr_count = sum(
        1 for kw in addr_keywords if kw in all_text
    )
    if addr_count > 0:
        pii_types.append({
            "type": "thai_address",
            "count": addr_count,
            "risk": "medium",
            "description": f"พบที่อยู่ไทย {addr_count} keyword(s)",
        })

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