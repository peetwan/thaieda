"""เตรียมข้อมูลก่อนส่งให้ LLM ตามโหมดความเป็นส่วนตัว (v0.9).

โหมด 4 แบบ (เรียงตามระดับความเสี่ยง):
  1. ``insight_only`` — ส่งเฉพาะสถิติสรุป + ข้อค้นพบเชิงลึก (ไม่ส่งข้อมูลดิบ)
     เป็นโหมดปลอดภัยที่สุด เพราะไม่มีข้อมูลดิบออกจากเครื่องเลย
  2. ``anonymized`` — ลบ PII (ชื่อ/เบอร์/บัตร) แล้วส่งข้อมูลที่ทำให้ไม่ระบุตัวได้
     ใช้ ``_anonymize.anonymize_dataframe`` ที่เรียก NER + regex
  3. ``dp_noise`` — สถิติสรุป + เสียงรบกวนแบบ differential privacy (Laplace mechanism)
     ปกปิดค่าสถิติที่อาจระบุตัวบุคคลได้ในข้อมูลขนาดเล็ก
  4. ``full`` — ส่งข้อมูลดิบทั้งหมด (ผู้ใช้ยอมรับความเสี่ยง)

หลักการ:
  * ไม่มี silent fallback — ถ้าโหมดไม่รู้จักจะ raise ValueError
  * vectorized — ใช้ pandas operation แทน loop ทุกที่ที่ได้
  * differential privacy: ใช้ Laplace mechanism พร้อม epsilon ที่ปรับได้
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from thaieda.llm._anonymize import anonymize_dataframe

# ----------------------------------------------------------------------------
# ค่าคงที่
# ----------------------------------------------------------------------------
# โหมดที่รองรับ
_VALID_MODES = frozenset({"insight_only", "synthetic", "anonymized", "dp_noise", "full"})

# ค่า epsilon เริ่มต้นสำหรับ differential privacy (ยิ่งน้อยยิ่งมั่นใจสูง แต่เสียงรบกวนมาก)
_DEFAULT_EPSILON = 1.0

# ค่าความไว (sensitivity) ของแต่ละสถิติ — ใช้ประมาณค่าสูงสุดที่เปลี่ยนได้เมื่อเพิ่ม/ลบ 1 แถว
_COUNT_SENSITIVITY = 1.0  # count เปลี่ยนได้ 1
_MEAN_SENSITIVITY = 1.0  # ประมาณ สำหรับค่าใน [0,1] หรือ std ต่ำ (ใช้ค่าประมาณ)


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def prepare_for_llm(
    df: pd.DataFrame,
    profile: Any | None = None,
    privacy_mode: str = "insight_only",
    *,
    epsilon: float = _DEFAULT_EPSILON,
) -> dict[str, Any]:
    """เตรียมข้อมูลตามโหมดความเป็นส่วนตัว — คืน dict ที่พร้อมส่งให้ ``build_prompt``.

    Args:
        df: DataFrame ที่จะวิเคราะห์.
        profile: DatasetProfile จาก ``thaieda.schema`` (optional — ใช้เพิ่มข้อมูล metadata).
        privacy_mode: โหมดความเป็นส่วนตัว — "insight_only" | "anonymized" | "dp_noise" | "full".
        epsilon: พารามิเตอร์ epsilon สำหรับ dp_noise (ยิ่งน้อย = เสียงรบกวนมากขึ้น).

    Returns:
        dict ที่มี key:
            * ``mode`` — โหมดที่ใช้
            * ``summary`` — สถิติสรุปของข้อมูล (เสมอ)
            * ``data`` — DataFrame (โหมด anonymized/full) หรือ None (insight_only/dp_noise)
            * ``token_map`` — mapping PII → token (โหมด anonymized) หรือ None
            * ``dp_noise`` — True ถ้าเพิ่มเสียงรบกวนแล้ว

    Raises:
        ValueError: ถ้า privacy_mode ไม่ใช่โหมดที่รองรับ.
    """
    if privacy_mode not in _VALID_MODES:
        supported = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"ไม่รองรับโหมด privacy {privacy_mode!r} — รองรับ: {supported}")

    # สถิติสรุป — คำนวณเสมอ (ใช้ในทุกโหมด)
    summary = _compute_summary(df)

    # ข้อมูลจาก profile (ถ้ามี) — เพิ่ม metadata โดยไม่ส่งข้อมูลดิบ
    profile_info = _extract_profile_info(profile)

    if privacy_mode == "insight_only":
        # โหมดปลอดภัยสุด — ส่งเฉพาะสถิติ + profile metadata ไม่ส่งข้อมูลดิบ
        return {
            "mode": "insight_only",
            "summary": summary,
            "profile_info": profile_info,
            "data": None,
            "token_map": None,
            "dp_noise": False,
        }

    if privacy_mode == "anonymized":
        # โหมดทำให้ไม่ระบุตัวได้ — ลบ PII แล้วส่งข้อมูลที่ปลอดภัย
        df_safe, token_map = anonymize_dataframe(df)
        return {
            "mode": "anonymized",
            "summary": summary,
            "profile_info": profile_info,
            "data": df_safe,
            "token_map": token_map,
            "dp_noise": False,
        }

    if privacy_mode == "synthetic":
        # v1.9: โหมด synthetic — สร้างข้อมูลจำลองจาก distribution จริง
        from thaieda.llm._synthetic import generate_synthetic_data

        df_synthetic = generate_synthetic_data(df)
        return {
            "mode": "synthetic",
            "summary": summary,
            "profile_info": profile_info,
            "data": df_synthetic,
            "token_map": None,
            "dp_noise": False,
        }

    if privacy_mode == "dp_noise":
        # โหมด differential privacy — สถิติ + เสียงรบกวน
        noisy_summary = _add_dp_noise(summary, epsilon)
        return {
            "mode": "dp_noise",
            "summary": noisy_summary,
            "profile_info": profile_info,
            "data": None,
            "token_map": None,
            "dp_noise": True,
            "epsilon": epsilon,
        }

    # privacy_mode == "full" — ส่งข้อมูลดิบทั้งหมด (ผู้ใช้ยอมรับความเสี่ยง)
    return {
        "mode": "full",
        "summary": summary,
        "profile_info": profile_info,
        "data": df.copy(),
        "token_map": None,
        "dp_noise": False,
    }


# ----------------------------------------------------------------------------
# สถิติสรุป (pure computation — ไม่ส่งข้อมูลดิบออก)
# ----------------------------------------------------------------------------
def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    """คำนวณสถิติสรุปของ DataFrame — ไม่ส่งข้อมูลดิบออก.

    คืน dict ที่มี:
      * ``shape`` — (n_rows, n_cols)
      * ``columns`` — รายชื่อคอลัมน์
      * ``dtypes`` — ชนิดข้อมูลต่อคอลัมน์
      * ``null_counts`` — จำนวนค่าว่างต่อคอลัมน์
      * ``numeric_stats`` — สถิติเชิงตัวเลข (count, mean, std, min, max, quartiles)
      * ``categorical_stats`` — สถิติเชิงหมวดหมู่ (unique count, top value, freq)
    """
    n_rows, n_cols = df.shape
    summary: dict[str, Any] = {
        "shape": (n_rows, n_cols),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "null_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
    }

    # สถิติเชิงตัวเลข — vectorized
    numeric = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    if numeric.shape[1] > 0:
        desc = numeric.describe()
        summary["numeric_stats"] = {
            str(col): {
                "count": int(desc.loc["count", col]),
                "mean": float(desc.loc["mean", col]),
                "std": float(desc.loc["std", col]) if not math.isnan(desc.loc["std", col]) else 0.0,
                "min": float(desc.loc["min", col]),
                "q25": float(desc.loc["25%", col]),
                "q50": float(desc.loc["50%", col]),
                "q75": float(desc.loc["75%", col]),
                "max": float(desc.loc["max", col]),
            }
            for col in numeric.columns
        }
    else:
        summary["numeric_stats"] = {}

    # สถิติเชิงหมวดหมู่ — vectorized (nunique + value_counts)
    categorical = df.select_dtypes(include=["object", "string", "category", "bool"])
    if categorical.shape[1] > 0:
        cat_stats: dict[str, Any] = {}
        for col in categorical.columns:
            vc = categorical[col].value_counts()
            cat_stats[str(col)] = {
                "unique_count": int(categorical[col].nunique()),
                "top_value": str(vc.index[0]) if len(vc) > 0 else "",
                "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
            }
        summary["categorical_stats"] = cat_stats
    else:
        summary["categorical_stats"] = {}

    return summary


def _extract_profile_info(profile: Any | None) -> dict[str, Any]:
    """ดึงข้อมูล metadata จาก DatasetProfile — ไม่ส่งข้อมูลดิบ.

    หาก profile เป็น DatasetProfile จะดึง:
      * จำนวนตาราง
      * จำนวนความสัมพันธ์
      * ข้อมูล orphan findings
      * รายชื่อตาราง + ชนิดคอลัมน์
    """
    if profile is None:
        return {}

    # ใช้ duck typing — ถ้ามี to_dict() ก็ใช้ ไม่งั้นถือว่าไม่มีข้อมูล
    if hasattr(profile, "to_dict"):
        try:
            d = profile.to_dict()
            return {
                "profile_available": True,
                "table_count": d.get("table_count", len(d.get("tables", []))),
                "relationship_count": d.get("relationship_count", 0),
                "orphan_findings": d.get("orphan_findings", []),
            }
        except Exception:  # noqa: BLE001 — ถ้า to_dict พัง ไม่ควรหยุดทำงาน
            return {"profile_available": False}
    return {"profile_available": False}


# ----------------------------------------------------------------------------
# Differential Privacy — Laplace mechanism
# ----------------------------------------------------------------------------
def _add_dp_noise(summary: dict[str, Any], epsilon: float) -> dict[str, Any]:
    """เพิ่มเสียงรบกวนแบบ Laplace ให้สถิติ — differential privacy.

    Laplace mechanism: noise ~ Laplace(sensitivity / epsilon)
    ถ้า epsilon น้อย = เสียงมากขึ้น = ปลอดภัยกว่า แต่ค่าเบี่ยงเบนมากขึ้น
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon ต้องเป็นบวก ได้รับ {epsilon!r}")

    noisy = {k: v for k, v in summary.items()}
    # คัดลอก dict ภายในเพื่อไม่แก้ของเดิม
    noisy_numeric = {}
    for col, stats in summary.get("numeric_stats", {}).items():
        noisy_numeric[col] = {k: v for k, v in stats.items()}
        # count — sensitivity = 1
        noisy_numeric[col]["count"] = max(
            0, int(stats["count"] + _laplace_noise(_COUNT_SENSITIVITY / epsilon))
        )
        # mean — sensitivity ≈ range/n (ประมาณ 1 สำหรับค่าปกติ)
        noisy_numeric[col]["mean"] = float(stats["mean"]) + _laplace_noise(
            _MEAN_SENSITIVITY / epsilon
        )
        # min/max — ปรับเสียงเล็กน้อย (เพื่อปกปิสุดขอบ)
        noisy_numeric[col]["min"] = float(stats["min"]) + _laplace_noise(
            _MEAN_SENSITIVITY / epsilon
        )
        noisy_numeric[col]["max"] = float(stats["max"]) + _laplace_noise(
            _MEAN_SENSITIVITY / epsilon
        )
        # ลบข้อมูลสถิติที่ไม่ได้ใส่ noise (เช่น std, quantiles) เพื่อรักษา Differential Privacy ป้องกันข้อมูลรั่วไหล
        for k in ["std", "25%", "50%", "75%", "q25", "q50", "q75", "median"]:
            noisy_numeric[col].pop(k, None)
    noisy["numeric_stats"] = noisy_numeric

    # categorical — unique_count sensitivity = 1, top_freq sensitivity = 1
    noisy_cat = {}
    for col, stats in summary.get("categorical_stats", {}).items():
        noisy_cat[col] = {k: v for k, v in stats.items()}
        noisy_cat[col]["unique_count"] = max(
            0, int(stats["unique_count"] + _laplace_noise(_COUNT_SENSITIVITY / epsilon))
        )
        noisy_cat[col]["top_freq"] = max(
            0, int(stats["top_freq"] + _laplace_noise(_COUNT_SENSITIVITY / epsilon))
        )
    noisy["categorical_stats"] = noisy_cat

    # shape — ปกปิดจำนวนแถวเช่นกัน (sensitivity = 1)
    n_rows, n_cols = summary["shape"]
    noisy["shape"] = (
        max(0, int(n_rows + _laplace_noise(_COUNT_SENSITIVITY / epsilon))),
        n_cols,
    )

    noisy["dp_epsilon"] = epsilon
    return noisy


def _laplace_noise(scale: float) -> float:
    """สุ่มค่าเสียงรบกวนจาก Laplace distribution ที่มี scale ที่กำหนด."""
    if scale <= 0:
        return 0.0
    return float(np.random.laplace(0.0, scale))


__all__ = ["prepare_for_llm"]
