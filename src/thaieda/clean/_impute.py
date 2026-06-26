"""ML-powered imputation — เติมค่าว่างด้วยโมเดล โดยอิงกลไกค่าว่าง (MCAR/MAR/MNAR) — v2.0.

ระบบ 3 ระดับผูกกับกลไกค่าว่างที่ ``thaieda.quality.detect_missing_mechanism`` ตรวจได้:
  * MCAR → median/mode (ง่าย ไม่สร้างอคติ — ค่าว่างสุ่มสมบูรณ์อยู่แล้ว)
  * MAR  → IterativeImputer (BayesianRidge) ใช้คอลัมน์อื่นทำนาย — ลดอคติเมื่อค่าว่างขึ้นกับค่าที่สังเกตได้
  * MNAR → flag เท่านั้น ไม่เติมค่า (การเติมจะสร้างอคติเพราะค่าว่างขึ้นกับค่าตัวเอง)

ข้อจำกัดสำหรับเครื่อง low-resource (4GB RAM, ไม่มี GPU):
  * จำกัด predictor columns ไม่เกิน 20 (subsample ถ้าเกิน)
  * จำกัด training rows ไม่เกิน 10,000 (fit บน subsample แล้ว transform ทั้งหมด)
  * ถ้า IterativeImputer ล้มเหลว → ถอยไป median (no silent crash)
  * ML imputation เฉพาะคอลัมน์ตัวเลข; categorical → mode fallback
  * เก็บ audit: เติมกี่ค่า ใช้วิธีใด (CleaningResult ต่อคอลัมน์)

หลักการ: lazy import sklearn (เป็น dep หลักอยู่แล้ว แต่ import เฉพาะตอนใช้เพื่อให้ core เบา)
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

from thaieda.clean import CleaningResult, handle_missing_values

# guardrails สำหรับ low-resource
_MAX_PREDICTORS = 20
_MAX_TRAIN_ROWS = 10_000


def ml_impute(
    df: pd.DataFrame,
    *,
    max_predictors: int = _MAX_PREDICTORS,
    max_train_rows: int = _MAX_TRAIN_ROWS,
    mechanism: str | None = None,
) -> tuple[pd.DataFrame, list[CleaningResult], list[str]]:
    """เติมค่าว่างด้วย ML แบบ 3 ระดับตามกลไกค่าว่าง — v2.0.

    Args:
        df: DataFrame ต้นฉบับ.
        max_predictors: จำนวนคอลัมน์ตัวทำนายสูงสุด (subsample ถ้าเกิน).
        max_train_rows: จำนวนแถวสูงสุดที่ใช้ fit (fit บน subsample แล้ว transform ทั้งหมด).
        mechanism: ระบุกลไกเอง ("MCAR"/"MAR_likely"/"MNAR_likely") — None = ตรวจอัตโนมัติ.

    Returns:
        (df ที่เติมค่าแล้ว, รายการ CleaningResult ต่อคอลัมน์, รายการคำเตือน).
    """
    out = df.copy()
    results: list[CleaningResult] = []
    warnings: list[str] = []

    cols_with_na = [str(c) for c in out.columns if out[c].isna().any()]
    if not cols_with_na:
        return out, results, warnings

    # 1. ตรวจกลไกค่าว่าง (ถ้าไม่ได้ระบุมา)
    if mechanism is None:
        mechanism = _detect_mechanism(df)

    # 2. MNAR → flag เท่านั้น (ไม่เติมค่า เพื่อกันอคติ)
    if mechanism.startswith("MNAR"):
        warnings.append("กลไกค่าว่างเป็น MNAR (ขึ้นกับค่าตัวเอง) — ข้ามการเติมค่าด้วย ML เพื่อกันอคติ (flag เท่านั้น)")
        for col in cols_with_na:
            results.append(
                CleaningResult(
                    operation="ml_impute",
                    rows_affected=0,
                    column=col,
                    description_th=f"[MNAR] คอลัมน์ '{col}' — ไม่เติมค่า (flag เท่านั้น) เพื่อกันอคติ",
                )
            )
        return out, results, warnings

    cat_na_cols = [c for c in cols_with_na if not pd.api.types.is_numeric_dtype(out[c])]
    num_na_cols = [c for c in cols_with_na if pd.api.types.is_numeric_dtype(out[c])]

    # categorical → mode fallback เสมอ (ทุกกลไก) — ML imputation ทำเฉพาะ numeric
    for col in cat_na_cols:
        out[col], res = handle_missing_values(out[col], "mode")
        res.operation = "ml_impute"
        res.description_th = f"[{mechanism}/categorical→mode] {res.description_th}"
        results.append(res)

    # 3. MCAR → median (numeric) — ง่าย ไม่สร้างอคติ
    if mechanism == "MCAR":
        for col in num_na_cols:
            out[col], res = handle_missing_values(out[col], "median")
            res.operation = "ml_impute"
            res.description_th = f"[MCAR/median] {res.description_th}"
            results.append(res)
        return out, results, warnings

    # 4. MAR → IterativeImputer (BayesianRidge) ใช้คอลัมน์ตัวเลขอื่นทำนาย
    if num_na_cols:
        imputed_results = _iterative_impute(
            out, num_na_cols, max_predictors=max_predictors, max_train_rows=max_train_rows
        )
        results.extend(imputed_results)

    return out, results, warnings


def _detect_mechanism(df: pd.DataFrame) -> str:
    """ตรวจกลไกค่าว่างผ่าน quality.detect_missing_mechanism — คืน 'MCAR' ถ้าข้อมูลไม่พอ."""
    from thaieda.quality import detect_missing_mechanism

    result = detect_missing_mechanism(df)
    if result is None:
        # ข้อมูลไม่พอวิเคราะห์กลไก — ใช้ median/mode ตรง ๆ (ปลอดภัยสุด)
        return "MCAR"
    return result.mechanism


def _iterative_impute(
    df: pd.DataFrame,
    num_na_cols: list[str],
    *,
    max_predictors: int,
    max_train_rows: int,
) -> list[CleaningResult]:
    """เติมค่าว่างคอลัมน์ตัวเลขด้วย IterativeImputer (in-place บน df) — มี guardrails + median fallback."""
    results: list[CleaningResult] = []

    numeric_df = df.select_dtypes(include=[np.number])
    predictors = [str(c) for c in numeric_df.columns]

    # guardrail: จำกัด predictor ไม่เกิน max_predictors — เก็บคอลัมน์ที่มี NA ไว้ก่อน
    # แล้วเติมคอลัมน์ที่ความแปรปรวนสูงสุดจนครบ (ตัวทำนายที่ให้ข้อมูลมากกว่า)
    if len(predictors) > max_predictors:
        must = list(num_na_cols)
        others = [c for c in predictors if c not in set(must)]
        variances = numeric_df[others].var(numeric_only=True).sort_values(ascending=False)
        ranked_others = [str(c) for c in variances.index]
        keep = must + ranked_others[: max(0, max_predictors - len(must))]
        keep_set = set(keep)
        predictors = [c for c in predictors if c in keep_set]

    record_before = {col: int(df[col].isna().sum()) for col in num_na_cols}

    try:
        from sklearn.experimental import (
            enable_iterative_imputer,  # noqa: F401  # เปิดใช้ IterativeImputer
        )
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        x = numeric_df[predictors].astype("float64")
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,
            random_state=0,
            sample_posterior=False,
        )

        # guardrail: fit บน subsample ถ้าข้อมูลใหญ่ แล้ว transform ทั้งหมด
        n = len(x)
        if n > max_train_rows:
            train = x.sample(n=max_train_rows, random_state=0)
            imputer.fit(train)
            arr = imputer.transform(x)
        else:
            arr = imputer.fit_transform(x)

        imputed = pd.DataFrame(arr, columns=predictors, index=x.index)
        train_note = f", fit บน {max_train_rows:,} แถว (subsample)" if n > max_train_rows else ""
        for col in num_na_cols:
            before = record_before[col]
            # คงชนิดข้อมูลเดิม (เช่น int) ถ้าเป็นไปได้
            df[col] = _restore_dtype(df[col], imputed[col])
            results.append(
                CleaningResult(
                    operation="ml_impute",
                    rows_affected=before,
                    column=col,
                    description_th=(
                        f"[MAR] เติมค่าว่าง {before:,} ค่าด้วย IterativeImputer (BayesianRidge, "
                        f"{len(predictors)} predictors{train_note})"
                    ),
                )
            )
    except Exception as exc:  # noqa: BLE001 — IterativeImputer ล้มเหลว → ถอยไป median (no silent crash)
        for col in num_na_cols:
            before = record_before[col]
            df[col], _ = handle_missing_values(df[col], "median")
            results.append(
                CleaningResult(
                    operation="ml_impute",
                    rows_affected=before,
                    column=col,
                    description_th=(
                        f"[MAR→median fallback] IterativeImputer ล้มเหลว ({exc}) — "
                        f"เติม {before:,} ค่าด้วย median แทน"
                    ),
                )
            )

    return results


def _restore_dtype(original: pd.Series, imputed: pd.Series) -> pd.Series:
    """คืนค่าที่เติมแล้วโดยพยายามคงชนิดข้อมูลเดิม (int ที่ไม่มี NA เหลือ → คงเป็น int)."""
    if pd.api.types.is_integer_dtype(original.dropna()):
        with contextlib.suppress(Exception):
            rounded = imputed.round()
            if np.isfinite(rounded.to_numpy()).all():
                return rounded.astype(original.dropna().dtype, errors="ignore")
    return imputed


__all__ = ["ml_impute"]
