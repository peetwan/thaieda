"""เปรียบเทียบชุดข้อมูลสองชุดแบบเคียงข้างกัน (side-by-side dataset comparison).

โมดูลนี้เปรียบเทียบ DataFrame สองชุด ตรวจหาความแตกต่างในด้าน:
  - Schema: คอลัมน์ที่เพิ่ม/หาย/เปลี่ยนชนิดข้อมูล
  - Row count: จำนวนแถวที่แตกต่างกัน
  - Per-column numeric stats: mean/median/std/min/max
  - Missing values: จำนวนค่าว่างต่อคอลัมน์
  - Distribution drift: KS statistic (Kolmogorov–Smirnov) สำหรับคอลัมน์ตัวเลข
  - Categorical drift: ความเปลี่ยนแปลงความถี่ค่าที่พบบ่อยในคอลัมน์หมวดหมู่

ตามหลักการของแพ็กเกจ: ไม่มี fallback แบบเงียบ — ถ้า scipy ไม่พร้อมใช้งาน
จะใช้การเปรียบเทียบ mean/std แบบง่ายแทน พร้อมระบุในผลลัพธ์ว่าใช้วิธีใด
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from thaieda._validation import ensure_unique_column_names

__all__ = ["compare", "compare_datasets", "compare_reports"]


def _finite_numeric(series: pd.Series) -> pd.Series:
    """Return numeric values excluding NaN and +/-inf."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return numeric
    finite = np.isfinite(numeric.to_numpy(dtype="float64"))
    return numeric[finite]


def _coerce_dataframe(obj: Any, name: str) -> pd.DataFrame:
    """Accept DataFrame, ProfileReport-like (.df), or EDAResult-like (.report.df)."""
    if isinstance(obj, pd.DataFrame):
        df = obj
    else:
        report = getattr(obj, "report", None)
        df = getattr(report, "df", None) if report is not None else getattr(obj, "df", None)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} ต้องเป็น pandas DataFrame, ProfileReport, หรือ EDAResult.")
    ensure_unique_column_names(df, context=name)
    return df


# ----------------------------------------------------------------------------
# Helper — ตรวจสอบ scipy ว่าพร้อมใช้งานหรือไม่
# ----------------------------------------------------------------------------
def _get_scipy_stats():
    """คืนโมดูล scipy.stats ถ้าติดตั้งแล้ว มิฉะนั้นคืน None.

    ไม่ใช่ silent fallback — compare_datasets จะระบุชัดเจนในผลลัพธ์ว่าใช้ scipy หรือไม่
    """
    try:
        import scipy.stats as st
    except ImportError:
        return None
    return st


# ----------------------------------------------------------------------------
# compare_datasets — เปรียบเทียบ DataFrame สองชุด คืน dict ผลลัพธ์
# ----------------------------------------------------------------------------
def compare_datasets(
    df1: Any,
    df2: Any,
    labels: tuple[str, ...] = ("A", "B"),
) -> dict[str, Any]:
    """เปรียบเทียบ DataFrame สองชุดแบบเคียงข้างกัน คืน dict ผลลัพธ์ที่มีโครงสร้าง.

    Args:
        df1: DataFrame ชุดแรก (ป้ายชื่อ labels[0]).
        df2: DataFrame ชุดที่สอง (ป้ายชื่อ labels[1]).
        labels: ป้ายชื่อของชุดข้อมูลทั้งสอง (default: ("A", "B")).

    Returns:
        dict ที่มี key ดังนี้:
            - "labels": ป้ายชื่อของชุดข้อมูลทั้งสอง
            - "schema_diff": {"columns_only_in_a", "columns_only_in_b", "type_changes"}
            - "row_count": {"a", "b", "diff"}
            - "numeric_stats_diff": dict ต่อคอลัมน์ตัวเลขที่มีในทั้งสองชุด
            - "missing_diff": dict ต่อคอลัมน์ที่มีในทั้งสองชุด
            - "distribution_drift": dict ต่อคอลัมน์ตัวเลข — KS statistic หรือ mean/std comparison
            - "categorical_drift": dict ต่อคอลัมน์หมวดหมู่ — ความถี่ค่าที่พบบ่อย

    Raises:
        TypeError: ถ้า df1 หรือ df2 ไม่ใช่ pandas DataFrame.
        ValueError: ถ้า labels ไม่ใช่ tuple ความยาว 2.
    """
    df1 = _coerce_dataframe(df1, "df1")
    df2 = _coerce_dataframe(df2, "df2")
    if not isinstance(labels, tuple) or len(labels) != 2:
        raise ValueError("labels ต้องเป็น tuple ความยาว 2")

    label_a, label_b = labels
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    common = cols1 & cols2

    # ---- Schema diff ----
    cols_only_in_a = sorted(cols1 - cols2, key=str)
    cols_only_in_b = sorted(cols2 - cols1, key=str)

    type_changes: list[dict[str, Any]] = []
    for col in sorted(common, key=str):
        dt1 = str(df1[col].dtype)
        dt2 = str(df2[col].dtype)
        if dt1 != dt2:
            type_changes.append(
                {"column": str(col), f"dtype_{label_a}": dt1, f"dtype_{label_b}": dt2}
            )

    schema_diff: dict[str, Any] = {
        f"columns_only_in_{label_a}": [str(c) for c in cols_only_in_a],
        f"columns_only_in_{label_b}": [str(c) for c in cols_only_in_b],
        "type_changes": type_changes,
    }

    # ---- Row count diff ----
    rows_a, rows_b = len(df1), len(df2)
    row_count: dict[str, Any] = {
        label_a: rows_a,
        label_b: rows_b,
        "diff": rows_b - rows_a,
    }

    # ---- Per-column numeric stats diff ----
    numeric_stats_diff: dict[str, Any] = {}
    missing_diff: dict[str, Any] = {}
    distribution_drift: dict[str, Any] = {}

    st = _get_scipy_stats()

    numeric_common = [
        c
        for c in sorted(common, key=str)
        if pd.api.types.is_numeric_dtype(df1[c]) and pd.api.types.is_numeric_dtype(df2[c])
    ]

    for col in numeric_common:
        s1 = _finite_numeric(df1[col])
        s2 = _finite_numeric(df2[col])

        def _safe_stat(func, s, default=None):
            if len(s) == 0:
                return default
            try:
                value = float(func(s))
                if not math.isfinite(value):
                    return default
                return round(value, 4)
            except (ValueError, OverflowError, ZeroDivisionError):
                return default

        stats_a = {
            "mean": _safe_stat(np.mean, s1),
            "median": _safe_stat(np.median, s1),
            "std": _safe_stat(np.std, s1, default=0.0),
            "min": _safe_stat(np.min, s1),
            "max": _safe_stat(np.max, s1),
            "count": int(len(s1)),
        }
        stats_b = {
            "mean": _safe_stat(np.mean, s2),
            "median": _safe_stat(np.median, s2),
            "std": _safe_stat(np.std, s2, default=0.0),
            "min": _safe_stat(np.min, s2),
            "max": _safe_stat(np.max, s2),
            "count": int(len(s2)),
        }

        numeric_stats_diff[str(col)] = {
            label_a: stats_a,
            label_b: stats_b,
            "mean_diff": _diff_or_none(stats_a["mean"], stats_b["mean"]),
            "std_diff": _diff_or_none(stats_a["std"], stats_b["std"]),
        }

        # ---- Distribution drift (KS statistic หรือ mean/std comparison) ----
        if len(s1) > 0 and len(s2) > 0:
            if st is not None:
                ks_stat, ks_pvalue = st.ks_2samp(s1.values, s2.values)
                distribution_drift[str(col)] = {
                    "method": "ks_2samp",
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(ks_pvalue), 6),
                    "drift_detected": float(ks_stat) > 0.10,
                }
            else:
                # ไม่มี scipy — ใช้ mean/std comparison แบบง่าย (ไม่ใช่ silent fallback:
                # ระบุชัดในผลลัพธ์ว่า method="mean_std")
                mean_diff = abs(float(s1.mean()) - float(s2.mean()))
                pooled_std = (
                    math.sqrt(float(s1.std() ** 2) + float(s2.std() ** 2)) / 2
                    if (float(s1.std()) > 0 or float(s2.std()) > 0)
                    else 0.0
                )
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
                distribution_drift[str(col)] = {
                    "method": "mean_std",
                    "mean_diff": round(mean_diff, 4),
                    "cohens_d": round(cohens_d, 4),
                    "drift_detected": abs(cohens_d) > 0.20,
                    "note": (
                        "scipy ไม่พร้อมใช้งาน จึงใช้การเปรียบเทียบ mean/std (ติดตั้ง thaieda[stats] เพื่อ KS)"
                    ),
                }

    # ---- Missing value diff (ทุกคอลัมน์ที่มีในทั้งสองชุด) ----
    for col in sorted(common, key=str):
        miss_a = int(df1[col].isna().sum())
        miss_b = int(df2[col].isna().sum())
        missing_diff[str(col)] = {
            label_a: miss_a,
            label_b: miss_b,
            "diff": miss_b - miss_a,
        }

    # ---- Categorical drift ----
    categorical_drift: dict[str, Any] = {}
    categorical_common = sorted(common, key=str)
    for col in categorical_common:
        s1 = df1[col].dropna()
        s2 = df2[col].dropna()
        # ถ้าเป็น numeric ทั้งคู่จะข้ามไป และจัดเป็นคอลัมน์ตัวเลขไปแล้ว
        is_num_1 = pd.api.types.is_numeric_dtype(df1[col])
        is_num_2 = pd.api.types.is_numeric_dtype(df2[col])
        if is_num_1 and is_num_2:
            continue
        if len(s1) == 0 and len(s2) == 0:
            continue

        vc1 = s1.astype(str).value_counts(normalize=True)
        vc2 = s2.astype(str).value_counts(normalize=True)

        # รวมค่าทั้งหมด
        all_values = set(vc1.index) | set(vc2.index)
        freq_shift: list[dict[str, Any]] = []
        for val in sorted(all_values, key=lambda x: -(vc1.get(x, 0) + vc2.get(x, 0))):
            f1 = float(vc1.get(val, 0.0))
            f2 = float(vc2.get(val, 0.0))
            if abs(f1 - f2) > 1e-9:
                freq_shift.append(
                    {
                        "value": val,
                        f"freq_{label_a}": round(f1, 4),
                        f"freq_{label_b}": round(f2, 4),
                        "shift": round(f2 - f1, 4),
                    }
                )

        if freq_shift:
            categorical_drift[str(col)] = {
                "top_values": freq_shift[:10],
                "total_shifted_values": len(freq_shift),
            }

    return {
        "labels": list(labels),
        "schema_diff": schema_diff,
        "row_count": row_count,
        "numeric_stats_diff": numeric_stats_diff,
        "missing_diff": missing_diff,
        "distribution_drift": distribution_drift,
        "categorical_drift": categorical_drift,
    }


# ----------------------------------------------------------------------------
# compare_reports — สร้างรายงาน HTML เคียงข้างกัน
# ----------------------------------------------------------------------------
def compare_reports(
    df1: Any,
    df2: Any,
    labels: tuple[str, ...] = ("A", "B"),
    lang: str = "th",
) -> str:
    """สร้างรายงาน HTML เปรียบเทียบชุดข้อมูลสองชุดแบบเคียงข้างกัน.

    Args:
        df1: DataFrame ชุดแรก.
        df2: DataFrame ชุดที่สอง.
        labels: ป้ายชื่อของชุดข้อมูล (default: ("A", "B")).
        lang: ภาษาของรายงาน — "th" (default) | "en".

    Returns:
        HTML string ของรายงานเปรียบเทียบ.
    """
    result = compare_datasets(df1, df2, labels=labels)
    label_a, label_b = result["labels"]

    if lang == "en":
        titles = {
            "title": "ThaiEDA Dataset Comparison",
            "schema": "Schema Diff",
            "row_count": "Row Count",
            "numeric_stats": "Numeric Stats Diff",
            "missing": "Missing Value Diff",
            "drift": "Distribution Drift",
            "cat_drift": "Categorical Drift",
            "only_in": "only in",
            "type_changes": "Type Changes",
            "no_diff": "No differences detected.",
        }
    else:
        titles = {
            "title": "รายงานเปรียบเทียบชุดข้อมูล ThaiEDA",
            "schema": "ความแตกต่างของโครงสร้าง",
            "row_count": "จำนวนแถว",
            "numeric_stats": "ความแตกต่างสถิติตัวเลข",
            "missing": "ความแตกต่างค่าว่าง",
            "drift": "การเลื่อนของการกระจายตัว",
            "cat_drift": "การเลื่อนของหมวดหมู่",
            "only_in": "มีเฉพาะใน",
            "type_changes": "การเปลี่ยนชนิดข้อมูล",
            "no_diff": "ไม่พบความแตกต่าง",
        }

    T = titles

    # ---- สร้าง HTML sections ----
    sections: list[str] = []

    # --- หัวข้อรายงาน ---
    sections.append(f"<h1>{T['title']}</h1>")
    sections.append(f"<p><strong>{label_a}</strong> vs <strong>{label_b}</strong></p>")

    has_any_diff = False

    # --- Schema diff ---
    schema = result["schema_diff"]
    only_a = schema[f"columns_only_in_{label_a}"]
    only_b = schema[f"columns_only_in_{label_b}"]
    type_changes = schema["type_changes"]
    if only_a or only_b or type_changes:
        has_any_diff = True
        sections.append(f"<h2>{T['schema']}</h2>")
        if only_a:
            sections.append(f"<p>{T['only_in']} {label_a}: {', '.join(only_a)}</p>")
        if only_b:
            sections.append(f"<p>{T['only_in']} {label_b}: {', '.join(only_b)}</p>")
        if type_changes:
            sections.append(f"<h3>{T['type_changes']}</h3><table border=1 cellpadding=4>")
            sections.append(f"<tr><th>Column</th><th>{label_a}</th><th>{label_b}</th></tr>")
            for tc in type_changes:
                sections.append(
                    f"<tr><td>{tc['column']}</td><td>{tc[f'dtype_{label_a}']}</td>"
                    f"<td>{tc[f'dtype_{label_b}']}</td></tr>"
                )
            sections.append("</table>")

    # --- Row count ---
    rc = result["row_count"]
    if rc["diff"] != 0:
        has_any_diff = True
        sections.append(f"<h2>{T['row_count']}</h2>")
        sections.append(
            f"<p>{label_a}: {rc[label_a]:,} | {label_b}: {rc[label_b]:,} "
            f"(diff: {rc['diff']:+,})</p>"
        )

    # --- Numeric stats diff ---
    nstats = result["numeric_stats_diff"]
    if nstats:
        sections.append(f"<h2>{T['numeric_stats']}</h2><table border=1 cellpadding=4>")
        sections.append(
            f"<tr><th>Column</th><th>Stat</th><th>{label_a}</th>"
            f"<th>{label_b}</th><th>Diff</th></tr>"
        )
        for col, data in nstats.items():
            for stat_key in ("mean", "std", "median", "min", "max"):
                va = data[label_a].get(stat_key)
                vb = data[label_b].get(stat_key)
                va_s = f"{va:.4f}" if isinstance(va, float | int) else "-"
                vb_s = f"{vb:.4f}" if isinstance(vb, float | int) else "-"
                diff_s = "-"
                if isinstance(va, float | int) and isinstance(vb, float | int):
                    diff_s = f"{vb - va:.4f}"
                    if abs(vb - va) > 1e-9:
                        has_any_diff = True
                sections.append(
                    f"<tr><td>{col}</td><td>{stat_key}</td><td>{va_s}</td>"
                    f"<td>{vb_s}</td><td>{diff_s}</td></tr>"
                )
        sections.append("</table>")

    # --- Missing diff ---
    mdiff = result["missing_diff"]
    if mdiff:
        any_missing_diff = False
        sections.append(f"<h2>{T['missing']}</h2><table border=1 cellpadding=4>")
        sections.append(
            f"<tr><th>Column</th><th>{label_a}</th><th>{label_b}</th><th>Diff</th></tr>"
        )
        for col, data in mdiff.items():
            if data["diff"] != 0:
                any_missing_diff = True
                has_any_diff = True
            sections.append(
                f"<tr><td>{col}</td><td>{data[label_a]}</td><td>{data[label_b]}</td>"
                f"<td>{data['diff']:+}</td></tr>"
            )
        sections.append("</table>")
        if not any_missing_diff:
            # ถ้าทุก diff=0 จะเพิ่ม note แต่ตารางยังแสดง
            pass

    # --- Distribution drift ---
    drift = result["distribution_drift"]
    if drift:
        sections.append(f"<h2>{T['drift']}</h2><table border=1 cellpadding=4>")
        sections.append(
            "<tr><th>Column</th><th>Method</th><th>Statistic</th><th>Drift Detected</th></tr>"
        )
        for col, data in drift.items():
            has_any_diff = has_any_diff or data.get("drift_detected", False)
            method = data.get("method", "?")
            stat_val = ""
            if "ks_statistic" in data:
                stat_val = f"KS={data['ks_statistic']:.4f}, p={data.get('p_value', '-')}"
            elif "cohens_d" in data:
                stat_val = f"Cohen's d={data['cohens_d']:.4f}"
            detected_str = "✓" if data.get("drift_detected") else "✗"
            sections.append(
                f"<tr><td>{col}</td><td>{method}</td><td>{stat_val}</td>"
                f"<td>{detected_str}</td></tr>"
            )
        sections.append("</table>")

    # --- Categorical drift ---
    cat_drift = result["categorical_drift"]
    if cat_drift:
        sections.append(f"<h2>{T['cat_drift']}</h2>")
        for col, data in cat_drift.items():
            has_any_diff = True
            sections.append(f"<h4>{col}</h4><table border=1 cellpadding=4>")
            sections.append(
                f"<tr><th>Value</th><th>freq {label_a}</th>"
                f"<th>freq {label_b}</th><th>Shift</th></tr>"
            )
            for tv in data.get("top_values", []):
                sections.append(
                    f"<tr><td>{tv['value']}</td><td>{tv.get(f'freq_{label_a}', 0):.4f}</td>"
                    f"<td>{tv.get(f'freq_{label_b}', 0):.4f}</td>"
                    f"<td>{tv.get('shift', 0):+.4f}</td></tr>"
                )
            sections.append("</table>")

    if not has_any_diff:
        sections.append(f"<p style='color:green'>{T['no_diff']}</p>")

    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{T['title']}</title>"
        "<style>"
        "body{font-family:sans-serif;margin:2em;}"
        "h1{color:#1a5276;} h2{border-bottom:2px solid #2980b9;padding-bottom:4px;}"
        "table{border-collapse:collapse;margin:10px 0;} "
        "th{background:#2980b9;color:#fff;} th,td{border:1px solid #ddd;}"
        "td{padding:4px 8px;}"
        "</style></head><body>"
    )
    html += "\n".join(sections)
    html += "</body></html>"
    return html


# ----------------------------------------------------------------------------
# Helper — คำนวณค่าผลต่าง ถ้าทั้งคู่ไม่ใช่ None
# ----------------------------------------------------------------------------
def _diff_or_none(a: float | int | None, b: float | int | None) -> float | None:
    """คืน b - a ถ้าทั้งคู่ไม่ใช่ None มิฉะนั้นคืน None."""
    if a is None or b is None:
        return None
    return round(float(b - a), 4)


compare = compare_datasets
