"""สร้าง prompt สำหรับ LLM จากข้อมูลที่เตรียมไว้ (v0.9).

สร้าง prompt ภาษาไทยหรืออังกฤษตามที่ผู้ใช้เลือก โดยประกอบด้วย:
  * บทบาทของตัวช่วย (system context)
  * สถิติสรุปของข้อมูล
  * ข้อค้นพบเชิงลึกจาก insight_engine (ถ้ามี)
  * ข้อมูลจาก profile (ถ้ามี)
  * ข้อมูลที่ทำให้ไม่ระบุตัวได้ หรือข้อมูลดิบ (ถ้าโหมดกำหนด)

หลักการ:
  * ไม่ฝังข้อมูล raw ถ้าโหมดเป็น insight_only/dp_noise (data = None)
  * ถ้ามี data DataFrame จะแปลงเป็น markdown table ก่อนฝังใน prompt
  * ไม่มี silent fallback — ถ้า insights ไม่ใช่ list/dict จะ raise TypeError
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

# ----------------------------------------------------------------------------
# ข้อความ template ภาษาไทย/อังกฤษ
# ----------------------------------------------------------------------------
_SYSTEM_TH = (
    "คุณเป็นผู้ช่วยวิเคราะห์ข้อมูลของ ThaiEDA — ไลบรารี AutoEDA สำหรับข้อมูลภาษาไทย.\n"
    "หน้าที่ของคุณ: วิเคราะห์สถิติและข้อค้นพบที่ให้มา แล้วสรุปเป็นข้อความที่อ่านง่าย "
    "ชี้ประเด็นสำคัญ ความเสี่ยง และคำแนะนำเชิงปฏิบัติ\n"
)

_SYSTEM_EN = (
    "You are a data analysis assistant for ThaiEDA — an AutoEDA library for Thai-language data.\n"
    "Your role: analyze the provided statistics and findings, then summarize them into clear, "
    "actionable insights highlighting key issues, risks, and recommendations.\n"
)

_PRIVACY_LABEL_TH: dict[str, str] = {
    "insight_only": "โหมดปกปิตัวตนสูง — ส่งเฉพาะสถิติและข้อค้นพบ ไม่ส่งข้อมูลดิบ",
    "anonymized": "โหมดทำให้ไม่ระบุตัวได้ — ข้อมูล PII ถูกแทนที่ด้วย token",
    "dp_noise": "โหมด differential privacy — สถิติมีเสียงรบกวนเพื่อปกปิดข้อมูลส่วนบุคคล",
    "full": "⚠️ โหมดข้อมูลดิบ — ผู้ใช้ยอมรับความเสี่ยง เนื้อหาต่อไปนี้มีข้อมูลดิบ",
}

_PRIVACY_LABEL_EN: dict[str, str] = {
    "insight_only": "High privacy mode — only statistics and findings shared, no raw data",
    "anonymized": "Anonymized mode — PII replaced with tokens",
    "dp_noise": "Differential privacy mode — statistics with added noise to protect privacy",
    "full": "⚠️ Full raw data mode — user accepted the risk; raw data follows",
}


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def build_prompt(
    prepared_data: dict[str, Any],
    profile: Any | None = None,
    insights: list[Any] | None = None,
    language: str = "th",
) -> str:
    """สร้าง prompt ภาษาไทย/อังกฤษจากข้อมูลที่เตรียมไว้.

    Args:
        prepared_data: dict จาก ``prepare_for_llm`` (มี mode, summary, data, token_map).
        profile: DatasetProfile (optional — ดึง metadata ถ้ายังไม่ได้ดึง).
        insights: รายการข้อค้นพบ แต่ละรายการเป็น dict ที่มี
            ``title_th``/``description_th`` หรือ ``title``/``description``.
        language: "th" (default) หรือ "en" — ภาษาของ prompt.

    Returns:
        ข้อความ prompt ที่พร้อมส่งให้ LLM (non-empty string).

    Raises:
        TypeError: ถ้า prepared_data ไม่ใช่ dict หรือ insights ไม่ใช่ list.
    """
    if not isinstance(prepared_data, dict):
        raise TypeError(f"prepared_data ต้องเป็น dict ได้รับ {type(prepared_data).__name__}")
    if insights is not None and not isinstance(insights, list):
        raise TypeError(f"insights ต้องเป็น list หรือ None ได้รับ {type(insights).__name__}")
    if language not in ("th", "en"):
        raise ValueError(f"ไม่รองรับภาษา {language!r} — รองรับ: 'th' (ไทย) หรือ 'en' (อังกฤษ)")

    use_thai = language == "th"
    system = _SYSTEM_TH if use_thai else _SYSTEM_EN

    privacy_mode = prepared_data.get("mode", "insight_only")
    label_map = _PRIVACY_LABEL_TH if use_thai else _PRIVACY_LABEL_EN
    privacy_label = label_map.get(privacy_mode, "")

    # เริ่มสร้าง prompt ทีละส่วน
    parts: list[str] = [system]

    # 1. โหมดความเป็นส่วนตัว
    if privacy_label:
        parts.append(f"\n## {'โหมดความเป็นส่วนตัว' if use_thai else 'Privacy Mode'}\n{privacy_label}")

    # 2. สถิติสรุป
    summary = prepared_data.get("summary", {})
    if summary:
        parts.append(_format_summary(summary, use_thai))

    # 3. ข้อมูลจาก profile
    prof_info = prepared_data.get("profile_info")
    if prof_info is None and profile is not None:
        # ยังไม่ได้ดึง ให้ดึงเอง
        from thaieda.llm._prepare import _extract_profile_info

        prof_info = _extract_profile_info(profile)
    if prof_info:
        parts.append(_format_profile_info(prof_info, use_thai))

    # 4. ข้อค้นพบเชิงลึก
    if insights:
        parts.append(_format_insights(insights, use_thai))

    # 5. ข้อมูลที่ทำให้ไม่ระบุตัวได้ หรือข้อมูลดิบ (ถ้ามี)
    data = prepared_data.get("data")
    if data is not None and isinstance(data, pd.DataFrame):
        header = "ข้อมูลตัวอย่าง" if use_thai else "Data Sample"
        parts.append(f"\n## {header}\n{_df_to_markdown(data)}")

    # 6. คำสั่งสำหรับ LLM
    instruction = (
        ("\n## คำสั่ง\nโปรดวิเคราะห์ข้อมูลข้างต้นและสรุปประเด็นสำคัญ ความเสี่ยง และคำแนะนำเชิงปฏิบัติ เป็นภาษาไทย")
        if use_thai
        else (
            "\n## Instructions\nPlease analyze the above data and summarize key findings, risks, "
            "and actionable recommendations in English."
        )
    )
    parts.append(instruction)

    return "\n".join(parts)


# ----------------------------------------------------------------------------
# helper สำหรับจัดรูปแบบแต่ละส่วน
# ----------------------------------------------------------------------------
def _format_summary(summary: dict[str, Any], use_thai: bool) -> str:
    """จัดรูปแบบสถิติสรุปเป็นข้อความ markdown."""
    title = "สถิติสรุปข้อมูล" if use_thai else "Data Summary"
    lines: list[str] = [f"\n## {title}"]

    # shape
    shape = summary.get("shape", (0, 0))
    if use_thai:
        lines.append(f"- ขนาด: {shape[0]:,} แถว × {shape[1]} คอลัมน์")
    else:
        lines.append(f"- Shape: {shape[0]:,} rows × {shape[1]} columns")

    # null counts (เฉพาะที่มี null)
    null_counts = summary.get("null_counts", {})
    nulls = {k: v for k, v in null_counts.items() if v > 0}
    if nulls:
        label = "ค่าว่าง" if use_thai else "Null counts"
        lines.append(f"- {label}: {json.dumps(nulls, ensure_ascii=False)}")

    # numeric stats
    numeric_stats = summary.get("numeric_stats", {})
    if numeric_stats:
        label = "สถิติเชิงตัวเลข" if use_thai else "Numeric Statistics"
        lines.append(f"\n### {label}")
        for col, stats in numeric_stats.items():
            mean_str = f"mean={stats['mean']:.2f}" if "mean" in stats else "mean=N/A"
            std_str = f"std={stats['std']:.2f}" if "std" in stats else "std=N/A"
            min_str = f"min={stats['min']:.2f}" if "min" in stats else "min=N/A"
            max_str = f"max={stats['max']:.2f}" if "max" in stats else "max=N/A"

            q25 = stats.get("q25", stats.get("25%"))
            q25_str = f"q25={q25:.2f}" if q25 is not None else "q25=N/A"

            q50 = stats.get("q50", stats.get("50%"))
            q50_str = f"q50={q50:.2f}" if q50 is not None else "q50=N/A"

            q75 = stats.get("q75", stats.get("75%"))
            q75_str = f"q75={q75:.2f}" if q75 is not None else "q75=N/A"

            lines.append(
                f"- **{col}**: "
                f"{mean_str}, {std_str}, {min_str}, {max_str}, {q25_str}, {q50_str}, {q75_str}"
            )

    # categorical stats
    cat_stats = summary.get("categorical_stats", {})
    if cat_stats:
        label = "สถิติเชิงหมวดหมู่" if use_thai else "Categorical Statistics"
        lines.append(f"\n### {label}")
        for col, stats in cat_stats.items():
            top = stats.get("top_value", "")
            freq = stats.get("top_freq", 0)
            uniq = stats.get("unique_count", 0)
            if use_thai:
                lines.append(f"- **{col}**: ค่าไม่ซ้ำ {uniq}, ค่าเด่น '{top}' ({freq} ครั้ง)")
            else:
                lines.append(f"- **{col}**: {uniq} unique, top '{top}' ({freq} times)")

    return "\n".join(lines)


def _format_profile_info(prof_info: dict[str, Any], use_thai: bool) -> str:
    """จัดรูปแบบข้อมูลจาก DatasetProfile."""
    if not prof_info or not prof_info.get("profile_available"):
        return ""
    title = "ข้อมูลมัลติตาราง" if use_thai else "Multi-table Profile"
    lines = [f"\n## {title}"]
    if "table_count" in prof_info:
        label = "จำนวนตาราง" if use_thai else "Tables"
        lines.append(f"- {label}: {prof_info['table_count']}")
    if "relationship_count" in prof_info:
        label = "จำนวนความสัมพันธ์" if use_thai else "Relationships"
        lines.append(f"- {label}: {prof_info['relationship_count']}")
    orphans = prof_info.get("orphan_findings", [])
    if orphans:
        label = "ข้อมูลกำพร้า" if use_thai else "Orphan findings"
        lines.append(f"\n### {label}")
        lines.extend(f"- {o}" for o in orphans[:5])
    return "\n".join(lines)


def _format_insights(insights: list[Any], use_thai: bool) -> str:
    """จัดรูปแบบข้อค้นพบเชิงลึกเป็นข้อความ markdown."""
    title = "ข้อค้นพบเชิงลึก" if use_thai else "Insights"
    lines = [f"\n## {title}"]
    for i, ins in enumerate(insights, start=1):
        if isinstance(ins, dict):
            # รองรับทั้ง key ไทยและอังกฤษ
            t = ins.get("title_th") or ins.get("title") or ""
            d = ins.get("description_th") or ins.get("description") or ""
            r = ins.get("recommendation_th") or ins.get("recommendation") or ""
        else:
            # ถ้าเป็น dataclass ที่มี attribute
            t = getattr(ins, "title_th", getattr(ins, "title", ""))
            d = getattr(ins, "description_th", getattr(ins, "description", ""))
            r = getattr(ins, "recommendation_th", getattr(ins, "recommendation", ""))
        lines.append(f"\n### {i}. {t}")
        if d:
            lines.append(f"**{'คำอธิบาย' if use_thai else 'Description'}**: {d}")
        if r:
            lines.append(f"**{'คำแนะนำ' if use_thai else 'Recommendation'}**: {r}")
    return "\n".join(lines)


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    """แปลง DataFrame เป็น markdown table (จำกัดจำนวนแถวเพื่อความปลอดภัยของ prompt)."""
    if df is None or len(df) == 0 or len(df.columns) == 0:
        return "(ไม่มีข้อมูล / no data)"

    # จำกัดจำนวนแถวก่อน
    sample = df.head(max_rows)
    # แปลงเป็น markdown table — ใช้ to_markdown (ต้องมี tabulate ถ้าไม่มีใช้ to_string)
    try:
        md = sample.to_markdown(index=False)
    except ImportError:
        # ไม่มี tabulate — ใช้ to_string แบบตารางแทน
        md = f"```\n{sample.to_string(index=False)}\n```"
    return str(md)


__all__ = ["build_prompt"]
