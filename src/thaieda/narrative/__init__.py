"""Template narrative — สร้าง story จากผล EDA โดยไม่ต้องใช้ LLM — v2.0.

ทำงานได้ offline, deterministic, ไม่ต้องมี API key — ใช้ template + jinja2 (มีอยู่แล้ว)
แปลงผลจาก insight engine + quality score + cleaning report เป็นบทสรุปผู้บริหาร
ข้อค้นพบสำคัญ คำแนะนำ และคำถามติดตาม (ภาษาไทย + อังกฤษ).

หลักการ:
  * deterministic — ผลลัพธ์เหมือนเดิมทุกครั้งสำหรับ input เดียวกัน (ไม่มีการสุ่ม)
  * ไม่ต้องต่ออินเทอร์เน็ต / ไม่ต้องมี LLM — ใช้เป็น fallback ของโหมด LLM ได้
  * รับ insight ได้หลายรูปแบบ (InsightCard / Insight / dict) ผ่าน duck typing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from jinja2 import Template

# จำนวนข้อค้นพบ/คำแนะนำสูงสุดที่นำมาแสดง
_MAX_FINDINGS = 5
_MAX_RECOMMENDATIONS = 6
_MAX_FOLLOWUPS = 5

# ----------------------------------------------------------------------------
# templates (jinja2) — บทสรุปผู้บริหาร 2 ภาษา
# ----------------------------------------------------------------------------
_EXEC_TH = Template(
    "จากการวิเคราะห์ข้อมูล พบข้อค้นพบสำคัญทั้งหมด {{ n_findings }} เรื่อง"
    "{% if n_critical %} (ในจำนวนนี้ {{ n_critical }} เรื่องร้ายแรงที่ควรรีบจัดการ)"
    "{% elif n_warning %} (มี {{ n_warning }} เรื่องที่ควรเฝ้าระวัง){% endif %}. "
    "{% if quality %}คะแนนคุณภาพข้อมูลอยู่ที่ {{ quality.score }}/100 "
    "(เกรด {{ quality.grade }}). {% endif %}"
    "{% if cleaning %}ข้อมูลผ่านการทำความสะอาด {{ cleaning.total_changes }} จุด "
    "({{ cleaning.rows_before }}→{{ cleaning.rows_after }} แถว). {% endif %}"
    "{% if top_finding %}ข้อค้นพบที่เด่นที่สุดคือ “{{ top_finding }}”. "
    "{% else %}ยังไม่พบข้อค้นพบเชิงลึกที่โดดเด่นจากชุดข้อมูลนี้. {% endif %}"
    "แนะนำให้พิจารณาคำแนะนำและคำถามติดตามด้านล่างเพื่อต่อยอดการวิเคราะห์.",
    autoescape=False,
)

_EXEC_EN = Template(
    "The analysis surfaced {{ n_findings }} key finding(s)"
    "{% if n_critical %} ({{ n_critical }} critical, needing prompt attention)"
    "{% elif n_warning %} ({{ n_warning }} worth monitoring){% endif %}. "
    "{% if quality %}Overall data quality scores {{ quality.score }}/100 "
    "(grade {{ quality.grade }}). {% endif %}"
    "{% if cleaning %}Cleaning applied {{ cleaning.total_changes }} change(s) "
    "({{ cleaning.rows_before }}→{{ cleaning.rows_after }} rows). {% endif %}"
    "{% if top_finding %}The most prominent finding is “{{ top_finding }}”. "
    "{% else %}No standout cross-column insight was detected for this dataset. {% endif %}"
    "Review the recommendations and follow-up questions below to go deeper.",
    autoescape=False,
)


@dataclass
class NarrativeResult:
    """ผลลัพธ์การสร้าง narrative — v2.0."""

    executive_summary_th: str  # สรุปภาษาไทย
    executive_summary_en: str  # English summary
    key_findings: list[str] = field(default_factory=list)  # ข้อค้นพบสำคัญ
    recommendations: list[str] = field(default_factory=list)  # คำแนะนำ
    follow_up_questions: list[str] = field(default_factory=list)  # คำถามติดตาม

    def to_dict(self) -> dict:
        return {
            "executive_summary_th": self.executive_summary_th,
            "executive_summary_en": self.executive_summary_en,
            "key_findings": self.key_findings,
            "recommendations": self.recommendations,
            "follow_up_questions": self.follow_up_questions,
        }

    def to_json(self, path: str | None = None) -> str:
        """ส่งออกเป็น JSON — ถ้าระบุ path จะเขียนไฟล์ด้วย."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path is not None:
            from pathlib import Path

            Path(path).write_text(text, encoding="utf-8")
        return text


def generate_narrative(
    insights: list[Any],
    quality_score: dict | None = None,
    cleaning_report: Any | None = None,
    language: str = "th",
) -> NarrativeResult:
    """สร้าง narrative จากผล EDA — ไม่ต้องใช้ LLM — v2.0.

    ทำงานได้ offline, deterministic, no API key needed.
    ใช้ template + jinja2 เพื่อสร้าง story จาก insights + quality + cleaning.

    Args:
        insights: รายการข้อค้นพบ (InsightCard / Insight / dict — duck typed).
        quality_score: dict จาก ``compute_quality_score`` ({score, grade, breakdown}) — optional.
        cleaning_report: CleaningReport จาก ``thaieda.clean`` — optional.
        language: ภาษาของ key_findings/recommendations/follow_up — "th" (default) | "en".
            (executive_summary มีทั้ง 2 ภาษาเสมอ).

    Returns:
        NarrativeResult.
    """
    lang = "en" if str(language).lower().startswith("en") else "th"

    norm = [_normalize_insight(i) for i in (insights or [])]
    # เรียงตาม score (มาก→น้อย) ถ้ามี; insight ที่ severity สูงมาก่อน
    norm.sort(key=_sort_key, reverse=True)

    n_critical = sum(1 for d in norm if d["severity"] == "critical")
    n_warning = sum(1 for d in norm if d["severity"] == "warning")
    quality_ctx = _quality_ctx(quality_score)
    cleaning_ctx = _cleaning_ctx(cleaning_report)
    top_finding = norm[0]["title_th"] if norm else None

    ctx = {
        "n_findings": len(norm),
        "n_critical": n_critical,
        "n_warning": n_warning,
        "quality": quality_ctx,
        "cleaning": cleaning_ctx,
        "top_finding": top_finding,
    }

    return NarrativeResult(
        executive_summary_th=_EXEC_TH.render(**ctx).strip(),
        executive_summary_en=_EXEC_EN.render(**ctx).strip(),
        key_findings=_key_findings(norm, lang),
        recommendations=_recommendations(norm, quality_ctx, cleaning_ctx, lang),
        follow_up_questions=_follow_ups(norm, quality_ctx, top_finding, lang),
    )


# ----------------------------------------------------------------------------
# helper: normalize input
# ----------------------------------------------------------------------------
def _normalize_insight(item: Any) -> dict:
    """แปลง insight หลายรูปแบบ (InsightCard / Insight / dict) → dict มาตรฐาน."""
    if isinstance(item, dict):
        d = item
    elif hasattr(item, "to_dict") and callable(item.to_dict):
        d = item.to_dict()
    else:
        d = {"title_th": str(item)}

    title = d.get("title_th") or d.get("title") or d.get("description_th") or ""
    score = d.get("score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    return {
        "title_th": str(title),
        "description_th": str(d.get("description_th") or d.get("description") or ""),
        "recommendation_th": str(d.get("recommendation_th") or d.get("recommendation") or ""),
        "severity": str(d.get("severity") or "info"),
        "score": score,
    }


_SEVERITY_RANK = {"critical": 3, "warning": 2, "info": 1}


def _sort_key(d: dict) -> tuple[float, float]:
    """จัดอันดับ: severity ก่อน แล้วตามด้วย score."""
    sev = _SEVERITY_RANK.get(d["severity"], 0)
    score = d["score"] if d["score"] is not None else 0.0
    return (float(sev), float(score))


def _quality_ctx(quality_score: dict | None) -> dict | None:
    """ดึง score/grade จาก quality_score dict — คืน None ถ้าไม่มี."""
    if not quality_score or not isinstance(quality_score, dict):
        return None
    if "score" not in quality_score:
        return None
    return {"score": quality_score.get("score"), "grade": quality_score.get("grade", "?")}


def _cleaning_ctx(cleaning_report: Any | None) -> dict | None:
    """ดึงตัวเลขจาก CleaningReport (หรือ dict) — คืน None ถ้าไม่มี."""
    if cleaning_report is None:
        return None
    if isinstance(cleaning_report, dict):
        d = cleaning_report
    elif hasattr(cleaning_report, "to_dict"):
        d = cleaning_report.to_dict()
    else:
        return None
    return {
        "rows_before": d.get("rows_before", 0),
        "rows_after": d.get("rows_after", 0),
        "total_changes": d.get("total_changes", 0),
        "warnings": d.get("warnings", []),
    }


# ----------------------------------------------------------------------------
# helper: สร้างเนื้อหาแต่ละส่วน (bilingual)
# ----------------------------------------------------------------------------
def _key_findings(norm: list[dict], lang: str) -> list[str]:
    """ข้อค้นพบสำคัญ — top N เรียงตามความสำคัญ."""
    findings: list[str] = []
    for d in norm[:_MAX_FINDINGS]:
        title = d["title_th"]
        desc = d["description_th"]
        if lang == "en":
            sev = d["severity"].upper()
            line = f"[{sev}] {title}"
            if desc and desc != title:
                line += f" — {desc}"
        else:
            line = title
            if desc and desc != title:
                line += f" — {desc}"
        findings.append(line)
    return findings


def _recommendations(
    norm: list[dict], quality: dict | None, cleaning: dict | None, lang: str
) -> list[str]:
    """คำแนะนำ — จากข้อค้นพบ + คุณภาพข้อมูล + คำเตือนการทำความสะอาด."""
    recs: list[str] = []
    seen: set[str] = set()

    for d in norm:
        rec = d["recommendation_th"].strip()
        if rec and rec not in seen:
            seen.add(rec)
            recs.append(rec)
        if len(recs) >= _MAX_RECOMMENDATIONS:
            break

    # คำแนะนำจากคะแนนคุณภาพ
    if quality and quality.get("grade") in ("C", "D", "F"):
        if lang == "en":
            recs.append(
                f"Improve data quality (grade {quality['grade']}, {quality['score']}/100) "
                "before drawing firm conclusions."
            )
        else:
            recs.append(
                f"ควรปรับปรุงคุณภาพข้อมูล (เกรด {quality['grade']}, {quality['score']}/100) "
                "ก่อนสรุปผลเชิงลึก"
            )

    # คำเตือนจากการทำความสะอาด
    if cleaning and cleaning.get("warnings"):
        for w in cleaning["warnings"][:2]:
            if w not in seen:
                seen.add(w)
                recs.append(w)

    if not recs:
        recs.append(
            "Data looks clean — proceed to deeper modelling or segmentation."
            if lang == "en"
            else "ข้อมูลดูสะอาดดี — สามารถต่อยอดไปสร้างโมเดลหรือแบ่งกลุ่ม (segmentation) ได้"
        )
    return recs[:_MAX_RECOMMENDATIONS]


def _follow_ups(
    norm: list[dict], quality: dict | None, top_finding: str | None, lang: str
) -> list[str]:
    """คำถามติดตาม — สร้างจากบริบทข้อมูล (deterministic)."""
    qs: list[str] = []
    if lang == "en":
        if top_finding:
            qs.append(f"What underlying factors best explain “{top_finding}”?")
        qs.append("Does this pattern hold when the data is split into smaller segments?")
        if quality and quality.get("grade") not in ("A", None):
            qs.append("How were missing or anomalous values produced during data collection?")
        qs.append(
            "Are there external variables (seasonality, promotions, region) worth joining in?"
        )
        qs.append("Is there time-series data available to confirm whether the trend is stable?")
    else:
        if top_finding:
            qs.append(f"ปัจจัยใดอธิบาย “{top_finding}” ได้ดีที่สุด?")
        qs.append("ข้อค้นพบนี้ยังคงเป็นจริงหรือไม่เมื่อแบ่งข้อมูลเป็นกลุ่มย่อย (segment)?")
        if quality and quality.get("grade") not in ("A", None):
            qs.append("ค่าว่างหรือค่าผิดปกติเกิดจากกระบวนการเก็บข้อมูลอย่างไร?")
        qs.append("มีตัวแปรภายนอก (ฤดูกาล โปรโมชัน พื้นที่) ที่ควรนำมาร่วมวิเคราะห์หรือไม่?")
        qs.append("มีข้อมูลอนุกรมเวลาเพิ่มเติมเพื่อยืนยันว่าแนวโน้มคงที่หรือไม่?")
    return qs[:_MAX_FOLLOWUPS]


__all__ = ["NarrativeResult", "generate_narrative"]
