"""LLM Q&A over computed profiles — การวิเคราะห์ข้อมูลด้วย LLM โดยปกปิดข้อมูลส่วนบุคคล (v0.9 + v1.9).

สถาปัตยกรรม 5 โหมดความเป็นส่วนตัว:
  1. ``insight_only`` (default) — ส่งเฉพาะสถิติสรุป + ข้อค้นพบเชิงลึก (ไม่ส่งข้อมูลดิบ)
  2. ``synthetic`` (v1.9) — สร้างข้อมูลจำลองจาก distribution จริง ไม่มีค่าจริงปน
  3. ``anonymized`` — ลบ PII (ชื่อ/เบอร์/บัตร) ด้วย NER + regex แล้วส่งข้อมูลที่ไม่ระบุตัวได้
  4. ``dp_noise`` — สถิติสรุป + เสียงรบกวนแบบ differential privacy (Laplace mechanism)
  5. ``full`` — ส่งข้อมูลดิบทั้งหมด (ผู้ใช้ยอมรับความเสี่ยง)

โมดูลภายใน:
  * ``_prepare`` — เตรียมข้อมูลตามโหมด (prepare_for_llm)
  * ``_anonymize`` — ทำให้ไม่ระบุตัวบุคคลได้ (anonymize_dataframe)
  * ``_synthetic`` (v1.9) — สร้างข้อมูลจำลอง (generate_synthetic_data, privacy_audit_report)
  * ``_prompt`` — สร้าง prompt ภาษาไทย/อังกฤษ (build_prompt)
  * ``_provider`` — เรียก LLM API ของผู้ให้บริการต่าง ๆ (call_llm)

หลักการสำคัญ:
  * lazy import ของเสริม — openai/anthropic/ollama ไม่ต้องติดตั้งตอน import โมดูล
  * ไม่มี silent fallback — ถ้าขาด dependency จะ raise พร้อมคำแนะนำ
  * ปกปิตข้อมูลดิบตามโหมดก่อนส่งให้ LLM เสมอ
  * คำนวณ insight_engine จากข้อมูลในเครื่อง ไม่ส่งข้อมูลดิบออกไปคำนวณ
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from thaieda.llm._prepare import prepare_for_llm
from thaieda.llm._prompt import build_prompt
from thaieda.llm._provider import call_llm
from thaieda.llm._synthetic import generate_synthetic_data, export_synthetic_data, privacy_audit_report

# ----------------------------------------------------------------------------
# ค่าที่ export
# ----------------------------------------------------------------------------
__all__ = [
    "analyze_with_llm",
    "prepare_for_llm",
    "build_prompt",
    "call_llm",
    "generate_synthetic_data",
    "export_synthetic_data",
    "privacy_audit_report",
]


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก — public API
# ----------------------------------------------------------------------------
def analyze_with_llm(
    df: pd.DataFrame,
    privacy: str = "insight_only",
    provider: str = "openai",
    model: str | None = None,
    language: str = "th",
    *,
    profile: Any | None = None,
    insights: list[Any] | None = None,
    epsilon: float = 1.0,
) -> str:
    """วิเคราะห์ข้อมูลด้วย LLM — เตรียมข้อมูลตามโหมดความเป็นส่วนตัว สร้าง prompt แล้วเรียก LLM.

    ขั้นตอน:
      1. เตรียมข้อมูลตามโหมด ``privacy`` (``prepare_for_llm``)
         - insight_only: สถิติสรุป + ข้อค้นพบเท่านั้น
         - anonymized: ลบ PII แล้วส่งข้อมูลที่ปลอดภัย
         - dp_noise: สถิติ + เสียงรบกวน
         - full: ข้อมูลดิบ
      2. คำนวณข้อค้นพบเชิงลึกด้วย ``thaieda.insight_engine.discover_insights`` (ถ้าไม่ได้ส่งมา)
      3. สร้าง prompt ด้วย ``build_prompt``
      4. เรียก LLM API ด้วย ``call_llm`` (ตาม provider ที่เลือก)

    Args:
        df: DataFrame ที่จะวิเคราะห์.
        privacy: โหมดความเป็นส่วนตัว —
            "insight_only" (default) | "synthetic" | "anonymized" | "dp_noise" | "full".
        provider: ผู้ให้บริการ LLM — "openai" (default) | "anthropic" | "ollama".
        model: ชื่อโมเดล — None = default ของ provider.
        language: ภาษาของ prompt — "th" (default) | "en".
        profile: DatasetProfile จาก ``thaieda.schema`` (optional).
        insights: รายการข้อค้นพบเชิงลึก (optional — ถ้า None จะคำนวณเอง).
        epsilon: พารามิเตอร์ epsilon สำหรับ dp_noise (default: 1.0).

    Returns:
        ข้อความตอบกลับจาก LLM (string).

    Raises:
        ValueError: ถ้า privacy หรือ provider ไม่รองรับ.
        ImportError: ถ้า package ของ provider ไม่ได้ติดตั้ง.
        RuntimeError: ถ้าเรียก LLM API ไม่สำเร็จ.

    Example::

        >>> import pandas as pd
        >>> from thaieda.llm import analyze_with_llm
        >>> df = pd.DataFrame({"name": ["สมชาย"], "age": [25]})
        >>> # ไม่ส่งข้อมูลดิบ (default)
        >>> response = analyze_with_llm(df, privacy="insight_only", provider="ollama")
    """
    # ขั้นตอน 1: เตรียมข้อมูลตามโหมดความเป็นส่วนตัว
    prepared = prepare_for_llm(df, profile, privacy, epsilon=epsilon)

    # ขั้นตอน 2: คำนวณ insights ถ้าไม่ได้ส่งมา
    insights_list = insights
    if insights_list is None:
        insights_list = _compute_insights(df)

    # ขั้นตอน 3: สร้าง prompt
    prompt = build_prompt(prepared, profile, insights_list, language)

    # ขั้นตอน 4: เรียก LLM
    return call_llm(prompt, provider, model)


# ----------------------------------------------------------------------------
# helper: คำนวณ insights จาก insight_engine
# ----------------------------------------------------------------------------
def _compute_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """คำนวณข้อค้นพบเชิงลึกด้วย ``thaieda.insight_engine.discover_insights``.

    ทำการ detect column types แล้วเรียก insight engine เพื่อหา cross-column insights
    แปลงผลจาก InsightCard เป็น dict สำหรับ ``build_prompt``
    """
    from thaieda.detect import detect_all  # lazy import — ไม่ใช่ optional แต่หนีงจาก import loop
    from thaieda.insight_engine import discover_insights

    if df is None or len(df) == 0 or len(df.columns) == 0:
        return []

    column_types = detect_all(df)
    result = discover_insights(df, column_types)
    return [card.to_dict() for card in result.cards]
