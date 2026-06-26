"""Auto insight — สรุปข้อค้นพบสำคัญเป็นภาษาไทยแบบอ่านเข้าใจง่าย.

โมดูลนี้ "ตีความ" ผลจาก quality/anomaly/text/target/cleaning ให้กลายเป็นข้อสรุปที่
บอกว่า "อะไรสำคัญ ควรทำอะไรต่อ" ไม่ใช่แค่ทวนตัวเลข — เพื่อให้ผู้ใช้รู้ทันทีว่าข้อมูล
มีสุขภาพดีแค่ไหน และควรแก้อะไรก่อนนำไปวิเคราะห์

ทุก Insight มี title/description/recommendation เป็นภาษาไทย และจัดระดับความรุนแรง
(critical/warning/info) เพื่อให้เรียงลำดับความสำคัญได้
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from thaieda.analysis import TargetAssociation
from thaieda.anomaly import AnomalyIssue
from thaieda.clean import CleaningResult
from thaieda.detect import ColumnType
from thaieda.ner import NERResult
from thaieda.quality import QualityIssue
from thaieda.text import TextMetrics
from thaieda.timeseries import TimeseriesResult

# ลำดับความรุนแรง (วิกฤตก่อน) สำหรับการเรียง
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# จำนวน insight สูงสุดที่แสดงในรายงาน (กัน insight overflow บนชุดข้อมูลคอลัมน์เยอะ — P1)
# ตัดให้เหลือเท่านี้โดยเก็บ critical ทั้งหมด → เติม warning → เติม info
_MAX_INSIGHTS = 30

# ประเภทคอลัมน์ที่ถือว่าเป็นข้อความ
_TEXT_TYPES = {ColumnType.THAI_TEXT, ColumnType.MIXED_TEXT, ColumnType.ENGLISH_TEXT}

# หัวข้อสั้น (ภาษาไทย) ของปัญหาคุณภาพแต่ละชนิด
_QUALITY_TITLES: dict[str, str] = {
    "buddhist_era": "พบปีพุทธศักราช (พ.ศ.) อาจปนกับคริสต์ศักราช (ค.ศ.)",
    "thai_numerals": "พบเลขไทยผสมเลขอารบิก",
    "zero_width_chars": "พบอักขระล่องหน (zero-width) ที่ทำให้ groupby พัง",
    "normalization": "พบปัญหาการ normalize ข้อความ",
    "whitespace": "พบปัญหาช่องว่าง",
    "mislabeled_thai_column": "คอลัมน์อาจติดป้ายภาษาผิด",
}

# หัวข้อสั้น (ภาษาไทย) ของความผิดปกติแต่ละชนิด
_ANOMALY_TITLES: dict[str, str] = {
    "numeric_outliers": "พบค่าผิดปกติเชิงตัวเลข (outlier)",
    "isolation_forest": "พบค่าผิดปกติจากโมเดล (Isolation Forest)",
    "local_outlier_factor": "พบค่าผิดปกติเชิงความหนาแน่น (LOF)",
    "encoding_mojibake": "พบข้อความเสีย encoding (mojibake)",
    "thai_mojibake": "พบข้อความไทยเสีย encoding (mojibake)",
    "garbled_text": "พบข้อความเสียหาย (อักขระแทนที่/ควบคุม)",
    "text_length_anomaly": "พบความยาวข้อความผิดปกติ",
    "excessive_repetition": "พบอักขระซ้ำมากผิดปกติ",
    "invalid_thai_sequence": "พบลำดับอักขระไทยไม่ถูกต้อง",
    "tone_mark_stacking": "พบวรรณยุกต์ซ้อนติดกัน",
    "diacritic_order": "พบลำดับสระ/วรรณยุกต์ไม่เป็นมาตรฐาน",
    "abnormal_script_mixing": "พบการปนสคริปต์ผิดปกติ",
    "rare_categories": "พบหมวดหมู่หายาก (<1%)",
    "fuzzy_duplicates": "พบหมวดหมู่ที่คล้ายกันจนน่าสงสัยว่าซ้ำ",
    "case_inconsistency": "พบการใช้ตัวพิมพ์ใหญ่/เล็กไม่สม่ำเสมอ",
    "type_mixing": "พบการปนชนิดข้อมูล (ตัวเลข/ข้อความ)",
    "mixed_date_formats": "พบรูปแบบวันที่ปนกัน",
    "high_null_spike": "คอลัมน์มีค่าว่างสูงผิดปกติ",
    "constant_column": "คอลัมน์มีค่าเดียวทุกแถว (ไม่มีข้อมูล)",
}

# anomaly_type -> หมวดของ insight
_ANOMALY_TYPE_CATEGORY: dict[str, str] = {
    "statistical": "anomaly",
    "encoding": "anomaly",
    "text": "text",
    "pattern": "structure",
    "categorical": "anomaly",
}

# ชื่อสั้น (ภาษาไทย) ของการดำเนินการทำความสะอาด
_CLEAN_OP_TH: dict[str, str] = {
    "normalize_encoding": "แก้ encoding ผิด",
    "remove_zero_width_chars": "ลบอักขระล่องหน",
    "strip_whitespace": "จัดช่องว่าง",
    "normalize_unicode": "normalize Unicode",
    "fix_tone_mark_stacking": "ลบวรรณยุกต์ซ้อน",
    "fix_repeated_chars": "ลดอักขระซ้ำเกิน",
    "normalize_thai_numerals": "แปลงเลขไทยเป็นอารบิก",
    "pythainlp_normalize": "จัดระเบียบข้อความไทย",
    "fix_keyboard_layout": "แก้การพิมพ์ผิดแป้นพิมพ์",
}

# จำนวนคู่คอลัมน์สูงสุดที่จะตรวจ co-missing (กัน O(n^2) ระเบิดบนตารางคอลัมน์เยอะ)
_MAX_COMISSING_COLS = 30
# จำนวนแถวขั้นต่ำที่ทำให้สัดส่วน cardinality ของข้อความมีความหมาย
_MIN_CARDINALITY_ROWS = 20
# เกณฑ์ cardinality สูงของข้อความ (อาจเป็น ID มากกว่าหมวดหมู่)
_HIGH_CARDINALITY = 0.9
# เกณฑ์สหสัมพันธ์การขาดหาย (co-missing) ที่ถือว่าน่าสงสัยว่าเป็น MNAR
_COMISSING_CORR = 0.9
# เกณฑ์สหสัมพันธ์ Pearson ที่ถือว่า "แรง" สำหรับ target
_STRONG_CORR = 0.5

# จำนวนค่าขั้นต่ำที่ทำให้สถิติการกระจาย (skew/kurtosis/bimodal) มีความหมาย
_MIN_DISTRIBUTION_SAMPLE = 20
# เกณฑ์ความเบ้ (|skewness|) ที่ถือว่า "เบ้มาก" — ควรพิจารณา log/transform
_SKEW_THRESHOLD = 1.0
# เกณฑ์ความโด่งส่วนเกิน (excess kurtosis) ที่ถือว่า "หางหนัก" — มี outlier มาก
_KURTOSIS_THRESHOLD = 7.0
# เกณฑ์สหสัมพันธ์ |r| ระหว่างคู่คอลัมน์ตัวเลขที่ถือว่า "สูง" (อาจซ้ำซ้อนกัน)
_HIGH_PAIR_CORR = 0.7
# จำนวนคอลัมน์ตัวเลขสูงสุดที่นำมาตรวจคู่สหสัมพันธ์ (กัน O(n^2) บนตารางกว้าง)
_MAX_CORR_COLS = 30
# จำนวนคู่สหสัมพันธ์สูงสุดที่รายงาน (เรียงจากแรงสุด)
_MAX_CORR_PAIRS = 8
# สัดส่วนค่าที่เป็นตัวเลขในคอลัมน์หมวดหมู่ ที่ถือว่า "เก็บตัวเลขเป็นข้อความ"
_TYPE_MISMATCH_RATIO = 0.8
# สัดส่วนแถวซ้ำที่ถือว่ารุนแรง (ยกระดับเป็น warning)
_DUP_WARN_RATIO = 0.01

# ---- เกณฑ์สำหรับ insight เฉพาะข้อความ (text-heavy datasets — IN-1) --------------
# จำนวนเซลล์/โทเคนขั้นต่ำที่ทำให้สถิติข้อความมีความหมาย
_MIN_TEXT_LENGTH_SAMPLE = 20
_MIN_VOCAB_TOKENS = 50
# มัธยฐานความยาวอักขระ: ต่ำกว่านี้ = ข้อความสั้น (social), สูงกว่านี้ = รีวิว/บทความยาว
_SHORT_TEXT_MEDIAN = 10
_LONG_TEXT_MEDIAN = 200
# ความยาวแปรปรวนสูง: เฉลี่ย > _LENGTH_SKEW_RATIO × มัธยฐาน และสูงสุด > _LENGTH_OUTLIER_RATIO × มัธยฐาน
_LENGTH_SKEW_RATIO = 2.0
_LENGTH_OUTLIER_RATIO = 8.0
# ความหลากหลายคำศัพท์ (unique/total tokens): ต่ำ = คำซ้ำเยอะ, สูง = หลากหลาย
_LOW_VOCAB_RICHNESS = 0.1
_HIGH_VOCAB_RICHNESS = 0.5
# NER: จำนวน entity ขั้นต่ำที่จะรายงานเป็น insight
_MIN_NER_ENTITIES = 1
# sentiment/rating: จำนวนค่าขั้นต่ำ, จำนวนระดับสูงสุด (0-10 → 11 ระดับ), เกณฑ์การกระจาย
_MIN_SENTIMENT_SAMPLE = 20
_MAX_RATING_LEVELS = 11
_SENTIMENT_DOMINANT = 0.5  # สัดส่วนที่ถือว่า "ส่วนใหญ่" กระจุกที่ขั้วเดียว
_SENTIMENT_BIMODAL_TAIL = 0.25  # สัดส่วนของแต่ละขั้ว (ต่ำสุด/สูงสุด) ที่ถือว่าเด่นใน bimodal
_SENTIMENT_BIMODAL_TOTAL = 0.6  # สัดส่วนรวมสองขั้วที่ถือว่าแบ่งเป็น 2 กลุ่ม

# ป้ายภาษาไทยของประเภท named entity (รองรับทั้ง thainer และ thainer-v2)
_ENTITY_TYPE_TH: dict[str, str] = {
    "PERSON": "ชื่อบุคคล",
    "PER": "ชื่อบุคคล",
    "LOCATION": "สถานที่",
    "LOC": "สถานที่",
    "ORGANIZATION": "องค์กร",
    "ORG": "องค์กร",
    "DATE": "วันที่",
    "TIME": "เวลา",
    "MONEY": "จำนวนเงิน",
}

# ชื่อคอลัมน์ที่บ่งชี้ว่าเป็นคะแนน/เรตติ้ง (ordinal) — ใช้คู่กับการตรวจค่าจำนวนเต็มช่วงแคบ
_RATING_NAME_RE = re.compile(
    r"(rating|score|star|sentiment|satisfaction|nps|csat|polarity|"
    r"คะแนน|ดาว|ความพึงพอใจ|เรตติ้ง)",
    re.IGNORECASE,
)
_DATE_COMPONENT_NAMES = {
    "year",
    "month",
    "month_number",
    "week",
    "week_number",
    "day",
    "day_of_week",
    "day_of_month",
    "quarter",
}
_DATE_DIMENSION_HINTS = {"date", "full_date", "calendar_date", "date_key", "day_date"}


def _is_date_component_name(name: str) -> bool:
    """ชื่อคอลัมน์ที่เป็นส่วนประกอบของวันที่ใน date dimension."""
    return str(name).strip().lower() in _DATE_COMPONENT_NAMES


def _has_date_dimension_context(df: pd.DataFrame) -> bool:
    """มี date column จริงหรือชื่อ table-like ที่ทำให้ year/month/week เป็น tautology."""
    names = {str(c).strip().lower() for c in df.columns}
    if names & _DATE_DIMENSION_HINTS:
        return True
    return any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)


def _column_from_description(text: str) -> str:
    """ดึงชื่อคอลัมน์จากข้อความ Insight รูปแบบ "คอลัมน์ 'col': ..."."""
    marker = "คอลัมน์ '"
    start = text.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = text.find("'", start)
    return text[start:end] if end >= 0 else ""


def _dedupe_insights(insights: list[Insight]) -> list[Insight]:
    """ตัด insight ซ้ำจาก quality/anomaly ตามหัวข้อปัญหา + คอลัมน์ โดยเก็บรายการแรกไว้."""
    out: list[Insight] = []
    seen: set[tuple[str, str]] = set()
    for insight in insights:
        column = _column_from_description(insight.description_th)
        key = (insight.title_th, column)
        if column and key in seen:
            continue
        if column:
            seen.add(key)
        out.append(insight)
    return out


def _dedupe_quality_anomaly(
    quality_issues: list[QualityIssue], anomaly_issues: list[AnomalyIssue]
) -> tuple[list[QualityIssue], list[AnomalyIssue]]:
    """ตัด issue ซ้ำตาม (check_name, column) โดยให้ quality มาก่อน anomaly."""
    seen: set[tuple[str, str]] = set()
    q_out: list[QualityIssue] = []
    for issue in quality_issues:
        key = (issue.check_name, issue.column)
        if key in seen:
            continue
        seen.add(key)
        q_out.append(issue)

    a_out: list[AnomalyIssue] = []
    for issue in anomaly_issues:
        key = (issue.check_name, issue.column)
        if key in seen:
            continue
        seen.add(key)
        a_out.append(issue)
    return q_out, a_out


def _cap_insights(insights: list[Insight], max_insights: int) -> list[Insight]:
    """ตัดรายการ insight ให้เหลือไม่เกิน max_insights — กัน insight overflow (P1).

    หลักการ (เรียงตามความสำคัญ): เก็บ critical "ทั้งหมด" (ห้ามตัด) → เติม warning จนเต็มโควตา
    → เติม info ที่เหลือ (info ตัดได้เต็มที่). คาดว่า ``insights`` ถูก sort ตามความรุนแรงมาแล้ว
    (critical → warning → info) ทำให้ลำดับภายในแต่ละระดับยังคงเดิม

    ถ้า critical มีมากกว่า max_insights จะคืน critical ทั้งหมด (ยอมให้เกินโควตา —
    correctness สำคัญกว่า completeness: ห้ามซ่อนเรื่องวิกฤต)
    """
    if max_insights <= 0 or len(insights) <= max_insights:
        return insights
    critical = [i for i in insights if i.severity == "critical"]
    warning = [i for i in insights if i.severity == "warning"]
    info = [i for i in insights if i.severity == "info"]

    kept = list(critical)  # critical เก็บทั้งหมดเสมอ
    budget = max_insights - len(kept)
    if budget > 0:
        kept.extend(warning[:budget])
        budget -= min(len(warning), budget)
    if budget > 0:
        kept.extend(info[:budget])
    return kept


# ----------------------------------------------------------------------------
# โครงสร้างผลลัพธ์
# ----------------------------------------------------------------------------
@dataclass
class Insight:
    """ข้อค้นพบเชิงลึกหนึ่งรายการ — ตีความแล้ว พร้อมคำแนะนำ (ภาษาไทย)."""

    category: str  # "quality" | "anomaly" | "text" | "structure" | "target"
    severity: str  # "critical" | "warning" | "info"
    title_th: str
    description_th: str
    recommendation_th: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity,
            "title_th": self.title_th,
            "description_th": self.description_th,
            "recommendation_th": self.recommendation_th,
        }


@dataclass
class InsightSummary:
    """สรุปข้อค้นพบทั้งหมด พร้อมบทสรุปผู้บริหาร (executive summary) ภาษาไทย."""

    total_insights: int
    critical_count: int
    warning_count: int
    info_count: int
    insights: list[Insight] = field(default_factory=list)
    executive_summary_th: str = ""
    # จำนวนข้อค้นพบที่สร้างได้ทั้งหมด "ก่อน" ตัดให้เหลือ max_insights (P1)
    # เท่ากับ total_insights เมื่อไม่มีการตัด; มากกว่าเมื่อถูกตัด — ใช้บอกผู้อ่านว่ามีทั้งหมดกี่ข้อ
    total_generated: int = 0

    def to_dict(self) -> dict:
        return {
            "total_insights": self.total_insights,
            "total_generated": self.total_generated,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "executive_summary_th": self.executive_summary_th,
            "insights": [i.to_dict() for i in self.insights],
        }


# ----------------------------------------------------------------------------
# ตัวแปลงจากผลตรวจแต่ละชนิด -> Insight
# ----------------------------------------------------------------------------
def _insight_from_quality(issue: QualityIssue) -> Insight:
    """แปลง QualityIssue หนึ่งรายการเป็น Insight (หมวด quality)."""
    title = _QUALITY_TITLES.get(issue.check_name, issue.check_name)
    desc = (
        f"คอลัมน์ '{issue.column}': {issue.description_th} "
        f"(พบ {issue.count:,} แถว, {issue.percentage:.1f}%)"
    )
    rec = issue.suggestion_th or f"ตรวจสอบคอลัมน์ '{issue.column}' และแก้ไขก่อนนำไปวิเคราะห์"
    return Insight("quality", issue.severity, title, desc, rec)


def _insight_from_anomaly(issue: AnomalyIssue) -> Insight:
    """แปลง AnomalyIssue หนึ่งรายการเป็น Insight (หมวดตาม anomaly_type)."""
    category = _ANOMALY_TYPE_CATEGORY.get(issue.anomaly_type, "anomaly")
    title = _ANOMALY_TITLES.get(issue.check_name, issue.check_name)
    desc = (
        f"คอลัมน์ '{issue.column}': {issue.description_th} "
        f"({issue.count:,} รายการ, {issue.percentage:.1f}%)"
    )
    rec = issue.suggestion_th or f"ตรวจสอบคอลัมน์ '{issue.column}'"
    return Insight(category, issue.severity, title, desc, rec)


def _text_columns(
    column_types: dict[str, ColumnType] | None, text_metrics: dict[str, TextMetrics]
) -> list[str]:
    """รวมรายชื่อคอลัมน์ข้อความจาก column_types และ text_metrics (กันกรณีอย่างใดอย่างหนึ่งว่าง)."""
    cols: list[str] = []
    if column_types:
        cols.extend(c for c, t in column_types.items() if t in _TEXT_TYPES)
    for c in text_metrics:
        if c not in cols:
            cols.append(c)
    return cols


def _high_cardinality_text_insights(
    df: pd.DataFrame,
    column_types: dict[str, ColumnType] | None,
    text_metrics: dict[str, TextMetrics],
) -> list[Insight]:
    """คอลัมน์ข้อความที่มีค่าไม่ซ้ำสูงมาก — น่าจะเป็น ID/ข้อความอิสระ ไม่ใช่หมวดหมู่."""
    out: list[Insight] = []
    for col in _text_columns(column_types, text_metrics):
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        n = len(non_null)
        if n < _MIN_CARDINALITY_ROWS:
            continue
        unique = int(non_null.astype(str).nunique())
        ratio = unique / n
        if ratio >= _HIGH_CARDINALITY:
            out.append(
                Insight(
                    "text",
                    "info",
                    "คอลัมน์ข้อความมีค่าไม่ซ้ำสูง",
                    f"คอลัมน์ '{col}' มีค่าไม่ซ้ำ {ratio * 100:.0f}% "
                    f"({unique:,} จาก {n:,} แถว) — อาจเป็น ID หรือข้อความอิสระ ไม่ใช่ตัวแปรหมวดหมู่",
                    f"ตรวจสอบว่าคอลัมน์ '{col}' ควรใช้เป็น ID/ข้อความอิสระ "
                    "และไม่นำไปจัดกลุ่ม (groupby) เป็นหมวดหมู่",
                )
            )
    return out


# ----------------------------------------------------------------------------
# insight เฉพาะข้อความ — ทำให้ text-heavy datasets มีข้อค้นพบมากกว่าแค่ quality/anomaly (IN-1)
# ----------------------------------------------------------------------------
def _text_length_distribution_insights(
    text_metrics: dict[str, TextMetrics],
) -> list[Insight]:
    """ตีความการกระจายความยาวข้อความต่อคอลัมน์ — สั้น/ยาว/แปรปรวนสูง (จาก text_metrics)."""
    out: list[Insight] = []
    for col, m in text_metrics.items():
        if m.non_null_cells < _MIN_TEXT_LENGTH_SAMPLE:
            continue
        median = float(m.median_char_length)
        if median <= 0:
            continue
        if median < _SHORT_TEXT_MEDIAN:
            out.append(
                Insight(
                    "text",
                    "info",
                    "ข้อมูลเป็นข้อความสั้น ๆ",
                    f"คอลัมน์ '{col}' มีความยาวมัธยฐานเพียง {median:.0f} อักขระ — "
                    "เป็นข้อความสั้น (เช่น โพสต์โซเชียล/แท็ก/ข้อความแชต)",
                    f"ใช้เทคนิคที่เหมาะกับข้อความสั้นกับ '{col}' (คีย์เวิร์ด/อิโมจิ/แฮชแท็ก) "
                    "แทนการวิเคราะห์เชิงบทความยาว",
                )
            )
        elif median > _LONG_TEXT_MEDIAN:
            out.append(
                Insight(
                    "text",
                    "info",
                    "ข้อมูลเป็นข้อความยาว (รีวิว/บทความ)",
                    f"คอลัมน์ '{col}' มีความยาวมัธยฐาน {median:.0f} อักขระ — เป็นรีวิว/บทความยาว",
                    f"พิจารณาสรุปความ/แบ่งประโยค หรือใช้โมเดลที่รองรับข้อความยาวกับ '{col}'",
                )
            )
        # ความยาวแปรปรวนสูง: เฉลี่ยเบ้ขวาจากมัธยฐาน และมีค่าสูงสุดยาวผิดปกติ (long tail)
        avg = float(m.avg_char_length)
        if (
            avg > _LENGTH_SKEW_RATIO * median
            and float(m.max_char_length) > _LENGTH_OUTLIER_RATIO * median
        ):
            out.append(
                Insight(
                    "text",
                    "info",
                    "ความยาวข้อความแปรปรวนสูง",
                    f"คอลัมน์ '{col}' มีความยาวข้อความแตกต่างกันมาก "
                    f"(มัธยฐาน {median:.0f}, เฉลี่ย {avg:.0f}, สูงสุด {m.max_char_length:,} อักขระ) — "
                    "อาจมีข้อความสั้นและยาวผิดปกติปนกัน",
                    f"ตรวจสอบข้อความที่ยาว/สั้นผิดปกติใน '{col}' "
                    "และพิจารณาแยกกลุ่มหรือตัดข้อความที่ผิดปกติก่อนวิเคราะห์",
                )
            )
    return out


def _vocabulary_richness_insights(
    text_metrics: dict[str, TextMetrics],
) -> list[Insight]:
    """ความหลากหลายของคำศัพท์ (unique/total tokens) — ต่ำ = คำซ้ำเยอะ, สูง = หลากหลาย."""
    out: list[Insight] = []
    for col, m in text_metrics.items():
        if m.total_tokens < _MIN_VOCAB_TOKENS:
            continue
        ratio = m.unique_tokens / m.total_tokens
        if ratio < _LOW_VOCAB_RICHNESS:
            out.append(
                Insight(
                    "text",
                    "info",
                    "คำศัพท์หลากหลายต่ำ (คำซ้ำเยอะ)",
                    f"คอลัมน์ '{col}' มีคำไม่ซ้ำเพียง {ratio * 100:.0f}% ของคำทั้งหมด "
                    f"({m.unique_tokens:,} จาก {m.total_tokens:,} คำ) — มีคำซ้ำเยอะ",
                    f"ข้อความใน '{col}' อาจเป็นเทมเพลต/ข้อความซ้ำ ๆ — "
                    "ตรวจสอบว่ามีคุณค่าต่อการวิเคราะห์เนื้อหาเพียงพอหรือไม่",
                )
            )
        elif ratio > _HIGH_VOCAB_RICHNESS:
            out.append(
                Insight(
                    "text",
                    "info",
                    "คำศัพท์หลากหลายสูง",
                    f"คอลัมน์ '{col}' มีคำไม่ซ้ำถึง {ratio * 100:.0f}% ของคำทั้งหมด "
                    f"({m.unique_tokens:,} จาก {m.total_tokens:,} คำ) — คำศัพท์หลากหลายสูง",
                    f"ข้อความใน '{col}' มีความหลากหลายของเนื้อหา "
                    "เหมาะกับการวิเคราะห์หัวข้อ (topic modeling) หรือสกัดคีย์เวิร์ด",
                )
            )
    return out


def _ner_summary_insights(ner_results: dict[str, NERResult] | None) -> list[Insight]:
    """สรุปชื่อเฉพาะ (named entities) ที่สกัดได้ — บ่งชี้ความหลากหลาย/ลักษณะเนื้อหา."""
    out: list[Insight] = []
    if not ner_results:
        return out
    for col, r in ner_results.items():
        if r.total_entities < _MIN_NER_ENTITIES or not r.entity_counts:
            continue
        # ประเภท entity ที่พบมากสุด (เรียงตามจำนวน แล้วตามชื่อ — ให้ผลคงที่เมื่อจำนวนเสมอกัน)
        top_type, top_count = max(r.entity_counts.items(), key=lambda kv: (kv[1], kv[0]))
        top_type_th = _ENTITY_TYPE_TH.get(top_type.upper(), top_type)
        out.append(
            Insight(
                "text",
                "info",
                "พบชื่อเฉพาะ (named entities) ในข้อความ",
                f"คอลัมน์ '{col}' มีชื่อเฉพาะ {r.total_entities:,} ตัว "
                f"(เด่นสุดคือประเภท{top_type_th} {top_count:,} ตัว) — "
                "บ่งชี้ว่าข้อความมีความหลากหลายของเนื้อหา",
                f"พิจารณาใช้ชื่อเฉพาะจาก '{col}' เป็นมิติเพิ่มเติม "
                "(เช่น จัดกลุ่มตามบุคคล/สถานที่/องค์กร) ในการวิเคราะห์",
            )
        )
        if top_type.upper() in ("PERSON", "PER"):
            out.append(
                Insight(
                    "text",
                    "info",
                    "ข้อความมีชื่อบุคคลเป็น entities หลัก",
                    f"คอลัมน์ '{col}' มีชื่อบุคคลเป็นชื่อเฉพาะหลัก ({top_count:,} ตัว)",
                    f"ระวังข้อมูลส่วนบุคคล (PII) ใน '{col}' — พิจารณา anonymize ก่อนเผยแพร่/วิเคราะห์",
                )
            )
        elif top_type.upper() in ("LOCATION", "LOC"):
            out.append(
                Insight(
                    "text",
                    "info",
                    "ข้อความเกี่ยวข้องกับสถานที่",
                    f"คอลัมน์ '{col}' มีชื่อสถานที่เป็นชื่อเฉพาะหลัก ({top_count:,} ตัว)",
                    f"พิจารณาวิเคราะห์เชิงพื้นที่ (geo) จากสถานที่ที่พบใน '{col}'",
                )
            )
    return out


def _looks_like_rating(series: pd.Series) -> bool:
    """ค่าดูเหมือนคะแนน/เรตติ้ง: จำนวนเต็มในช่วงแคบ (เช่น 1-5, 0-10) มีหลายระดับแต่ไม่มากเกินไป."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < _MIN_SENTIMENT_SAMPLE:
        return False
    n_levels = int(numeric.nunique())
    if n_levels < 2 or n_levels > _MAX_RATING_LEVELS:
        return False
    arr = numeric.to_numpy(dtype="float64")
    if not np.allclose(arr, np.round(arr)):  # ต้องเป็นจำนวนเต็มทั้งหมด
        return False
    return float(arr.min()) >= 0 and float(arr.max()) <= 10


def _sentiment_distribution_insights(
    df: pd.DataFrame, column_types: dict[str, ColumnType] | None
) -> list[Insight]:
    """การกระจายของคอลัมน์คะแนน/เรตติ้ง — เชิงบวก/ลบ/แบ่งเป็น 2 กลุ่ม (sentiment)."""
    out: list[Insight] = []
    date_dimension = _has_date_dimension_context(df)
    for col in df.columns:
        name = str(col)
        if not _RATING_NAME_RE.search(name.lower()):
            continue
        if date_dimension and _is_date_component_name(name):
            continue
        if column_types is not None:
            ctype = column_types.get(name)
            if ctype is not None and ctype not in (ColumnType.NUMERIC, ColumnType.CATEGORICAL):
                continue
        if not _looks_like_rating(df[col]):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        shares = {float(k): float(v) for k, v in numeric.value_counts(normalize=True).items()}
        lo, hi = min(shares), max(shares)
        top_value = max(shares, key=lambda k: shares[k])
        top_share = shares[top_value]
        lo_share, hi_share = shares.get(lo, 0.0), shares.get(hi, 0.0)

        if top_value == hi and top_share >= _SENTIMENT_DOMINANT:
            out.append(
                Insight(
                    "text",
                    "info",
                    "ส่วนใหญ่เป็นรีวิวเชิงบวก",
                    f"คอลัมน์ '{col}' ส่วนใหญ่ ({top_share * 100:.0f}%) ให้คะแนนสูงสุด "
                    f"({top_value:.0f}) — เป็นความเห็นเชิงบวกเป็นหลัก",
                    f"หากใช้ '{col}' เป็น target ระวัง class imbalance "
                    "(พิจารณา resampling/weighting) และตรวจว่าคะแนนสูงสะท้อนความจริง",
                )
            )
        elif top_value == lo and top_share >= _SENTIMENT_DOMINANT:
            out.append(
                Insight(
                    "text",
                    "warning",
                    "ส่วนใหญ่เป็นรีวิวเชิงลบ",
                    f"คอลัมน์ '{col}' ส่วนใหญ่ ({top_share * 100:.0f}%) ให้คะแนนต่ำสุด "
                    f"({top_value:.0f}) — เป็นความเห็นเชิงลบเป็นหลัก อาจเป็นปัญหาที่ต้องตรวจสอบ",
                    f"ตรวจสอบสาเหตุของคะแนนต่ำใน '{col}' (คุณภาพสินค้า/บริการ) เป็นลำดับแรก",
                )
            )
        elif (
            lo_share >= _SENTIMENT_BIMODAL_TAIL
            and hi_share >= _SENTIMENT_BIMODAL_TAIL
            and (lo_share + hi_share) >= _SENTIMENT_BIMODAL_TOTAL
        ):
            out.append(
                Insight(
                    "text",
                    "info",
                    "คะแนนแบ่งเป็น 2 กลุ่มชัดเจน",
                    f"คอลัมน์ '{col}' มีคะแนนกระจุกที่สองขั้ว — ต่ำสุด ({lo:.0f}) {lo_share * 100:.0f}% "
                    f"และสูงสุด ({hi:.0f}) {hi_share * 100:.0f}% (ความเห็นแบ่งเป็นชอบมาก/ไม่ชอบมาก)",
                    f"วิเคราะห์แยกสองกลุ่มของ '{col}' เพื่อหาสาเหตุของความเห็นที่ขั้วตรงข้าม",
                )
            )
    return out


def _comissing_insights(df: pd.DataFrame) -> list[Insight]:
    """คู่คอลัมน์ที่ค่าว่าง "เกิดพร้อมกัน" (สหสัมพันธ์การขาดหายสูง) — อาจเป็น MNAR."""
    na = df.isna()
    candidates = [c for c in df.columns if 0.05 <= float(na[c].mean()) < 1.0]
    candidates = candidates[:_MAX_COMISSING_COLS]
    out: list[Insight] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            corr = float(na[a].astype(float).corr(na[b].astype(float)))
            if math.isnan(corr) or corr < _COMISSING_CORR:
                continue
            out.append(
                Insight(
                    "structure",
                    "info",
                    "ค่าว่างเกิดพร้อมกันหลายคอลัมน์",
                    f"คอลัมน์ '{a}' และ '{b}' มักมีค่าว่างพร้อมกัน "
                    f"(สหสัมพันธ์การขาดหาย ≈ {corr:.2f}) — อาจเป็น Missing Not At Random (MNAR)",
                    f"ตรวจสอบกระบวนการเก็บข้อมูลของ '{a}' และ '{b}' ว่าเหตุใดจึงขาดพร้อมกัน",
                )
            )
    return out


def _cleaning_insight(cleaning_results: list[CleaningResult]) -> Insight | None:
    """สรุปสิ่งที่ถูกทำความสะอาด — รวมจำนวนเซลล์และการดำเนินการที่มีผลมากสุด."""
    effective = [r for r in cleaning_results if r.rows_affected > 0]
    if not effective:
        return None
    by_op: dict[str, int] = {}
    for r in effective:
        by_op[r.operation] = by_op.get(r.operation, 0) + r.rows_affected
    total = sum(by_op.values())
    top = sorted(by_op.items(), key=lambda x: -x[1])
    parts = ", ".join(f"{_CLEAN_OP_TH.get(op, op)} {cnt:,} เซลล์" for op, cnt in top[:4])
    return Insight(
        "quality",
        "info",
        "สรุปการทำความสะอาดข้อมูล",
        f"ทำความสะอาดแล้วรวม {total:,} เซลล์ — {parts}",
        "ตรวจสอบรายละเอียดก่อน/หลังในส่วน 'การทำความสะอาด' ของรายงาน",
    )


def _target_insights(target_associations: list[TargetAssociation]) -> list[Insight]:
    """ความสัมพันธ์กับ target ที่แรงหรือมีนัยสำคัญ — ใช้เป็นเบาะแสฟีเจอร์สำคัญ."""
    out: list[Insight] = []
    for a in target_associations:
        p_ok = not math.isnan(a.p_value) and a.p_value < 0.05
        if a.association_type == "correlation":
            if abs(a.score) < _STRONG_CORR and not p_ok:
                continue
            out.append(
                Insight(
                    "target",
                    "info",
                    "พบความสัมพันธ์ชัดเจนกับตัวแปรเป้าหมาย",
                    f"คอลัมน์ '{a.column}' มีสหสัมพันธ์ {a.score:.2f} กับเป้าหมาย '{a.target}'",
                    f"พิจารณาใช้ '{a.column}' เป็นฟีเจอร์สำคัญในการสร้างโมเดล/วิเคราะห์",
                )
            )
        elif p_ok:
            out.append(
                Insight(
                    "target",
                    "info",
                    "พบความสัมพันธ์มีนัยสำคัญกับตัวแปรเป้าหมาย",
                    f"คอลัมน์ '{a.column}' สัมพันธ์กับเป้าหมาย '{a.target}' "
                    f"อย่างมีนัยสำคัญ (p={a.p_value:.4f})",
                    f"พิจารณาใช้ '{a.column}' เป็นฟีเจอร์สำคัญในการสร้างโมเดล/วิเคราะห์",
                )
            )
    return out


# ----------------------------------------------------------------------------
# insight การกระจายของคอลัมน์ตัวเลข (skewness/kurtosis/bimodal)
# ----------------------------------------------------------------------------
def _is_bimodal(values: np.ndarray) -> bool:
    """ตรวจแบบ heuristic ว่าการแจกแจงมี 2 จุดยอด (bimodal) หรือไม่.

    วิธี: ทำฮิสโทแกรม -> ปรับเรียบเล็กน้อย -> นับจุดยอดเฉพาะที่ (local maxima) ที่เด่นพอ
    แล้วตรวจว่าระหว่างจุดยอดสองอันที่สูงสุดมี "หุบเขา" ที่ลึกพอหรือไม่ (กัน false positive)
    """
    if values.size < 2 * _MIN_DISTRIBUTION_SAMPLE:
        return False
    hist, _ = np.histogram(values, bins=20)
    if hist.max() == 0:
        return False
    # ปรับเรียบด้วยหน้าต่างกว้าง 3 เพื่อลด noise
    smooth = np.convolve(hist.astype("float64"), np.ones(3) / 3.0, mode="same")
    peak = float(smooth.max())
    # จุดยอดเฉพาะที่ที่สูงอย่างน้อย 25% ของจุดยอดหลัก
    peaks = [
        i
        for i in range(1, len(smooth) - 1)
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] >= 0.25 * peak
    ]
    if len(peaks) < 2:
        return False
    # เอาสองจุดยอดที่สูงสุด แล้วดูหุบเขาระหว่างกลาง
    peaks.sort(key=lambda i: -smooth[i])
    a, b = sorted(peaks[:2])
    valley = float(smooth[a : b + 1].min())
    lower_peak = min(smooth[a], smooth[b])
    return valley < 0.6 * lower_peak


def _distribution_insights(
    df: pd.DataFrame, column_types: dict[str, ColumnType] | None
) -> list[Insight]:
    """วิเคราะห์การกระจายของคอลัมน์ตัวเลข — ความเบ้ (skew), หางหนัก (kurtosis), 2 กลุ่ม (bimodal)."""
    if not column_types:
        return []
    out: list[Insight] = []
    date_dimension = _has_date_dimension_context(df)
    for col, ctype in column_types.items():
        if ctype != ColumnType.NUMERIC or col not in df.columns:
            continue
        if date_dimension and _is_date_component_name(col):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(numeric) < _MIN_DISTRIBUTION_SAMPLE or numeric.nunique() <= 1:
            continue

        skew = float(numeric.skew())
        if not math.isnan(skew) and abs(skew) > _SKEW_THRESHOLD:
            direction = "ขวา (หางยาวด้านมาก)" if skew > 0 else "ซ้าย (หางยาวด้านน้อย)"
            out.append(
                Insight(
                    "distribution",
                    "warning",
                    "การกระจายเบ้มาก",
                    f"คอลัมน์ '{col}' มีการกระจายเบ้{direction} (skew={skew:.2f}) — "
                    "ค่าเฉลี่ยอาจไม่สะท้อนค่ากลางที่แท้จริง",
                    f"พิจารณาแปลง '{col}' ด้วย log/sqrt/Box-Cox ก่อนวิเคราะห์หรือสร้างโมเดล",
                )
            )

        kurt = float(numeric.kurt())
        if not math.isnan(kurt) and kurt > _KURTOSIS_THRESHOLD:
            out.append(
                Insight(
                    "distribution",
                    "info",
                    "การกระจายหางหนัก (heavy tail)",
                    f"คอลัมน์ '{col}' มีหางหนัก (kurtosis={kurt:.2f}) — มีค่าสุดโต่ง (outlier) มากกว่าปกติ",
                    f"ตรวจสอบค่าสุดโต่งของ '{col}' และพิจารณาวิธีที่ทนต่อ outlier (median/robust stats)",
                )
            )

        if _is_bimodal(numeric.to_numpy(dtype="float64")):
            out.append(
                Insight(
                    "distribution",
                    "info",
                    "อาจมี 2 กลุ่มข้อมูล (bimodal)",
                    f"คอลัมน์ '{col}' มีลักษณะการแจกแจงแบบ 2 จุดยอด — อาจมีกลุ่มย่อย 2 กลุ่มปนกัน",
                    f"พิจารณาแยกวิเคราะห์ '{col}' ตามกลุ่ม หรือหาตัวแปรที่อธิบายการแบ่งกลุ่ม",
                )
            )
    return out


# ----------------------------------------------------------------------------
# insight สหสัมพันธ์ระหว่างคู่คอลัมน์ตัวเลข (อาจซ้ำซ้อนกัน)
# ----------------------------------------------------------------------------
def _correlation_insights(df: pd.DataFrame) -> list[Insight]:
    """คู่คอลัมน์ตัวเลขที่สหสัมพันธ์สูง (|r| > 0.7) — อาจซ้ำซ้อน (multicollinearity)."""
    numeric = df.select_dtypes(include="number")
    if _has_date_dimension_context(df):
        numeric = numeric.drop(
            columns=[c for c in numeric.columns if _is_date_component_name(str(c))],
            errors="ignore",
        )
    if numeric.shape[1] < 2:
        return []
    if numeric.shape[1] > _MAX_CORR_COLS:
        numeric = numeric.iloc[:, :_MAX_CORR_COLS]

    corr = numeric.corr(numeric_only=True)
    cols = list(corr.columns)
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iat[i, j])
            if not math.isnan(r) and abs(r) > _HIGH_PAIR_CORR:
                pairs.append((str(cols[i]), str(cols[j]), r))

    # เรียงจากแรงสุด แล้วจำกัดจำนวนเพื่อไม่ให้รายงานรก
    pairs.sort(key=lambda p: -abs(p[2]))
    out: list[Insight] = []
    for a, b, r in pairs[:_MAX_CORR_PAIRS]:
        kind = "เชิงบวก" if r > 0 else "เชิงลบ"
        out.append(
            Insight(
                "structure",
                "info",
                "คู่คอลัมน์สหสัมพันธ์สูง",
                f"คอลัมน์ '{a}' และ '{b}' สหสัมพันธ์{kind}สูง (r={r:.2f}) — อาจให้ข้อมูลซ้ำซ้อนกัน",
                f"ในการสร้างโมเดล พิจารณาเลือกใช้คอลัมน์ใดคอลัมน์หนึ่งระหว่าง '{a}' และ '{b}' "
                "เพื่อลด multicollinearity",
            )
        )
    return out


# ----------------------------------------------------------------------------
# insight แถวซ้ำ และคอลัมน์เก็บตัวเลขเป็นข้อความ
# ----------------------------------------------------------------------------
def _duplicate_row_insight(df: pd.DataFrame) -> Insight | None:
    """แถวซ้ำทั้งแถว — บิดเบือนสถิติ/การกระจาย ควรพิจารณาลบ."""
    if len(df) < 2:
        return None
    dup = int(df.duplicated().sum())
    if dup <= 0:
        return None
    ratio = dup / len(df)
    severity = "warning" if ratio >= _DUP_WARN_RATIO else "info"
    return Insight(
        "structure",
        severity,
        "พบแถวซ้ำ",
        f"พบแถวที่ซ้ำกันทั้งแถว {dup:,} แถว ({ratio * 100:.1f}%) — อาจทำให้สถิติและการกระจายคลาดเคลื่อน",
        "ตรวจสอบว่าเป็นการซ้ำที่ตั้งใจหรือข้อผิดพลาด แล้วพิจารณา drop_duplicates() ก่อนวิเคราะห์",
    )


def _type_mismatch_insights(
    df: pd.DataFrame, column_types: dict[str, ColumnType] | None
) -> list[Insight]:
    """คอลัมน์หมวดหมู่ที่จริง ๆ เก็บ "ตัวเลขเป็นข้อความ" — ควรแปลงเป็น numeric ก่อนวิเคราะห์."""
    if not column_types:
        return []
    out: list[Insight] = []
    for col, ctype in column_types.items():
        if ctype != ColumnType.CATEGORICAL or col not in df.columns:
            continue
        non_null = df[col].dropna().astype(str)
        if len(non_null) < _MIN_DISTRIBUTION_SAMPLE:
            continue
        numeric_ratio = float(pd.to_numeric(non_null, errors="coerce").notna().mean())
        if numeric_ratio >= _TYPE_MISMATCH_RATIO:
            out.append(
                Insight(
                    "structure",
                    "warning",
                    "คอลัมน์เก็บตัวเลขเป็นข้อความ",
                    f"คอลัมน์ '{col}' มีค่าที่เป็นตัวเลข {numeric_ratio * 100:.0f}% แต่ถูกเก็บเป็นข้อความ — "
                    "การคำนวณทางสถิติจะไม่ทำงาน",
                    f"แปลง '{col}' เป็นชนิดตัวเลขด้วย pd.to_numeric(..., errors='coerce') ก่อนวิเคราะห์",
                )
            )
    return out


# ----------------------------------------------------------------------------
# insight จากการวิเคราะห์ timeseries
# ----------------------------------------------------------------------------
def _timeseries_insights(ts_results: dict[str, TimeseriesResult]) -> list[Insight]:
    """แปลงผล timeseries analysis เป็น Insight (หมวด timeseries)."""
    out: list[Insight] = []
    for col, r in ts_results.items():
        if not r.is_timeseries:
            continue

        if r.has_trend and r.trend_direction != "stable":
            out.append(
                Insight(
                    "timeseries",
                    "info",
                    f"พบแนวโน้ม{r.trend_direction_th}ตามเวลา",
                    f"คอลัมน์ '{col}' มีแนวโน้ม{r.trend_direction_th}อย่างต่อเนื่องตามเวลา",
                    f"พิจารณาถอดแนวโน้ม (detrend) ของ '{col}' ก่อนวิเคราะห์ความสัมพันธ์/พยากรณ์",
                )
            )
        if r.has_seasonality and r.seasonal_period > 0:
            out.append(
                Insight(
                    "timeseries",
                    "info",
                    "พบรูปแบบตามฤดูกาล (seasonality)",
                    f"คอลัมน์ '{col}' มีรูปแบบซ้ำเป็นรอบ {r.seasonal_period} จุด ({r.frequency_th})",
                    f"ใช้โมเดลที่รองรับ seasonality (เช่น SARIMA/STL) ในการพยากรณ์ '{col}'",
                )
            )
        if r.gap_count > 0:
            out.append(
                Insight(
                    "timeseries",
                    "warning",
                    "ข้อมูลเวลาไม่ต่อเนื่อง (มีช่องว่าง)",
                    f"คอลัมน์ '{col}' มีช่องว่างของเวลา {r.gap_count} ช่วง — ข้อมูลบางช่วงขาดหาย",
                    f"เติมช่วงเวลาที่ขาด (resample/reindex) ของ '{col}' ก่อนวิเคราะห์อนุกรมเวลา",
                )
            )
        if r.anomalies:
            out.append(
                Insight(
                    "timeseries",
                    "warning",
                    "พบค่าผิดปกติเฉพาะช่วง (spike)",
                    f"คอลัมน์ '{col}' มีค่าผิดปกติเฉพาะช่วง {len(r.anomalies)} จุด "
                    "(spike/level shift จาก residual)",
                    f"ตรวจสอบเหตุการณ์ในช่วงเวลาดังกล่าวของ '{col}' ว่าผิดปกติจริงหรือเป็นข้อมูลพิเศษ",
                )
            )
        if not r.has_trend and not r.has_seasonality:
            out.append(
                Insight(
                    "timeseries",
                    "info",
                    "ไม่พบแนวโน้มหรือ seasonality",
                    f"คอลัมน์ '{col}' ไม่มีแนวโน้มหรือรูปแบบตามฤดูกาลชัดเจน — อาจเป็น random walk/ข้อมูลนิ่ง",
                    f"การพยากรณ์ '{col}' อาจใช้วิธีพื้นฐาน (naive/mean) เป็นฐานเปรียบเทียบ",
                )
            )
    return out


# ----------------------------------------------------------------------------
# executive summary
# ----------------------------------------------------------------------------
def _build_executive_summary(
    df: pd.DataFrame,
    insights: list[Insight],
    quality_issues: list[QualityIssue],
    anomaly_issues: list[AnomalyIssue],
    timeseries_results: dict[str, TimeseriesResult] | None = None,
    *,
    total_generated: int | None = None,
    shown: int | None = None,
) -> str:
    """สร้างบทสรุปผู้บริหาร 2-3 ประโยค ที่ระบุปัญหาเด่นและคำตัดสินสุขภาพข้อมูลโดยรวม."""
    timeseries_results = timeseries_results or {}
    rows, cols = df.shape
    parts: list[str] = [f"ชุดข้อมูลมี {rows:,} แถว × {cols} คอลัมน์"]

    crit = [i for i in insights if i.severity == "critical"]
    warn = [i for i in insights if i.severity == "warning"]

    if quality_issues:
        n_crit_q = sum(1 for i in quality_issues if i.severity == "critical")
        seg = f"พบปัญหาคุณภาพ {len(quality_issues)} ข้อ"
        if n_crit_q:
            seg += f" ({n_crit_q} วิกฤต)"
        parts.append(seg)

    if anomaly_issues:
        cols_with = len({a.column for a in anomaly_issues})
        parts.append(f"พบความผิดปกติ {len(anomaly_issues)} จุดใน {cols_with} คอลัมน์")

    # สรุปอนุกรมเวลา (timeseries) — ระบุคอลัมน์ที่มีแนวโน้ม/seasonality เด่น
    if timeseries_results:
        n_trend = sum(1 for r in timeseries_results.values() if r.has_trend)
        n_season = sum(1 for r in timeseries_results.values() if r.has_seasonality)
        seg = f"วิเคราะห์อนุกรมเวลา {len(timeseries_results)} คอลัมน์"
        extra = []
        if n_trend:
            extra.append(f"มีแนวโน้ม {n_trend}")
        if n_season:
            extra.append(f"มี seasonality {n_season}")
        if extra:
            seg += f" ({', '.join(extra)})"
        parts.append(seg)

    # ระบุหัวข้อวิกฤตเด่น ๆ ให้เป็นรูปธรรม (สูงสุด 3 ข้อ)
    if crit:
        top_titles = "; ".join(dict.fromkeys(i.title_th for i in crit[:3]))
        parts.append(f"ประเด็นที่ควรแก้ก่อน: {top_titles}")

    # ระบุว่ามีข้อค้นพบทั้งหมดกี่ข้อ แต่แสดงเพียงส่วนที่สำคัญที่สุด (เมื่อถูกตัด — P1)
    if total_generated is not None and shown is not None and total_generated > shown:
        parts.append(f"พบข้อค้นพบทั้งหมด {total_generated:,} ข้อ แสดงเฉพาะ {shown:,} ข้อที่สำคัญที่สุด")

    # คำตัดสินสุขภาพข้อมูลโดยรวม
    if crit:
        verdict = "ควรแก้ปัญหาวิกฤต (โดยเฉพาะ encoding/ศักราช) ก่อนนำไปวิเคราะห์"
    elif warn:
        verdict = "ข้อมูลใช้งานได้ แต่ควรตรวจสอบจุดที่เตือนก่อนวิเคราะห์เชิงลึก"
    elif quality_issues or anomaly_issues:
        verdict = "ข้อมูลมีคุณภาพดีโดยรวม มีเพียงข้อสังเกตเล็กน้อย"
    else:
        verdict = "ไม่พบปัญหาคุณภาพหรือความผิดปกติที่สำคัญ ข้อมูลพร้อมนำไปวิเคราะห์"
    parts.append(verdict)

    return " ".join(parts)


# ----------------------------------------------------------------------------
# ฟังก์ชันหลัก
# ----------------------------------------------------------------------------
def generate_insights(
    df: pd.DataFrame,
    quality_issues: list[QualityIssue],
    anomaly_issues: list[AnomalyIssue],
    text_metrics: dict[str, TextMetrics],
    target_associations: list[TargetAssociation] | None = None,
    cleaning_results: list[CleaningResult] | None = None,
    column_types: dict[str, ColumnType] | None = None,
    timeseries_results: dict[str, TimeseriesResult] | None = None,
    ner_results: dict[str, NERResult] | None = None,
    extra_insights: list[Insight] | None = None,
    max_insights: int = _MAX_INSIGHTS,
) -> InsightSummary:
    """สร้างสรุปข้อมูลเชิงลึกอัตโนมัติเป็นภาษาไทย.

    รวบรวมผลจากทุกส่วน (คุณภาพ/ความผิดปกติ/ข้อความ/การกระจาย/โครงสร้าง/เป้าหมาย/อนุกรมเวลา/
    การทำความสะอาด) แล้วตีความเป็นข้อค้นพบที่บอก "อะไรสำคัญ ควรทำอะไรต่อ" จัดเรียงตามความรุนแรง
    พร้อมบทสรุปผู้บริหารสั้น ๆ

    Args:
        df: ข้อมูลที่วิเคราะห์.
        quality_issues: ผลจาก run_quality_checks.
        anomaly_issues: ผลจาก detect_anomalies.
        text_metrics: สถิติข้อความต่อคอลัมน์ (จาก text_metrics).
        target_associations: ผลวิเคราะห์ target (ถ้ามี).
        cleaning_results: ผลการทำความสะอาด (ถ้ามี).
        column_types: ประเภทคอลัมน์ (จาก detect_all) — ใช้ตรวจ cardinality/การกระจาย/ชนิดข้อมูล.
        timeseries_results: ผลวิเคราะห์ timeseries ต่อคอลัมน์ (ถ้ามี).
        ner_results: ผลสกัดชื่อเฉพาะ (NER) ต่อคอลัมน์ (ถ้ามี) — ใช้สร้าง insight เฉพาะข้อความ.
        extra_insights: Insight เพิ่มเติมจากภายนอก (เช่น cross-column insight engine v0.6)
            ที่ถูกแปลงเป็น Insight แล้ว — จะถูกรวมและจัดเรียงร่วมกับข้อค้นพบอื่น.
        max_insights: จำนวน Insight สูงสุดที่คืน (กัน insight overflow — P1). ตัดโดยเก็บ
            critical ทั้งหมด → เติม warning → เติม info. ใช้ค่า <= 0 เพื่อปิดการตัด.

    Returns:
        InsightSummary พร้อมรายการ Insight (เรียงวิกฤตก่อน) และบทสรุปผู้บริหารภาษาไทย.
        ``total_generated`` บอกจำนวนข้อค้นพบทั้งหมดก่อนถูกตัด (>= total_insights).
    """
    target_associations = target_associations or []
    cleaning_results = cleaning_results or []
    timeseries_results = timeseries_results or {}
    if _has_date_dimension_context(df):
        quality_issues = [i for i in quality_issues if not _is_date_component_name(i.column)]
        anomaly_issues = [i for i in anomaly_issues if not _is_date_component_name(i.column)]

    quality_issues, anomaly_issues = _dedupe_quality_anomaly(quality_issues, anomaly_issues)

    insights: list[Insight] = []
    insights.extend(_insight_from_quality(i) for i in quality_issues)
    insights.extend(_insight_from_anomaly(a) for a in anomaly_issues)
    insights.extend(_high_cardinality_text_insights(df, column_types, text_metrics))
    # insight เฉพาะข้อความ — ทำให้ text-heavy datasets มีข้อค้นพบมากกว่าแค่ quality/anomaly (IN-1)
    insights.extend(_text_length_distribution_insights(text_metrics))
    insights.extend(_vocabulary_richness_insights(text_metrics))
    insights.extend(_ner_summary_insights(ner_results))
    insights.extend(_sentiment_distribution_insights(df, column_types))
    insights.extend(_distribution_insights(df, column_types))
    insights.extend(_correlation_insights(df))
    insights.extend(_type_mismatch_insights(df, column_types))
    insights.extend(_comissing_insights(df))
    insights.extend(_target_insights(target_associations))
    insights.extend(_timeseries_insights(timeseries_results))

    dup = _duplicate_row_insight(df)
    if dup is not None:
        insights.append(dup)

    cleaning = _cleaning_insight(cleaning_results)
    if cleaning is not None:
        insights.append(cleaning)

    # ข้อค้นพบจาก cross-column insight engine (v0.6) — แปลงเป็น Insight มาแล้วจากภายนอก
    if extra_insights:
        insights.extend(extra_insights)

    insights = _dedupe_insights(insights)

    # เรียงตามความรุนแรง (วิกฤตก่อน) — เสถียร จึงรักษาลำดับการเพิ่มภายในระดับเดียวกัน
    insights.sort(key=lambda i: _SEVERITY_ORDER.get(i.severity, 99))

    # ตัดให้เหลือไม่เกิน max_insights (P1) — เก็บ critical ทั้งหมด → warning → info
    total_generated = len(insights)
    insights = _cap_insights(insights, max_insights)

    critical_count = sum(1 for i in insights if i.severity == "critical")
    warning_count = sum(1 for i in insights if i.severity == "warning")
    info_count = sum(1 for i in insights if i.severity == "info")

    summary = _build_executive_summary(
        df,
        insights,
        quality_issues,
        anomaly_issues,
        timeseries_results,
        total_generated=total_generated,
        shown=len(insights),
    )

    return InsightSummary(
        total_insights=len(insights),
        critical_count=critical_count,
        warning_count=warning_count,
        info_count=info_count,
        insights=insights,
        executive_summary_th=summary,
        total_generated=total_generated,
    )


__all__ = [
    "Insight",
    "InsightSummary",
    "generate_insights",
]
