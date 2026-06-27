"""Report generation — ประกอบทุกส่วนเป็นรายงาน HTML/JSON/dict.

ProfileReport เป็นจุดเชื่อมหลัก: ตรวจประเภทคอลัมน์ -> ตรวจคุณภาพ -> สถิติข้อความ
-> สร้างกราฟ -> เรนเดอร์ HTML แบบ self-contained (CSS ฝัง, รูป base64)
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from thaieda import __version__
from thaieda.analysis import TargetAssociation, analyze_target
from thaieda.anomaly import AnomalyIssue, detect_anomalies
from thaieda.clean import (
    CleaningResult,
    clean_thai_text,
    handle_missing_values,
    remove_duplicate_rows,
)
from thaieda.detect import ColumnType, _detect_language, detect_all
from thaieda.i18n import TECHNICAL_TO_PLAIN, label
from thaieda.insight import Insight, InsightSummary, generate_insights
from thaieda.insight_engine import InsightEngineResult, discover_insights
from thaieda.ner import NERResult, extract_entities, ner_available
from thaieda.quality import QualityIssue, run_quality_checks
from thaieda.report._template import REPORT_TEMPLATE
from thaieda.text import TextMetrics, text_metrics
from thaieda.timeseries import (
    TimeseriesResult,
    analyze_timeseries,
    detect_timeseries_columns,
)

# จำนวนกราฟต่อคอลัมน์สูงสุดที่สร้าง (กันรายงานใหญ่เกินไปบนชุดข้อมูลที่มีคอลัมน์ข้อความเยอะ)
_MAX_CHART_COLUMNS = 20

# จำนวนกราฟ (base64) สูงสุดต่อรายงาน — กัน HTML บวมจนเบราว์เซอร์ค้าง (P2)
# เมื่อเกิน จะตัดกราฟที่สำคัญน้อยสุดก่อน: valuecounts/distribution ต่อคอลัมน์ → acf/decomposition
_MAX_CHARTS_PER_REPORT = 40
# ขนาดรวมสูงสุด (ไบต์) ของกราฟ base64 ที่ฝังใน HTML — กัน HTML > 2MB แม้กราฟไม่ถึง 40 รูป (P2)
# กราฟบางชนิด (decomposition timeseries) ใหญ่มาก จึงจำกัดด้วย "ขนาด" ควบคู่กับ "จำนวน"
# เผื่อพื้นที่ ~0.4MB ให้ CSS/ข้อความ/ตาราง เพื่อให้ไฟล์รวมยังต่ำกว่า 2MB
_MAX_CHART_BYTES = 1_600_000

# กราฟระดับชุดข้อมูล (สำคัญสูง — เก็บไว้เสมอเมื่อตัดงบกราฟ)
_DATASET_LEVEL_CHART_KEYS = (
    "correlation_heatmap",
    "scatter_matrix",
    "boxplot",
    "violinplot",
    "missing_matrix",
    "missing_heatmap",
)

# จำนวนคอลัมน์สูงสุดที่ยังแสดงการ์ดรายคอลัมน์เต็ม — เกินนี้สรุปเป็นตารางแทน (P2)
_MAX_COLUMN_CARDS = 60

# ลำดับความสำคัญในรายงาน — ให้เรื่องที่กระทบการตัดสินใจขึ้นก่อน
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# ประเภทที่ต้องวิเคราะห์ข้อความ (ต้องใช้ tokenizer)
_TEXT_METRIC_TYPES = {ColumnType.THAI_TEXT, ColumnType.MIXED_TEXT}
# ประเภทที่ถือว่าเป็นข้อความสำหรับสร้างกราฟ
_TEXT_TYPES = {ColumnType.THAI_TEXT, ColumnType.MIXED_TEXT, ColumnType.ENGLISH_TEXT}
# ประเภทคอลัมน์ที่ควรลองเสนอการทำความสะอาดข้อความ — เฉพาะคอลัมน์ที่เป็น "ข้อความ" จริง ๆ
# ไม่รวม NUMERIC/ID/DATETIME/PHONE_NUMBER: การทำความสะอาดข้อความ (เช่น ยุบเลขซ้ำ) ไม่เหมาะกับ
# คอลัมน์ตัวเลข/ตัวระบุ และเคยทำให้เกิด error บนคอลัมน์ int64 (เช่น customer_id 1111 → 111)
_CLEANABLE_TYPES = {
    ColumnType.THAI_TEXT,
    ColumnType.MIXED_TEXT,
    ColumnType.ENGLISH_TEXT,
    ColumnType.CATEGORICAL,
}

_MONEY_COLUMN_RE = re.compile(
    r"(amount|amt|price|revenue|sales|total|subtotal|net|gross|cost|paid|"
    r"เงิน|ราคา|ยอด|รายได้|ขาย|มูลค่า)",
    re.IGNORECASE,
)
_NON_AMOUNT_COLUMN_RE = re.compile(
    r"(^|_)(method|type|channel|status)($|_)|payment_?(method|type|channel|status)",
    re.IGNORECASE,
)
_TRANSACTION_COLUMN_RE = re.compile(r"(transaction|txn|order|invoice|receipt|bill)", re.IGNORECASE)
_ENTITY_ID_RE = re.compile(
    r"(customer|user|member|product|item|sku|employee|store|branch|account|entity|supplier|vendor|"
    r"ลูกค้า|สินค้า|สมาชิก|พนักงาน|สาขา).*(id|code|no|number)?$|^(id|.+_id)$",
    re.IGNORECASE,
)
_RATING_COLUMN_RE = re.compile(
    r"(rating|score|satisfaction|nps|csat|survey|feedback|review|comment|"
    r"response|question|answer|^q\d+$|คะแนน|ความพึงพอใจ|รีวิว|ความคิดเห็น|คำถาม|คำตอบ)",
    re.IGNORECASE,
)
_NON_RESPONSE_TEXT_RE = re.compile(
    r"(^|_)(name|first_name|last_name|fullname|person|passenger|customer|user|employee|"
    r"ticket|code|id|uuid|guid|email|phone|address)($|_)",
    re.IGNORECASE,
)
_UNNAMED_INDEX_RE = re.compile(r"^Unnamed:\s*\d+$", re.IGNORECASE)
_CODE_LIKE_COLUMN_RE = re.compile(
    r"(^|_)(code|status|type|method|category|class|segment|flag|level)($|_)",
    re.IGNORECASE,
)
_DATE_COMPONENT_RE = re.compile(
    r"^(year|month|month_number|week|week_number|day|day_of_week|day_of_month|quarter)$",
    re.IGNORECASE,
)
_CONDITIONAL_MISSING_RE = re.compile(
    r"(holiday|event|promotion|promo|campaign|เทศกาล|วันหยุด|เหตุการณ์)",
    re.IGNORECASE,
)


def _is_index_artifact_column(name: str) -> bool:
    """คอลัมน์ index artifact จาก CSV เช่น ``Unnamed: 0`` ที่ไม่ควรถูกวิเคราะห์."""
    return bool(_UNNAMED_INDEX_RE.match(str(name).strip()))


def _is_money_measure_name(name: str) -> bool:
    """ชื่อคอลัมน์ที่สื่อถึงยอดเงิน/ตัววัดเงินจริง ไม่ใช่ payment_method/type."""
    col = str(name).lower()
    return bool(_MONEY_COLUMN_RE.search(col)) and not _NON_AMOUNT_COLUMN_RE.search(col)


def _is_id_like_column(name: str, ctype: ColumnType | None = None) -> bool:
    """คอลัมน์ที่เป็น ID/FK ตามชื่อหรือผล detect — ไม่ควรใช้เป็น metric/measure."""
    col = str(name).strip().lower()
    return ctype == ColumnType.ID or col == "id" or col.endswith("_id")


def _is_date_component_column(name: str) -> bool:
    """คอลัมน์ส่วนประกอบของวันที่ใน date dimension (year/month/week/day)."""
    return bool(_DATE_COMPONENT_RE.match(str(name).strip().lower()))


def _is_low_cardinality_code_or_boolean(series: pd.Series, name: str) -> bool:
    """True เมื่อ numeric column ดูเป็น code/flag/boolean มากกว่าค่าที่ควรวิเคราะห์เป็น metric."""
    non_null = series.dropna()
    if non_null.empty:
        return True
    unique = int(non_null.nunique(dropna=True))
    if unique <= 1:
        return True
    vals = set(pd.to_numeric(non_null, errors="coerce").dropna().unique().tolist())
    if vals and vals.issubset({0, 1}):
        return True
    if _CODE_LIKE_COLUMN_RE.search(str(name)):
        ratio = unique / max(len(non_null), 1)
        return unique <= 20 and ratio <= 0.2
    return False


def _is_conditional_missing_column(name: str) -> bool:
    """ชื่อคอลัมน์ที่มักว่างตามเงื่อนไข เช่น holiday_name/event_name."""
    return bool(_CONDITIONAL_MISSING_RE.search(str(name)))


_DATA_TYPE_GUIDANCE: dict[str, dict[str, Any]] = {
    "transaction": {
        "label": "Transaction Data",
        "label_th": "ข้อมูลธุรกรรม",
        "summary": "ข้อมูลนี้เป็นรายการเหตุการณ์/การขายทีละรายการ จึงควรอ่านเป็นพฤติกรรมการซื้อขายและยอดเงิน",
        "focus": [
            "ดู revenue pattern ตามเวลา/สินค้า/ลูกค้า",
            "หา peak hours หรือช่วงที่ยอดขายผิดปกติถ้ามีคอลัมน์เวลา",
            "ตรวจ outlier transactions ก่อนสรุปยอดหรือทำ forecast",
        ],
    },
    "registry": {
        "label": "Registry Data",
        "label_th": "ข้อมูลทะเบียน/มาสเตอร์",
        "summary": "ข้อมูลนี้เป็นรายการ entity พร้อมคุณลักษณะ ควรเน้นความครบถ้วนและความไม่ซ้ำของระเบียน",
        "focus": [
            "ตรวจ completeness ของ attribute สำคัญ",
            "ตรวจ uniqueness ของรหัส/คีย์หลัก",
            "ดู category distribution เพื่อหา master data ที่ผิดรูปแบบหรือกระจุกตัว",
        ],
    },
    "survey": {
        "label": "Survey Data",
        "label_th": "ข้อมูลแบบสอบถาม/รีวิว",
        "summary": "ข้อมูลนี้มีคะแนนและข้อความตอบกลับ ควรอ่านร่วมกันทั้งระดับคะแนนและธีมของคำตอบ",
        "focus": [
            "ดู rating distribution และคะแนนที่ต่ำ/สูงผิดปกติ",
            "สรุป sentiment หรือธีมของข้อความตอบกลับ",
            "เชื่อมคะแนนกับกลุ่มลูกค้า/สินค้าเพื่อหา pain point",
        ],
    },
    "timeseries": {
        "label": "Timeseries Data",
        "label_th": "ข้อมูลอนุกรมเวลา",
        "summary": "ข้อมูลนี้มีแกนเวลาและตัวเลขวัดผล เหมาะกับการดูแนวโน้ม ฤดูกาล และจุดกระโดด",
        "focus": [
            "ดู trend และ seasonality ก่อน forecasting",
            "ตรวจช่วงเวลาที่ข้อมูลขาดหรือความถี่ไม่สม่ำเสมอ",
            "หา spike/drop ที่อาจเป็นเหตุการณ์สำคัญหรือ data issue",
        ],
    },
    "mixed": {
        "label": "Mixed Data",
        "label_th": "ข้อมูลผสม",
        "summary": "ข้อมูลนี้มีหลายลักษณะผสมกัน ควรแยกวิเคราะห์ตามกลุ่มคอลัมน์ก่อนสรุปภาพรวม",
        "focus": [
            "แบ่งคอลัมน์เป็น ID/วันที่/ตัวเลข/ข้อความก่อนทำ EDA",
            "เริ่มจากคุณภาพข้อมูลและความหมายของคีย์หลัก",
            "เลือกกราฟและสถิติตามชนิดคอลัมน์ ไม่ใช้วิธีเดียวกับทุกคอลัมน์",
        ],
    },
}


def _plain_language(text: str) -> str:
    """แทนศัพท์เทคนิคในข้อความด้วยคำอธิบายภาษาคนอ่าน โดยคงคำเดิมไว้ในวงเล็บ."""
    out = str(text or "")
    for technical, plain in sorted(TECHNICAL_TO_PLAIN.items(), key=lambda kv: -len(kv[0])):
        pattern = re.compile(re.escape(technical), re.IGNORECASE)
        _plain = plain  # bind loop variable for closure

        def repl(match: re.Match[str], _p: str = _plain) -> str:
            original = match.group(0)
            if _p in original:
                return original
            return f"{_p} ({original})"

        out = pattern.sub(repl, out)
    return out


def _detect_data_type(df: pd.DataFrame) -> dict[str, Any]:
    """จำแนกประเภทข้อมูลก่อน EDA เพื่อบอกคนอ่านว่าควรสำรวจแบบไหน.

    ใช้ heuristic จากชื่อคอลัมน์ + dtype + รูปแบบข้อมูล โดยตั้งใจให้ conservative:
    ถ้าคะแนนใกล้กันหลายแบบจะคืน Mixed แทนการฟันธงผิดทาง
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("_detect_data_type expects a pandas DataFrame.")

    rows, cols = df.shape
    all_col_names = [str(c) for c in df.columns]
    index_artifact_cols = [c for c in all_col_names if _is_index_artifact_column(c)]
    # ข้ามคอลัมน์ index artifact (เช่น Unnamed: 0 จาก CSV) ใน heuristic ทุกชนิด
    col_names = [c for c in all_col_names if c not in index_artifact_cols]
    language_info = _detect_language(df)
    detected_language = str(language_info.get("language", "numeric"))

    id_cols = [
        c for c in col_names if c.lower().endswith("_id") or c.lower() in {"id", "uuid", "guid"}
    ]
    entity_id_cols = [c for c in col_names if _ENTITY_ID_RE.search(c.lower())]
    transaction_cols = [c for c in col_names if _TRANSACTION_COLUMN_RE.search(c.lower())]
    amount_cols = [c for c in col_names if _is_money_measure_name(c)]
    rating_cols = [c for c in col_names if _RATING_COLUMN_RE.search(c.lower())]

    numeric_cols = [
        c
        for c in col_names
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    text_cols = [
        c
        for c in col_names
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]

    datetime_cols: list[str] = []
    for c in col_names:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(c)
            continue
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(80)
            if len(sample) >= 3:
                # format="mixed": parse แต่ละค่าแยกกัน เลี่ยง UserWarning "Could not infer format" (U1)
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if float(parsed.notna().mean()) >= 0.8:
                    datetime_cols.append(c)

    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    has_datetime = bool(datetime_cols) or has_datetime_index

    unique_ratios: dict[str, float] = {}
    for c in col_names:
        non_null = int(df[c].notna().sum())
        unique_ratios[c] = (float(df[c].nunique(dropna=True)) / non_null) if non_null else 0.0

    likely_key_cols = [c for c in id_cols if rows and unique_ratios.get(c, 0.0) >= 0.9]
    non_key_id_cols = [c for c in id_cols if c not in likely_key_cols]

    low_scale_rating_cols: list[str] = []
    for c in numeric_cols:
        name_hit = _RATING_COLUMN_RE.search(c.lower()) is not None
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        min_v = float(vals.min())
        max_v = float(vals.max())
        unique = int(vals.nunique())
        if name_hit and 0 <= min_v <= 10 and 1 <= max_v <= 10 and unique <= 11:
            low_scale_rating_cols.append(c)

    amount_numeric_cols = [c for c in amount_cols if c in numeric_cols]
    if not amount_numeric_cols:
        for c in numeric_cols:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) >= 5 and float(vals.max()) > 10 and int(vals.nunique()) > 10:
                is_amount = (
                    not _is_id_like_column(c)
                    and c not in low_scale_rating_cols
                    and not _is_low_cardinality_code_or_boolean(df[c], c)
                    and not _is_date_component_column(c)
                )
                if is_amount:
                    amount_numeric_cols.append(c)

    has_survey_name_evidence = bool(rating_cols)
    text_response_cols: list[str] = []
    for c in text_cols:
        if _NON_RESPONSE_TEXT_RE.search(c.lower()) or _is_id_like_column(c):
            continue
        name_hit = _RATING_COLUMN_RE.search(c.lower()) is not None
        median_len = df[c].dropna().astype(str).str.len().median() if len(df[c].dropna()) else 0
        if name_hit or (has_survey_name_evidence and median_len >= 20):
            text_response_cols.append(c)

    scores = {
        "transaction": 0,
        "registry": 0,
        "survey": 0,
        "timeseries": 0,
    }

    if transaction_cols:
        scores["transaction"] += 3
    if len(id_cols) >= 2:
        scores["transaction"] += 2
    elif id_cols:
        scores["transaction"] += 1
    if amount_numeric_cols:
        scores["transaction"] += 3
    if has_datetime:
        scores["transaction"] += 1
    if rows >= 100 and non_key_id_cols and amount_numeric_cols:
        scores["transaction"] += 1

    if likely_key_cols:
        scores["registry"] += 3
    if entity_id_cols:
        scores["registry"] += 2
    if cols and len(amount_numeric_cols) <= max(1, cols // 4):
        scores["registry"] += 1
    categorical_like = [
        c for c in col_names if unique_ratios.get(c, 0.0) < 0.5 and c not in numeric_cols
    ]
    if len(categorical_like) >= 2:
        scores["registry"] += 1

    if low_scale_rating_cols or rating_cols:
        scores["survey"] += 3
    if text_response_cols and (low_scale_rating_cols or rating_cols):
        scores["survey"] += 3
    if (low_scale_rating_cols or rating_cols or text_response_cols) and text_cols and numeric_cols:
        scores["survey"] += 1

    if has_datetime_index:
        scores["timeseries"] += 4
    if has_datetime and numeric_cols:
        scores["timeseries"] += 3
    if has_datetime and rows >= 20:
        scores["timeseries"] += 1
    if len(datetime_cols) >= 1 and len(numeric_cols) >= 2:
        scores["timeseries"] += 1

    # Transaction แบบ coffee-chain TRANSACTION.csv ไม่มีวันที่ แต่ชื่อคอลัมน์ชัด: transaction/order + revenue
    if transaction_cols and amount_cols and id_cols:
        scores["transaction"] += 2

    best_type, best_score = max(scores.items(), key=lambda kv: kv[1])
    strong_types = [k for k, v in scores.items() if v >= 5]
    if best_score < 4 or (
        len(strong_types) >= 2 and max(scores.values()) - sorted(scores.values())[-2] <= 1
    ):
        best_type = "mixed"

    config = _DATA_TYPE_GUIDANCE[best_type]
    focus = list(config["focus"])
    thai_recommendations: list[str] = []
    language_impact = ""
    if detected_language == "thai":
        thai_recommendations = [
            "เปิดการตรวจปี พ.ศ. และการแปลงศักราชให้สม่ำเสมอ",
            "ตรวจเลขไทย (๐–๙) และ normalize เป็นเลขอารบิกก่อนคำนวณ",
            "ตรวจอักขระล่องหน/การ normalize ข้อความไทยก่อน groupby/join",
        ]
        language_impact = (
            "ข้อมูลมีภาษาไทยเด่น จึงเปิด Thai-specific checks เช่น พ.ศ., เลขไทย และ normalization"
        )
    elif detected_language == "mixed":
        thai_recommendations = [
            "ตรวจทั้งกติกาภาษาไทยและอังกฤษ เพราะข้อมูลมีสองภาษา",
            "แยกคอลัมน์ไทย/อังกฤษก่อนทำ text analytics หรือ tokenization",
            "ตรวจ พ.ศ., เลขไทย, encoding และรูปแบบวันที่ทั้งสองภาษา",
        ]
        language_impact = "ข้อมูลผสมไทย+อังกฤษ ควรตรวจคุณภาพและทำความสะอาดทั้งสองภาษา"
    elif detected_language == "english":
        language_impact = "ข้อมูลเป็นอังกฤษล้วน จึงข้าม Thai-specific checks อัตโนมัติ"
    else:
        language_impact = "ไม่พบข้อความชัดเจน เน้นวิเคราะห์ตัวเลข/วันที่และข้าม Thai-specific checks"
    if thai_recommendations:
        focus.extend(thai_recommendations)

    evidence = []
    if language_info.get("evidence"):
        evidence.extend(str(e) for e in language_info["evidence"][:3])
    if id_cols:
        evidence.append(f"พบคอลัมน์ ID {len(id_cols)} คอลัมน์: {', '.join(id_cols[:4])}")
    if amount_cols:
        evidence.append(f"พบคอลัมน์ยอดเงิน/ตัววัดธุรกิจ: {', '.join(amount_cols[:4])}")
    if has_datetime:
        time_desc = "datetime index" if has_datetime_index else ", ".join(datetime_cols[:3])
        evidence.append(f"พบแกนเวลา: {time_desc}")
    if low_scale_rating_cols or rating_cols:
        evidence.append("พบคอลัมน์คะแนน/รีวิว: " + ", ".join((low_scale_rating_cols + rating_cols)[:4]))
    if text_response_cols:
        evidence.append(f"พบข้อความตอบกลับ/รีวิว: {', '.join(text_response_cols[:4])}")
    if index_artifact_cols:
        evidence.append(f"ข้ามคอลัมน์ index artifact: {', '.join(index_artifact_cols[:4])}")
    if not evidence:
        evidence.append("โครงสร้างข้อมูลไม่เข้ากับรูปแบบเดียวชัดเจน จึงจัดเป็นข้อมูลผสม")

    return {
        "key": best_type,
        "label": config["label"],
        "label_th": config["label_th"],
        "summary": config["summary"],
        "focus": focus,
        "language": language_info,
        "language_impact": language_impact,
        "thai_recommendations": thai_recommendations,
        "show_thai_specific": detected_language in {"thai", "mixed"},
        "evidence": evidence[:5],
        "scores": scores,
        "signals": {
            "id_columns": id_cols,
            "amount_columns": amount_cols,
            "datetime_columns": datetime_cols,
            "rating_columns": list(dict.fromkeys(low_scale_rating_cols + rating_cols)),
            "text_columns": text_response_cols,
            "index_artifact_columns": index_artifact_cols,
        },
    }


class ProfileReport:
    """รายงานวิเคราะห์ข้อมูล ThaiEDA สำหรับ DataFrame หนึ่งชุด."""

    def __init__(
        self,
        df: pd.DataFrame,
        lang: str = "th",
        tokenizer_engine: str = "auto",
        max_sample: int = 5000,
        make_charts: bool = True,
        target_column: str | None = None,
        clean: bool = False,
        timeseries: bool = True,
        insights_engine: bool = True,
        insights_top: int = 8,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("ProfileReport expects a pandas DataFrame.")
        if target_column is not None and target_column not in df.columns:
            raise KeyError(f"target_column {target_column!r} not found in DataFrame.")
        self.df = df
        self.lang = lang
        self.tokenizer_engine = tokenizer_engine
        self.max_sample = max_sample
        self.make_charts = make_charts
        self.target_column = target_column
        # clean=True: ทำความสะอาดข้อความก่อนวิเคราะห์ และเก็บ diff (ก่อน/หลัง) ไว้แสดงในรายงาน
        self.clean = clean
        # timeseries=True: ตรวจหา datetime column แล้ววิเคราะห์คอลัมน์ตัวเลขเป็นอนุกรมเวลาอัตโนมัติ
        self.timeseries = timeseries
        # insights_engine=True: ค้นหาข้อค้นพบจากการผสมคอลัมน์ (group-by + aggregate + scoring)
        self.insights_engine = insights_engine
        self.insights_top = insights_top
        # progress: callback(ข้อความ) เรียกระหว่างแต่ละขั้นตอน — ใช้แสดงความคืบหน้าบนไฟล์ใหญ่
        self._progress_cb = progress

        self._ran = False
        self._column_types: dict[str, ColumnType] = {}
        self._quality_issues: list[QualityIssue] = []
        self._anomalies: list[AnomalyIssue] = []
        self._cleaning: list[CleaningResult] = []
        # cleaning_diff: การทำความสะอาดที่ "ลงมือทำจริง" (เมื่อ clean=True) — ต่างจาก _cleaning (dry-run)
        self._cleaning_diff: list[CleaningResult] = []
        self._text_metrics: dict[str, TextMetrics] = {}
        self._ner: dict[str, NERResult] = {}
        self._target_associations: list[TargetAssociation] = []
        # ผลวิเคราะห์อนุกรมเวลา ต่อคอลัมน์ตัวเลข (เมื่อมี datetime column)
        self._timeseries: dict[str, TimeseriesResult] = {}
        # ซีรีส์ที่ index ด้วยเวลาแล้ว (เก็บไว้สร้างกราฟ timeseries) + ชื่อคอลัมน์เวลา
        self._ts_indexed: dict[str, pd.Series] = {}
        self._ts_time_col: str | None = None
        # กราฟ timeseries ต่อคอลัมน์ (line/decomposition/acf)
        self._timeseries_charts: dict[str, dict[str, str]] = {}
        # กราฟ cross-column insight ต่อ card index (v0.7)
        self._insight_charts: dict[int, str] = {}
        self._insights: InsightSummary | None = None
        # ผลจาก cross-column insight engine (v0.6) — None ถ้าปิดใช้งานหรือยังไม่ได้รัน
        self._insight_engine: InsightEngineResult | None = None
        self._overview: dict[str, Any] = {}
        self._charts: dict[str, dict[str, str]] = {}
        # กราฟระดับชุดข้อมูล (correlation/scatter/box/violin/missing) จาก auto_select_charts
        self._dataset_charts: dict[str, str] = {}
        self._basic_stats: dict[str, dict[str, Any]] = {}
        self._data_type: dict[str, Any] = {}
        self._notes: list[str] = []
        self._ignored_columns: set[str] = {
            str(c) for c in self.df.columns if _is_index_artifact_column(str(c))
        }

    def _emit_progress(self, key: str) -> None:
        """แจ้งความคืบหน้าหนึ่งขั้นตอน (localized ตามภาษาของรายงาน) ถ้ามี callback."""
        if self._progress_cb is not None:
            self._progress_cb(label(key, self.lang))

    def _mark_ignored_columns(self) -> None:
        """ระบุคอลัมน์ที่เป็น artifact/derived date component เพื่อไม่ส่งเข้า anomaly/timeseries/insight."""
        for col in self.df.columns:
            name = str(col)
            if _is_index_artifact_column(name):
                self._ignored_columns.add(name)
                continue
            if _is_date_component_column(name):
                self._ignored_columns.add(name)

        index_cols = [c for c in self._ignored_columns if _is_index_artifact_column(c)]
        if index_cols:
            self._notes.append(
                "Ignored CSV index artifact column(s): " + ", ".join(sorted(index_cols))
            )

    def _analysis_columns(self) -> list[str]:
        """คอลัมน์ที่ควรวิเคราะห์จริง (ตัด index/date-component artifact ออก)."""
        return [str(c) for c in self.df.columns if str(c) not in self._ignored_columns]

    def _analysis_df(self) -> pd.DataFrame:
        """DataFrame สำหรับ analysis ที่ไม่รวมคอลัมน์ artifact."""
        cols = self._analysis_columns()
        if len(cols) == len(self.df.columns):
            return self.df
        return self.df.loc[:, cols]

    def _analysis_column_types(self) -> dict[str, ColumnType]:
        """column_types เฉพาะคอลัมน์ที่วิเคราะห์จริง."""
        return {c: t for c, t in self._column_types.items() if c not in self._ignored_columns}

    def _refine_column_types_for_report(self) -> None:
        """ปรับ type เฉพาะรายงาน: คอลัมน์ review/comment สั้น ๆ ยังเป็น text response ไม่ใช่ category."""
        for col in self.df.columns:
            name = str(col)
            if self._column_types.get(name) != ColumnType.CATEGORICAL:
                continue
            series = self.df[col]
            if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
                continue
            if not _RATING_COLUMN_RE.search(name.lower()) or _NON_RESPONSE_TEXT_RE.search(
                name.lower()
            ):
                continue
            lang = str(_detect_language(pd.DataFrame({name: series})).get("language", "numeric"))
            if lang == "thai":
                self._column_types[name] = ColumnType.THAI_TEXT
            elif lang == "mixed":
                self._column_types[name] = ColumnType.MIXED_TEXT
            elif lang == "english":
                self._column_types[name] = ColumnType.ENGLISH_TEXT

    # ------------------------------------------------------------------ run
    def run(self) -> ProfileReport:
        """รันการวิเคราะห์ทั้งหมด (idempotent — เรียกซ้ำได้)."""
        # ถ้า clean=True ทำความสะอาดข้อความก่อน แล้ววิเคราะห์ข้อมูลที่สะอาดแล้ว (เก็บ diff ไว้แสดง)
        if self.clean:
            self._emit_progress("prog_clean")
            self._apply_cleaning()

        self._emit_progress("prog_detect")
        self._data_type = _detect_data_type(self.df)
        self._column_types = detect_all(self.df)
        self._refine_column_types_for_report()
        self._mark_ignored_columns()
        self._overview = self._compute_overview()
        self._emit_progress("prog_quality")
        self._quality_issues = run_quality_checks(
            self.df,
            self._column_types,
            language_info=self._data_type.get("language"),
        )

        # text metrics
        thai_cols = [c for c, t in self._column_types.items() if t in _TEXT_METRIC_TYPES]
        tokenizer = None
        if thai_cols:
            tokenizer = self._try_get_tokenizer()

        if tokenizer is not None:
            for col in thai_cols:
                self._text_metrics[col] = text_metrics(
                    self.df[col], tokenizer, max_sample=self.max_sample
                )

        # ความผิดปกติ (statistical/text/encoding/categorical) — text checks ต้องมี tokenizer
        self._emit_progress("prog_anomaly")
        analysis_df = self._analysis_df()
        self._anomalies = detect_anomalies(
            analysis_df, self._analysis_column_types(), tokenizer, notes=self._notes
        )
        self._note_if_ml_skipped()

        # Named entities (NER) — เฉพาะคอลัมน์ข้อความไทย และเมื่อมี NER engine ที่ใช้ได้
        self._compute_ner(thai_cols, tokenizer)

        # การวิเคราะห์ตัวแปรเป้าหมาย (target analysis) — เมื่อผู้ใช้ระบุ target_column
        self._compute_target_analysis()

        # ค้นหาข้อค้นพบจากการผสมคอลัมน์ (cross-column insight engine, v0.6)
        if self.insights_engine:
            self._emit_progress("prog_insights_engine")
        self._compute_insight_engine()

        # การวิเคราะห์อนุกรมเวลา (timeseries) — เมื่อมี datetime column และเปิดใช้งาน
        if self.timeseries:
            self._emit_progress("prog_timeseries")
        self._compute_timeseries()

        # คำแนะนำการทำความสะอาด (dry-run — ไม่แก้ข้อมูลจริง)
        # เมื่อ clean=True ข้อมูลถูกทำความสะอาดไปแล้วใน _apply_cleaning (ใช้ DEFAULT_OPERATIONS
        # ชุดเดียวกัน) การ dry-run ซ้ำบนข้อมูลที่สะอาดแล้วจะได้ผล rows_affected=0 ทุก op
        # (operations เหล่านี้ idempotent) → suggestions ว่างเสมอ จึงข้ามได้เพื่อความเร็ว
        # โดยไม่กระทบ output. เมื่อ clean=False ยัง dry-run บนข้อมูลดิบตามเดิม
        self._cleaning = [] if self.clean else self._compute_cleaning_suggestions()

        # สถิติพื้นฐานของทุกคอลัมน์
        for col in self.df.columns:
            self._basic_stats[str(col)] = self._compute_basic_stats(str(col))

        # สรุปข้อค้นพบสำคัญอัตโนมัติ (ตีความผลทั้งหมดเป็นภาษาไทย) — ทำหลังวิเคราะห์ครบ
        self._emit_progress("prog_insights")
        self._insights = generate_insights(
            self._analysis_df(),
            self._quality_issues,
            self._anomalies,
            self._text_metrics,
            target_associations=self._target_associations,
            cleaning_results=self._cleaning_diff,
            column_types=self._analysis_column_types(),
            timeseries_results=self._timeseries,
            ner_results=self._ner,
            extra_insights=self._business_insights(),
        )

        # กราฟ
        if self.make_charts:
            self._emit_progress("prog_charts")
            # กราฟต่อคอลัมน์ข้อความ (word cloud/top tokens/length) — ต้องมี tokenizer
            if tokenizer is not None:
                self._build_charts(tokenizer)
            # กราฟระดับชุดข้อมูล (correlation/box/violin/missing/distribution) — ไม่ต้องใช้ tokenizer
            self._build_dataset_charts()
            # กราฟอนุกรมเวลา (line/decomposition/acf) — เมื่อมีผลวิเคราะห์ timeseries
            if self._timeseries:
                self._build_timeseries_charts()
            # กราฟ cross-column insights (outstanding/attribution/comparison/trend) — v0.7
            if self._insight_engine is not None and self._insight_engine.cards:
                self._build_insight_charts()
            # จำกัดจำนวนกราฟรวมไม่ให้ HTML บวมเกินไป (P2)
            self._enforce_chart_budget()

        self._ran = True
        return self

    def _count_embedded_charts(self) -> int:
        """นับจำนวนกราฟ (base64) ที่จะถูกฝังใน HTML จากทุกแหล่งรวมกัน."""
        dc = self._dataset_charts
        n = sum(1 for k in _DATASET_LEVEL_CHART_KEYS if dc.get(k))
        n += sum(1 for k in dc if k.startswith(("distribution::", "valuecounts::")))
        n += sum(len(v) for v in self._timeseries_charts.values())
        n += len(self._insight_charts)
        n += sum(len(v) for v in self._charts.values())
        return n

    def _embedded_chart_bytes(self) -> int:
        """ขนาดรวม (ไบต์) ของสตริง base64 ที่จะถูกฝังใน HTML — ใช้คุมขนาดไฟล์รวม."""
        dc = self._dataset_charts
        total = sum(len(v) for v in dc.values())
        total += sum(len(v) for charts in self._timeseries_charts.values() for v in charts.values())
        total += sum(len(v) for v in self._insight_charts.values())
        total += sum(len(v) for charts in self._charts.values() for v in charts.values())
        return total

    def _over_chart_budget(self) -> bool:
        """เกินงบกราฟไหม — เกินทั้ง 'จำนวน' หรือ 'ขนาดรวม' ถือว่าเกิน (P2)."""
        return (
            self._count_embedded_charts() > _MAX_CHARTS_PER_REPORT
            or self._embedded_chart_bytes() > _MAX_CHART_BYTES
        )

    def _enforce_chart_budget(self) -> None:
        """ตัดกราฟที่สำคัญน้อยสุดให้อยู่ในงบทั้งจำนวน (``_MAX_CHARTS_PER_REPORT``) และ
        ขนาดรวม (``_MAX_CHART_BYTES``) — กัน HTML บวมเกิน 2MB หรือกราฟเยอะจนเบราว์เซอร์ค้าง (P2).

        ลำดับการตัด (จากสำคัญน้อยไปมา): ACF/decomposition ของอนุกรมเวลา (เก็บกราฟเส้นไว้) →
        กราฟต่อคอลัมน์ข้อความ → กราฟเส้นอนุกรมเวลาที่เหลือ → กราฟ value-counts/histogram
        ต่อคอลัมน์ (เก็บไว้สำหรับ Columns tab ให้นานที่สุด). กราฟระดับชุดข้อมูล
        (correlation/box/violin/missing) และกราฟ insight เชิงธุรกิจจะถูกเก็บไว้เสมอ
        """
        if not self._over_chart_budget():
            return
        dropped = 0

        # 1) ตัด ACF แล้ว decomposition ของอนุกรมเวลา (เก็บกราฟเส้นไว้ — สำคัญสุด)
        for sub in ("acf", "decomposition"):
            for col in list(self._timeseries_charts):
                if not self._over_chart_budget():
                    break
                if self._timeseries_charts[col].pop(sub, None) is not None:
                    dropped += 1

        # 3) ถ้ายังเกิน ตัดกราฟต่อคอลัมน์ข้อความ แล้วกราฟเส้นอนุกรมเวลาที่เหลือ
        for store in (self._charts, self._timeseries_charts):
            for col in list(store):
                if not self._over_chart_budget():
                    break
                if store[col]:
                    dropped += len(store[col])
                    store[col] = {}

        # 4) สุดท้าย ถ้ายังเกิน ตัด value-counts แล้ว distribution ต่อคอลัมน์
        for prefix in ("valuecounts::", "distribution::"):
            for key in [k for k in self._dataset_charts if k.startswith(prefix)]:
                if not self._over_chart_budget():
                    break
                del self._dataset_charts[key]
                dropped += 1

        if dropped:
            self._notes.append(
                f"จำกัดกราฟไว้ {_MAX_CHARTS_PER_REPORT} รูป / "
                f"{_MAX_CHART_BYTES / 1_000_000:.1f}MB (ตัด {dropped} รูปที่สำคัญน้อยสุด) "
                "เพื่อลดขนาดไฟล์ HTML"
            )

    def _compute_timeseries(self) -> None:
        """ตรวจหาแกนเวลาแล้ววิเคราะห์ทุกคอลัมน์ตัวเลขเป็นอนุกรมเวลา — แบบอัตโนมัติ.

        เลือก datetime column แรกที่เหมาะเป็นแกนเวลา, index ข้อมูลด้วยเวลานั้น (เรียงตามเวลา)
        แล้ววิเคราะห์ทุกคอลัมน์ตัวเลข เก็บทั้งผลวิเคราะห์และซีรีส์ที่ index แล้ว (ไว้สร้างกราฟ)
        ไม่บังคับ: ถ้าไม่มี datetime column หรือผู้ใช้ปิด timeseries=False จะข้ามไป
        """
        if not self.timeseries:
            return
        try:
            ts_cols = detect_timeseries_columns(self.df)
        except Exception as exc:  # noqa: BLE001 — การตรวจ timeseries พังไม่ควรล้มทั้งรายงาน
            self._notes.append(f"timeseries detection failed: {exc}")
            return
        if not ts_cols:
            return

        time_col = next(iter(ts_cols))
        # format="mixed": parse แต่ละค่าแยกกัน เลี่ยง UserWarning "Could not infer format" (U1)
        time_values = pd.to_datetime(self.df[time_col], errors="coerce", format="mixed")
        valid = time_values.notna()
        if int(valid.sum()) < 5:
            return
        indexed = self.df.loc[valid].copy()
        indexed.index = pd.DatetimeIndex(time_values[valid])
        indexed = indexed.sort_index()

        numeric_cols: list[str] = []
        skipped_metric_cols: list[str] = []
        for c in self.df.columns:
            col = str(c)
            ctype = self._column_types.get(col)
            if col == str(time_col) or col in self._ignored_columns or ctype != ColumnType.NUMERIC:
                continue
            if _is_id_like_column(col, ctype) or _is_low_cardinality_code_or_boolean(
                self.df[c], col
            ):
                skipped_metric_cols.append(col)
                continue
            numeric_cols.append(col)
        if skipped_metric_cols:
            self._notes.append(
                "timeseries skipped non-measure ID/code column(s): "
                + ", ".join(skipped_metric_cols[:8])
            )
        if not numeric_cols:
            return
        self._ts_time_col = str(time_col)
        # ข้อมูลยาวมาก (>200K จุด): analyze_timeseries จะใช้ decomposition พื้นฐานแทน STL
        # โดยอัตโนมัติเพื่อความเร็ว — แจ้งให้ผู้ใช้ทราบครั้งเดียว
        if len(indexed) > 200_000:
            self._notes.append(
                f"timeseries: ข้อมูล {len(indexed):,} แถว — ใช้การแยกองค์ประกอบแบบพื้นฐาน "
                "(ข้าม STL/statsmodels) เพื่อความเร็ว"
            )
        for col in numeric_cols[:20]:
            try:
                result = analyze_timeseries(indexed[col])
            except Exception as exc:  # noqa: BLE001 — วิเคราะห์ timeseries พังไม่ควรล้มทั้งรายงาน
                self._notes.append(f"timeseries analysis failed for '{col}': {exc}")
                continue
            if result.is_timeseries:
                self._timeseries[col] = result
                self._ts_indexed[col] = indexed[col]

    def _build_timeseries_charts(self) -> None:
        """สร้างกราฟ timeseries ต่อคอลัมน์ — เส้น (พร้อม trend), STL decomposition, ACF."""
        from thaieda.viz import (
            create_acf_plot,
            create_decomposition_plot,
            create_timeseries_plot,
            get_thai_font_path,
        )

        font_path = get_thai_font_path()
        for col, result in self._timeseries.items():
            series = self._ts_indexed.get(col)
            if series is None:
                continue
            charts: dict[str, str] = {}
            try:
                charts["line"] = create_timeseries_plot(series, title=col, font_path=font_path)
            except Exception as exc:  # noqa: BLE001 — กราฟพังไม่ควรล้มทั้งรายงาน
                self._notes.append(f"timeseries plot failed for '{col}': {exc}")
            try:
                components = {k: c.values for k, c in result.components.items()}
                charts["decomposition"] = create_decomposition_plot(
                    components, title=col, font_path=font_path
                )
            except Exception as exc:  # noqa: BLE001
                self._notes.append(f"decomposition plot failed for '{col}': {exc}")
            try:
                charts["acf"] = create_acf_plot(series, title=col, font_path=font_path)
            except Exception as exc:  # noqa: BLE001
                self._notes.append(f"ACF plot failed for '{col}': {exc}")
            self._timeseries_charts[col] = {k: v for k, v in charts.items() if v}

    def _build_insight_charts(self) -> None:
        """สร้างกราฟสำหรับ cross-column insight cards — แต่ละ card ได้กราฟตาม pattern (v0.7).

        outstanding → bar chart, attribution → donut, comparison → box plot, trend → line
        กราฟที่สร้างไม่ได้ (ข้อมูลไม่พอ) จะถูกข้าม — card นั้นไม่มีกราฟ
        """
        from thaieda.viz import create_insight_chart, get_thai_font_path

        if self._insight_engine is None:
            return
        font_path = get_thai_font_path()
        for i, card in enumerate(self._insight_engine.cards):
            try:
                img = create_insight_chart(card.to_dict(), df=self.df, font_path=font_path)
                if img:
                    self._insight_charts[i] = img
            except Exception as exc:  # noqa: BLE001 — กราฟพังไม่ควรล้มทั้งรายงาน
                self._notes.append(f"insight chart failed for card #{i}: {exc}")

    def _note_if_ml_skipped(self) -> None:
        """ถ้ามีคอลัมน์ตัวเลขใหญ่ (>100 แถว) แต่ไม่มี scikit-learn ให้บันทึก note ว่าข้ามวิธี ML."""
        from thaieda.anomaly import sklearn_available

        if sklearn_available():
            return
        big_numeric = [
            str(c)
            for c in self.df.columns
            if self._column_types.get(str(c)) == ColumnType.NUMERIC
            and str(c) not in self._ignored_columns
            and int(self.df[c].notna().sum()) > 100
        ]
        if big_numeric:
            self._notes.append(
                "ML-based anomaly detection (Isolation Forest / LOF) skipped: "
                "install pip install thaieda[ml] (scikit-learn)"
            )

    def _build_dataset_charts(self) -> None:
        """สร้างกราฟระดับชุดข้อมูลด้วย auto_select_charts (เลือกกราฟที่เหมาะกับข้อมูลให้อัตโนมัติ).

        ส่ง text_columns=[] เพราะกราฟต่อคอลัมน์ข้อความ (word cloud/top tokens/length) สร้างแยกใน
        _build_charts แล้ว — ที่นี่จึงเอาเฉพาะกราฟตัวเลข/ค่าว่าง/หมวดหมู่ (รวม scatter matrix)
        """
        from thaieda.viz import auto_select_charts, get_thai_font_path

        try:
            self._dataset_charts = auto_select_charts(
                self.df, tokenizer=None, font_path=get_thai_font_path(), text_columns=[]
            )
        except Exception as exc:  # noqa: BLE001 — กราฟพังไม่ควรล้มทั้งรายงาน
            self._notes.append(f"dataset charts failed: {exc}")
            self._dataset_charts = {}

    def _compute_ner(self, thai_cols: list[str], tokenizer) -> None:
        """สกัด named entities จากคอลัมน์ข้อความไทย — ทำเฉพาะเมื่อมี NER engine ที่ใช้ได้.

        ไม่บังคับ: ถ้าไม่มี backend (เช่น python-crfsuite/transformers) จะข้ามเงียบ ๆ
        เพราะ NER เป็น optional (thaieda[ner]) ไม่ใช่ส่วนหลักของรายงาน
        """
        if not thai_cols or not ner_available():
            return
        for col in thai_cols:
            try:
                result = extract_entities(self.df[col], tokenizer, max_sample=self.max_sample)
            except Exception as exc:  # noqa: BLE001 — NER พังไม่ควรล้มทั้งรายงาน
                self._notes.append(f"NER failed for '{col}': {exc}")
                continue
            if result.total_entities > 0:
                self._ner[col] = result

    def _compute_target_analysis(self) -> None:
        """วิเคราะห์ความสัมพันธ์ของทุกคอลัมน์กับ target column (ถ้าระบุ)."""
        if self.target_column is None:
            return
        try:
            self._target_associations = analyze_target(self.df, self.target_column)
        except Exception as exc:  # noqa: BLE001 — การวิเคราะห์ target พังไม่ควรล้มทั้งรายงาน
            self._notes.append(f"target analysis failed: {exc}")
            self._target_associations = []

    def _compute_insight_engine(self) -> None:
        """ค้นหาข้อค้นพบจากการผสมคอลัมน์ (group-by + aggregate + statistical scoring)."""
        if not self.insights_engine:
            return
        try:
            self._insight_engine = discover_insights(
                self._analysis_df(),
                self._analysis_column_types(),
                top_n=self.insights_top,
                progress=self._progress_cb,
            )
        except Exception as exc:  # noqa: BLE001 — เอนจินพังไม่ควรล้มทั้งรายงาน
            self._notes.append(f"cross-column insight engine failed: {exc}")
            self._insight_engine = None

    def _business_insights(self) -> list[Insight]:
        """แปลง InsightCard เด่นสุด 3 อันดับ → Insight (หมวด business) เพื่อป้อนบทสรุปผู้บริหาร."""
        if self._insight_engine is None or not self._insight_engine.cards:
            return []
        out: list[Insight] = []
        for card in self._insight_engine.cards[:3]:
            out.append(
                Insight(
                    category="business",
                    severity=card.severity,
                    title_th=card.title_th,
                    description_th=card.description_th,
                    recommendation_th=card.recommendation_th,
                )
            )
        return out

    def _apply_cleaning(self) -> None:
        """ทำความสะอาดข้อมูลจริง — ลบแถวซ้ำ, จัดการ missing, และทำความสะอาดข้อความ.

        ใช้ DEFAULT_OPERATIONS ของ clean_thai_text (แก้ encoding/zw/ช่องว่าง/normalize ฯลฯ)
        เก็บเฉพาะการดำเนินการที่มีผล (>0 แถว) ไว้ใน self._cleaning_diff เพื่อแสดงก่อน/หลังในรายงาน
        """
        cleaned_df = self.df.copy()
        diffs: list[CleaningResult] = []

        try:
            cleaned_df, dup_result = remove_duplicate_rows(cleaned_df)
            if dup_result.rows_affected > 0:
                diffs.append(dup_result)
        except Exception as exc:  # noqa: BLE001 — การลบแถวซ้ำพังไม่ควรล้มทั้งรายงาน
            self._notes.append(f"duplicate-row cleaning failed: {exc}")

        for col in cleaned_df.columns:
            series = cleaned_df[col]
            try:
                cleaned_series, missing_result = handle_missing_values(series, strategy="flag")
            except Exception as exc:  # noqa: BLE001 — การจัดการ missing พังไม่ควรล้มทั้งรายงาน
                self._notes.append(f"missing-value cleaning failed for '{col}': {exc}")
                continue
            cleaned_df[col] = cleaned_series
            if missing_result.rows_affected > 0:
                diffs.append(missing_result)

        for col in cleaned_df.columns:
            series = cleaned_df[col]
            # ทำความสะอาดเฉพาะคอลัมน์ข้อความ — ข้ามตัวเลข/วันที่/บูลีน
            if (
                pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_bool_dtype(series)
            ):
                continue
            try:
                cleaned, results = clean_thai_text(series)
            except Exception as exc:  # noqa: BLE001 — การทำความสะอาดพังไม่ควรล้มทั้งรายงาน
                self._notes.append(f"cleaning failed for '{col}' (dtype {series.dtype}): {exc}")
                continue
            cleaned_df[col] = cleaned
            diffs.extend(r for r in results if r.rows_affected > 0)
        self.df = cleaned_df
        self._cleaning_diff = diffs

    def _compute_cleaning_suggestions(self) -> list[CleaningResult]:
        """ลองทำความสะอาดคอลัมน์ข้อความแบบ dry-run และเก็บเฉพาะการดำเนินการที่มีผล (>0 แถว)."""
        suggestions: list[CleaningResult] = []
        for col in self.df.columns:
            ctype = self._column_types.get(str(col), ColumnType.EMPTY)
            if ctype not in _CLEANABLE_TYPES:
                continue
            series = self.df[col]
            # กันคอลัมน์ชนิดตัวเลข/วันที่/บูลีน (เช่น CATEGORICAL ที่เก็บเป็น int) — ไม่ใช่ข้อความ
            if (
                pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_bool_dtype(series)
            ):
                continue
            try:
                _, results = clean_thai_text(series)
            except Exception as exc:  # noqa: BLE001 — การทำความสะอาดพังไม่ควรล้มทั้งรายงาน
                self._notes.append(
                    f"cleaning suggestions failed for '{col}' (dtype {series.dtype}): {exc}"
                )
                continue
            suggestions.extend(r for r in results if r.rows_affected > 0)
        return suggestions

    def _ensure_ran(self) -> None:
        if not self._ran:
            self.run()

    def _try_get_tokenizer(self):
        """พยายามสร้าง tokenizer — ถ้าไม่มี engine บันทึก note แล้วคืน None.

        คุณภาพ (the moat) ยังทำงานได้โดยไม่ต้องมี tokenizer จึงไม่ทำให้รายงานทั้งฉบับล่ม
        แต่จะระบุชัดเจนว่าสถิติข้อความต้องติดตั้ง thaieda[thai]
        """
        from thaieda.tokenize import get_tokenizer

        try:
            return get_tokenizer(self.tokenizer_engine)
        except ImportError as exc:
            self._notes.append(
                f"Text metrics & word clouds skipped: {exc} "
                "(คอลัมน์ข้อความไทยต้องติดตั้ง pip install thaieda[thai])"
            )
            return None

    # -------------------------------------------------------------- compute
    def _compute_overview(self) -> dict[str, Any]:
        df = self.df
        rows, cols = df.shape
        total_cells = int(rows * cols)
        missing = int(df.isna().sum().sum())
        type_counts: dict[str, int] = {}
        for t in self._column_types.values():
            type_counts[t.value] = type_counts.get(t.value, 0) + 1
        return {
            "rows": int(rows),
            "columns": int(cols),
            "total_cells": total_cells,
            "missing_cells": missing,
            "missing_pct": round((missing / total_cells * 100.0) if total_cells else 0.0, 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "ignored_columns": sorted(self._ignored_columns),
            "type_counts": type_counts,
        }

    def _compute_basic_stats(self, col: str) -> dict[str, Any]:
        series = self.df[col]
        ctype = self._column_types.get(col, ColumnType.EMPTY)
        non_null = series.dropna()
        stats: dict[str, Any] = {
            "count": int(len(non_null)),
            "missing": int(series.isna().sum()),
            "unique": int(non_null.nunique()),
        }

        if ctype == ColumnType.NUMERIC:
            if col in self._ignored_columns:
                stats["note"] = "ignored: index/date component artifact"
                return stats
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) > 0:
                stats.update(
                    {
                        "mean": round(float(numeric.mean()), 4),
                        "std": round(float(numeric.std()), 4) if len(numeric) > 1 else 0.0,
                        "min": float(numeric.min()),
                        "max": float(numeric.max()),
                    }
                )
        elif ctype == ColumnType.DATETIME:
            # format="mixed": parse แต่ละค่าแยกกัน เลี่ยง UserWarning "Could not infer format" (U1)
            dt = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
            if len(dt) > 0:
                stats["min"] = str(dt.min())
                stats["max"] = str(dt.max())
        elif ctype in (ColumnType.CATEGORICAL, ColumnType.ID):
            vc = non_null.astype(str).value_counts().head(10)
            stats["top_values"] = [(str(k), int(v)) for k, v in vc.items()]

        return stats

    def _build_charts(self, tokenizer) -> None:
        from thaieda.viz import (
            create_length_histogram,
            create_top_tokens_chart,
            create_wordcloud,
            get_thai_font_path,
        )

        font_path = get_thai_font_path()
        # จำกัดจำนวนคอลัมน์ที่สร้างกราฟ (top N) — กันรายงานใหญ่เกินบนข้อมูลคอลัมน์ข้อความเยอะ
        text_items = list(self._text_metrics.items())
        if len(text_items) > _MAX_CHART_COLUMNS:
            self._notes.append(
                f"charts limited to first {_MAX_CHART_COLUMNS} of {len(text_items)} text columns"
            )
            text_items = text_items[:_MAX_CHART_COLUMNS]
        for col, metrics in text_items:
            charts: dict[str, str] = {}
            non_null = self.df[col].dropna().astype(str)

            # กราฟแท่งคำที่พบบ่อย
            try:
                if metrics.top_tokens:
                    charts["top_tokens"] = create_top_tokens_chart(
                        metrics.top_tokens, font_path=font_path
                    )
            except Exception as exc:  # noqa: BLE001 — กราฟพังไม่ควรล้มทั้งรายงาน
                self._notes.append(f"top-tokens chart failed for '{col}': {exc}")

            # ฮิสโทแกรมความยาว
            try:
                lengths = [len(s) for s in non_null]
                if lengths:
                    charts["length_hist"] = create_length_histogram(lengths, font_path=font_path)
            except Exception as exc:  # noqa: BLE001
                self._notes.append(f"length histogram failed for '{col}': {exc}")

            # word cloud (ต้องมี wordcloud package)
            try:
                sample = non_null.head(self.max_sample)
                joined = " ".join(sample)
                if joined.strip():
                    charts["wordcloud"] = create_wordcloud(joined, tokenizer, font_path=font_path)
            except ImportError:
                # ไม่มี wordcloud — ข้ามเงียบ ๆ (เป็น optional extra)
                pass
            except Exception as exc:  # noqa: BLE001
                self._notes.append(f"word cloud failed for '{col}': {exc}")

            self._charts[col] = charts

    # ------------------------------------------------------------ properties
    @property
    def column_types(self) -> dict[str, ColumnType]:
        self._ensure_ran()
        return self._column_types

    @property
    def quality_issues(self) -> list[QualityIssue]:
        self._ensure_ran()
        return self._quality_issues

    @property
    def anomalies(self) -> list[AnomalyIssue]:
        self._ensure_ran()
        return self._anomalies

    @property
    def cleaning_suggestions(self) -> list[CleaningResult]:
        self._ensure_ran()
        return self._cleaning

    @property
    def cleaning_diff(self) -> list[CleaningResult]:
        """การทำความสะอาดที่ลงมือทำจริง (มีค่าเมื่อสร้างรายงานด้วย clean=True)."""
        self._ensure_ran()
        return self._cleaning_diff

    @property
    def insights(self) -> InsightSummary | None:
        """สรุปข้อค้นพบสำคัญอัตโนมัติ (InsightSummary) — None ถ้ายังไม่ได้รัน."""
        self._ensure_ran()
        return self._insights

    @property
    def insight_engine(self) -> InsightEngineResult | None:
        """ผลจาก cross-column insight engine (v0.6) — None ถ้าปิดใช้งาน insights_engine."""
        self._ensure_ran()
        return self._insight_engine

    @property
    def text_metrics(self) -> dict[str, TextMetrics]:
        self._ensure_ran()
        return self._text_metrics

    @property
    def overview(self) -> dict[str, Any]:
        self._ensure_ran()
        return self._overview

    @property
    def notes(self) -> list[str]:
        self._ensure_ran()
        return self._notes

    @property
    def ner(self) -> dict[str, NERResult]:
        self._ensure_ran()
        return self._ner

    @property
    def target_associations(self) -> list[TargetAssociation]:
        self._ensure_ran()
        return self._target_associations

    @property
    def timeseries_results(self) -> dict[str, TimeseriesResult]:
        """ผลวิเคราะห์อนุกรมเวลาต่อคอลัมน์ตัวเลข (ว่างถ้าไม่มี datetime column)."""
        self._ensure_ran()
        return self._timeseries

    # --------------------------------------------------------------- export
    def to_dict(self) -> dict[str, Any]:
        """ส่งออกข้อมูลแบบมีโครงสร้าง (ไม่รวมรูป base64 เพื่อให้กระชับ)."""
        self._ensure_ran()
        columns: dict[str, Any] = {}
        for col in self.df.columns:
            name = str(col)
            entry: dict[str, Any] = {
                "type": self._column_types[name].value,
                "basic_stats": self._basic_stats.get(name, {}),
            }
            if name in self._text_metrics:
                entry["text_metrics"] = self._text_metrics[name].to_dict()
            columns[name] = entry

        result = {
            "thaieda_version": __version__,
            "overview": self._overview,
            "data_type": self._data_type,
            "key_findings": self._top_findings(),
            "column_types": {k: v.value for k, v in self._column_types.items()},
            "insights": self._insights.to_dict() if self._insights is not None else None,
            "insight_engine": (
                self._insight_engine.to_dict() if self._insight_engine is not None else None
            ),
            "quality_issues": [i.to_dict() for i in self._quality_issues],
            "anomalies": [a.to_dict() for a in self._anomalies],
            "cleaning_suggestions": [c.to_dict() for c in self._cleaning],
            "columns": columns,
            "notes": self._notes,
        }
        if self._cleaning_diff:
            result["cleaning_diff"] = [c.to_dict() for c in self._cleaning_diff]
        if self._ner:
            result["ner"] = {col: r.to_dict() for col, r in self._ner.items()}
        if self._timeseries:
            result["timeseries"] = {
                "time_column": self._ts_time_col,
                "columns": {col: r.to_dict() for col, r in self._timeseries.items()},
            }
        if self.target_column is not None:
            result["target_analysis"] = {
                "target_column": self.target_column,
                "associations": [a.to_dict() for a in self._target_associations],
            }
        return result

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        """ส่งออกเป็น JSON string (เขียนไฟล์ด้วยถ้าระบุ path)."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_html(self, path: str | None = None) -> str:
        """เรนเดอร์เป็น HTML (เขียนไฟล์ด้วยถ้าระบุ path) — คืน HTML string."""
        self._ensure_ran()
        html = self._render_html()
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
        return html

    def _repr_html_(self) -> str:
        """สำหรับแสดงผลใน Jupyter notebook."""
        return self.to_html()

    # ---------------------------------------------------------- render helpers
    def _build_report_summary(self) -> dict[str, Any]:
        """สรุปสุขภาพข้อมูลแบบภาษาคนอ่าน — ใช้เปิดรายงานแทนตัวเลขล้วน."""
        rows = int(self._overview.get("rows", 0))
        missing_pct = float(self._overview.get("missing_pct", 0.0))
        duplicate_rows = int(self._overview.get("duplicate_rows", 0))
        duplicate_pct = round((duplicate_rows / rows * 100.0) if rows else 0.0, 2)

        critical_count = sum(1 for i in self._quality_issues if i.severity == "critical")
        critical_count += sum(1 for a in self._anomalies if a.severity == "critical")
        warning_count = sum(1 for i in self._quality_issues if i.severity == "warning")
        warning_count += sum(1 for a in self._anomalies if a.severity == "warning")

        if critical_count or missing_pct >= 30:
            status = "critical"
            verdict = "ควรแก้ประเด็นสำคัญก่อนใช้ตัดสินใจจริง"
        elif warning_count or missing_pct >= 5 or duplicate_pct >= 1:
            status = "warning"
            verdict = "ข้อมูลใช้ต่อได้ แต่ควรตรวจจุดที่เตือนก่อนวิเคราะห์เชิงลึก"
        else:
            status = "good"
            verdict = "ข้อมูลโดยรวมดูพร้อมใช้ ไม่พบประเด็นเร่งด่วน"

        highlights = [
            f"มีข้อมูล {rows:,} แถว × {self._overview.get('columns', 0)} คอลัมน์",
            f"ค่าว่าง {missing_pct:.2f}% ของข้อมูลทั้งหมด",
            f"แถวซ้ำ {duplicate_rows:,} แถว ({duplicate_pct:.2f}%)",
        ]
        if self._insights is not None and self._insights.total_insights:
            highlights.append(f"พบข้อค้นพบที่ควรดู {self._insights.total_insights} เรื่อง")
        if self._insight_engine is not None and self._insight_engine.cards:
            highlights.append(f"มี insight เชิงธุรกิจ {len(self._insight_engine.cards)} เรื่อง")

        return {
            "status": status,
            "verdict": verdict,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "missing_pct": missing_pct,
            "duplicate_pct": duplicate_pct,
            "highlights": highlights,
        }

    def _sorted_quality_issues(self) -> list[dict[str, Any]]:
        """เรียง quality issue ตามความสำคัญและผลกระทบ เพื่อให้คนอ่านเห็นเรื่องใหญ่ก่อน."""
        issues = sorted(
            self._quality_issues,
            key=lambda i: (_SEVERITY_ORDER.get(i.severity, 99), -i.percentage, -i.count),
        )
        out = []
        for issue in issues:
            entry = issue.to_dict()
            entry["description_th_plain"] = _plain_language(entry["description_th"])
            entry["suggestion_th_plain"] = _plain_language(entry.get("suggestion_th", ""))
            entry["business"] = self._translate_to_business(entry)
            out.append(entry)
        return out

    def _sorted_anomalies(self, L) -> list[dict[str, Any]]:
        """เรียง anomaly ตามความสำคัญ พร้อม label ประเภท."""
        anomalies = sorted(
            self._anomalies,
            key=lambda a: (_SEVERITY_ORDER.get(a.severity, 99), -a.percentage, -a.count),
        )
        out = []
        for a in anomalies:
            entry = a.to_dict()
            entry["type_label"] = L(f"antype_{a.anomaly_type}")
            entry["description_th_plain"] = _plain_language(entry["description_th"])
            entry["suggestion_th_plain"] = _plain_language(entry.get("suggestion_th", ""))
            entry["business"] = self._translate_to_business(entry)
            out.append(entry)
        return out

    def _build_priority_actions(
        self,
        insights: list[dict[str, Any]],
        business_cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """เลือก action ที่ควรทำก่อนจริง ๆ จาก insight ทั้งหมด — จำกัดไม่ให้รายงานรก."""
        actions: list[dict[str, Any]] = []
        seen: set[str] = set()

        for ins in insights:
            if len(actions) >= 5:
                break
            key = str(ins.get("recommendation_th") or ins.get("title_th"))
            if key in seen:
                continue
            if ins.get("severity") in {"critical", "warning"} or len(actions) < 3:
                seen.add(key)
                actions.append(
                    {
                        "severity": ins.get("severity", "info"),
                        "title": ins.get("title_th", ""),
                        "why": ins.get("description_th", ""),
                        "action": ins.get("recommendation_th", ""),
                    }
                )

        for card in business_cards:
            if len(actions) >= 5:
                break
            key = str(card.get("recommendation_th") or card.get("title_th"))
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                {
                    "severity": card.get("severity", "info"),
                    "title": card.get("title_th", ""),
                    "why": card.get("description_th", ""),
                    "action": card.get("recommendation_th", ""),
                }
            )

        if not actions and self._cleaning:
            top = max(self._cleaning, key=lambda c: c.rows_affected)
            actions.append(
                {
                    "severity": "info",
                    "title": f"ทำความสะอาดคอลัมน์ {top.column}",
                    "why": f"พบ {top.description_th} {top.rows_affected:,} แถว",
                    "action": "ตรวจตัวอย่างก่อน/หลัง แล้วใช้ clean=True หรือ clean_thai_text() กับข้อมูลจริง",
                }
            )

        return actions

    def _conditional_missing_notes(self) -> list[dict[str, Any]]:
        """เพิ่ม key finding แบบ info เมื่อคอลัมน์มีชื่อที่น่าจะว่างตามเงื่อนไข เช่น holiday/event."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*self._quality_issues, *self._anomalies]:
            column = str(item.column)
            if column in seen or not _is_conditional_missing_column(column):
                continue
            if item.percentage < 50 or not (
                "missing" in item.check_name
                or "null" in item.check_name
                or "ค่าว่าง" in item.description_th
                or "ขาด" in item.description_th
            ):
                continue
            seen.add(column)
            out.append(
                {
                    "source": "report_note",
                    "severity": "info",
                    "column": column,
                    "check_name": "conditional_missing_possible",
                    "count": item.count,
                    "percentage": item.percentage,
                    "description_th": (
                        f"คอลัมน์ '{column}' มีค่าว่างสูง แต่อาจเป็น conditional missing "
                        "(ว่างเมื่อไม่มีวันหยุด/เหตุการณ์/แคมเปญ)"
                    ),
                    "suggestion_th": (
                        "ตรวจ business rule ก่อนเติมค่า/ลบแถว เพราะค่าว่างอาจมีความหมายว่า 'ไม่เกิดเหตุการณ์'"
                    ),
                }
            )
        return out

    def _translate_to_business(self, finding: dict[str, Any]) -> str:
        """แปล finding เชิงเทคนิคให้เป็นภาษาผลกระทบทางธุรกิจ/การตัดสินใจ."""
        column = str(finding.get("column") or finding.get("title_th") or "คอลัมน์นี้")
        check_name = str(finding.get("check_name") or "")
        category = str(finding.get("category") or finding.get("anomaly_type") or "")
        description = str(finding.get("description_th") or finding.get("description") or "")
        suggestion = str(finding.get("suggestion_th") or finding.get("recommendation_th") or "")
        pct = float(finding.get("percentage") or 0.0)
        count = int(finding.get("count") or 0)
        pct_text = f"{pct:.1f}%" if pct else (f"{count:,} รายการ" if count else "บางส่วน")
        col_lower = column.lower()

        if _is_index_artifact_column(column):
            return (
                f"{column} น่าจะเป็นคอลัมน์ index ที่ติดมาจากไฟล์ CSV "
                "ไม่ใช่ข้อมูลธุรกิจ จึงควร ignore/drop ก่อนวิเคราะห์"
            )

        if (
            check_name == "constant_column"
            or "ค่าเดียว" in description
            or "only one unique" in description
        ):
            return (
                f"{column} ไม่มีข้อมูลที่ช่วยแยกความแตกต่าง — ทุกแถวเหมือนกัน ลบออกได้หากไม่ได้ใช้เป็น flag ทางธุรกิจ"
            )

        if (
            check_name == "numeric_outliers"
            or category == "statistical"
            or "ค่าผิดปกติ" in description
        ):
            if _MONEY_COLUMN_RE.search(col_lower):
                return f"{column} {pct_text} มีค่าผิดปกติ — ควรตรวจก่อนนำไป forecast คำนวณยอดรวม หรือวัด performance"  # noqa: E501
            return f"{column} {pct_text} อยู่ไกลจากช่วงปกติ — ควรแยกว่าเป็นเคสพิเศษจริงหรือข้อมูลผิดก่อนใช้ในโมเดล/สรุปผล"  # noqa: E501

        if "missing" in check_name or "ค่าว่าง" in description or "ขาด" in description:
            if _is_conditional_missing_column(column):
                return (
                    f"{column} อาจว่างตามเงื่อนไข (เช่น ไม่มีวันหยุด/เหตุการณ์) {pct_text} — "
                    "ตรวจตรรกะธุรกิจก่อนเติมค่า ไม่ควรสรุปว่าเป็น data quality issue ทันที"
                )
            return f"{column} ขาดข้อมูล {pct_text} — อาจทำให้การแบ่งกลุ่ม รายงาน หรือโมเดลเอนเอียง ควรกำหนดวิธีเติม/ตัดก่อนใช้จริง"  # noqa: E501

        if "duplicate" in check_name or "ซ้ำ" in description:
            return f"{column} มีความซ้ำ {pct_text} — เสี่ยงนับยอด/จำนวนลูกค้าซ้ำ ควร deduplicate ก่อนทำ KPI"

        if "buddhist" in check_name or "พ.ศ" in description:
            return f"{column} มีปี พ.ศ./ค.ศ. ปนกัน — เสี่ยงจัดช่วงเวลาและ trend ผิด ควรแปลงเป็นศักราชเดียวก่อนวิเคราะห์"  # noqa: E501

        if "thai_numerals" in check_name or "เลขไทย" in description:
            return (
                f"{column} มีเลขไทยปน — อาจทำให้ระบบอ่านเป็นข้อความแทนตัวเลข ควรแปลงเป็นเลขอารบิกก่อนคำนวณ"
            )

        if "zero_width" in check_name or "ล่องหน" in description:
            return (
                f"{column} มีอักขระที่มองไม่เห็น — เสี่ยง join/groupby แล้วแยกกลุ่มผิด ควรล้างข้อความก่อนรวมข้อมูล"
            )

        if "placeholder" in check_name or "placeholder" in description.lower():
            return f"{column} ใช้ข้อความแทนค่าว่าง — ควรแปลงเป็น missing จริง เพื่อให้รายงานและโมเดลตีความถูก"

        if category == "business":
            return suggestion or description

        if suggestion:
            return suggestion
        return f"{column} มีประเด็นที่อาจกระทบการวิเคราะห์ — ควรตรวจตัวอย่างและตัดสินใจวิธีจัดการก่อนใช้ข้อมูลจริง"

    def _finding_score(self, finding: dict[str, Any]) -> tuple[float, float, float, str]:
        """คำนวณคะแนนจัดอันดับ finding จาก severity + impact + actionability."""
        severity = str(finding.get("severity") or "info")
        severity_score = {"critical": 100.0, "warning": 60.0, "info": 25.0}.get(severity, 10.0)
        impact = float(finding.get("percentage") or 0.0)
        if not impact and finding.get("count") and self._overview.get("rows"):
            impact = min(float(finding["count"]) / float(self._overview["rows"]) * 100.0, 100.0)
        check_name = str(finding.get("check_name") or "")
        actionable_terms = (
            "missing",
            "duplicate",
            "constant",
            "outlier",
            "thai_numerals",
            "zero_width",
            "placeholder",
            "buddhist",
            "encoding",
            "mojibake",
        )
        actionability = 18.0 if any(t in check_name for t in actionable_terms) else 8.0
        if finding.get("suggestion_th") or finding.get("recommendation_th"):
            actionability += 7.0
        score = severity_score + min(impact, 100.0) * 0.8 + actionability
        return (
            score,
            severity_score,
            impact,
            str(finding.get("column") or finding.get("title_th") or ""),
        )

    def _top_findings(self) -> list[dict[str, Any]]:
        """เลือก Key Findings 3–5 ข้อที่สำคัญที่สุด แทนการโชว์ list ยาวด้านบน."""
        raw: list[dict[str, Any]] = []
        seen_qa: set[tuple[str, str]] = set()
        for issue in self._quality_issues:
            key = (issue.check_name, issue.column)
            if key in seen_qa:
                continue
            seen_qa.add(key)
            entry = issue.to_dict()
            entry["source"] = "quality"
            raw.append(entry)
        for anomaly in self._anomalies:
            key = (anomaly.check_name, anomaly.column)
            if key in seen_qa:
                continue
            seen_qa.add(key)
            entry = anomaly.to_dict()
            entry["source"] = "anomaly"
            raw.append(entry)
        if self._insight_engine is not None:
            for card in self._insight_engine.cards[:5]:
                entry = card.to_dict()
                entry["source"] = "business"
                perspective = entry.get("perspective", {})
                column = perspective.get("measure") or perspective.get("breakdown")
                if column and str(column) in self._ignored_columns:
                    continue
                entry["column"] = column
                raw.append(entry)

        raw.extend(self._conditional_missing_notes())

        if not raw and self._insights is not None:
            for insight in self._insights.insights[:5]:
                entry = insight.to_dict()
                entry["source"] = "insight"
                raw.append(entry)

        ranked = sorted(raw, key=self._finding_score, reverse=True)
        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for item in ranked:
            key = (
                str(item.get("source") or ""),
                str(item.get("column") or item.get("title_th") or ""),
                str(item.get("check_name") or item.get("pattern") or item.get("title_th") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            technical = (
                item.get("description_th") or item.get("description") or item.get("title_th") or ""
            )
            recommendation = item.get("suggestion_th") or item.get("recommendation_th") or ""
            enriched = {
                **item,
                "technical": _plain_language(str(technical)),
                "business": self._translate_to_business(item),
                "recommendation": _plain_language(str(recommendation)),
                "impact": float(item.get("percentage") or 0.0),
            }
            selected.append(enriched)
            if len(selected) >= 5:
                break

        if len(selected) > 3:
            return selected[:5]
        return selected

    def _build_issue_summary(self) -> dict[str, Any]:
        """นับจำนวน issue ตามระดับความรุนแรง สำหรับแสดงแถบสรุปด้านบน."""
        items = [*self._quality_issues, *self._anomalies]
        return {
            "critical": sum(1 for i in items if i.severity == "critical"),
            "warning": sum(1 for i in items if i.severity == "warning"),
            "info": sum(1 for i in items if i.severity == "info"),
            "total": len(items),
        }

    def _build_top_columns_to_watch(self) -> list[dict[str, Any]]:
        """รวมคอลัมน์ที่มีประเด็นเยอะ/รุนแรงที่สุด เพื่อช่วยให้คนอ่านเริ่มตรวจถูกจุด."""
        scores: dict[str, dict[str, Any]] = {}

        def add(column: str, severity: str, label_text: str, count: int) -> None:
            score = {"critical": 5, "warning": 3, "info": 1}.get(severity, 1)
            entry = scores.setdefault(
                column,
                {"column": column, "score": 0, "severity": severity, "reasons": []},
            )
            entry["score"] += score + min(count, 1000) / 1000
            if _SEVERITY_ORDER.get(severity, 99) < _SEVERITY_ORDER.get(entry["severity"], 99):
                entry["severity"] = severity
            if len(entry["reasons"]) < 3:
                entry["reasons"].append(label_text)

        for issue in self._quality_issues:
            add(issue.column, issue.severity, issue.description_th, issue.count)
        for anomaly in self._anomalies:
            add(anomaly.column, anomaly.severity, anomaly.description_th, anomaly.count)

        ranked = sorted(scores.values(), key=lambda x: (-x["score"], x["column"]))
        return ranked[:6]

    # --------------------------------------------------------------- render
    def _render_html(self) -> str:
        from jinja2 import Environment

        env = Environment(autoescape=True)
        template = env.from_string(REPORT_TEMPLATE)

        lang = self.lang

        def L(key: str) -> str:
            return label(key, lang)

        # การกระจายประเภทคอลัมน์
        type_distribution = [
            (t, L(f"type_{t}"), cnt)
            for t, cnt in sorted(self._overview["type_counts"].items(), key=lambda x: -x[1])
        ]

        # เตรียมข้อมูลต่อคอลัมน์
        columns = [self._render_column_context(str(col), L) for col in self.df.columns]

        # ความผิดปกติ — เรียงจากเรื่องเร่งด่วนก่อน และแนบ label ของประเภทไว้ใช้ในเทมเพลต
        quality_issues = self._sorted_quality_issues()
        anomalies = self._sorted_anomalies(L)

        dc = self._dataset_charts
        # Plotly interactive correlation heatmap (lazy — fallback to static if not available)
        plotly_heatmap = ""
        try:
            from thaieda.viz._interactive import create_correlation_heatmap_interactive

            num_df = self.df.select_dtypes(include="number")
            if len(num_df.columns) >= 2:
                plotly_heatmap = create_correlation_heatmap_interactive(num_df)
        except Exception:
            pass
        dist_charts = {
            "correlation_heatmap": dc.get("correlation_heatmap", ""),
            "correlation_heatmap_plotly": plotly_heatmap,
            "scatter_matrix": dc.get("scatter_matrix", ""),
            "boxplot": dc.get("boxplot", ""),
            "violinplot": dc.get("violinplot", ""),
        }
        missing_charts = {
            "missing_matrix": dc.get("missing_matrix", ""),
            "missing_heatmap": dc.get("missing_heatmap", ""),
        }

        # named entities — แนบ label ของประเภทไว้ใช้ในเทมเพลต
        ner_sections = [
            {"column": col, "result": result.to_dict()} for col, result in self._ner.items()
        ]

        # timeseries — สรุปผล + กราฟ (line/decomposition/acf) ต่อคอลัมน์
        timeseries_section = None
        if self._timeseries:
            timeseries_section = {
                "time_column": self._ts_time_col,
                # นับคอลัมน์ที่มีแนวโน้ม/ฤดูกาล ไว้แสดงเป็นแบนเนอร์สรุปด้านบน
                "trend_count": sum(1 for r in self._timeseries.values() if r.has_trend),
                "seasonal_count": sum(1 for r in self._timeseries.values() if r.has_seasonality),
                "columns": [
                    {
                        "column": col,
                        "result": r.to_dict(),
                        "charts": self._timeseries_charts.get(col, {}),
                    }
                    for col, r in self._timeseries.items()
                ],
            }

        # target analysis — แนบ label ของชนิดความสัมพันธ์
        target_section = None
        if self.target_column is not None:
            target_section = {
                "target_column": self.target_column,
                "associations": [
                    {**a.to_dict(), "type_label": L(f"assoc_{a.association_type}")}
                    for a in self._target_associations
                ],
            }

        # auto insights — แนบ label ของหมวดหมู่ไว้ใช้ในเทมเพลต
        insight_section = None
        insight_items: list[dict[str, Any]] = []
        if self._insights is not None:
            insight_items = [
                {**i.to_dict(), "category_label": L(f"icat_{i.category}")}
                for i in self._insights.insights
            ]
            insight_section = {
                "executive_summary_th": self._insights.executive_summary_th,
                "total_insights": self._insights.total_insights,
                # จำนวนข้อค้นพบทั้งหมดก่อนถูกตัด (P1) — ใช้บอกผู้อ่านว่าแสดงเพียงส่วนสำคัญ
                "total_generated": self._insights.total_generated,
                "critical_count": self._insights.critical_count,
                "warning_count": self._insights.warning_count,
                "info_count": self._insights.info_count,
                "insights": insight_items,
            }

        # cross-column insights (insight engine, v0.6) — แนบ label ของ pattern และกราฟ (v0.7)
        business_section = None
        business_cards: list[dict[str, Any]] = []
        if self._insight_engine is not None and self._insight_engine.cards:
            business_cards = [
                {
                    **c.to_dict(),
                    "pattern_label": L(f"pattern_{c.pattern}"),
                    "chart": self._insight_charts.get(i, ""),
                }
                for i, c in enumerate(self._insight_engine.cards)
            ]
            business_section = {
                "total": self._insight_engine.total,
                "cards": business_cards,
            }

        # cleaning diff — สรุปการทำความสะอาดที่ลงมือทำจริง (เมื่อ clean=True)
        cleaning_diff = [c.to_dict() for c in self._cleaning_diff]
        cleaning_diff_summary = None
        if self._cleaning_diff:
            total_changed = sum(c.rows_affected for c in self._cleaning_diff)
            top = max(self._cleaning_diff, key=lambda c: c.rows_affected)
            cleaning_diff_summary = {
                "total_cells_changed": total_changed,
                "most_impactful_op": top.operation,
                "most_impactful_th": top.description_th,
                "most_impactful_rows": top.rows_affected,
            }

        context = {
            "lang": lang,
            "L": L,
            "version": __version__,
            # ไอคอนตามความรุนแรง — ใช้ในการ์ด insight/quality/anomaly
            "sev_icons": {"critical": "🔴", "warning": "🟡", "info": "🔵"},
            "overview": self._overview,
            "data_type": self._data_type,
            "report_summary": self._build_report_summary(),
            "issue_summary": self._build_issue_summary(),
            "key_findings": self._top_findings(),
            "priority_actions": self._build_priority_actions(insight_items, business_cards),
            "top_columns_to_watch": self._build_top_columns_to_watch(),
            "type_distribution": type_distribution,
            "insight_section": insight_section,
            "business_section": business_section,
            "quality_issues": quality_issues,
            "anomalies": anomalies,
            "cleaning_diff": cleaning_diff,
            "cleaning_diff_summary": cleaning_diff_summary,
            "cleaning_suggestions": [c.to_dict() for c in self._cleaning],
            "columns": columns,
            "notes": self._notes,
            "dist_charts": dist_charts,
            "has_dist_charts": any(dist_charts.values()),
            "missing_charts": missing_charts,
            "has_missing_charts": any(missing_charts.values()),
            "ner_sections": ner_sections,
            "target_section": target_section,
            "timeseries_section": timeseries_section,
        }
        return template.render(**context)

    def _render_column_context(self, name: str, L) -> dict[str, Any]:
        ctype = self._column_types[name]
        is_text = name in self._text_metrics
        metrics = self._text_metrics.get(name)
        stats = self._basic_stats.get(name, {})

        # สถิติพื้นฐานแบบ (label, value) สำหรับ template
        basic_pairs: list[tuple[str, Any]] = [
            (L("non_null"), f"{stats.get('count', 0):,}"),
            (L("missing_cells"), f"{stats.get('missing', 0):,}"),
            (L("unique"), f"{stats.get('unique', 0):,}"),
        ]
        for key, lbl in (
            ("mean", "mean"),
            ("std", "std"),
            ("min", "min"),
            ("max", "max"),
        ):
            if key in stats:
                basic_pairs.append((L(lbl), stats[key]))

        return {
            "name": name,
            "type_key": ctype.value,
            "type_label": L(f"type_{ctype.value}"),
            "is_text": is_text,
            "metrics": metrics.to_dict() if metrics is not None else None,
            "charts": self._charts.get(name, {}),
            "dist_chart": self._dataset_charts.get(f"distribution::{name}", ""),
            "valuecounts_chart": self._dataset_charts.get(f"valuecounts::{name}", ""),
            "basic_stats": basic_pairs,
            "top_values": stats.get("top_values"),
        }


def profile(
    df: pd.DataFrame,
    lang: str = "th",
    tokenizer_engine: str = "auto",
    make_charts: bool = True,
    target_column: str | None = None,
    clean: bool = False,
    timeseries: bool = True,
    insights_engine: bool = True,
    insights_top: int = 8,
    progress: Callable[[str], None] | None = None,
) -> ProfileReport:
    """สร้าง ProfileReport และรันการวิเคราะห์ทันที — ฟังก์ชันอำนวยความสะดวกหลัก.

    ระบุ target_column เพื่อเพิ่มส่วน "การวิเคราะห์ตัวแปรเป้าหมาย" (ความสัมพันธ์ของทุกคอลัมน์กับเป้าหมาย)
    ระบุ clean=True เพื่อทำความสะอาดข้อความก่อนวิเคราะห์ และแสดงส่วน "การทำความสะอาด" (ก่อน/หลัง) ในรายงาน
    ระบุ timeseries=False เพื่อข้ามการวิเคราะห์อนุกรมเวลา (เร็วขึ้นบนข้อมูลที่ไม่ใช่ timeseries)
    ระบุ insights_engine=False เพื่อข้ามการค้นหาข้อค้นพบจากการผสมคอลัมน์ (cross-column insights)
    """
    report = ProfileReport(
        df,
        lang=lang,
        tokenizer_engine=tokenizer_engine,
        make_charts=make_charts,
        target_column=target_column,
        clean=clean,
        timeseries=timeseries,
        insights_engine=insights_engine,
        insights_top=insights_top,
        progress=progress,
    )
    report.run()
    return report


__all__ = ["ProfileReport", "profile", "_detect_data_type"]
