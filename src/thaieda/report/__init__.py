"""Report generation — ประกอบทุกส่วนเป็นรายงาน HTML/JSON/dict.

ProfileReport เป็นจุดเชื่อมหลัก: ตรวจประเภทคอลัมน์ -> ตรวจคุณภาพ -> สถิติข้อความ
-> สร้างกราฟ -> เรนเดอร์ HTML แบบ self-contained (CSS ฝัง, รูป base64)
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from thaieda import __version__
from thaieda.analysis import TargetAssociation, analyze_target
from thaieda.anomaly import AnomalyIssue, detect_anomalies
from thaieda.clean import CleaningResult, clean_thai_text
from thaieda.detect import ColumnType, detect_all
from thaieda.i18n import label
from thaieda.insight import InsightSummary, generate_insights
from thaieda.ner import NERResult, extract_entities, ner_available
from thaieda.quality import QualityIssue, run_quality_checks
from thaieda.report._template import REPORT_TEMPLATE
from thaieda.text import TextMetrics, text_metrics

# ประเภทที่ต้องวิเคราะห์ข้อความ (ต้องใช้ tokenizer)
_TEXT_METRIC_TYPES = {ColumnType.THAI_TEXT, ColumnType.MIXED_TEXT}
# ประเภทที่ถือว่าเป็นข้อความสำหรับสร้างกราฟ
_TEXT_TYPES = {ColumnType.THAI_TEXT, ColumnType.MIXED_TEXT, ColumnType.ENGLISH_TEXT}
# ประเภทคอลัมน์ที่ควรลองเสนอการทำความสะอาดข้อความ
_CLEANABLE_TYPES = {
    ColumnType.THAI_TEXT,
    ColumnType.MIXED_TEXT,
    ColumnType.ENGLISH_TEXT,
    ColumnType.CATEGORICAL,
    ColumnType.ID,
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
        self._insights: InsightSummary | None = None
        self._overview: dict[str, Any] = {}
        self._charts: dict[str, dict[str, str]] = {}
        # กราฟระดับชุดข้อมูล (correlation/scatter/box/violin/missing) จาก auto_select_charts
        self._dataset_charts: dict[str, str] = {}
        self._basic_stats: dict[str, dict[str, Any]] = {}
        self._notes: list[str] = []

    # ------------------------------------------------------------------ run
    def run(self) -> ProfileReport:
        """รันการวิเคราะห์ทั้งหมด (idempotent — เรียกซ้ำได้)."""
        # ถ้า clean=True ทำความสะอาดข้อความก่อน แล้ววิเคราะห์ข้อมูลที่สะอาดแล้ว (เก็บ diff ไว้แสดง)
        if self.clean:
            self._apply_cleaning()

        self._column_types = detect_all(self.df)
        self._overview = self._compute_overview()
        self._quality_issues = run_quality_checks(self.df, self._column_types)

        # text metrics — ต้องใช้ tokenizer เฉพาะเมื่อมีคอลัมน์ข้อความไทย/ผสม
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
        self._anomalies = detect_anomalies(self.df, self._column_types, tokenizer)
        self._note_if_ml_skipped()

        # Named entities (NER) — เฉพาะคอลัมน์ข้อความไทย และเมื่อมี NER engine ที่ใช้ได้
        self._compute_ner(thai_cols, tokenizer)

        # การวิเคราะห์ตัวแปรเป้าหมาย (target analysis) — เมื่อผู้ใช้ระบุ target_column
        self._compute_target_analysis()

        # คำแนะนำการทำความสะอาด (dry-run — ไม่แก้ข้อมูลจริง)
        self._cleaning = self._compute_cleaning_suggestions()

        # สถิติพื้นฐานของทุกคอลัมน์
        for col in self.df.columns:
            self._basic_stats[str(col)] = self._compute_basic_stats(str(col))

        # สรุปข้อค้นพบสำคัญอัตโนมัติ (ตีความผลทั้งหมดเป็นภาษาไทย) — ทำหลังวิเคราะห์ครบ
        self._insights = generate_insights(
            self.df,
            self._quality_issues,
            self._anomalies,
            self._text_metrics,
            target_associations=self._target_associations,
            cleaning_results=self._cleaning_diff,
            column_types=self._column_types,
        )

        # กราฟ
        if self.make_charts:
            # กราฟต่อคอลัมน์ข้อความ (word cloud/top tokens/length) — ต้องมี tokenizer
            if tokenizer is not None:
                self._build_charts(tokenizer)
            # กราฟระดับชุดข้อมูล (correlation/box/violin/missing/distribution) — ไม่ต้องใช้ tokenizer
            self._build_dataset_charts()

        self._ran = True
        return self

    def _note_if_ml_skipped(self) -> None:
        """ถ้ามีคอลัมน์ตัวเลขใหญ่ (>100 แถว) แต่ไม่มี scikit-learn ให้บันทึก note ว่าข้ามวิธี ML."""
        from thaieda.anomaly import sklearn_available

        if sklearn_available():
            return
        big_numeric = [
            str(c)
            for c in self.df.columns
            if self._column_types.get(str(c)) == ColumnType.NUMERIC
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

    def _apply_cleaning(self) -> None:
        """ทำความสะอาดข้อความทุกคอลัมน์ (object/string) จริง — แทนที่ self.df และเก็บ diff.

        ใช้ DEFAULT_OPERATIONS ของ clean_thai_text (แก้ encoding/zw/ช่องว่าง/normalize ฯลฯ)
        เก็บเฉพาะการดำเนินการที่มีผล (>0 แถว) ไว้ใน self._cleaning_diff เพื่อแสดงก่อน/หลังในรายงาน
        """
        cleaned_df = self.df.copy()
        diffs: list[CleaningResult] = []
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
                self._notes.append(f"cleaning failed for '{col}': {exc}")
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
            try:
                _, results = clean_thai_text(self.df[col])
            except Exception as exc:  # noqa: BLE001 — การทำความสะอาดพังไม่ควรล้มทั้งรายงาน
                self._notes.append(f"cleaning suggestions failed for '{col}': {exc}")
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
            dt = pd.to_datetime(series, errors="coerce").dropna()
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
        for col, metrics in self._text_metrics.items():
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
            "column_types": {k: v.value for k, v in self._column_types.items()},
            "insights": self._insights.to_dict() if self._insights is not None else None,
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

        # ความผิดปกติ — แนบ label ของประเภท (anomaly_type) ไว้ใช้ในเทมเพลต
        anomalies = []
        for a in self._anomalies:
            entry = a.to_dict()
            entry["type_label"] = L(f"antype_{a.anomaly_type}")
            anomalies.append(entry)

        dc = self._dataset_charts
        dist_charts = {
            "correlation_heatmap": dc.get("correlation_heatmap", ""),
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
        if self._insights is not None:
            insight_section = {
                "executive_summary_th": self._insights.executive_summary_th,
                "total_insights": self._insights.total_insights,
                "critical_count": self._insights.critical_count,
                "warning_count": self._insights.warning_count,
                "info_count": self._insights.info_count,
                "insights": [
                    {**i.to_dict(), "category_label": L(f"icat_{i.category}")}
                    for i in self._insights.insights
                ],
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
            "overview": self._overview,
            "type_distribution": type_distribution,
            "insight_section": insight_section,
            "quality_issues": [i.to_dict() for i in self._quality_issues],
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
) -> ProfileReport:
    """สร้าง ProfileReport และรันการวิเคราะห์ทันที — ฟังก์ชันอำนวยความสะดวกหลัก.

    ระบุ target_column เพื่อเพิ่มส่วน "การวิเคราะห์ตัวแปรเป้าหมาย" (ความสัมพันธ์ของทุกคอลัมน์กับเป้าหมาย)
    ระบุ clean=True เพื่อทำความสะอาดข้อความก่อนวิเคราะห์ และแสดงส่วน "การทำความสะอาด" (ก่อน/หลัง) ในรายงาน
    """
    report = ProfileReport(
        df,
        lang=lang,
        tokenizer_engine=tokenizer_engine,
        make_charts=make_charts,
        target_column=target_column,
        clean=clean,
    )
    report.run()
    return report


__all__ = ["ProfileReport", "profile"]
