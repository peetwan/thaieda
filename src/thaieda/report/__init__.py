"""Report generation — ประกอบทุกส่วนเป็นรายงาน HTML/JSON/dict.

ProfileReport เป็นจุดเชื่อมหลัก: ตรวจประเภทคอลัมน์ -> ตรวจคุณภาพ -> สถิติข้อความ
-> สร้างกราฟ -> เรนเดอร์ HTML แบบ self-contained (CSS ฝัง, รูป base64)
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from thaieda import __version__
from thaieda.anomaly import AnomalyIssue, detect_anomalies
from thaieda.clean import CleaningResult, clean_thai_text
from thaieda.detect import ColumnType, detect_all
from thaieda.i18n import label
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
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("ProfileReport expects a pandas DataFrame.")
        self.df = df
        self.lang = lang
        self.tokenizer_engine = tokenizer_engine
        self.max_sample = max_sample
        self.make_charts = make_charts

        self._ran = False
        self._column_types: dict[str, ColumnType] = {}
        self._quality_issues: list[QualityIssue] = []
        self._anomalies: list[AnomalyIssue] = []
        self._cleaning: list[CleaningResult] = []
        self._text_metrics: dict[str, TextMetrics] = {}
        self._overview: dict[str, Any] = {}
        self._charts: dict[str, dict[str, str]] = {}
        self._basic_stats: dict[str, dict[str, Any]] = {}
        self._notes: list[str] = []

    # ------------------------------------------------------------------ run
    def run(self) -> ProfileReport:
        """รันการวิเคราะห์ทั้งหมด (idempotent — เรียกซ้ำได้)."""
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

        # คำแนะนำการทำความสะอาด (dry-run — ไม่แก้ข้อมูลจริง)
        self._cleaning = self._compute_cleaning_suggestions()

        # สถิติพื้นฐานของทุกคอลัมน์
        for col in self.df.columns:
            self._basic_stats[str(col)] = self._compute_basic_stats(str(col))

        # กราฟ (เฉพาะคอลัมน์ข้อความที่มี metrics)
        if self.make_charts and tokenizer is not None:
            self._build_charts(tokenizer)

        self._ran = True
        return self

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

        return {
            "thaieda_version": __version__,
            "overview": self._overview,
            "column_types": {k: v.value for k, v in self._column_types.items()},
            "quality_issues": [i.to_dict() for i in self._quality_issues],
            "anomalies": [a.to_dict() for a in self._anomalies],
            "cleaning_suggestions": [c.to_dict() for c in self._cleaning],
            "columns": columns,
            "notes": self._notes,
        }

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

        context = {
            "lang": lang,
            "L": L,
            "version": __version__,
            "overview": self._overview,
            "type_distribution": type_distribution,
            "quality_issues": [i.to_dict() for i in self._quality_issues],
            "anomalies": anomalies,
            "cleaning_suggestions": [c.to_dict() for c in self._cleaning],
            "columns": columns,
            "notes": self._notes,
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
            "basic_stats": basic_pairs,
            "top_values": stats.get("top_values"),
        }


def profile(
    df: pd.DataFrame,
    lang: str = "th",
    tokenizer_engine: str = "auto",
    make_charts: bool = True,
) -> ProfileReport:
    """สร้าง ProfileReport และรันการวิเคราะห์ทันที — ฟังก์ชันอำนวยความสะดวกหลัก."""
    report = ProfileReport(
        df, lang=lang, tokenizer_engine=tokenizer_engine, make_charts=make_charts
    )
    report.run()
    return report


__all__ = ["ProfileReport", "profile"]
