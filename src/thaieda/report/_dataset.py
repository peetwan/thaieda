"""Dataset report — รายงาน HTML รวมหลายตาราง + ความสัมพันธ์ระหว่างตาราง (v0.5).

DatasetReport รับ DatasetProfile (จาก thaieda.schema.profile_dataset) แล้วเรนเดอร์เป็น
รายงาน HTML แบบ self-contained: ภาพรวม schema, แผนผัง ER (Mermaid.js ผ่าน CDN),
สรุปแต่ละตาราง, ตารางความสัมพันธ์, และรายการข้อมูลกำพร้า — ใช้ธีม/สไตล์เดียวกับ ProfileReport
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from thaieda import __version__
from thaieda.i18n import label
from thaieda.report._dataset_template import DATASET_TEMPLATE

if TYPE_CHECKING:
    from thaieda.schema import DatasetProfile


# ความรุนแรงของความสัมพันธ์ตามความมั่นใจ — ใช้เลือกสีขอบการ์ด
def _confidence_class(confidence: float) -> str:
    """แปลงคะแนนความมั่นใจเป็นระดับสี (info/warning) สำหรับการแสดงผล."""
    if confidence >= 0.9:
        return "info"
    if confidence >= 0.7:
        return "warning"
    return "critical"


class DatasetReport:
    """รายงาน HTML รวมหลายตาราง + ความสัมพันธ์ระหว่างตาราง."""

    def __init__(self, dataset: DatasetProfile, lang: str = "th") -> None:
        self.dataset = dataset
        self.lang = lang

    # --------------------------------------------------------------- export
    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        """ส่งออกข้อมูลชุดข้อมูลเป็น JSON (รวมข้อความ Mermaid)."""
        data = self.dataset.to_dict()
        data["thaieda_version"] = __version__
        data["mermaid"] = self.dataset.to_mermaid()
        text = json.dumps(data, ensure_ascii=False, indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_html(self, path: str | None = None) -> str:
        """เรนเดอร์เป็น HTML (เขียนไฟล์ด้วยถ้าระบุ path) — คืน HTML string."""
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
        template = env.from_string(DATASET_TEMPLATE)
        lang = self.lang

        def L(key: str) -> str:
            return label(key, lang)

        ds = self.dataset

        # ภาพรวม
        total_rows = sum(t.row_count for t in ds.tables)
        overview = {
            "table_count": len(ds.tables),
            "relationship_count": len(ds.relationships),
            "orphan_count": len(ds.orphan_findings),
            "total_rows": total_rows,
        }

        # ตารางแต่ละตาราง — แนบ type distribution + key candidates ที่อ่านง่าย
        tables_ctx: list[dict[str, Any]] = []
        for t in ds.tables:
            type_counts: dict[str, int] = {}
            for ctype in t.column_types.values():
                type_counts[ctype] = type_counts.get(ctype, 0) + 1
            type_dist = [
                (tk, L(f"type_{tk}"), cnt)
                for tk, cnt in sorted(type_counts.items(), key=lambda x: -x[1])
            ]
            keys = [
                {
                    "column": k.column,
                    "is_unique": k.is_unique,
                    "name_hint": k.name_hint,
                    "cardinality": k.cardinality,
                    "dtype": k.dtype,
                    "role": L("primary_key") if k.is_unique else L("foreign_key"),
                }
                for k in t.key_candidates
            ]
            tables_ctx.append(
                {
                    "name": t.name,
                    "row_count": t.row_count,
                    "column_count": t.column_count,
                    "type_dist": type_dist,
                    "key_candidates": keys,
                    "notes": t.notes,
                }
            )

        # ความสัมพันธ์ — แนบ label/สี
        rels_ctx = []
        for r in ds.relationships:
            rels_ctx.append(
                {
                    **r.to_dict(),
                    "conf_class": _confidence_class(r.confidence),
                    "conf_pct": round(r.confidence * 100),
                    "overlap_pct": round(r.overlap_ratio * 100, 1),
                    "method_label": L("validated") if r.is_validated else L("name_only"),
                }
            )

        context = {
            "lang": lang,
            "L": L,
            "version": __version__,
            "overview": overview,
            "mermaid": ds.to_mermaid(),
            "tables": tables_ctx,
            "relationships": rels_ctx,
            "orphan_findings": ds.orphan_findings,
            "notes": ds.notes,
        }
        return template.render(**context)


__all__ = ["DatasetReport"]
