"""ThaiEDA — AutoEDA สำหรับข้อมูลภาษาไทย.

Exploratory data analysis that speaks Thai.

usage แบบ one-liner::

    import pandas as pd, thaieda
    df = pd.read_csv("data.csv")
    result = thaieda.run(df)              # ทำ EDA ครบทุกขั้นตอน + รายงาน HTML
    result.report.to_html("report.html")

    # เพิ่ม LLM analysis (ต้องติดตั้ง thaieda[all])
    result = thaieda.run(df, llm=True, privacy="insight_only")
    print(result.llm_response)

Alias: ``thaieda.EDA(df)`` เทียบเท่า ``thaieda.run(df)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

    from thaieda.report import ProfileReport

__version__ = "0.6.0"
__all__ = [
    "profile",
    "ProfileReport",
    "extract_entities",
    "analyze_target",
    "generate_insights",
    "Insight",
    "InsightSummary",
    "discover_insights",
    "InsightCard",
    "InsightEngineResult",
    "Perspective",
    "analyze_timeseries",
    "analyze_dataframe_timeseries",
    "detect_timeseries_columns",
    "TimeseriesResult",
    "TimeseriesComponent",
    "read_data",
    "detect_encoding",
    "detect_format",
    "profile_dataset",
    "DatasetProfile",
    "Relationship",
    "KeyCandidate",
    "TableProfile",
    "DatasetReport",
    "run",
    "EDA",
    "EDAResult",
    "__version__",
]


# ----------------------------------------------------------------------------
# EDAResult — ผลลัพธ์จาก one-liner API
# ----------------------------------------------------------------------------
@dataclass
class EDAResult:
    """ผลลัพธ์การวิเคราะห์ข้อมูลแบบครบวงจรจาก ``thaieda.run()``.

    Attributes:
        report: ProfileReport ที่รันการวิเคราะห์ครบแล้ว (detect, quality, insights, viz).
        llm_response: ข้อความตอบกลับจาก LLM (ถ้าเรียกด้วย llm=True) มิฉะนั้น None.
        notes: รายการหมายเหตุ/คำเตือนที่เกิดระหว่างการวิเคราะห์.

    ใช้ ``result.report`` เพื่อเข้าถึงข้อมูลทั้งหมด (overview, quality_issues,
    insights, anomaly, to_html, to_dict, ฯลฯ) หรือ ``result.cleaned_df`` เพื่อ
    ดึง DataFrame ที่ทำความสะอาดแล้ว (เมื่อ clean=True).
    """

    report: ProfileReport
    llm_response: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def cleaned_df(self) -> pd.DataFrame:
        """DataFrame ที่ทำความสะอาดแล้ว (เมื่อ clean=True) หรือ DataFrame เดิม."""
        return self.report.df

    @property
    def overview(self) -> dict[str, Any]:
        """ภาพรวมข้อมูล: แถว, คอลัมน์, เซลล์ว่าง, ประเภทคอลัมน์."""
        return self.report.overview

    @property
    def insights(self):
        """สรุปข้อค้นพบสำคัญอัตโนมัติ (InsightSummary) — None ถ้าไม่มี."""
        return self.report.insights

    @property
    def quality_issues(self):
        """รายการปัญหาคุณภาพข้อมูลที่ตรวจพบ."""
        return self.report.quality_issues

    @property
    def anomalies(self):
        """รายการความผิดปกติที่ตรวจพบ."""
        return self.report.anomalies

    def to_html(self, path: str | None = None) -> str:
        """สร้างรายงาน HTML — เรียก ``report.to_html()`` ให้โดยอัตโนมัติ."""
        return self.report.to_html(path)

    def to_dict(self) -> dict[str, Any]:
        """ส่งออกข้อมูลแบบมีโครงสร้าง — เรียก ``report.to_dict()`` ให้โดยอัตโนมัติ."""
        return self.report.to_dict()

    def to_json(self, path: str | None = None, indent: int = 2) -> str:
        """ส่งออกเป็น JSON — เรียก ``report.to_json()`` ให้โดยอัตโนมัติ."""
        return self.report.to_json(path, indent=indent)

    # ------------------------------------------------------------------
    # Jupyter rich display — แสดง report ใน notebook ได้เลย ไม่ต้องเปิด browser
    # ------------------------------------------------------------------
    def _repr_html_(self) -> str:
        """แสดง HTML report ใน Jupyter notebook โดยอัตโนมัติ.

        เมื่อพิมพ์ ``result`` ใน cell สุดท้ายของ Jupyter จะแสดง HTML report
        แบบย่อ (overview + quality + insights) โดยไม่ต้องเรียก ``to_html()``
        """
        return self.report.to_html()

    def _repr_pretty_(self, pp, cycle: bool) -> None:
        """แสดงสรุปแบบ text ใน IPython REPL."""
        overview = self.report.overview
        pp.text(f"ThaiEDA EDAResult({overview['rows']:,} rows × {overview['columns']} cols)")
        pp.text(f"  Quality issues: {len(self.report.quality_issues)}")
        if self.report.insights:
            pp.text(f"  Insights: {self.report.insights.total_insights}")
        if self.llm_response:
            pp.text(f"  LLM: ✓ ({len(self.llm_response)} chars)")
        pp.text("  → call .to_html('report.html') to save full report")


# ----------------------------------------------------------------------------
# run() — one-liner API หลัก
# ----------------------------------------------------------------------------
def run(
    df: pd.DataFrame,
    *,
    clean: bool = True,
    lang: str = "th",
    tokenizer_engine: str = "auto",
    make_charts: bool = True,
    target_column: str | None = None,
    timeseries: bool = True,
    insights_engine: bool = True,
    insights_top: int = 8,
    progress: Callable[[str], None] | None = None,
    llm: bool = False,
    privacy: str = "insight_only",
    provider: str = "openai",
    model: str | None = None,
    llm_language: str | None = None,
    epsilon: float = 1.0,
) -> EDAResult:
    """วิเคราะห์ข้อมูลแบบครบวงจรในบรรทัดเดียว — detect → clean → quality → insights → viz → report.

    ขั้นตอนที่ทำอัตโนมัติ:
      1. **detect** — ตรวจประเภทคอลัมน์ (ตัวเลข/ข้อความไทย/วันที่/หมวดหมู่/เบอร์/ID)
      2. **clean** — ทำความสะอาดข้อความไทย (encoding, zero-width, เลขไทย, ช่องว่าง) เมื่อ ``clean=True``
      3. **quality** — ตรวจสอบคุณภาพข้อมูล (Buddhist Era, placeholder, constant, ฯลฯ)
      4. **insights** — สรุปข้อค้นพบสำคัญ + cross-column insight engine (group-by + statistical scoring)
      5. **viz** — สร้างกราฟ (correlation, distribution,
         word cloud, timeseries) เมื่อ ``make_charts=True``
      6. **report** — สร้าง ProfileReport พร้อมเรนเดอร์ HTML/JSON
      7. **LLM** (ถ้า ``llm=True``) — วิเคราะห์ด้วย LLM โดยปกปิดข้อมูลส่วนบุคคลตามโหมด ``privacy``

    Args:
        df: DataFrame ที่จะวิเคราะห์.
        clean: ทำความสะอาดข้อความก่อนวิเคราะห์ (default: True).
        lang: ภาษาของรายงาน — "th" (default) | "en".
        tokenizer_engine: เครื่องมือตัดคำ — "auto" (default) | "pythainlp" | "nlpo3" | "attacut".
        make_charts: สร้างกราฟ (default: True — ตั้ง False เพื่อความเร็ว).
        target_column: คอลัมน์เป้าหมายสำหรับการวิเคราะห์ความสัมพันธ์ (optional).
        timeseries: วิเคราะห์อนุกรมเวลา (default: True).
        insights_engine: ค้นหาข้อค้นพบจากการผสมคอลัมน์ (default: True).
        insights_top: จำนวนข้อค้นพบสูงสุดที่แสดง (default: 8).
        progress: callback แสดงความคืบหน้า (optional).
        llm: เรียก LLM analysis หลังวิเคราะห์เสร็จ (default: False).
        privacy: โหมดความเป็นส่วนตัวของ LLM — "insight_only" (default) | "anonymized" | "dp_noise"
            | "full".
        provider: ผู้ให้บริการ LLM — "openai" (default) | "anthropic" | "ollama".
        model: ชื่อโมเดล LLM (None = default ของ provider).
        llm_language: ภาษาของ LLM prompt — None = ใช้ค่าเดียวกับ ``lang``.
        epsilon: พารามิเตอร์ epsilon สำหรับ dp_noise (default: 1.0).

    Returns:
        EDAResult — มี ``.report`` (ProfileReport), ``.llm_response`` (ถ้า llm=True), ``.notes``

    Raises:
        TypeError: ถ้า ``df`` ไม่ใช่ pandas DataFrame.
        KeyError: ถ้า ``target_column`` ไม่มีใน DataFrame.
        ImportError: ถ้าเรียก ``llm=True`` แต่ไม่ได้ติดตั้ง ``thaieda[llm]`` หรือ ``thaieda[all]``.

    Example::

        >>> import pandas as pd, thaieda
        >>> df = pd.DataFrame({"review": ["อร่อยมาก"], "rating": [5]})
        >>> result = thaieda.run(df)
        >>> result.overview
        {'rows': 1, 'columns': 2, ...}
        >>> result.report.to_html("report.html")
    """
    # lazy import — ไม่โหลด report module จนกว่าจะเรียก run()
    from thaieda.report import profile

    if not isinstance(df, pd.DataFrame):
        raise TypeError("thaieda.run() ต้องรับ pandas DataFrame.")

    report = profile(
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

    notes: list[str] = list(report.notes)

    llm_response: str | None = None
    if llm:
        from thaieda.llm import analyze_with_llm

        llm_response = analyze_with_llm(
            report.df,
            privacy=privacy,
            provider=provider,
            model=model,
            language=llm_language or lang,
            insights=[i.to_dict() for i in report.insights.insights] if report.insights else None,
            epsilon=epsilon,
        )

    return EDAResult(report=report, llm_response=llm_response, notes=notes)


# alias — thaieda.EDA(df) เทียบเท่า thaieda.run(df)
EDA = run


def __getattr__(name: str):
    """Lazy import เพื่อให้ core ไม่ต้องโหลด dependencies หนักทั้งหมด."""
    if name == "profile":
        from thaieda.report import profile

        return profile
    if name == "ProfileReport":
        from thaieda.report import ProfileReport

        return ProfileReport
    if name in ("extract_entities", "NERResult", "NEREntity", "ner_available"):
        import thaieda.ner as _ner

        return getattr(_ner, name)
    if name in ("analyze_target", "TargetAssociation"):
        import thaieda.analysis as _analysis

        return getattr(_analysis, name)
    if name in ("generate_insights", "Insight", "InsightSummary"):
        import thaieda.insight as _insight

        return getattr(_insight, name)
    if name in (
        "discover_insights",
        "InsightCard",
        "InsightEngineResult",
        "Perspective",
    ):
        import thaieda.insight_engine as _insight_engine

        return getattr(_insight_engine, name)
    if name in (
        "analyze_timeseries",
        "analyze_dataframe_timeseries",
        "detect_timeseries_columns",
        "TimeseriesResult",
        "TimeseriesComponent",
    ):
        import thaieda.timeseries as _timeseries

        return getattr(_timeseries, name)
    if name in ("read_data", "detect_encoding", "detect_format"):
        import thaieda.io as _io

        return getattr(_io, name)
    if name in (
        "profile_dataset",
        "DatasetProfile",
        "Relationship",
        "KeyCandidate",
        "TableProfile",
    ):
        import thaieda.schema as _schema

        return getattr(_schema, name)
    if name == "DatasetReport":
        from thaieda.report._dataset import DatasetReport

        return DatasetReport
    raise AttributeError(f"module 'thaieda' has no attribute {name!r}")
