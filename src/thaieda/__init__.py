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

    # วิเคราะห์ทุกไฟล์ในโฟลเดอร์ (CSV/Excel/JSON) — one-liner
    results = thaieda.run_folder("data/")  # สร้าง report ให้ทุกไฟล์

Alias: ``thaieda.EDA(df)`` เทียบเท่า ``thaieda.run(df)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    "run_folder",
    "FolderResult",
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


# ----------------------------------------------------------------------------
# FolderResult — ผลลัพธ์จาก run_folder()
# ----------------------------------------------------------------------------

import re as _re


def _extract_body(html: str) -> str:
    """ดึงเฉพาะเนื้อหาใน <body> จาก HTML report เต็ม."""
    match = _re.search(r"<body[^>]*>(.*)</body>", html, _re.DOTALL | _re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # fallback: ถ้าไม่มี <body> คืนทั้งหมด
    return html


_MASTER_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ThaiEDA Master Report — {folder}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Sarabun', 'Segoe UI', sans-serif; background: #f5f5f5; }}
  .layout {{ display: flex; min-height: 100vh; }}
  /* Sidebar */
  .sidebar {{
    width: 250px; background: #1a1a2e; color: #eee;
    padding: 20px 15px; position: fixed; top: 0; left: 0;
    height: 100vh; overflow-y: auto;
  }}
  .sidebar h1 {{ font-size: 18px; margin-bottom: 15px; color: #e94560; }}
  .sidebar ul {{ list-style: none; }}
  .sidebar li {{ margin: 8px 0; }}
  .sidebar a {{ color: #eee; text-decoration: none; font-size: 14px; }}
  .sidebar a:hover {{ color: #e94560; }}
  .sidebar .stats {{ margin-top: 20px; padding-top: 15px; border-top: 1px solid #333; font-size: 13px; color: #aaa; }}
  /* Main content */
  .main {{ margin-left: 250px; padding: 30px; flex: 1; }}
  .overview {{ background: #fff; border-radius: 10px; padding: 25px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .overview h2 {{ margin-bottom: 15px; color: #1a1a2e; }}
  .overview table {{ width: 100%; border-collapse: collapse; }}
  .overview th, .overview td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 14px; }}
  .overview th {{ background: #f8f8f8; font-weight: 600; }}
  .overview a {{ color: #e94560; text-decoration: none; }}
  .overview a:hover {{ text-decoration: underline; }}
  /* File sections */
  .file-section {{
    background: #fff; border-radius: 10px; padding: 30px;
    margin-bottom: 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  .file-section h2 {{
    font-size: 24px; color: #1a1a2e; margin-bottom: 8px;
    border-bottom: 3px solid #e94560; padding-bottom: 8px;
  }}
  .file-section .meta {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
  .file-section .error {{ color: #e94560; font-size: 16px; padding: 20px; }}
  /* Embed individual report styles */
  .file-section table {{ border-collapse: collapse; }}
  .file-section th, .file-section td {{ padding: 8px; }}
</style>
<link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap" rel="stylesheet">
</head>
<body>
<div class="layout">
  <nav class="sidebar">
    <h1>📋 ThaiEDA</h1>
    <p style="font-size:12px;color:#888;margin-bottom:10px;">{folder}</p>
    <ul>
      <li><a href="#overview">📊 ภาพรวม</a></li>
      {nav}
    </ul>
    <div class="stats">
      <p>ไฟล์ทั้งหมด: {total}</p>
      <p>✅ สำเร็จ: {success}</p>
      <p>❌ พัง: {failed}</p>
    </div>
  </nav>
  <main class="main">
    <div class="overview" id="overview">
      <h2>📊 ภาพรวมทุกไฟล์</h2>
      <table>
        <thead>
          <tr><th>ไฟล์</th><th>แถว</th><th>คอลัมน์</th><th>Insights</th><th>Quality Issues</th><th>สถานะ</th></tr>
        </thead>
        <tbody>
          {summary_table}
        </tbody>
      </table>
    </div>
    {sections}
  </main>
</div>
</body>
</html>"""


@dataclass
class _FileResult:
    """ผลลัพธ์ของไฟล์เดียวในโฟลเดอร์."""

    filename: str
    result: EDAResult | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.error is None


@dataclass
class FolderResult:
    """ผลลัพธ์การวิเคราะห์ทุกไฟล์ในโฟลเดอร์ด้วย ``thaieda.run_folder()``.

    Attributes:
        folder: พาธโฟลเดอร์ที่วิเคราะห์.
        results: รายการผลลัพธ์แต่ละไฟล์ (list[_FileResult]).
        total_files: จำนวนไฟล์ทั้งหมดที่พบ.
        success: จำนวนไฟล์ที่วิเคราะห์สำเร็จ.
        failed: จำนวนไฟล์ที่พัง.
    """

    folder: str
    results: list[_FileResult] = field(default_factory=list)
    total_files: int = 0
    success: int = 0
    failed: int = 0

    def to_html(self, output_dir: str | None = None) -> list[str]:
        """บันทึก HTML report ให้ทุกไฟล์ที่สำเร็จ — คืน list ของพาธที่บันทึก.

        Args:
            output_dir: โฟลเดอร์ปลายทาง (None = โฟลเดอร์เดียวกับที่อ่าน).
        """
        out = Path(output_dir) if output_dir else Path(self.folder)
        out.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for fr in self.results:
            if fr.ok and fr.result:
                stem = Path(fr.filename).stem
                p = out / f"{stem}-report.html"
                fr.result.to_html(str(p))
                saved.append(str(p))
        return saved

    def to_master_html(self, path: str | None = None) -> str:
        """สร้าง master HTML ที่รวมทุกไฟล์เป็นหน้าเดียว — มี sidebar นำทาง.

        แต่ละไฟล์จะอยู่ใน section ของตัวเอง มี anchor link ให้คลิกข้ามได้
        รวมสรุปภาพรวมที่ด้านบน (ตารางสรุปทุกไฟล์)

        Args:
            path: พาธไฟล์ HTML ปลายทาง (None = คืน HTML string ไม่บันทึก).

        Returns:
            HTML string ของ master report.

        Example::

            >>> results = thaieda.run_folder("data/")
            >>> results.to_master_html("master-report.html")
        """
        sections: list[str] = []
        nav_items: list[str] = []
        summary_rows: list[str] = []

        for i, fr in enumerate(self.results):
            anchor = f"file-{i}"
            if fr.ok and fr.result:
                ov = fr.result.overview
                rows = ov.get("rows", 0)
                cols = ov.get("columns", 0)
                n_insights = fr.result.insights.total_insights if fr.result.insights else 0
                n_quality = len(fr.result.quality_issues) if fr.result.quality_issues else 0

                # Summary row
                summary_rows.append(
                    f"<tr>"
                    f"<td><a href='#{anchor}'>{fr.filename}</a></td>"
                    f"<td>{rows:,}</td><td>{cols}</td>"
                    f"<td>{n_insights}</td><td>{n_quality}</td>"
                    f"<td>✅</td></tr>"
                )

                # Nav item
                nav_items.append(
                    f"<li><a href='#{anchor}'>{fr.filename}</a></li>"
                )

                # Section: extract body content from individual report
                full_html = fr.result.report.to_html()
                # ดึงเฉพาะ body content (ตัด <html>, <head>, ฯลฯ)
                body = _extract_body(full_html)
                sections.append(
                    f"<section id='{anchor}' class='file-section'>"
                    f"<h2>{fr.filename}</h2>"
                    f"<p class='meta'>{rows:,} rows × {cols} cols · {n_insights} insights · {n_quality} quality issues</p>"
                    f"{body}"
                    f"</section>"
                )
            else:
                err = fr.error or "ไม่ทราบสาเหตุ"
                summary_rows.append(
                    f"<tr><td>{fr.filename}</td><td colspan='4'>—</td><td>❌ {err}</td></tr>"
                )
                nav_items.append(f"<li><a href='#{anchor}'>{fr.filename} ❌</a></li>")
                sections.append(
                    f"<section id='{anchor}' class='file-section'>"
                    f"<h2>{fr.filename}</h2><p class='error'>❌ {err}</p>"
                    f"</section>"
                )

        master_html = _MASTER_HTML_TEMPLATE.format(
            folder=self.folder,
            total=self.total_files,
            success=self.success,
            failed=self.failed,
            nav="\n".join(nav_items),
            summary_table="\n".join(summary_rows),
            sections="\n".join(sections),
        )

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(master_html, encoding="utf-8")

        return master_html

    def summary(self) -> str:
        """สรุปผลแบบ text."""
        lines = [
            f"ThaiEDA FolderResult — {self.folder}",
            f"  Files: {self.total_files} (✅ {self.success} / ❌ {self.failed})",
        ]
        for fr in self.results:
            status = "✅" if fr.ok else "❌"
            info = ""
            if fr.ok and fr.result:
                ov = fr.result.overview
                info = f" — {ov.get('rows', 0):,} rows × {ov.get('columns', 0)} cols"
                if fr.result.insights:
                    info += f", {fr.result.insights.total_insights} insights"
            elif fr.error:
                info = f" — {fr.error}"
            lines.append(f"  {status} {fr.filename}{info}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """แสดงสรุปใน Jupyter."""
        rows_html = []
        for fr in self.results:
            status = "✅" if fr.ok else "❌"
            info = ""
            if fr.ok and fr.result:
                ov = fr.result.overview
                info = f"{ov.get('rows', 0):,} rows × {ov.get('columns', 0)} cols"
            elif fr.error:
                info = fr.error
            rows_html.append(
                f"<tr><td>{status}</td><td>{fr.filename}</td><td>{info}</td></tr>"
            )
        return (
            f"<h3>ThaiEDA Folder Report — {self.folder}</h3>"
            f"<p>{self.success}/{self.total_files} files analyzed</p>"
            f"<table><tr><th></th><th>File</th><th>Info</th></tr>{''.join(rows_html)}</table>"
        )


# ----------------------------------------------------------------------------
# run_folder() — one-liner สำหรับวิเคราะห์ทุกไฟล์ในโฟลเดอร์
# ----------------------------------------------------------------------------
_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".tsv"}

# พารามิเตอร์ที่ส่งต่อไปยัง run()
_RUN_PARAMS = (
    "clean",
    "lang",
    "tokenizer_engine",
    "make_charts",
    "target_column",
    "timeseries",
    "insights_engine",
    "insights_top",
    "llm",
    "privacy",
    "provider",
    "model",
    "llm_language",
    "epsilon",
)


def run_folder(
    folder: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
    output_dir: str | None = None,
    save_html: bool = True,
    progress: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> FolderResult:
    """วิเคราะห์ทุกไฟล์ในโฟลเดอร์ในบรรทัดเดียว — อ่าน → run() → บันทึก HTML.

    สแกนโฟลเดอร์หาไฟล์ CSV/Excel/JSON แล้วรัน ``thaieda.run()`` ให้ทุกไฟล์
    พร้อมบันทึก HTML report อัตโนมัติ.

    Args:
        folder: พาธโฟลเดอร์ที่จะวิเคราะห์.
        pattern: glob pattern สำหรับกรองชื่อไฟล์ (default: ``"*"`` = ทุกไฟล์ที่รองรับ).
        recursive: ค้นหาในโฟลเดอร์ย่อยด้วย (default: False).
        output_dir: โฟลเดอร์สำหรับบันทึก HTML (None = โฟลเดอร์เดียวกับที่อ่าน).
        save_html: บันทึก HTML report อัตโนมัติ (default: True).
        progress: callback สำหรับแสดงความคืบหน้า (optional).
        **kwargs: พารามิเตอร์อื่น ๆ ส่งต่อให้ ``thaieda.run()`` (clean, lang, llm, ฯลฯ).

    Returns:
        FolderResult — มี ``.results``, ``.summary()``, ``.to_html()``.

    Raises:
        FileNotFoundError: ถ้าไม่พบโฟลเดอร์.
        ValueError: ถ้าไม่พบไฟล์ที่รองรับในโฟลเดอร์.

    Example::

        >>> import thaieda
        >>> results = thaieda.run_folder("data/")
        >>> print(results.summary())
        >>> results.to_html("reports/")
    """
    from thaieda.io import read_data

    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {folder_path}")

    # สแกนไฟล์
    files: list[Path] = []
    glob_fn = folder_path.rglob if recursive else folder_path.glob
    for p in glob_fn(pattern):
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
            files.append(p)

    if not files:
        raise ValueError(
            f"ไม่พบไฟล์ที่รองรับ ({', '.join(sorted(_SUPPORTED_EXTENSIONS))}) "
            f"ในโฟลเดอร์: {folder_path}"
        )

    files.sort(key=lambda x: x.name)

    # กรอง kwargs เฉพาะที่ run() รองรับ
    run_kwargs = {k: v for k, v in kwargs.items() if k in _RUN_PARAMS}

    file_results: list[_FileResult] = []
    for i, fpath in enumerate(files):
        fname = fpath.name
        if progress:
            progress(f"[{i + 1}/{len(files)}] {fname}")
        elif i == 0:
            # ถ้าไม่มี progress callback ก็ไม่ print อะไร
            pass

        try:
            df = read_data(fpath)
            if df.empty:
                file_results.append(_FileResult(fname, None, "ไฟล์ว่าง (0 rows)"))
                continue
            res = run(df, progress=progress, **run_kwargs)
            file_results.append(_FileResult(fname, res, None))
        except Exception as exc:  # noqa: BLE001 — เก็บ error ไว้ ไม่พังทั้งโฟลเดอร์
            file_results.append(_FileResult(fname, None, str(exc)))

    result = FolderResult(
        folder=str(folder_path),
        results=file_results,
        total_files=len(file_results),
        success=sum(1 for fr in file_results if fr.ok),
        failed=sum(1 for fr in file_results if not fr.ok),
    )

    if save_html:
        result.to_html(output_dir)

    return result


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
