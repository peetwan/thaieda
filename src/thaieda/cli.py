"""Command-line interface สำหรับ ThaiEDA.

ตัวอย่าง:
    # วิเคราะห์ → รายงาน HTML
    thaieda profile data.csv -o report.html --lang th --tokenizer auto

    # ครบจบในคำสั่งเดียว: ทำความสะอาด → วิเคราะห์ → รายงาน + ไฟล์ที่สะอาดแล้ว
    thaieda run data.csv -o report.html --cleaned-output cleaned.csv

    # ทำความสะอาดอย่างเดียว
    thaieda clean data.csv -o cleaned.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from thaieda import __version__
from thaieda.i18n import label

# ตัวเลือก format ที่รองรับสำหรับการอ่านไฟล์
_FORMAT_CHOICES = ["auto", "csv", "json", "jsonl"]

# นามสกุลไฟล์ที่นับเป็น "ตาราง" เมื่อสแกนไดเรกทอรี (โหมด dataset หลายตาราง)
_DATASET_EXTS = (".csv", ".tsv", ".json", ".jsonl", ".ndjson")

# ขนาดไฟล์ที่ถือว่า "ใหญ่" (10MB) — แสดงจำนวนแถวหลังอ่าน และแนะนำ flag เพื่อความเร็ว
_LARGE_FILE_BYTES = 10 * 1024 * 1024

# ----------------------------------------------------------------------------
# helper: จัดรูปแบบ output บนเทอร์มินัล (สี + หัวข้อ box-drawing)
# ----------------------------------------------------------------------------
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "critical": "\033[91m",
    "warning": "\033[93m",
    "info": "\033[96m",
    "ok": "\033[92m",
}


def _supports_color() -> bool:
    """ใช้สีก็ต่อเมื่อ stdout เป็น tty และไม่ได้ตั้ง NO_COLOR (มาตรฐาน no-color.org)."""
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _paint(text: str, style: str, color: bool) -> str:
    """ใส่สี ANSI ให้ข้อความถ้า color=True — ไม่งั้นคืนข้อความเดิม (กัน escape code รั่วใส่ไฟล์/pipe)."""
    if not color or style not in _ANSI:
        return text
    return f"{_ANSI[style]}{text}{_ANSI['reset']}"


def _section(title: str, color: bool) -> str:
    """หัวข้อส่วนแบบ box-drawing — '━━━ title ━━━'."""
    bar = _paint("━━━", "dim", color)
    return f"\n{bar} {_paint(title, 'bold', color)} {bar}"


def _make_progress(quiet: bool, color: bool):
    """สร้าง callback แสดงความคืบหน้าทีละขั้น (flush ทันทีเพื่อให้เห็นบนไฟล์ใหญ่). quiet -> None."""
    if quiet:
        return None

    def cb(message: str) -> None:
        print(f"  {_paint('…', 'dim', color)} {message}", flush=True)

    return cb


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    """เพิ่มอาร์กิวเมนต์การอ่านไฟล์ที่ใช้ร่วมกัน (--format, --encoding)."""
    parser.add_argument(
        "--format",
        choices=_FORMAT_CHOICES,
        default="auto",
        help="รูปแบบไฟล์ (เริ่มต้น: auto — เดาจากนามสกุล/เนื้อหา)",
    )
    parser.add_argument(
        "--encoding",
        default="auto",
        help="encoding ของไฟล์ (เริ่มต้น: auto — ลอง utf-8/tis-620/cp874/cp1252)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thaieda",
        description="AutoEDA สำหรับข้อมูลภาษาไทย — Exploratory data analysis that speaks Thai",
    )
    parser.add_argument("--version", action="version", version=f"thaieda {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="command")

    # ----- profile -----
    p_profile = sub.add_parser("profile", help="วิเคราะห์ไฟล์ CSV/JSON และสร้างรายงาน HTML")
    p_profile.add_argument("input", help="พาธไฟล์ CSV/JSON ที่ต้องการวิเคราะห์")
    p_profile.add_argument(
        "-o",
        "--output",
        default=None,
        help="พาธไฟล์ HTML ผลลัพธ์ (ค่าเริ่มต้น: <input>.thaieda.html)",
    )
    p_profile.add_argument(
        "--lang", choices=["th", "en"], default="th", help="ภาษาของรายงาน (เริ่มต้น: th)"
    )
    p_profile.add_argument(
        "--tokenizer",
        choices=["auto", "pythainlp", "nlpo3", "attacut"],
        default="auto",
        help="เครื่องมือตัดคำภาษาไทย (เริ่มต้น: auto)",
    )
    p_profile.add_argument("--target", default=None, help="คอลัมน์เป้าหมายสำหรับการวิเคราะห์ความสัมพันธ์")
    p_profile.add_argument(
        "--clean", action="store_true", help="ทำความสะอาดข้อความก่อนวิเคราะห์ และแสดง diff ในรายงาน"
    )
    p_profile.add_argument(
        "--no-timeseries",
        action="store_true",
        help="ข้ามการวิเคราะห์อนุกรมเวลา (เร็วขึ้นบนข้อมูลที่ไม่ใช่ timeseries)",
    )
    p_profile.add_argument(
        "--no-insights",
        action="store_true",
        help="ข้ามการค้นหาข้อค้นพบจากการผสมคอลัมน์ (cross-column insights)",
    )
    p_profile.add_argument(
        "--insights-top",
        type=int,
        default=8,
        metavar="N",
        help="จำนวนข้อค้นพบจากการผสมคอลัมน์สูงสุดที่แสดง (เริ่มต้น: 8)",
    )
    p_profile.add_argument("--json", default=None, help="ส่งออกข้อมูลเป็น JSON ไปยังพาธที่ระบุด้วย")
    p_profile.add_argument("--no-charts", action="store_true", help="ไม่สร้างกราฟ (เร็วขึ้น)")
    p_profile.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="สุ่มตัวอย่าง N แถวก่อนวิเคราะห์ (เหมาะกับไฟล์ใหญ่ — ค่าเริ่มต้น: ใช้ข้อมูลทั้งหมด)",
    )
    p_profile.add_argument("--quiet", action="store_true", help="แสดงผลแบบย่อ (พิมพ์เฉพาะพาธไฟล์ผลลัพธ์)")
    _add_io_args(p_profile)

    # ----- run (single-command pipeline) -----
    p_run = sub.add_parser(
        "run", help="ครบจบในคำสั่งเดียว: ทำความสะอาด → วิเคราะห์ → รายงาน + ไฟล์ที่สะอาดแล้ว"
    )
    p_run.add_argument("input", help="พาธไฟล์ CSV/JSON ที่ต้องการประมวลผล")
    p_run.add_argument(
        "-o",
        "--output",
        default=None,
        help="พาธไฟล์ HTML ผลลัพธ์ (ค่าเริ่มต้น: <input>.thaieda.html)",
    )
    p_run.add_argument(
        "--cleaned-output",
        default=None,
        help="พาธไฟล์ CSV ข้อมูลที่สะอาดแล้ว (ค่าเริ่มต้น: <input>.cleaned.csv)",
    )
    p_run.add_argument("--target", default=None, help="คอลัมน์เป้าหมายสำหรับการวิเคราะห์ความสัมพันธ์")
    p_run.add_argument(
        "--no-clean", action="store_true", help="ข้ามขั้นตอนทำความสะอาด (วิเคราะห์อย่างเดียว)"
    )
    p_run.add_argument(
        "--no-timeseries",
        action="store_true",
        help="ข้ามการวิเคราะห์อนุกรมเวลา (เร็วขึ้นบนข้อมูลที่ไม่ใช่ timeseries)",
    )
    p_run.add_argument(
        "--no-insights",
        action="store_true",
        help="ข้ามการค้นหาข้อค้นพบจากการผสมคอลัมน์ (cross-column insights)",
    )
    p_run.add_argument(
        "--insights-top",
        type=int,
        default=8,
        metavar="N",
        help="จำนวนข้อค้นพบจากการผสมคอลัมน์สูงสุดที่แสดง (เริ่มต้น: 8)",
    )
    p_run.add_argument(
        "--lang", choices=["th", "en"], default="th", help="ภาษาของรายงาน (เริ่มต้น: th)"
    )
    p_run.add_argument(
        "--tokenizer",
        choices=["auto", "pythainlp", "nlpo3", "attacut"],
        default="auto",
        help="เครื่องมือตัดคำภาษาไทย (เริ่มต้น: auto)",
    )
    p_run.add_argument("--json", default=None, help="ส่งออกข้อมูลเป็น JSON ไปยังพาธที่ระบุด้วย")
    p_run.add_argument("--no-charts", action="store_true", help="ไม่สร้างกราฟ (เร็วขึ้น)")
    p_run.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="สุ่มตัวอย่าง N แถวก่อนประมวลผล (เหมาะกับไฟล์ใหญ่ — ค่าเริ่มต้น: ใช้ข้อมูลทั้งหมด)",
    )
    p_run.add_argument("--quiet", action="store_true", help="แสดงผลแบบย่อ (พิมพ์เฉพาะพาธไฟล์ผลลัพธ์)")
    _add_io_args(p_run)

    # ----- clean -----
    p_clean = sub.add_parser("clean", help="ทำความสะอาดข้อความไทยในไฟล์ CSV/JSON แล้วบันทึกไฟล์ใหม่")
    p_clean.add_argument("input", help="พาธไฟล์ CSV/JSON ที่ต้องการทำความสะอาด")
    p_clean.add_argument(
        "-o",
        "--output",
        default=None,
        help="พาธไฟล์ CSV ผลลัพธ์ (ค่าเริ่มต้น: <input>.cleaned.csv)",
    )
    p_clean.add_argument(
        "--operations",
        default="all",
        help=(
            "การดำเนินการ คั่นด้วยจุลภาค (เริ่มต้น: all) — "
            "เลือกได้: all, encoding, zwspace, whitespace, unicode, "
            "tonemarks, repeat, numerals, phone"
        ),
    )
    _add_io_args(p_clean)

    # ----- dataset (multi-file schema discovery) -----
    p_dataset = sub.add_parser("dataset", help="วิเคราะห์หลายไฟล์พร้อมกัน ค้นหาความสัมพันธ์ระหว่างตาราง")
    p_dataset.add_argument(
        "input",
        nargs="+",
        help="พาธไดเรกทอรี หรือรายการไฟล์ CSV/JSON (เว้นวรรคคั่น)",
    )
    p_dataset.add_argument(
        "-o",
        "--output",
        default=None,
        help="พาธไฟล์ HTML ผลลัพธ์ (ค่าเริ่มต้น: <dir>/dataset.thaieda.html)",
    )
    p_dataset.add_argument(
        "--lang", choices=["th", "en"], default="th", help="ภาษาของรายงาน (เริ่มต้น: th)"
    )
    p_dataset.add_argument(
        "--no-validate",
        action="store_true",
        help="ข้ามการตรวจสอบ value overlap (เร็วขึ้น — จับคู่ด้วยชื่อคอลัมน์อย่างเดียว)",
    )
    p_dataset.add_argument("--json", default=None, help="ส่งออกข้อมูลเป็น JSON ไปยังพาธที่ระบุด้วย")
    p_dataset.add_argument("--quiet", action="store_true", help="แสดงผลแบบย่อ (พิมพ์เฉพาะพาธไฟล์ผลลัพธ์)")
    return parser


# ----------------------------------------------------------------------------
# helper: อ่านไฟล์ + จัดการ error เป็นภาษาไทย
# ----------------------------------------------------------------------------
def _read_input(args: argparse.Namespace, quiet: bool = False, color: bool = False):
    """อ่านไฟล์ input ผ่าน thaieda.io.read_data — คืน (df, exit_code).

    ถ้าอ่านสำเร็จคืน (DataFrame, 0). ถ้าผิดพลาดคืน (None, exit_code) พร้อมพิมพ์ error เป็นภาษาไทย
    บนไฟล์ใหญ่ (>10MB) จะแสดงจำนวนแถวหลังอ่านเพื่อบอกว่าไม่ได้ค้าง
    """
    from thaieda.i18n import label
    from thaieda.io import read_data

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"error: ไม่พบไฟล์ {input_path}", file=sys.stderr)
        return None, 2

    lang = getattr(args, "lang", "th")
    if not quiet:
        print(f"  {_paint('…', 'dim', color)} {label('prog_read', lang)}", flush=True)

    try:
        df = read_data(input_path, format=args.format, encoding=args.encoding)
    except ValueError as exc:
        print(
            f"error: อ่านไฟล์ไม่สำเร็จ: {exc} — "
            "ลองระบุ --format csv/json/jsonl หรือ --encoding tis-620/cp874",
            file=sys.stderr,
        )
        return None, 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: อ่านไฟล์ไม่สำเร็จ: {exc}", file=sys.stderr)
        return None, 1

    if df.empty:
        print("error: ไฟล์ว่างเปล่า (ไม่มีข้อมูล)", file=sys.stderr)
        return None, 1

    # ไฟล์ใหญ่ -> บอกจำนวนแถว/คอลัมน์ทันที (ผู้ใช้จะได้รู้ว่ากำลังทำงาน ไม่ใช่ค้าง)
    if not quiet:
        try:
            is_large = input_path.stat().st_size > _LARGE_FILE_BYTES
        except OSError:
            is_large = False
        if is_large:
            print(f"    {len(df):,} แถว × {len(df.columns)} คอลัมน์", flush=True)

    return df, 0


def _maybe_sample(df, sample, quiet: bool, color: bool):
    """สุ่มตัวอย่าง sample แถวถ้าระบุและน้อยกว่าจำนวนแถวจริง — คืน DataFrame (อาจถูกสุ่ม)."""
    if sample is None or sample <= 0 or sample >= len(df):
        return df
    total = len(df)
    sampled = df.sample(n=sample, random_state=42).reset_index(drop=True)
    if not quiet:
        print(
            _paint(f"  ⚠ สุ่มตัวอย่าง {sample:,} จาก {total:,} แถว", "warning", color),
            flush=True,
        )
    return sampled


def _maybe_large_file_hint(input_path: Path, args: argparse.Namespace, quiet: bool) -> None:
    """ไฟล์ใหญ่ + ยังไม่ได้เปิด flag เร่งความเร็ว -> แนะนำ flag (Improve 3)."""
    if quiet:
        return
    try:
        size = input_path.stat().st_size
    except OSError:
        return
    if size <= _LARGE_FILE_BYTES:
        return
    tips: list[str] = []
    if not getattr(args, "no_charts", False):
        tips.append("--no-charts")
    if not getattr(args, "no_timeseries", False):
        tips.append("--no-timeseries")
    if getattr(args, "sample", None) is None:
        tips.append("--sample 10000")
    if tips:
        mb = size / (1024 * 1024)
        print(f"  ℹ ไฟล์ใหญ่ ({mb:.0f}MB) — เพื่อความเร็วลองใช้: {' '.join(tips)}", flush=True)


# ----------------------------------------------------------------------------
# profile
# ----------------------------------------------------------------------------
def _run_profile(args: argparse.Namespace) -> int:
    from thaieda.report import ProfileReport

    quiet = args.quiet
    color = _supports_color()

    # input เป็นไดเรกทอรีที่มีหลายไฟล์ → วิเคราะห์เป็นชุดข้อมูล (multi-file schema) อัตโนมัติ
    routed = _maybe_route_dataset(args)
    if routed is not None:
        return routed

    df, code = _read_input(args, quiet=quiet, color=color)
    if df is None:
        return code

    if args.target is not None and args.target not in df.columns:
        print(f"error: ไม่พบคอลัมน์เป้าหมาย '{args.target}' ในข้อมูล", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    output_path = args.output or str(input_path.with_suffix(".thaieda.html"))

    _maybe_large_file_hint(input_path, args, quiet)
    df = _maybe_sample(df, args.sample, quiet, color)

    report = ProfileReport(
        df,
        lang=args.lang,
        tokenizer_engine=args.tokenizer,
        make_charts=not args.no_charts,
        target_column=args.target,
        clean=args.clean,
        timeseries=not args.no_timeseries,
        insights_engine=not args.no_insights,
        insights_top=args.insights_top,
        progress=_make_progress(quiet, color),
    )
    report.run()
    if not quiet:
        print(f"  {_paint('…', 'dim', color)} {label('prog_report', args.lang)}", flush=True)
    report.to_html(output_path)

    if args.json:
        report.to_json(args.json)

    if quiet:
        _print_quiet(output_path, args.json, None)
    else:
        _print_summary(report, input_path, output_path, args.json, color)
    return 0


def _print_quiet(html_path: str, json_path: str | None, cleaned_path: str | None) -> None:
    """โหมด --quiet: พิมพ์เฉพาะพาธไฟล์ผลลัพธ์ (อย่างละบรรทัด)."""
    print(html_path)
    if json_path:
        print(json_path)
    if cleaned_path:
        print(cleaned_path)


def _severity_counts(issues) -> dict[str, int]:
    counts = {"critical": 0, "warning": 0, "info": 0}
    for i in issues:
        counts[i.severity] = counts.get(i.severity, 0) + 1
    return counts


def _print_summary_table(report, color: bool) -> None:
    """ตารางสรุปท้ายรายงาน — ประเภทคอลัมน์ / ปัญหา / ความผิดปกติ / ข้อค้นพบ (Improve 2)."""
    from thaieda.i18n import label as L

    overview = report.overview
    sev = _severity_counts(report.quality_issues)
    types = ", ".join(
        f"{t} {cnt}" for t, cnt in sorted(overview["type_counts"].items(), key=lambda x: -x[1])
    )
    crit = _paint(f"วิกฤต {sev['critical']}", "critical", color)
    warn = _paint(f"เตือน {sev['warning']}", "warning", color)
    info = _paint(f"ข้อมูล {sev['info']}", "info", color)
    rows = [
        (L("column_types", "th"), types),
        (L("quality_issues", "th"), f"{len(report.quality_issues)} ({crit}, {warn}, {info})"),
        (L("anomalies", "th"), str(len(report.anomalies))),
    ]
    insights = report.insights
    if insights is not None:
        rows.append((L("auto_insights", "th"), str(insights.total_insights)))
    print(_section(L("summary_table", "th"), color))
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}")


def _print_summary(
    report, input_path: Path, output_path: str, json_path: str | None, color: bool
) -> None:
    overview = report.overview
    print(_section(f"วิเคราะห์ไฟล์: {input_path}", color))
    print(
        f"  แถว: {overview['rows']:,} | คอลัมน์: {overview['columns']} | "
        f"เซลล์ว่าง: {overview['missing_pct']}%"
    )
    if report.cleaning_diff:
        total = sum(c.rows_affected for c in report.cleaning_diff)
        print(f"  ทำความสะอาดแล้ว: {len(report.cleaning_diff)} การดำเนินการ (รวม {total:,} เซลล์)")
    else:
        print(f"  คำแนะนำการทำความสะอาด: {len(report.cleaning_suggestions)}")
    _print_timeseries_highlights(report)
    _print_business_highlights(report)
    _print_insight_highlights(report)
    for note in report.notes:
        print(_paint(f"  ⚠ {note}", "warning", color))
    _print_summary_table(report, color)
    print(_paint(f"✓ บันทึกรายงาน HTML: {output_path}", "ok", color))
    if json_path:
        print(_paint(f"✓ บันทึก JSON: {json_path}", "ok", color))


def _print_timeseries_highlights(report) -> None:
    """พิมพ์สรุปการวิเคราะห์อนุกรมเวลา (ถ้ามี datetime column)."""
    ts = report.timeseries_results
    if not ts:
        return
    print(f"  อนุกรมเวลา: วิเคราะห์ {len(ts)} คอลัมน์")
    for col, r in list(ts.items())[:5]:
        bits: list[str] = [r.frequency_th]
        if r.has_trend:
            bits.append(f"แนวโน้ม{r.trend_direction_th}")
        if r.has_seasonality:
            bits.append(f"seasonality รอบ {r.seasonal_period}")
        if r.gap_count:
            bits.append(f"ช่องว่าง {r.gap_count} ช่วง")
        if r.anomalies:
            bits.append(f"spike {len(r.anomalies)} จุด")
        print(f"    • {col}: {', '.join(bits)}")


def _print_business_highlights(report) -> None:
    """พิมพ์ข้อค้นพบจากการผสมคอลัมน์ (cross-column insights) เด่น ๆ สูงสุด 5 ข้อ."""
    engine = report.insight_engine
    if engine is None or engine.total == 0:
        return
    print(f"  ข้อค้นพบคอลัมน์ผสม: {engine.total}")
    for card in engine.cards[:5]:
        print(f"    • {card.title_th}")


def _print_insight_highlights(report) -> None:
    """พิมพ์บทสรุปผู้บริหารและหัวข้อข้อค้นพบเด่น ๆ (สูงสุด 5 ข้อ)."""
    insights = report.insights
    if insights is None or insights.total_insights == 0:
        return
    print(
        f"  ข้อค้นพบ: {insights.total_insights} "
        f"(วิกฤต {insights.critical_count}, เตือน {insights.warning_count}, "
        f"ข้อมูล {insights.info_count})"
    )
    print(f"  สรุป: {insights.executive_summary_th}")
    for ins in insights.insights[:5]:
        print(f"    • [{ins.severity}] {ins.title_th}")


# ----------------------------------------------------------------------------
# run (single-command pipeline)
# ----------------------------------------------------------------------------
def _run_run(args: argparse.Namespace) -> int:
    from thaieda.report import ProfileReport

    quiet = args.quiet
    color = _supports_color()

    # input เป็นไดเรกทอรีที่มีหลายไฟล์ → วิเคราะห์เป็นชุดข้อมูล (multi-file schema) อัตโนมัติ
    routed = _maybe_route_dataset(args)
    if routed is not None:
        return routed

    df, code = _read_input(args, quiet=quiet, color=color)
    if df is None:
        return code

    if args.target is not None and args.target not in df.columns:
        print(f"error: ไม่พบคอลัมน์เป้าหมาย '{args.target}' ในข้อมูล", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    output_path = args.output or str(input_path.with_suffix(".thaieda.html"))
    do_clean = not args.no_clean

    _maybe_large_file_hint(input_path, args, quiet)
    df = _maybe_sample(df, args.sample, quiet, color)

    report = ProfileReport(
        df,
        lang=args.lang,
        tokenizer_engine=args.tokenizer,
        make_charts=not args.no_charts,
        target_column=args.target,
        clean=do_clean,
        timeseries=not args.no_timeseries,
        insights_engine=not args.no_insights,
        insights_top=args.insights_top,
        progress=_make_progress(quiet, color),
    )
    report.run()
    if not quiet:
        print(f"  {_paint('…', 'dim', color)} {label('prog_report', args.lang)}", flush=True)
    report.to_html(output_path)

    if args.json:
        report.to_json(args.json)

    # บันทึกข้อมูลที่สะอาดแล้ว (report.df คือ DataFrame ที่ทำความสะอาดแล้วเมื่อ clean=True)
    cleaned_path: str | None = None
    if do_clean:
        cleaned_path = args.cleaned_output or str(input_path.with_suffix(".cleaned.csv"))
        # แปลงคอลัมน์เบอร์โทรเป็น string ก่อนเขียน CSV กัน leading zero หาย
        from thaieda.detect import ColumnType, detect_all

        col_types = detect_all(report.df)
        for col, ctype in col_types.items():
            if ctype == ColumnType.PHONE_NUMBER and col in report.df.columns:
                report.df[col] = report.df[col].astype(str)
        report.df.to_csv(cleaned_path, index=False, encoding="utf-8")

    if quiet:
        _print_quiet(output_path, args.json, cleaned_path)
    else:
        _print_run_summary(report, input_path, output_path, cleaned_path, args.json, color)
    return 0


def _print_run_summary(
    report,
    input_path: Path,
    html_path: str,
    cleaned_path: str | None,
    json_path: str | None,
    color: bool,
) -> None:
    overview = report.overview
    print(_section(f"ประมวลผลไฟล์: {input_path}", color))
    print(
        f"  แถว: {overview['rows']:,} | คอลัมน์: {overview['columns']} | "
        f"เซลล์ว่าง: {overview['missing_pct']}%"
    )
    if report.cleaning_diff:
        total = sum(c.rows_affected for c in report.cleaning_diff)
        print(f"  ทำความสะอาดแล้ว: {len(report.cleaning_diff)} การดำเนินการ (รวม {total:,} เซลล์)")
    _print_timeseries_highlights(report)
    _print_business_highlights(report)
    _print_insight_highlights(report)
    for note in report.notes:
        print(_paint(f"  ⚠ {note}", "warning", color))
    _print_summary_table(report, color)
    print(_paint(f"✓ บันทึกรายงาน HTML: {html_path}", "ok", color))
    if json_path:
        print(_paint(f"✓ บันทึก JSON: {json_path}", "ok", color))
    if cleaned_path:
        print(_paint(f"✓ บันทึกข้อมูลที่สะอาดแล้ว: {cleaned_path}", "ok", color))


# ----------------------------------------------------------------------------
# clean
# ----------------------------------------------------------------------------
def _run_clean(args: argparse.Namespace) -> int:
    import pandas as pd

    from thaieda.clean import clean_thai_text
    from thaieda.detect import ColumnType, detect_column_type, normalize_phone_number

    df, code = _read_input(args, color=_supports_color())
    if df is None:
        return code

    operations = [op.strip() for op in args.operations.split(",") if op.strip()]
    input_path = Path(args.input)
    output_path = args.output or str(input_path.with_suffix(".cleaned.csv"))

    total_affected = 0
    results_by_op: dict[str, int] = {}
    for col in df.columns:
        col_type = detect_column_type(df[col])

        # เบอร์โทร: แปลงเป็น string ก่อน (กัน leading zero หาย) แล้วทำความสะอาด
        if col_type == ColumnType.PHONE_NUMBER:
            original_vals = df[col].copy()
            df[col] = df[col].apply(normalize_phone_number)
            changed = int((df[col].astype(str) != original_vals.astype(str)).sum())
            if changed > 0:
                results_by_op["normalize_phone_numbers"] = (
                    results_by_op.get("normalize_phone_numbers", 0) + changed
                )
                total_affected += changed
            continue

        # ข้าม numeric/datetime — เป็นข้อมูลเชิงสถิติ ไม่ใช่ข้อความ
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        try:
            cleaned, results = clean_thai_text(df[col], operations=operations)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        df[col] = cleaned
        for r in results:
            if r.rows_affected > 0:
                results_by_op[r.operation] = results_by_op.get(r.operation, 0) + r.rows_affected
                total_affected += r.rows_affected

    # เก็บคอลัมน์เบอร์โทรเป็น string ตอนเขียน CSV (กัน leading zero หาย)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✓ ทำความสะอาดไฟล์: {input_path}")
    print(f"  การดำเนินการ: {', '.join(operations)}")
    if results_by_op:
        print("  สรุปการเปลี่ยนแปลง:")
        for op, cnt in sorted(results_by_op.items(), key=lambda x: -x[1]):
            print(f"    - {op}: {cnt:,} แถว")
    else:
        print("  ไม่พบสิ่งที่ต้องทำความสะอาด")
    print(f"  รวมเซลล์ที่เปลี่ยน: {total_affected:,}")
    print(f"✓ บันทึกไฟล์ที่สะอาดแล้ว: {output_path}")
    return 0


# ----------------------------------------------------------------------------
# dataset (multi-file schema discovery)
# ----------------------------------------------------------------------------
def _dataset_core(
    ds_input: str | list[str],
    output_path: str,
    lang: str,
    validate: bool,
    json_path: str | None,
    quiet: bool,
    color: bool,
) -> int:
    """รัน profile_dataset → DatasetReport แล้วบันทึก HTML/JSON + พิมพ์สรุป."""
    from thaieda.report._dataset import DatasetReport
    from thaieda.schema import profile_dataset

    dataset = profile_dataset(
        ds_input, lang=lang, validate_values=validate, progress=_make_progress(quiet, color)
    )
    report = DatasetReport(dataset, lang=lang)
    if not quiet:
        print(f"  {_paint('…', 'dim', color)} {label('prog_report', lang)}", flush=True)
    report.to_html(output_path)
    if json_path:
        report.to_json(json_path)

    if quiet:
        print(output_path)
        if json_path:
            print(json_path)
    else:
        _print_dataset_summary(dataset, output_path, json_path, color)
    return 0


def _print_dataset_summary(dataset, html_path: str, json_path: str | None, color: bool) -> None:
    """สรุปผลโหมด dataset บนเทอร์มินัล — ตาราง, ความสัมพันธ์, ข้อมูลกำพร้า."""
    print(_section("วิเคราะห์ชุดข้อมูลหลายตาราง", color))
    print(
        f"  ตาราง: {len(dataset.tables)} | "
        f"ความสัมพันธ์: {len(dataset.relationships)} | "
        f"ข้อมูลกำพร้า: {len(dataset.orphan_findings)}"
    )
    for t in dataset.tables:
        print(f"    • {t.name}: {t.row_count:,} แถว × {t.column_count} คอลัมน์")
    if dataset.relationships:
        print("  ความสัมพันธ์ที่พบ:")
        for r in dataset.relationships[:12]:
            print(f"    • {r.description_th} (มั่นใจ {r.confidence * 100:.0f}%)")
        if len(dataset.relationships) > 12:
            print(f"    … และอีก {len(dataset.relationships) - 12} ความสัมพันธ์")
    for o in dataset.orphan_findings[:5]:
        print(_paint(f"  ⚠ {o}", "warning", color))
    for note in dataset.notes:
        print(_paint(f"  ⚠ {note}", "warning", color))
    print(_paint(f"✓ บันทึกรายงาน HTML: {html_path}", "ok", color))
    if json_path:
        print(_paint(f"✓ บันทึก JSON: {json_path}", "ok", color))


def _run_dataset(args: argparse.Namespace) -> int:
    quiet = args.quiet
    color = _supports_color()

    paths = [Path(p) for p in args.input]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"error: ไม่พบพาธ: {', '.join(missing)}", file=sys.stderr)
        return 2

    if len(paths) == 1 and paths[0].is_dir():
        ds_input: str | list[str] = str(paths[0])
        default_out = str(paths[0] / "dataset.thaieda.html")
    else:
        ds_input = [str(p) for p in paths]
        default_out = str(paths[0].parent / "dataset.thaieda.html")

    output_path = args.output or default_out
    return _dataset_core(
        ds_input, output_path, args.lang, not args.no_validate, args.json, quiet, color
    )


def _maybe_route_dataset(args: argparse.Namespace) -> int | None:
    """ถ้า input เป็นไดเรกทอรีที่มีไฟล์รองรับ >= 2 → วิเคราะห์เป็นชุดข้อมูล (คืน exit code) มิฉะนั้น None."""
    input_path = Path(args.input)
    if not input_path.is_dir():
        return None
    files = [
        f for f in sorted(input_path.iterdir()) if f.is_file() and f.suffix.lower() in _DATASET_EXTS
    ]
    if len(files) < 2:
        return None
    quiet = getattr(args, "quiet", False)
    color = _supports_color()
    if not quiet:
        msg = f"  ℹ พบไดเรกทอรี ({len(files)} ไฟล์) — วิเคราะห์เป็นชุดข้อมูลหลายตาราง"
        print(_paint(msg, "info", color), flush=True)
    output_path = args.output or str(input_path / "dataset.thaieda.html")
    return _dataset_core(
        str(input_path),
        output_path,
        getattr(args, "lang", "th"),
        True,
        getattr(args, "json", None),
        quiet,
        color,
    )


# ----------------------------------------------------------------------------
# entrypoint
# ----------------------------------------------------------------------------
_RUNNERS = {
    "profile": _run_profile,
    "run": _run_run,
    "clean": _run_clean,
    "dataset": _run_dataset,
}


def main(argv: list[str] | None = None) -> int:
    """จุดเริ่มต้นของคำสั่ง `thaieda`."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    runner = _RUNNERS.get(args.command)
    if runner is not None:
        try:
            return runner(args)
        except KeyboardInterrupt:
            print("\nยกเลิกแล้ว", file=sys.stderr)
            return 130
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
