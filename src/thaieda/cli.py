"""Command-line interface สำหรับ ThaiEDA.

ตัวอย่าง:
    # one-liner (ค่าเริ่มต้น): อ่าน → clean → blueprint report → HTML
    thaieda data.csv
    thaieda data.csv -o report.html --target clicked
    thaieda .                              # batch ทุกไฟล์ในโฟลเดอร์
    thaieda folder/ --output-dir reports/   # batch → โฟลเดอร์ reports/

    # วิเคราะห์ → รายงาน HTML (โหมด explore เต็มรูปแบบ)
    thaieda profile data.csv -o report.html --lang th --tokenizer auto

    # ครบจบในคำสั่งเดียว: ทำความสะอาด → วิเคราะห์ → รายงาน + ไฟล์ที่สะอาดแล้ว
    thaieda run data.csv -o report.html --cleaned-output cleaned.csv

    # ทำความสะอาดอย่างเดียว
    thaieda clean data.csv -o cleaned.csv
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

from thaieda import __version__
from thaieda.i18n import label

# ตัวเลือก format ที่รองรับสำหรับการอ่านไฟล์
_FORMAT_CHOICES = ["auto", "csv", "tsv", "json", "jsonl", "excel", "parquet"]

# นามสกุลไฟล์ที่นับเป็น "ตาราง" เมื่อสแกนไดเรกทอรี (โหมด dataset หลายตาราง)
_DATASET_EXTS = (".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet")

# ขนาดไฟล์ที่ถือว่า "ใหญ่" (10MB) — แสดงจำนวนแถวหลังอ่าน และแนะนำ flag เพื่อความเร็ว
_LARGE_FILE_BYTES = 10 * 1024 * 1024

# ชื่อคอลัมน์ที่ใช้เดา target อัตโนมัติ (one-liner) — ตรงแบบ case-insensitive
_TARGET_NAME_CANDIDATES = (
    "target",
    "label",
    "clicked",
    "churn",
    "survived",
    "y",
    "class",
    "outcome",
    "income",
    "quality",
    "response",
    "default",
    "fraud",
    "exit",
)

# รูปแบบชื่อคอลัมน์ target แบบ partial (prefix/suffix) — ใช้หลัง exact match
_TARGET_PARTIAL_PATTERNS = (
    "target",
    "label",
    "churn",
    "survived",
    "outcome",
    "quality",
    "income",
    "clicked",
)

# นามสกุลไฟล์สำหรับโหมด batch one-liner (โฟลเดอร์ → รายงานต่อไฟล์)
_BATCH_EXTS = frozenset({".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet"})

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


def _configure_stdio() -> None:
    """บังคับ stdout/stderr เป็น UTF-8 บน Windows — กัน charmap error ตอนพิมพ์ไทย/×."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")


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


def _build_oneliner_parser() -> argparse.ArgumentParser:
    """Parser สำหรับ one-liner: ``thaieda data.csv`` (ไม่ต้องระบุ subcommand)."""
    parser = argparse.ArgumentParser(
        prog="thaieda",
        description="AutoEDA สำหรับข้อมูลภาษาไทย — วิเคราะห์ไฟล์แล้วสร้างรายงาน HTML อัตโนมัติ",
    )
    parser.add_argument("--version", action="version", version=f"thaieda {__version__}")
    parser.add_argument(
        "input",
        nargs="?",
        default=".",
        help="พาธไฟล์ CSV/TSV/JSON/Excel/Parquet หรือโฟลเดอร์ (batch รายงานต่อไฟล์)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="พาธไฟล์ HTML ผลลัพธ์ (ไฟล์เดียวเท่านั้น — ค่าเริ่มต้น: <stem>-report.html)",
    )
    parser.add_argument(
        "-t",
        "--target",
        default=None,
        metavar="COLUMN",
        help=(
            "คอลัมน์เป้าหมายที่อยากทำนาย/วิเคราะห์ (เช่น clicked, churn, Survived) / "
            "target column for ML blueprint. "
            "ใช้ thaieda data.csv --columns เพื่อดูรายชื่อคอลัมน์ก่อน"
        ),
    )
    parser.add_argument(
        "--columns",
        "--preview",
        action="store_true",
        dest="columns_preview",
        help="แสดงรายชื่อคอลัมน์ + dtype + target ที่น่าจะเป็น (ไม่รัน EDA, ออกทันที)",
    )
    parser.add_argument(
        "--folder",
        action="store_true",
        help="บังคับโหมด batch — ประมวลผลทุกไฟล์ในโฟลเดอร์ (เหมือน thaieda .)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="โฟลเดอร์สำหรับบันทึก HTML ในโหมด batch (ค่าเริ่มต้น: ข้างไฟล์ต้นทาง)",
    )
    parser.add_argument(
        "-y",
        "--no-interactive",
        action="store_true",
        help="ข้าม prompt เลือก target (เหมาะกับ batch/CI — ไม่ถามบน TTY)",
    )
    parser.add_argument(
        "--lang", choices=["th", "en"], default="th", help="ภาษาของรายงาน (เริ่มต้น: th)"
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="ใช้รายงาน explore แบบเต็ม (ค่าเริ่มต้น: blueprint — สั้น เน้น actionable)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="ข้ามขั้นตอนทำความสะอาด (วิเคราะห์ข้อมูลดิบ)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="วิเคราะห์ด้วย LLM หลัง EDA (ต้องติดตั้ง thaieda[llm] และมี API key)",
    )
    parser.add_argument("--no-charts", action="store_true", help="ไม่สร้างกราฟ (เร็วขึ้น)")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="สุ่มตัวอย่าง N แถวก่อนวิเคราะห์ (เหมาะกับไฟล์ใหญ่)",
    )
    parser.add_argument("--quiet", action="store_true", help="แสดงผลแบบย่อ (พิมพ์เฉพาะพาธไฟล์ผลลัพธ์)")
    _add_io_args(parser)
    return parser


def _target_match_reason(col: str) -> str | None:
    """เหตุผลที่คอลัมน์นี้น่าจะเป็น target — คืน None ถ้าไม่ตรง."""
    col_lower = str(col).lower()
    for name in _TARGET_NAME_CANDIDATES:
        if col_lower == name:
            return f"ชื่อตรงกับ '{name}'"
    for pat in _TARGET_PARTIAL_PATTERNS:
        if col_lower == pat or col_lower.endswith(f"_{pat}") or col_lower.startswith(f"{pat}_"):
            return f"ชื่อมี '{pat}'"
    return None


def _guess_target_column(columns) -> str | None:
    """เดาคอลัมน์เป้าหมายจากชื่อที่พบบ่อย — คืน None ถ้าไม่พบ."""
    for col in columns:
        if _target_match_reason(str(col)) is not None:
            return str(col)
    return None


def _target_candidates(columns) -> list[tuple[str, str]]:
    """รายการ (คอลัมน์, เหตุผล) ที่น่าจะเป็น target — เรียงตามลำดับในข้อมูล."""
    out: list[tuple[str, str]] = []
    for col in columns:
        reason = _target_match_reason(str(col))
        if reason is not None:
            out.append((str(col), reason))
    return out


def _likely_target_columns(columns) -> set[str]:
    """คอลัมน์ที่น่าจะเป็น target — ใช้ mark * ใน prompt interactive."""
    return {col for col, _ in _target_candidates(columns)}


def _dtype_hint(series) -> str:
    """ชื่อ dtype สั้น ๆ สำหรับแสดงบนเทอร์มินัล."""
    dtype = str(series.dtype)
    if dtype.startswith("int") or dtype.startswith("uint"):
        return "int"
    if dtype.startswith("float"):
        return "float"
    if dtype == "bool":
        return "bool"
    if dtype.startswith("datetime"):
        return "datetime"
    if dtype == "object":
        return "object"
    return dtype.split("[")[0][:12]


def _target_source_label(source: str | None) -> str:
    """แปลงแหล่งที่มาของ target เป็นข้อความสั้น ๆ."""
    labels = {
        "flag": "จาก --target",
        "auto": "auto",
        "interactive": "interactive",
    }
    return labels.get(source or "", source or "")


def _print_target_line(target: str | None, source: str | None, *, quiet: bool) -> None:
    """พิมพ์บรรทัด Target: ... หลังรันสำเร็จ (รวมกรณีไม่ระบุ target)."""
    if quiet:
        return
    if target:
        print(f"Target: {target} ({_target_source_label(source)})", flush=True)
    else:
        print("Target: (ไม่ระบุ — ไม่มีแผนสร้างโมเดล)", flush=True)


def _print_columns_table(df, *, title: str | None = None) -> None:
    """แสดงตารางคอลัมน์แบบ dry-run (--columns / --preview) — ไม่รัน EDA."""
    cols = [str(c) for c in df.columns]
    candidates = {col: reason for col, reason in _target_candidates(cols)}
    header = title or f"Columns ({len(cols)}):"
    print(header, flush=True)
    print(f"  {'#':>3}  {'Column':<22}  {'dtype':<8}  {'unique':>6}  target?", flush=True)
    for i, col in enumerate(cols, start=1):
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        target_cell = ""
        if col in candidates:
            target_cell = f"★ likely target (auto: {candidates[col]})"
        print(
            f"  {i:>3}  {col:<22}  {_dtype_hint(series):<8}  {n_unique:>6}  {target_cell}",
            flush=True,
        )


def _prompt_target_column(df) -> str | None:
    """ถามผู้ใช้เลือกคอลัมน์ target แบบ interactive (TTY เท่านั้น)."""
    cols = [str(c) for c in df.columns]
    candidates = {col: reason for col, reason in _target_candidates(cols)}
    print(
        "target คือคอลัมน์ที่อยากทำนาย/วิเคราะห์ เช่น ยอดขาย, คลิก, churn, Survived",
        flush=True,
    )
    print("คอลัมน์ที่มี:", flush=True)
    for i, col in enumerate(cols, start=1):
        mark = ""
        if col in candidates:
            mark = f" * (auto: {candidates[col]})"
        dtype = _dtype_hint(df[col])
        print(f"  {i}. {col}  [{dtype}]{mark}", flush=True)
    if candidates:
        print("  (* = ชื่อที่น่าจะเป็น target อัตโนมัติ)", flush=True)
    try:
        answer = input("เลือกคอลัมน์ target (Enter=ข้าม, หรือพิมพ์ชื่อ/เลข): ").strip()
    except EOFError:
        return None
    if not answer:
        return None
    if answer.isdigit():
        idx = int(answer)
        if 1 <= idx <= len(cols):
            return cols[idx - 1]
        print(f"  ⚠ เลข {idx} ไม่อยู่ในรายการ — ข้าม target", flush=True)
        return None
    if answer in cols:
        return answer
    lower_map = {c.lower(): c for c in cols}
    if answer.lower() in lower_map:
        return lower_map[answer.lower()]
    print(f"  ⚠ ไม่พบคอลัมน์ '{answer}' — ข้าม target", flush=True)
    return None


def _resolve_target_column(
    df,
    explicit_target: str | None,
    *,
    interactive: bool,
) -> tuple[str | None, str | None]:
    """คืน (target_column, source) — source เป็น flag | auto | interactive | None."""
    columns = df.columns
    if explicit_target is not None:
        return explicit_target, "flag"
    candidates = _target_candidates(columns)
    if len(candidates) == 1:
        return candidates[0][0], "auto"
    if len(candidates) > 1:
        if interactive and sys.stdin.isatty():
            chosen = _prompt_target_column(df)
            if chosen:
                return chosen, "interactive"
        return candidates[0][0], "auto"
    if interactive and sys.stdin.isatty():
        chosen = _prompt_target_column(df)
        if chosen:
            return chosen, "interactive"
    return None, None


def _is_batch_data_file(path: Path) -> bool:
    """True ถ้า path เป็นไฟล์ข้อมูลที่ควรประมวลผลในโหมด batch."""
    if not path.is_file():
        return False
    name = path.name
    if name.startswith("."):
        return False
    if name.endswith("-report.html") or name.endswith(".thaieda.html"):
        return False
    return path.suffix.lower() in _BATCH_EXTS


def _find_batch_files(folder: Path) -> list[Path]:
    """รายการไฟล์ข้อมูลในโฟลเดอร์ (ไม่ recursive, ไม่รวม hidden/report)."""
    return sorted(p for p in folder.iterdir() if _is_batch_data_file(p))


def _default_oneliner_output(input_path: Path) -> str:
    """ค่าเริ่มต้น: ``<stem>-report.html`` ในโฟลเดอร์เดียวกับไฟล์ input."""
    return str(input_path.with_name(f"{input_path.stem}-report.html"))


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
            "ลองระบุ --format csv/tsv/json/jsonl/excel/parquet หรือ --encoding tis-620/cp874",
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
# one-liner (thaieda data.csv)
# ----------------------------------------------------------------------------
def _run_oneliner_file(
    df,
    input_path: Path,
    args: argparse.Namespace,
    *,
    target: str | None,
    target_source: str | None,
    output_path: str | None = None,
    quiet: bool | None = None,
    color: bool | None = None,
    show_summary: bool = True,
) -> tuple[int, str | None]:
    """รัน pipeline one-liner สำหรับ DataFrame เดียว — คืน (exit_code, html_path)."""
    from thaieda import run as thaieda_run

    quiet = args.quiet if quiet is None else quiet
    color = _supports_color() if color is None else color

    if target is not None and target not in df.columns:
        print(f"error: ไม่พบคอลัมน์เป้าหมาย '{target}' ในข้อมูล", file=sys.stderr)
        return 2, None

    out = output_path or args.output or _default_oneliner_output(input_path)
    report_mode = "explore" if args.explore else "blueprint"
    do_clean = not args.no_clean

    _maybe_large_file_hint(input_path, args, quiet)
    df = _maybe_sample(df, args.sample, quiet, color)

    try:
        result = thaieda_run(
            df,
            clean=do_clean,
            handle_missing="flag",
            lang=args.lang,
            make_charts=not args.no_charts,
            target_column=target,
            report_mode=report_mode,
            llm=args.llm,
            progress=_make_progress(quiet, color),
        )
    except (TypeError, KeyError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1, None

    if not quiet:
        print(f"  {_paint('…', 'dim', color)} {label('prog_report', args.lang)}", flush=True)
    result.to_html(out)

    abs_output = str(Path(out).resolve())
    _print_target_line(target, target_source, quiet=quiet)
    if show_summary and not quiet:
        _print_oneliner_summary(
            result, input_path, abs_output, target, target_source, report_mode, color
        )
    return 0, abs_output


def _resolve_batch_global_target(
    file_dfs: list[tuple[Path, object]],
    explicit_target: str | None,
    *,
    interactive: bool,
) -> tuple[str | None, str | None]:
    """เดา/ถาม target ครั้งเดียวสำหรับทั้ง batch — ใช้เมื่อคอลัมน์คล้ายกัน."""
    if explicit_target is not None:
        return explicit_target, "flag"

    file_columns = [(p, [str(c) for c in df.columns]) for p, df in file_dfs if not df.empty]
    autodetected = [_guess_target_column(cols) for _, cols in file_columns if cols]
    if autodetected and all(a == autodetected[0] and a is not None for a in autodetected):
        return autodetected[0], "auto"

    valid = [(p, df) for p, df in file_dfs if not df.empty]
    if not valid:
        return None, None

    col_sets = {tuple(str(c) for c in df.columns) for _, df in valid}
    if len(col_sets) == 1:
        df0 = valid[0][1]
        candidates = _target_candidates(df0.columns)
        if len(candidates) == 1:
            return candidates[0][0], "auto"
        if len(candidates) > 1:
            if interactive and sys.stdin.isatty():
                chosen = _prompt_target_column(df0)
                if chosen:
                    return chosen, "interactive"
            return candidates[0][0], "auto"
        if interactive and sys.stdin.isatty():
            chosen = _prompt_target_column(df0)
            if chosen:
                return chosen, "interactive"
        return None, None

    if interactive and sys.stdin.isatty():
        print(
            "  ℹ คอลัมน์ต่างกันระหว่างไฟล์ — เลือก target ครั้งเดียว (ใช้กับไฟล์ที่มีคอลัมน์นี้)",
            flush=True,
        )
        chosen = _prompt_target_column(valid[0][1])
        if chosen:
            return chosen, "interactive"
    return None, None


def _batch_output_path(
    fpath: Path,
    output_dir: Path | None,
) -> Path:
    """พาธ HTML สำหรับไฟล์หนึ่งในโหมด batch."""
    stem = f"{fpath.stem}-report.html"
    if output_dir is not None:
        return output_dir / stem
    return fpath.with_name(stem)


def _run_oneliner_batch(args: argparse.Namespace) -> int:
    """Batch one-liner: ประมวลผลทุกไฟล์ในโฟลเดอร์ → {stem}-report.html ต่อไฟล์."""
    from thaieda.io import read_data

    quiet = args.quiet
    color = _supports_color()
    interactive = not args.no_interactive
    folder = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not folder.is_dir():
        print(f"error: ไม่พบโฟลเดอร์ {folder}", file=sys.stderr)
        return 2

    files = _find_batch_files(folder)
    if not files:
        exts = ", ".join(sorted(_BATCH_EXTS))
        print(
            f"error: ไม่พบไฟล์ข้อมูล ({exts}) ใน {folder}\n"
            "  ใส่ไฟล์ CSV/TSV/JSON/Excel/Parquet "
            "หรือใช้ `thaieda dataset folder/` สำหรับ schema หลายตาราง",
            file=sys.stderr,
        )
        return 2

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(_section(f"Batch: {folder} ({len(files)} ไฟล์)", color), flush=True)

    file_columns: list[tuple[Path, list[str]]] = []
    file_peeks: list[tuple[Path, object]] = []
    for fpath in files:
        try:
            df_peek = read_data(fpath, format=args.format, encoding=args.encoding)
            file_columns.append((fpath, [str(c) for c in df_peek.columns]))
            file_peeks.append((fpath, df_peek))
        except Exception as exc:  # noqa: BLE001
            file_columns.append((fpath, []))
            file_peeks.append((fpath, None))
            if not quiet:
                print(f"  ⚠ อ่าน {fpath.name} ไม่สำเร็จ (peek): {exc}", flush=True)

    valid_peeks = [(p, df) for p, df in file_peeks if df is not None and not df.empty]
    global_target, global_source = _resolve_batch_global_target(
        valid_peeks, args.target, interactive=interactive
    )
    if global_target and not quiet:
        _print_target_line(global_target, global_source, quiet=False)

    ok_paths: list[str] = []
    fail_count = 0
    total = len(files)

    for i, fpath in enumerate(files):
        label_idx = f"[{i + 1}/{total}]"
        t0 = time.perf_counter()
        out_path = _batch_output_path(fpath, output_dir)

        if not quiet:
            print(f"{label_idx} processing {fpath.name} ...", flush=True)

        try:
            df = read_data(fpath, format=args.format, encoding=args.encoding)
        except Exception as exc:  # noqa: BLE001
            fail_count += 1
            if not quiet:
                print(f"{label_idx} {fpath.name} ... FAIL ({time.perf_counter() - t0:.1f}s): {exc}")
            continue

        if df.empty:
            fail_count += 1
            if not quiet:
                print(
                    f"{label_idx} {fpath.name} ... FAIL ({time.perf_counter() - t0:.1f}s): ไฟล์ว่าง"
                )
            continue

        target, target_source = global_target, global_source
        if target is None:
            target, target_source = _resolve_target_column(df, None, interactive=interactive)
        elif target not in df.columns:
            if not quiet:
                print(
                    f"  ⚠ {fpath.name}: ไม่มีคอลัมน์ '{target}' — ข้าม target",
                    flush=True,
                )
            if args.target is not None:
                target, target_source = None, None
            else:
                target, target_source = _resolve_target_column(df, None, interactive=interactive)

        code, html_path = _run_oneliner_file(
            df,
            fpath,
            args,
            target=target,
            target_source=target_source,
            output_path=str(out_path),
            show_summary=False,
        )
        elapsed = time.perf_counter() - t0

        if code != 0 or html_path is None:
            fail_count += 1
            if not quiet:
                print(f"{label_idx} {fpath.name} ... FAIL ({elapsed:.1f}s)")
            continue

        ok_paths.append(html_path)
        if quiet:
            print(html_path)
        else:
            print(f"{label_idx} {fpath.name} ... OK → {Path(html_path).name} ({elapsed:.1f}s)")

    success = len(ok_paths)
    if not quiet:
        print(_section("สรุป batch", color))
        print(f"  สำเร็จ: {success}/{total} | ล้มเหลว: {fail_count}/{total}")
        for p in ok_paths:
            print(f"    • {p}")

    return 0 if success > 0 else 1


def _run_columns_preview(args: argparse.Namespace) -> int:
    """Dry-run: อ่านไฟล์แล้วแสดงตารางคอลัมน์ — ไม่รัน EDA."""
    input_path = Path(args.input)
    if input_path.is_dir() or args.folder:
        print("error: --columns ใช้กับไฟล์เดียว (ไม่ใช่โฟลเดอร์)", file=sys.stderr)
        return 2
    df, code = _read_input(args, quiet=True)
    if df is None:
        return code
    _print_columns_table(df)
    return 0


def _run_oneliner(args: argparse.Namespace) -> int:
    """One-liner: อ่านไฟล์/โฟลเดอร์ → thaieda.run() → บันทึก HTML blueprint report."""
    if getattr(args, "columns_preview", False):
        return _run_columns_preview(args)

    input_path = Path(args.input)
    interactive = not args.no_interactive

    if input_path.is_dir() or args.folder:
        if args.output and not args.output_dir:
            print(
                "error: ใช้ --output-dir สำหรับโหมด batch (ไม่ใช่ -o)",
                file=sys.stderr,
            )
            return 2
        return _run_oneliner_batch(args)

    quiet = args.quiet
    color = _supports_color()

    df, code = _read_input(args, quiet=quiet, color=color)
    if df is None:
        return code

    target, target_source = _resolve_target_column(df, args.target, interactive=interactive)
    if target_source == "flag" and target not in df.columns:
        print(f"error: ไม่พบคอลัมน์เป้าหมาย '{target}' ในข้อมูล", file=sys.stderr)
        return 2

    rc, html_path = _run_oneliner_file(
        df,
        input_path,
        args,
        target=target,
        target_source=target_source,
    )
    if rc == 0 and quiet and html_path:
        print(html_path)
    return rc


def _print_oneliner_summary(
    result,
    input_path: Path,
    output_path: str,
    target: str | None,
    target_source: str | None,
    report_mode: str,
    color: bool,
) -> None:
    """สรุปผล one-liner บนเทอร์มินัล."""
    overview = result.overview
    mode_label = "blueprint" if report_mode == "blueprint" else "explore"
    print(_section(f"วิเคราะห์ไฟล์: {input_path}", color))
    print(
        f"  แถว: {overview['rows']:,} | คอลัมน์: {overview['columns']} | "
        f"เซลล์ว่าง: {overview['missing_pct']}% | โหมด: {mode_label}"
    )
    if target:
        print(f"  Target: {target} ({_target_source_label(target_source)})")
    else:
        print("  Target: (ไม่ระบุ — ไม่มีแผนสร้างโมเดล)")
    if result.cleaning_report and result.cleaning_report.operations_run:
        print(f"  ทำความสะอาดแล้ว: {len(result.cleaning_report.operations_run)} การดำเนินการ")
    _print_insight_highlights(result.report)
    for note in result.notes:
        print(_paint(f"  ⚠ {note}", "warning", color))
    _print_summary_table(result.report, color)
    print(_paint(f"✓ บันทึกรายงาน HTML: {output_path}", "ok", color))


# ----------------------------------------------------------------------------
# entrypoint
# ----------------------------------------------------------------------------
_RUNNERS = {
    "profile": _run_profile,
    "run": _run_run,
    "clean": _run_clean,
    "dataset": _run_dataset,
}
_SUBCOMMANDS = frozenset(_RUNNERS)


def main(argv: list[str] | None = None) -> int:
    """จุดเริ่มต้นของคำสั่ง `thaieda`.

    ถ้า argv[0] เป็น subcommand ที่รู้จัก → ใช้โหมดเดิม (profile/run/clean/dataset)
    มิฉะนั้น → one-liner ``thaieda <input>`` สร้าง blueprint report อัตโนมัติ
    """
    _configure_stdio()
    argv = list(argv if argv is not None else sys.argv[1:])

    if not argv:
        _build_parser().print_help()
        return 0

    if argv[0] in ("--help", "-h") and len(argv) == 1:
        _build_oneliner_parser().print_help()
        return 0

    if "--version" in argv or "-V" in argv:
        print(f"thaieda {__version__}")
        raise SystemExit(0)

    if argv[0] in _SUBCOMMANDS:
        parser = _build_parser()
        args = parser.parse_args(argv)
        runner = _RUNNERS.get(args.command)
        assert runner is not None
        try:
            return runner(args)
        except KeyboardInterrupt:
            print("\nยกเลิกแล้ว", file=sys.stderr)
            return 130
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}", file=sys.stderr)
            return 1

    parser = _build_oneliner_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    try:
        return _run_oneliner(args)
    except KeyboardInterrupt:
        print("\nยกเลิกแล้ว", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
