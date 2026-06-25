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
import sys
from pathlib import Path

from thaieda import __version__

# ตัวเลือก format ที่รองรับสำหรับการอ่านไฟล์
_FORMAT_CHOICES = ["auto", "csv", "json", "jsonl"]


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
    p_profile.add_argument("--json", default=None, help="ส่งออกข้อมูลเป็น JSON ไปยังพาธที่ระบุด้วย")
    p_profile.add_argument("--no-charts", action="store_true", help="ไม่สร้างกราฟ (เร็วขึ้น)")
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
    return parser


# ----------------------------------------------------------------------------
# helper: อ่านไฟล์ + จัดการ error เป็นภาษาไทย
# ----------------------------------------------------------------------------
def _read_input(args: argparse.Namespace):
    """อ่านไฟล์ input ผ่าน thaieda.io.read_data — คืน (df, exit_code).

    ถ้าอ่านสำเร็จคืน (DataFrame, 0). ถ้าผิดพลาดคืน (None, exit_code) พร้อมพิมพ์ error เป็นภาษาไทย
    """
    from thaieda.io import read_data

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"error: ไม่พบไฟล์ {input_path}", file=sys.stderr)
        return None, 2

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

    return df, 0


# ----------------------------------------------------------------------------
# profile
# ----------------------------------------------------------------------------
def _run_profile(args: argparse.Namespace) -> int:
    from thaieda.report import ProfileReport

    df, code = _read_input(args)
    if df is None:
        return code

    if args.target is not None and args.target not in df.columns:
        print(f"error: ไม่พบคอลัมน์เป้าหมาย '{args.target}' ในข้อมูล", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    output_path = args.output or str(input_path.with_suffix(".thaieda.html"))

    report = ProfileReport(
        df,
        lang=args.lang,
        tokenizer_engine=args.tokenizer,
        make_charts=not args.no_charts,
        target_column=args.target,
        clean=args.clean,
    )
    report.run()
    report.to_html(output_path)

    if args.json:
        report.to_json(args.json)

    _print_summary(report, input_path, output_path, args.json)
    return 0


def _print_summary(report, input_path: Path, output_path: str, json_path: str | None) -> None:
    overview = report.overview
    issues = report.quality_issues
    sev_counts: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
    for i in issues:
        sev_counts[i.severity] = sev_counts.get(i.severity, 0) + 1

    print(f"✓ วิเคราะห์ไฟล์: {input_path}")
    print(
        f"  แถว: {overview['rows']:,} | คอลัมน์: {overview['columns']} | "
        f"เซลล์ว่าง: {overview['missing_pct']}%"
    )
    print("  ประเภทคอลัมน์:")
    for t, cnt in sorted(overview["type_counts"].items(), key=lambda x: -x[1]):
        print(f"    - {t}: {cnt}")
    print(
        f"  ปัญหาคุณภาพที่พบ: {len(issues)} "
        f"(วิกฤต {sev_counts['critical']}, เตือน {sev_counts['warning']}, "
        f"ข้อมูล {sev_counts['info']})"
    )
    print(f"  ความผิดปกติที่พบ: {len(report.anomalies)}")
    if report.cleaning_diff:
        total = sum(c.rows_affected for c in report.cleaning_diff)
        print(f"  ทำความสะอาดแล้ว: {len(report.cleaning_diff)} การดำเนินการ (รวม {total:,} เซลล์)")
    else:
        print(f"  คำแนะนำการทำความสะอาด: {len(report.cleaning_suggestions)}")
    _print_insight_highlights(report)
    for note in report.notes:
        print(f"  ⚠ {note}")
    print(f"✓ บันทึกรายงาน HTML: {output_path}")
    if json_path:
        print(f"✓ บันทึก JSON: {json_path}")


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

    df, code = _read_input(args)
    if df is None:
        return code

    if args.target is not None and args.target not in df.columns:
        print(f"error: ไม่พบคอลัมน์เป้าหมาย '{args.target}' ในข้อมูล", file=sys.stderr)
        return 2

    input_path = Path(args.input)
    output_path = args.output or str(input_path.with_suffix(".thaieda.html"))
    do_clean = not args.no_clean

    report = ProfileReport(
        df,
        lang=args.lang,
        tokenizer_engine=args.tokenizer,
        make_charts=not args.no_charts,
        target_column=args.target,
        clean=do_clean,
    )
    report.run()
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

    _print_run_summary(report, input_path, output_path, cleaned_path, args.json)
    return 0


def _print_run_summary(
    report,
    input_path: Path,
    html_path: str,
    cleaned_path: str | None,
    json_path: str | None,
) -> None:
    overview = report.overview
    print(f"✓ ประมวลผลไฟล์: {input_path}")
    print(
        f"  แถว: {overview['rows']:,} | คอลัมน์: {overview['columns']} | "
        f"เซลล์ว่าง: {overview['missing_pct']}%"
    )
    print(
        f"  ปัญหาคุณภาพ: {len(report.quality_issues)} | "
        f"ความผิดปกติ: {len(report.anomalies)}"
    )
    if report.cleaning_diff:
        total = sum(c.rows_affected for c in report.cleaning_diff)
        print(f"  ทำความสะอาดแล้ว: {len(report.cleaning_diff)} การดำเนินการ (รวม {total:,} เซลล์)")
    _print_insight_highlights(report)
    for note in report.notes:
        print(f"  ⚠ {note}")
    print(f"✓ บันทึกรายงาน HTML: {html_path}")
    if json_path:
        print(f"✓ บันทึก JSON: {json_path}")
    if cleaned_path:
        print(f"✓ บันทึกข้อมูลที่สะอาดแล้ว: {cleaned_path}")


# ----------------------------------------------------------------------------
# clean
# ----------------------------------------------------------------------------
def _run_clean(args: argparse.Namespace) -> int:
    import pandas as pd

    from thaieda.clean import clean_thai_text
    from thaieda.detect import ColumnType, detect_column_type, normalize_phone_number

    df, code = _read_input(args)
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
# entrypoint
# ----------------------------------------------------------------------------
_RUNNERS = {
    "profile": _run_profile,
    "run": _run_run,
    "clean": _run_clean,
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
