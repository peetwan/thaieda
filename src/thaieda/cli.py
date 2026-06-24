"""Command-line interface สำหรับ ThaiEDA.

ตัวอย่าง:
    thaieda profile data.csv -o report.html --lang th --tokenizer auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thaieda import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thaieda",
        description="AutoEDA สำหรับข้อมูลภาษาไทย — Exploratory data analysis that speaks Thai",
    )
    parser.add_argument("--version", action="version", version=f"thaieda {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="command")

    p_profile = sub.add_parser("profile", help="วิเคราะห์ไฟล์ CSV และสร้างรายงาน HTML")
    p_profile.add_argument("input", help="พาธไฟล์ CSV ที่ต้องการวิเคราะห์")
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
    p_profile.add_argument("--encoding", default="utf-8", help="encoding ของไฟล์ CSV (เริ่มต้น: utf-8)")
    p_profile.add_argument("--json", default=None, help="ส่งออกข้อมูลเป็น JSON ไปยังพาธที่ระบุด้วย")
    p_profile.add_argument("--no-charts", action="store_true", help="ไม่สร้างกราฟ (เร็วขึ้น)")

    p_clean = sub.add_parser("clean", help="ทำความสะอาดข้อความไทยในไฟล์ CSV แล้วบันทึกไฟล์ใหม่")
    p_clean.add_argument("input", help="พาธไฟล์ CSV ที่ต้องการทำความสะอาด")
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
            "เลือกได้: all, encoding, zwspace, whitespace, unicode, tonemarks, repeat, numerals"
        ),
    )
    p_clean.add_argument("--encoding", default="utf-8", help="encoding ของไฟล์ CSV (เริ่มต้น: utf-8)")
    return parser


def _run_profile(args: argparse.Namespace) -> int:
    import pandas as pd

    from thaieda.report import ProfileReport

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"error: ไม่พบไฟล์ {input_path}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(input_path, encoding=args.encoding)
    except UnicodeDecodeError:
        print(
            f"error: อ่านไฟล์ด้วย encoding '{args.encoding}' ไม่สำเร็จ — "
            "ลองระบุ --encoding tis-620 หรือ cp874 สำหรับไฟล์ไทยเก่า",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: อ่าน CSV ไม่สำเร็จ: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("error: ไฟล์ CSV ว่างเปล่า (ไม่มีข้อมูล)", file=sys.stderr)
        return 1

    output_path = args.output or str(input_path.with_suffix(".thaieda.html"))

    report = ProfileReport(
        df,
        lang=args.lang,
        tokenizer_engine=args.tokenizer,
        make_charts=not args.no_charts,
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
    print(f"  คำแนะนำการทำความสะอาด: {len(report.cleaning_suggestions)}")
    for note in report.notes:
        print(f"  ⚠ {note}")
    print(f"✓ บันทึกรายงาน HTML: {output_path}")
    if json_path:
        print(f"✓ บันทึก JSON: {json_path}")


def _run_clean(args: argparse.Namespace) -> int:
    import pandas as pd

    from thaieda.clean import clean_thai_text

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"error: ไม่พบไฟล์ {input_path}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(input_path, encoding=args.encoding)
    except UnicodeDecodeError:
        print(
            f"error: อ่านไฟล์ด้วย encoding '{args.encoding}' ไม่สำเร็จ — "
            "ลองระบุ --encoding tis-620 หรือ cp874 สำหรับไฟล์ไทยเก่า",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: อ่าน CSV ไม่สำเร็จ: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("error: ไฟล์ CSV ว่างเปล่า (ไม่มีข้อมูล)", file=sys.stderr)
        return 1

    operations = [op.strip() for op in args.operations.split(",") if op.strip()]
    output_path = args.output or str(input_path.with_suffix(".cleaned.csv"))

    total_affected = 0
    results_by_op: dict[str, int] = {}
    for col in df.columns:
        # ทำความสะอาดเฉพาะคอลัมน์ข้อความ (object/StringDtype) — pandas 3.0 ใช้ StringDtype โดยปริยาย
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


def main(argv: list[str] | None = None) -> int:
    """จุดเริ่มต้นของคำสั่ง `thaieda`."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in ("profile", "clean"):
        runner = _run_profile if args.command == "profile" else _run_clean
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
