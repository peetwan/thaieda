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
    for note in report.notes:
        print(f"  ⚠ {note}")
    print(f"✓ บันทึกรายงาน HTML: {output_path}")
    if json_path:
        print(f"✓ บันทึก JSON: {json_path}")


def main(argv: list[str] | None = None) -> int:
    """จุดเริ่มต้นของคำสั่ง `thaieda`."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "profile":
        try:
            return _run_profile(args)
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
