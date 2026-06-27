# -*- coding: utf-8 -*-
"""สคริปต์สำหรับรัน ThaiEDA AutoEDA และสร้างรายงาน HTML สำหรับข้อมูลจริง 3 ชุด."""

import sys
from pathlib import Path
import pandas as pd

# เพิ่ม src เข้า sys.path เพื่อให้สามารถ import thaieda ได้โดยตรง
sys.path.insert(0, str(Path(__file__).parent / "src"))

import thaieda
from thaieda.io import read_data


def process_dataset(file_path: Path, output_dir: Path) -> Path:
    """อ่านข้อมูล รัน ThaiEDA และบันทึกรายงานเป็น HTML.

    Args:
        file_path: Path ไปยังไฟล์ข้อมูลอินพุต
        output_dir: Path ของโฟลเดอร์สำหรับบันทึกผลลัพธ์

    Returns:
        Path ของไฟล์ HTML ที่สร้างขึ้น
    """
    print(f"กำลังเริ่มประมวลผลไฟล์: {file_path.name}")

    # ใช้ตัวอ่านข้อมูลของ ThaiEDA ที่รองรับ auto encoding และ format detection
    df = read_data(file_path)
    print(f"โหลดข้อมูลสำเร็จ: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")

    # รัน ThaiEDA process (detect -> clean -> quality -> insights -> viz -> report)
    # หมายเหตุ: เราไม่ส่งพารามิเตอร์ llm=True เนื่องจากต้องการรันแบบ offline/local เป็นหลัก
    result = thaieda.run(df, clean=True)

    # ตรวจสอบและสร้าง output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # ตั้งชื่อไฟล์รายงาน HTML
    report_name = f"{file_path.stem}_report.html"
    report_path = output_dir / report_name

    # บันทึกเป็น HTML
    result.to_html(str(report_path))
    print(f"บันทึกรายงานเรียบร้อยแล้วที่: {report_path}")

    return report_path


def main() -> None:
    """ฟังก์ชันหลักสำหรับประมวลผลข้อมูลทั้ง 3 ชุด."""
    base_dir = Path(__file__).parent
    output_dir = base_dir / "eval_reports"

    # นิยามรายการไฟล์ข้อมูลอินพุต
    datasets = [
        base_dir / "data-example/thai-datasets/thai-ecommerce-15k.csv",
        base_dir / "data-example/public-datasets/superstore.csv",
        base_dir / "data-example/thai-datasets/thai-restaurant-hybrid-20k.csv",
    ]

    print("--- เริ่มต้นการรัน AutoEDA สำหรับข้อมูลประเมินผล ---")

    for path in datasets:
        if not path.exists():
            print(f"ข้อผิดพลาด: ไม่พบไฟล์ข้อมูลที่ {path}", file=sys.stderr)
            sys.exit(1)

        try:
            process_dataset(path, output_dir)
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการประมวลผล {path.name}: {e}", file=sys.stderr)
            raise e

    print("--- ทำงานเสร็จสิ้นทั้งหมดเรียบร้อยแล้ว ---")


if __name__ == "__main__":
    main()
