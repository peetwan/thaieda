"""สร้าง fixtures สำหรับ eval — รันครั้งเดียวเพื่อ "ตรึง" ข้อมูลทดสอบให้ทำซ้ำได้.

มี 3 ส่วน:
  1. dirty-thai-labeled.csv / clean-thai.csv — สร้างด้วยมือ (defects ที่รู้คำตอบ) → commit ลง repo
  2. coffee-chain/  — ตัวอย่างจากชุด Coffee-Chain จริง แต่ "ย่อขนาดแบบรักษาความสัมพันธ์"
       (relational downsample) เพราะไฟล์จริง ORDER/TRANSACTION/INVENTORY รวมกัน ~200MB
       ใหญ่เกินกว่าจะ commit; การย่อขนาดนี้ "รักษาทุกเส้น FK→PK" จึงได้ผล P/R/F1 เท่าเดิม
  3. superstore.csv — คัดลอกตรง ๆ (เล็กพอ ~2MB) สำหรับทดสอบ tautology ของ insight engine

ส่วนที่ 1 รันได้เสมอ (ไม่พึ่งข้อมูลภายนอก). ส่วนที่ 2-3 จะรันก็ต่อเมื่อพบ data-example/
(อยู่นอก worktree, ถูก gitignore) — ถ้าไม่พบจะข้ามแบบสุภาพ เพราะไฟล์ที่ commit แล้วคือของจริงที่ eval ใช้

วิธีรัน (จาก worktree root):
    PYTHONPATH="src" .../.venv/Scripts/python.exe eval/fixtures/build_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent
# data-example อยู่ในโฟลเดอร์ repo หลัก (นอก worktree) — เดาตำแหน่งจากโครงสร้างที่รู้
# worktree ซ้อนอยู่ใต้ .../thaieda/_worktrees/<branch>/eval-framework/eval/fixtures
# จึงไล่หา data-example/ ขึ้นไปทุกชั้นของ parent (ทนต่อความลึกของ path ที่ต่างกัน)
_CANDIDATE_SOURCES = [parent / "data-example" for parent in FIXTURES_DIR.parents]

# อักขระความกว้างศูนย์ (zero-width space, U+200B) ที่มองไม่เห็นแต่ทำให้ groupby/join พัง
ZW = "​"


# ----------------------------------------------------------------------------
# ส่วนที่ 1: fixtures ภาษาไทยที่ติดป้ายคำตอบไว้
# ----------------------------------------------------------------------------
def build_dirty_thai(path: Path) -> int:
    """สร้าง dirty-thai-labeled.csv (80 แถว) ที่มี defect ตรงตาม manifest.

    คอลัมน์ที่มี defect (ตั้งใจฝัง):
      phone        -> thai_numerals  (เลขไทยปนอารบิก)
      price        -> thai_numerals
      date         -> buddhist_era    (พ.ศ. ปน ค.ศ. = critical)
      product_name -> zero_width_chars (critical)
      city         -> zero_width_chars (critical; ทำให้ unique หดเมื่อ clean)
    คอลัมน์ที่ "สะอาด" (ไม่ควรมี issue):
      customer_name (ชื่อไทยสะอาด), amount (เลขจำนวนเต็ม เลี่ยงช่วงปี พ.ศ. 2440-2599)
    """
    n = 80
    # ชื่อไทยสะอาด — เว้นวรรคเดียว ไม่มีอักขระแปลก
    names = [
        "สมชาย ใจดี",
        "สมหญิง รักไทย",
        "วิชัย มั่นคง",
        "นภา สดใส",
        "ประพันธ์ ทองคำ",
        "กนกพร แสงเดือน",
        "ธนา พงษ์ไทย",
        "ศิริพร บุญมา",
        "อนันต์ วงศ์ใหญ่",
        "มาลี ดอกไม้",
    ]
    products_clean = ["กาแฟ", "ชาไทย", "โกโก้", "ชาเขียว", "นมสด", "เอสเพรสโซ", "ลาเต้", "มอคค่า"]
    # เมืองสะอาด 4 ค่า + ตัวแปร zero-width ของ "กรุงเทพฯ" (จะยุบรวมเป็น 4 หลัง clean)
    cities_clean = ["กรุงเทพฯ", "เชียงใหม่", "ภูเก็ต", "ขอนแก่น"]

    rows = []
    for i in range(n):
        # phone: หมุนเวียน 4 รูปแบบ — รูปแบบที่ 3 เป็นเลขไทย (ฝัง thai_numerals)
        phone_variants = [
            "081-234-5678",
            "+66812345678",
            "๐๘๑-๒๓๔-๕๖๗๘",  # เลขไทย
            "08 1234 5678",
        ]
        phone = phone_variants[i % 4]

        # price: สลับเลขอารบิก/เลขไทย (ฝัง thai_numerals แบบปนอารบิก -> warning)
        price_arabic = ["80", "95", "100", "120", "150"]
        price_thai = ["๘๐", "๙๕", "๑๐๐", "๑๒๐", "๑๕๐"]
        price = price_thai[i % 5] if i % 2 == 0 else price_arabic[i % 5]

        # date: สลับ พ.ศ. (2566-2567) กับ ค.ศ. (2023-2024) -> mixed = critical
        if i % 2 == 0:
            date = ["2567-01-15", "2566-06-30", "2567-11-02"][i % 3]  # พ.ศ.
        else:
            date = ["2024-02-20", "2023-09-12", "2024-07-08"][i % 3]  # ค.ศ.

        # product_name: ทุก ๆ 3 แถวฝัง zero-width space ต่อท้าย
        pname = products_clean[i % len(products_clean)]
        if i % 3 == 0:
            pname = pname + ZW

        # city: ทุก ๆ 7 แถวใช้ "กรุงเทพ<zw>ฯ" (ตัวแปร zero-width ของ กรุงเทพฯ — จะยุบรวมหลัง clean)
        city = "กรุงเทพ" + ZW + "ฯ" if i % 7 == 0 else cities_clean[i % len(cities_clean)]

        # amount: เลขจำนวนเต็มสะอาด เลี่ยงช่วง 2440-2599 (กันถูกตีความเป็นปี พ.ศ.)
        amount = [50, 65, 120, 180, 320, 450, 780, 1200, 1850, 3200][i % 10]

        rows.append(
            {
                "customer_name": names[i % len(names)],
                "phone": phone,
                "price": price,
                "date": date,
                "product_name": pname,
                "city": city,
                "amount": amount,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")
    return len(df)


def build_clean_thai(path: Path) -> int:
    """สร้าง clean-thai.csv (100 แถว) — ข้อมูลไทยสะอาด ควรได้ 0 issue ระดับ critical/warning.

    เบอร์โทรมาตรฐาน (0 นำหน้า เลขอารบิกล้วน), ราคาเลขอารบิก, วันที่ ค.ศ. ล้วน, ไม่มี zero-width
    """
    n = 100
    names = [
        "สมชาย ใจดี",
        "สมหญิง รักไทย",
        "วิชัย มั่นคง",
        "นภา สดใส",
        "ประพันธ์ ทองคำ",
        "กนกพร แสงเดือน",
        "ธนา พงษ์ไทย",
        "ศิริพร บุญมา",
        "อนันต์ วงศ์ใหญ่",
        "มาลี ดอกไม้",
        "ชัยวัฒน์ เจริญสุข",
        "ปรียา ทองแท้",
    ]
    products = ["กาแฟ", "ชาไทย", "โกโก้", "ชาเขียว", "นมสด", "เอสเพรสโซ", "ลาเต้", "มอคค่า"]
    cities = ["กรุงเทพฯ", "เชียงใหม่", "ภูเก็ต", "ขอนแก่น", "นครราชสีมา"]
    # เบอร์มาตรฐาน 10 หลักขึ้นต้น 0 เลขอารบิกล้วน
    phones = [f"08{d}1234{d}{d}{d}" for d in range(10)]

    rows = []
    for i in range(n):
        rows.append(
            {
                "customer_name": names[i % len(names)],
                "phone": phones[i % len(phones)],
                "price": [80, 95, 100, 120, 150, 200][i % 6],
                "date": ["2024-02-20", "2023-09-12", "2024-07-08", "2023-12-01"][i % 4],
                "product_name": products[i % len(products)],
                "city": cities[i % len(cities)],
                "amount": [50, 65, 120, 180, 320, 450, 780, 1200, 1850, 3200][i % 10],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")
    return len(df)


# ----------------------------------------------------------------------------
# ส่วนที่ 2: Coffee-Chain — ย่อขนาดแบบรักษาความสัมพันธ์ (relational downsample)
# ----------------------------------------------------------------------------
def _find_coffee_src() -> Path | None:
    for base in _CANDIDATE_SOURCES:
        cand = base / "Coffee-Chain-Hackathon" / "train"
        if (cand / "ORDER.csv").is_file():
            return cand
    return None


def build_coffee_chain(dst: Path) -> dict | None:
    """ย่อขนาด Coffee-Chain ลง eval/fixtures/coffee-chain/ โดยรักษาทุกเส้น FK→PK.

    กลยุทธ์:
      * ตารางแม่ (PK) เล็กอยู่แล้ว -> คัดลอกเต็ม: CUSTOMER, STORE, PRODUCT, DATE_DIM, LOCAL_EVENT
      * ORDER (1.4M) -> สุ่ม 10,000 แถว (seed=42); order_id ยัง unique = PK ได้
      * TRANSACTION (2.9M) -> เก็บเฉพาะแถวที่ order_id อยู่ในตัวอย่าง ORDER
            (รักษาเส้น TRANSACTION.order_id → ORDER.order_id ให้ overlap = 100%)
      * INVENTORY (804K) -> สุ่ม 10,000 แถว; FK ทุกตัวชี้ตารางแม่ที่ยังเต็ม -> overlap 100%
      * PROMOTION (31K) -> สุ่ม 8,000 แถว; FK ชี้ตารางแม่ที่ยังเต็ม

    คืน dict สรุปจำนวนแถวที่เขียน หรือ None ถ้าไม่พบข้อมูลต้นทาง
    """
    src = _find_coffee_src()
    if src is None:
        return None

    dst.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    # ตารางแม่/ตารางเล็ก — คัดลอกเต็ม
    for t in ["CUSTOMER", "STORE", "PRODUCT", "DATE_DIM", "LOCAL_EVENT"]:
        df = pd.read_csv(src / f"{t}.csv")
        df.to_csv(dst / f"{t}.csv", index=False, encoding="utf-8")
        summary[t] = len(df)

    # ORDER -> สุ่ม
    order = pd.read_csv(src / "ORDER.csv")
    order_s = order.sample(n=min(10_000, len(order)), random_state=42).sort_index()
    order_s.to_csv(dst / "ORDER.csv", index=False, encoding="utf-8")
    summary["ORDER"] = len(order_s)
    order_ids = set(order_s["order_id"].tolist())

    # TRANSACTION -> กรองตาม order_id ที่สุ่มไว้ (รักษาเส้น TXN→ORDER)
    txn = pd.read_csv(src / "TRANSACTION.csv")
    txn_s = txn[txn["order_id"].isin(order_ids)]
    txn_s.to_csv(dst / "TRANSACTION.csv", index=False, encoding="utf-8")
    summary["TRANSACTION"] = len(txn_s)

    # INVENTORY -> สุ่ม (FK ชี้ตารางแม่เต็ม)
    inv = pd.read_csv(src / "INVENTORY.csv")
    inv_s = inv.sample(n=min(10_000, len(inv)), random_state=42).sort_index()
    inv_s.to_csv(dst / "INVENTORY.csv", index=False, encoding="utf-8")
    summary["INVENTORY"] = len(inv_s)

    # PROMOTION -> สุ่ม (FK ชี้ตารางแม่เต็ม)
    promo = pd.read_csv(src / "PROMOTION.csv")
    promo_s = promo.sample(n=min(8_000, len(promo)), random_state=42).sort_index()
    promo_s.to_csv(dst / "PROMOTION.csv", index=False, encoding="utf-8")
    summary["PROMOTION"] = len(promo_s)

    return summary


# ----------------------------------------------------------------------------
# ส่วนที่ 3: Superstore — คัดลอกตรง ๆ (เล็กพอ) สำหรับทดสอบ tautology
# ----------------------------------------------------------------------------
def build_superstore(dst: Path) -> int | None:
    for base in _CANDIDATE_SOURCES:
        cand = base / "public-datasets" / "superstore.csv"
        if cand.is_file():
            df = pd.read_csv(cand)
            df.to_csv(dst, index=False, encoding="utf-8")
            return len(df)
    return None


# ----------------------------------------------------------------------------
def main() -> None:
    print("=== สร้าง fixtures ===")
    d = build_dirty_thai(FIXTURES_DIR / "dirty-thai-labeled.csv")
    print(f"  dirty-thai-labeled.csv : {d} แถว")
    c = build_clean_thai(FIXTURES_DIR / "clean-thai.csv")
    print(f"  clean-thai.csv         : {c} แถว")

    coffee = build_coffee_chain(FIXTURES_DIR / "coffee-chain")
    if coffee is None:
        print("  coffee-chain/          : ข้าม (ไม่พบ data-example/ — ใช้ของที่ commit ไว้)")
    else:
        print(f"  coffee-chain/          : {coffee}")

    ss = build_superstore(FIXTURES_DIR / "superstore.csv")
    if ss is None:
        print("  superstore.csv         : ข้าม (ไม่พบ data-example/ — ใช้ของที่ commit ไว้)")
    else:
        print(f"  superstore.csv         : {ss} แถว")
    print("เสร็จสิ้น")


if __name__ == "__main__":
    main()
