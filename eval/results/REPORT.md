# ThaiEDA Eval Report

วันที่รัน (run date): 2026-06-27 19:54  
เวอร์ชัน (version): thaieda 2.0.0 · pandas 3.0.3

> รายงานนี้สร้างอัตโนมัติจาก `eval/run_eval.py` และวัดพฤติกรรม **จริง** ของไลบรารี บน stack ปัจจุบัน — ตัวเลขไม่ได้ถูกปรับให้สวย ช่องว่างที่พบถูกบันทึกเป็น *ข้อค้นพบ* ด้านล่าง

## S1: การตรวจจับปัญหาข้อมูลไทย (Thai quality detection)

| Metric | ผล (result) | เป้าหมาย (target) | สถานะ |
|--------|------|---------|------|
| Detection Recall | 1.00 | 1.00 | ✅ |
| Detection Precision | 1.00 | 1.00 | ✅ |
| Severity Accuracy | 1.00 | ≥0.80 | ✅ |
| Clean Control | ผ่าน (0 false positives) | 0 false positives | ✅ |

- ตรวจพบถูกต้อง (TP): city:zero_width_chars, date:buddhist_era, phone:thai_numerals, price:thai_numerals, product_name:zero_width_chars

### Silent Corruption Demo
- `city`: 5 unique → 4 unique หลัง clean (ยุบ 1 กลุ่มที่ตาเห็นเหมือนกันแต่ถูกแยกด้วยอักขระล่องหน)

## S2: การค้นหาความสัมพันธ์ระหว่างตาราง (Relationship discovery)

| Metric | ผล (result) | เป้าหมาย (target) | สถานะ |
|--------|------|---------|------|
| Precision | 1.00 | 1.00 | ✅ |
| Recall | 1.00 | ≥0.90 | ✅ |
| F1 | 1.00 | ≥0.90 | ✅ |
| False Positives (hard neg) | 0 | 0 | ✅ |

- ค้นพบ 12 เส้น จากเส้นจริง 12 เส้น (TP=12, FP=0, FN=0)
- หมายเหตุ (blind spot): ไม่พบ CUSTOMER.preferred_store_id -> STORE.store_id (ชื่อคอลัมน์ต่างกัน — name match พลาด, รอแก้ v0.7; ไม่นับใน recall)

## S3: ความน่าเชื่อถือของ Insight (Insight honesty)

| Metric | ผล (result) | เป้าหมาย (target) | สถานะ |
|--------|------|---------|------|
| Noise FDR (insights บนข้อมูลสุ่ม) | 0 | ≤2 | ✅ |
| Determinism (รัน 2 ครั้ง) | เหมือนกันทุกครั้ง | เหมือนกัน | ✅ |
| Tautology (ID/รหัส เป็น measure) | 0 | 0 | ✅ |
| Planted signal found | พบ | พบ | ✅ |


## ข้อค้นพบและข้อจำกัด (Findings & limitations)

1. **[S2] blind spot `preferred_store_id → store_id`** — การจับคู่อาศัยชื่อคอลัมน์ตรงกัน จึงพลาดคู่ที่ชื่อต่าง (รอแก้ด้วย fuzzy/semantic matching ใน v0.7)

## วิธีทำซ้ำ (Reproduce)

```bash
PYTHONPATH="src" .venv/Scripts/python.exe eval/run_eval.py
```

> ข้อมูล Coffee-Chain ใน `eval/fixtures/coffee-chain/` เป็นตัวอย่าง **ย่อขนาดแบบรักษา ความสัมพันธ์** (relational downsample) จากชุดจริง (ORDER/TRANSACTION/INVENTORY รวม ~200MB) — ทุกเส้น FK→PK ถูกรักษาไว้ ผล P/R/F1 จึงเท่ากับบนข้อมูลเต็ม ดู `eval/fixtures/build_fixtures.py`
