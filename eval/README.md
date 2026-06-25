# ThaiEDA Eval Framework

ชุดวัดความสามารถของ ThaiEDA บนข้อมูลจริง/สังเคราะห์ที่ "รู้คำตอบ" (labeled ground truth)
แล้วสรุปออกมาเป็นตัวเลขที่ตรวจซ้ำได้ — ใช้เป็นทั้ง **regression guard** (ผ่าน `results.json`)
และ **แหล่งตัวเลขสำหรับ README** (ผ่าน `results/REPORT.md`)

หลักการ: *รายงานตามจริง* — ตัวเลขคือพฤติกรรมจริงของไลบรารีบน stack ปัจจุบัน ไม่ปรับให้สวย;
ช่องว่างที่เจอถูกบันทึกเป็น "ข้อค้นพบ" (findings) เพื่อเป็นการบ้านของเวอร์ชันถัดไป

---

## วิธีรัน (How to run)

ต้องรันผ่าน **venv ของโปรเจกต์** (มี pandas/numpy/scipy) และตั้ง `PYTHONPATH=src` ให้ import `thaieda` ได้:

```bash
# จาก worktree root
PYTHONPATH="src" .venv/Scripts/python.exe eval/run_eval.py
```

> บนเครื่องนี้ venv อยู่ที่ `C:/Users/User/Desktop/Hermes_Peet/scratch/thaieda/.venv/Scripts/python.exe`

ผลลัพธ์จะถูกเขียนไปที่:
- `eval/results/results.json` — machine-readable (สำหรับ CI/regression; ไม่มี timestamp จึง diff เสถียร)
- `eval/results/REPORT.md` — human-readable (ตาราง + การตีความ + ข้อค้นพบ; commit ไว้ลิงก์จาก README หลัก)

รันทีละ scenario เพื่อ debug ก็ได้:

```bash
PYTHONPATH="src" .venv/Scripts/python.exe eval/scenarios/s1_thai_quality.py
```

### ข้อกำหนด (Requirements)

| สิ่งที่ใช้ | สถานะใน venv | หมายเหตุ |
|-----------|-------------|---------|
| pandas, numpy, scipy | ✅ จำเป็น | scipy ทำให้ p-value ใน insight engine ถูกต้อง |
| PyYAML | ❌ ไม่มีใน venv | manifest จึงเป็น **JSON** (stdlib `json`) ไม่ใช่ YAML |

---

## scenario แต่ละตัววัดอะไร (What each scenario measures)

### S1 — `s1_thai_quality.py` · การตรวจจับปัญหาข้อมูลไทย
รัน `detect_all()` + `run_quality_checks()` แบบ end-to-end บน `fixtures/dirty-thai-labeled.csv`
(80 แถว ที่ฝัง defect ไว้และระบุคำตอบใน `manifests/dirty-thai-labeled.expected.json`) แล้ววัด:
- **Recall / Precision / F1** ของการตรวจจับ (ระดับคู่ `(คอลัมน์, ชนิด issue)`)
- **Severity accuracy** — ความรุนแรงตรงกับที่ระบุไหม (เช่น พ.ศ.ปน ค.ศ. ต้องเป็น `critical`)
- **Silent corruption demo** — `city.nunique()` ก่อน/หลังลบ zero-width (โชว์ว่าอักขระล่องหนทำกลุ่มแตก)
- **Clean control** — รันบน `fixtures/clean-thai.csv` (100 แถวสะอาด) ต้องได้ **0** critical/warning

### S2 — `s2_relationships.py` · การค้นหาความสัมพันธ์ระหว่างตาราง
รัน `profile_dataset()` บน `fixtures/coffee-chain/` แล้วเทียบเส้น FK→PK ที่ค้นพบกับ
`manifests/coffee-chain-schema.expected.json`:
- **Precision / Recall / F1** เทียบ `true_edges`
- **hard_negative_violations** — คู่ที่ "ชื่อพ้องแต่ไม่ใช่ความสัมพันธ์" ต้องไม่ถูกค้นพบ (target 0)
- **known_blind_spot** — ความสัมพันธ์ที่ชื่อคอลัมน์ต่างกัน (จับคู่ด้วยชื่อจึงพลาด) — เป็นข้อมูลประกอบ ไม่หักคะแนน

### S3 — `s3_insight_honesty.py` · ความน่าเชื่อถือของ insight engine
กดดัน `discover_insights()` 4 ด้าน:
- **Noise FDR** — ข้อมูลสุ่มล้วน (seed 42, 500 แถว, 5 categorical × 5 numeric i.i.d.) ต้องได้ insight ≈ 0 (BH correction กรองออก)
- **Determinism** — รันซ้ำต้องได้ card เหมือนกันเป๊ะ
- **Tautology** — บน Superstore (มี `Row ID`, `Postal Code`) ต้องไม่ใช้คอลัมน์ ID/รหัสเป็น measure
- **Sanity recall** — สัญญาณที่ฝังชัด (กลุ่มเดียว ~3 เท่า) ต้องโผล่ใน top-5

---

## การตีความผล (How to interpret)

ทุก metric ใน `REPORT.md` มีคอลัมน์ **ผล / เป้าหมาย / สถานะ** (✅ ถึงเป้า, ⚠️ ต่ำกว่าเป้า)
- ✅ ทั้งหมด = ความสามารถนั้นทำงานตามที่ออกแบบ
- ⚠️ = พบช่องว่างจริง — อ่านรายละเอียดในหัวข้อ **ข้อค้นพบและข้อจำกัด** ของ `REPORT.md`

`results.json` ใช้สำหรับ CI: เทียบค่า metric ใน `s1_thai_quality` / `s2_relationships` /
`s3_insight_honesty` กับ baseline เพื่อจับ regression (เช่น recall ตก, hard-negative หลุด)

> ค่าใน `meta` (เวอร์ชัน thaieda/pandas) ช่วยอธิบายผลที่ขึ้นกับ stack — เช่น พฤติกรรม dtype ของ pandas 3.x

---

## fixtures มาจากไหน (Provenance)

| ไฟล์ | ที่มา | ขนาด |
|------|------|------|
| `dirty-thai-labeled.csv` / `clean-thai.csv` | สร้างด้วยมือ (deterministic) ใน `build_fixtures.py` | ~80 / ~100 แถว |
| `coffee-chain/*.csv` | **ย่อขนาดแบบรักษาความสัมพันธ์** จากชุด Coffee-Chain จริง | ~2.5 MB |
| `superstore.csv` | คัดลอกตรงจาก public-datasets | ~2.3 MB |

**ทำไม Coffee-Chain ถึงย่อขนาด?** ไฟล์จริง `ORDER`/`TRANSACTION`/`INVENTORY` รวมกัน ~200 MB
(หลายล้านแถว) ใหญ่เกินกว่าจะ commit และขัดกับหลัก "fixtures ต้องเล็ก ตรวจด้วยมือได้".
`build_fixtures.py` จึงสุ่ม `ORDER`/`INVENTORY`/`PROMOTION` (seed 42) และ **กรอง `TRANSACTION`
ให้เหลือเฉพาะ order ที่สุ่มไว้** เพื่อรักษาเส้น `TRANSACTION.order_id → ORDER.order_id`
ส่วนตารางแม่ (PK) เล็กอยู่แล้วจึงคัดลอกเต็ม — **ทุกเส้น FK→PK ถูกรักษาไว้ครบ** ผล P/R/F1
จึงเท่ากับบนข้อมูลเต็ม

### สร้าง fixtures ใหม่ (Regenerate)

```bash
PYTHONPATH="src" .venv/Scripts/python.exe eval/fixtures/build_fixtures.py
```

ส่วนข้อความไทยสร้างได้เสมอ; ส่วน coffee-chain/superstore จะสร้างก็ต่อเมื่อพบโฟลเดอร์
`data-example/` ของ repo หลัก (อยู่นอก worktree, ถูก gitignore) — ถ้าไม่พบจะข้าม แล้วใช้ไฟล์ที่
commit ไว้แทน

---

## โครงสร้าง (Layout)

```
eval/
  README.md                 # ไฟล์นี้
  run_eval.py               # orchestrator → results/REPORT.md + results.json
  manifests/                # ground truth (JSON)
    dirty-thai-labeled.expected.json
    coffee-chain-schema.expected.json
  scenarios/
    s1_thai_quality.py
    s2_relationships.py
    s3_insight_honesty.py
  fixtures/
    build_fixtures.py       # สร้าง/ย่อขนาด fixtures (provenance)
    dirty-thai-labeled.csv
    clean-thai.csv
    superstore.csv
    coffee-chain/*.csv
  results/                  # generated (committed)
    REPORT.md
    results.json
```
