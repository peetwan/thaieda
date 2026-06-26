# ThaiEDA QA Defect List — 2026-06-26

## จาก QA pipeline v2 (9 datasets)

### Category 1: Clean/Quality — False Positives

**Defect C1: Placeholder over-flagging on "-" (single dash)**
- Files: `quality/__init__.py` (`_PLACEHOLDER_SET`, `check_placeholder_values`)
- Symptom: Superstore มี 302 ค่า "-" ใน 13 คอลัมน์ ทุกคอลัมน์ถูก flag เป็น placeholder ทำให้ report verbose มาก (13 insights ที่ซ้ำกัน)
- Root cause: `"-"` ใน `_PLACEHOLDER_SET` ถูกตรวจเจอในทุกคอลัมน์ที่มี missing แทนด้วย dash
- Fix: ถ้าคอลัมน์ไหนมี "-" มากกว่า 5% ให้ flag ครั้งเดียวใน overview ไม่ใช่ทุกคอลัมน์ หรือเพิ่ม threshold (ต้องมี >=2 placeholder types หรือ >=1% เท่านั้น)

**Defect C2: repeated-char spam false positive on ID columns**
- Files: `quality/__init__.py` (`_skip_repeated_spam_check`, `_REPEAT_SPAM_RE`)
- Symptom: Superstore Product ID 8,038 แถว (78.1%) ถูก flag ว่ามี normalization issue
- Root cause: `_skip_repeated_spam_check` skip เฉพาะ text สั้น <15 chars แต่ Product ID ยาวกว่านั้นและมี repeated chars (เช่น "FUR-BO-10001798")
- Fix: ขยาย skip rule ให้ครอบ ID-like patterns (contains digits + hyphens, alphanumeric code format)

### Category 2: Detect — Misclassification

**Defect D1: Order ID ถูกระบุเป็น date column**
- Files: `detect/__init__.py` (`_looks_like_datetime`)
- Symptom: Superstore "Order ID" ถูกระบุว่ามี "5 รูปแบบวันที่" (CA-9999-9999 ฯลฯ)
- Root cause: `_looks_like_datetime` เห็น pattern 9999-9999 แล้วเข้าใจผิดเป็น date format
- Fix: เพิ่ม guard — ถ้าคอลัมน์มี prefix ตัวอักษร (CA-, US-) ก่อนตัวเลข ไม่ใช่ date

### Category 3: Report — Rendering Issues

**Defect R1: Cross-Column Insights labels ว่าง (Breakdown/Measure)**
- Files: `report/_template.py` (line 355)
- Symptom: "Breakdown: · Measure:" โดยไม่มีชื่อคอลัมน์ ในหลาย insight cards
- Root cause: `{{ c.perspective.breakdown }}` และ `{{ c.perspective.measure }}` เป็นค่าว่าง แต่ template ยังแสดง label อยู่
- Fix: เพิ่ม `{% if c.perspective.breakdown %}` guard ก่อนแสดง Breakdown label เหมือนที่ measure ทำอยู่แล้ว

### Category 4: Performance

**Defect P1: online-retail 541K rows ใช้ 382s (6.4 min)**
- Files: `insight_engine/__init__.py` (group-by loop), `timeseries/__init__.py`
- Symptom: ช้าเกินไปสำหรับ production use
- Root cause: ต้อง profile เพื่อหา bottleneck — น่าจะเป็น insight_engine group-by combinations หรือ timeseries decomposition
- Fix: profile แต่ละ step ด้วย time.time() แล้ว optimize bottleneck (vectorize/sampling/early-exit)

### Category 5: UX / Report Quality

**Defect U1: Date parsing warnings ทุก dataset ที่มี date-like column**
- Files: `clean/__init__.py` (`normalize_dates`) หรือ `timeseries/__init__.py`
- Symptom: `UserWarning: Could not infer format` จาก pandas ทุก dataset ที่มี date column
- Root cause: ThaiEDA ไม่ได้ pass `format=` ให้ `pd.to_datetime()`
- Fix: ใช้ `pd.to_datetime(series, format='mixed')` หรือ infer format ครั้งเดียว

**Defect U2: insight_engine สร้าง insights เยอะเกินบนข้อมูลใหญ่**
- Symptom: Superstore มี 57 insights ทำให้ report ยาวมาก (2.2MB)
- Fix: อาจจะต้องมี `max_insights` parameter หรือ deduplicate similar insights