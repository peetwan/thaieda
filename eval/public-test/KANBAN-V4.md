# Kanban — ThaiEDA QA v4 (Thai/Hybrid Datasets)

## Pipeline Result: 5/5 OK, 0 defects — แต่มี hidden issues 4 ข้อ

---

## 🔴 CRITICAL — Bug

### BD-1: Regex backreference `\1` พังบน pandas 3.x Arrow backend
- **Symptom**: `⚠ cleaning failed for '<col>' (dtype str): Invalid regular expression: invalid escape sequence: \1`
- **Root cause**: pandas 3.0.3 ใช้ `string[pyarrow]` เป็น default string dtype, และ Arrow regex engine ไม่รองรับ `\1` backreference
- **Affected files**:
  - `src/thaieda/clean/__init__.py` line 355: `s.str.replace(_TONE_STACK_RE.pattern, r"\1", regex=True)`
  - `src/thaieda/clean/__init__.py` line 331: `work.str.replace(pattern, lambda m: m.group(1) * max_repeat, regex=True)`
- **Fix approach**:
  - Option A: Cast series to `object` dtype ก่อน `.str.replace` (fallback to Python regex engine)
  - Option B: ใช้ `re.sub` ผ่าน `.map()` แทน `.str.replace(regex=True)` (ช้ากว่านิดหน่อย แต่ถูกทุก version)
  - Option C: ใช้ lambda function แทน raw `\1` string: `s.str.replace(pattern, lambda m: m.group(1), regex=True)` — Arrow รองรับ lambda/callable
- **Priority**: P0 — ทำให้ cleaning พังทุก text column ทุก dataset

---

## 🟠 HIGH — Performance

### PF-1: Quality checks วน per-row per-char (check_normalization)
- **Symptom**: wongnai-reviews-40k (40k rows × 540 chars avg) ใช้ 724s; wisesight (27k rows × short text) ใช้ 20s
- **Root cause**: `check_normalization` (quality/__init__.py:443) วนทุก value แล้วเรียก `_has_combining_order_issue` ที่วนทุกตัวอักษร = ~21.6M iterations
- **Also affected**: `check_zero_width`, `check_whitespace`, `check_thai_numerals` ก็วน per-row
- **Affected file**: `src/thaieda/quality/__init__.py` lines 443-460
- **Fix approach**:
  - Vectorize ด้วย `.str.contains()` / `.str.match()` สำหรับ regex-based checks
  - สำหรับ `_has_combining_order_issue`: ใช้ regex แทน per-char loop (เช่น `^[่้๊๋]` หรือ `[่้๊๋]{2,}`)
  - ใช้ short-circuit: ถ้า column ไม่มี Thai combining marks เลย ข้ามทั้ง check

### PF-2: Anomaly detection วน per-row (text anomaly checks)
- **Symptom**: 40k long reviews ช้ามาก ใน anomaly detection phase
- **Root cause**: หลาย functions วน `for pos, s in items:` ทุกแถว + บาง check วน per-char
- **Affected file**: `src/thaieda/anomaly/__init__.py` lines 585, 733, 784-785, 819
- **Hotspots**:
  - `_has_orphan_combining` (line 733): per-char loop ทุกแถว
  - NFC normalize check (line 785): `unicodedata.normalize("NFC", s)` ทุกแถว — ช้ามากสำหรับ long text
  - Script composition (line 819): per-char ratio ทุกแถว
- **Fix approach**:
  - ใช้ `.str.contains()` / regex vectorized แทน per-row loop
  - สำหรับ NFC check: ทำเฉพาะ sample (เช่น 1000 แถว) แทนทั้ง column
  - สำหรับ script composition: ใช้ `.str.count()` แทน per-char loop

---

## 🟡 MEDIUM — Insight Quality

### IN-1: Insights ล็อคที่ 7 สำหรับ text-heavy datasets
- **Symptom**: ทุก Thai text dataset ได้ 7 insights; English datasets ได้ 11-679
- **Root cause**: 
  - `_distribution_insights` และ `_correlation_insights` ต้องการ numeric columns — text datasets มีน้อย/ไม่มี
  - Cross-column insight engine (extra_insights) ต้องการ 2+ numeric columns
  - text datasets (2 cols: text + rating) มีแค่ 1 numeric column
- **Affected file**: `src/thaieda/insight/__init__.py` lines 449-502, 508+
- **Fix approach**:
  - เพิ่ม text-specific insights: sentiment distribution, text length distribution, top keywords, language mixing ratio
  - เพิ่ม insight จาก text metrics (avg length, token diversity, vocabulary richness)
  - ใช้ NER results สร้าง insights (เช่น "พบ entities 1,931 ตัว บ่งชี้ข้อมูลหลากหลาย")
- **Note**: บางส่วนของปัญหานี้เกิดจาก BD-1 — cleaning failed ทำให้ quality issues ถูก report แต่ cleaning suggestions หายไป

### IN-2: Cross-column insights ว่างสำหรับ text datasets
- **Symptom**: "ข้อค้นพบจากการวิเคราะห์คอลัมน์ผสม (1)" — มีแค่ 1 cross-column insight สำหรับ wongnai
- **Root cause**: insight_engine ต้องการ 2+ numeric/categorical columns; text datasets มีแค่ 2 cols (text + rating)
- **Fix**: อาจไม่ใช่ bug — เป็น limitation ตามธรรมชาติของ 2-col dataset. แต่ถ้าเพิ่ม text-text insights ได้ จะดีขึ้น

---

## 🟢 LOW — Cosmetic / Enhancement

### VS-1: Cross-column insight charts ว่าง (label ว่าง)
- **Symptom**: "จัดกลุ่มตาม: · ตัววัด: " — labels ว่างเปล่าใน HTML
- **Root cause**: อาจเป็น jinja template issue หรือข้อมูลไม่ถูกส่งเข้า template
- **Affected file**: น่าจะใน `src/thaieda/report/__init__.py` (template rendering)

### VS-2: "ไม่มีคำแนะนำการทำความสะอาด" ทุก dataset
- **Symptom**: Cleaning section ว่าง ในหลาย dataset — ทั้งที่ quality checks พบปัญหา
- **Root cause**: อาจเกี่ยวข้องกับ BD-1 — cleaning failed ทำให้ไม่มี suggestions

---

## ✅ DONE — No Action Needed

- HTML rendering: ไม่มี mojibake, ไม่มี tofu, ไม่มี broken images
- Thai font rendering: ถูกต้อง
- NER: ทำงานถูกต้อง (1,931 entities สำหรับ thai-ecommerce)
- Timeseries: ทำงานถูกต้อง
- Column type detection: ถูกต้อง
- All 5 datasets: 0 errors, 0 crashes