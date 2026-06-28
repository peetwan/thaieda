<p align="center">
  <img src="docs/og-image.png" alt="ThaiEDA — AutoEDA & Smart Data Cleaning for Thai Data" width="100%" />
</p>

# ThaiEDA

> **AutoEDA และเครื่องมือทำความสะอาดข้อมูลอัจฉริยะสำหรับข้อมูลภาษาไทยและข้อมูลผสมภาษา**
>
> **One-line Exploratory Data Analysis and Smart Data Cleaning for Thai and Mixed-Language Datasets.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions/workflows/ci.yml)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

**ThaiEDA** ออกแบบมาเพื่อตอบคำถามเดียว: *"ชุดข้อมูลนี้มีความน่าเชื่อถือเพียงใด และควรเริ่มต้นสำรวจจุดไหนก่อน?"*

เครื่องมือวิเคราะห์ข้อมูล (profiling) ทั่วไปมักจะตรวจนับค่าว่าง (missing values) และวาดกราฟมาตรฐานได้ดี แต่มักจะมองข้ามปัญหาเฉพาะตัวของข้อมูลภาษาไทย เช่น ปี พ.ศ., เลขไทย, อักขระเว้นวรรคที่มองไม่เห็น (zero-width space), ปัญหาการแสดงผลตัวอักษรผิดเพี้ยน (mojibake), รูปแบบเบอร์โทรศัพท์/บัตรประชาชน หรือแม้กระทั่งฟอนต์ในกราฟที่อ่านไม่ออก ซึ่ง ThaiEDA สามารถจัดการกับปัญหาเหล่านี้ให้โดยอัตโนมัติในกระบวนการทำงานปกติ (pipeline) ทำให้คุณไม่ต้องเสียเวลาเตรียมข้อมูล (preprocess) เองทีละคอลัมน์

ThaiEDA answers one simple question: **"Can I trust this dataset, and what should I explore first?"** It treats Thai-specific data issues — Buddhist Era dates, Thai numerals, zero-width spaces, encoding errors, local phone formats, and chart fonts — as normal problems to fix automatically.

- **Repository:** [github.com/peetwan/thaieda](https://github.com/peetwan/thaieda)
- **PyPI:** [pypi.org/project/thaieda](https://pypi.org/project/thaieda/)
- **License:** Apache-2.0
- **Current version:** v2.3.0

---

## คุณสมบัติหลัก / Key Features

| หมวด | ไทย | English |
|------|-----|---------|
| **Detection** | ตรวจสอบประเภทคอลัมน์ภาษาไทย/ภาษาอังกฤษ, ค้นหาตัวเลขที่แฝงอยู่ในข้อความ, ตรวจจับปี พ.ศ., ตรวจสอบรูปแบบเบอร์โทรศัพท์และบัตรประชาชน รวมถึงข้อมูลที่มีหลายภาษาผสมกัน | Smart column & type detection for Thai/mixed data |
| **One-liner API** | `run()` / `EDA()` — ตรวจจับ ทำความสะอาด ตรวจสอบคุณภาพ ค้นหาข้อมูลเชิงลึก แสดงแผนภูมิ และรายงานผลลัพธ์เป็น HTML ได้ในคำสั่งเดียว | Full AutoEDA pipeline in one function call |
| **Blueprint mode** | รายงานสรุปที่กระชับและนำไปใช้งานต่อได้ทันที เหมาะสำหรับงาน Machine Learning หรือข้อมูลตารางทั่วไปเมื่อกำหนดคอลัมน์เป้าหมาย (`target_column`) | Shorter actionable reports with modeling blueprint |
| **Cleaning** | จัดการ Unicode, อักขระเว้นวรรคที่มองไม่เห็น (zero-width), แปลงเลขไทยเป็นอารบิก, แปลงปี พ.ศ. → ค.ศ., ปรับรูปแบบสกุลเงิน (พร้อมตัวกันไม่ให้แปลงคอลัมน์ข้อความอิสระที่เอ่ยถึงราคา เช่น รีวิว ทั้งคอลัมน์เป็น NaN), ลบแถวซ้ำ, จัดการค่าว่าง (รวมถึงการเติมค่าด้วย Machine Learning) และการลดขนาดหน่วยความจำ (downcast) — พร้อมเตือนเมื่อการแปลง/ลบแถวกระทบข้อมูลเกินเกณฑ์ | Thai-aware cleaning with audit trail + free-text/row-loss guardrails |
| **Insights** | ระบบวิเคราะห์ความสัมพันธ์ระหว่างคอลัมน์ — ตรวจจับความสัมพันธ์ของข้อมูล, ค่าผิดปกติ (outliers) แบบเลือกวิธีตามการกระจาย (z-score / MAD / IQR / GESD เพื่อรองรับข้อมูลเบ้), แนวโน้ม, ความขัดแย้งของซิมป์สัน (Simpson's paradox) และปัญหาข้อมูลรั่วไหล (**target leakage**) ทั้งในระดับ Tier A และ Tier B พร้อมกรองข้อค้นพบที่ไม่มีความหมายออก (รหัส/ดัชนีแถว, คอลัมน์ที่แทบคงที่, รหัส/รุ่นที่คล้ายกัน) — และรู้จักคอลัมน์ตัวเลขที่ไม่ใช่ค่าวัด (id/code/zip/lat/long) จึงไม่นำไปคิดสถิติ/correlation/outlier แต่ให้คำแนะนำตามบริบทแทน (geo → วิเคราะห์เชิงพื้นที่, id → ใช้เป็นคีย์เชื่อม) เพื่อลดสัญญาณรบกวน | Statistical insight discovery with BH correction + distribution-aware robust outliers + geo/ID-aware advisories, low false positives |
| **Quality score** | คะแนนคุณภาพ (0–100, เกรด A–F) ถ่วงน้ำหนักตาม **ขนาดของปัญหา** (severity × สัดส่วนที่กระทบ) และ normalize ด้วยจำนวนคอลัมน์ จึงสะท้อนปัญหาจริง — คอลัมน์ว่างทั้งคอลัมน์หรือค่าหายจำนวนมากหักคะแนนอย่างมีนัยสำคัญ ไม่ใช่ปล่อยให้ dataset ใหญ่ได้ A ฟรี ๆ | Magnitude-weighted data quality score |
| **Reports** | รายงานรูปแบบ HTML รองรับทั้งภาษาไทยและภาษาอังกฤษ, บทสรุปแบบบรรยายสำหรับผู้บริหาร (offline narrative) และการแสดงผลบน Jupyter Notebook ที่สวยงาม — รวมข้อค้นพบของคอลัมน์เดียวเป็นการ์ดเดียว (critical → warning → info) และยุบการ์ด info ที่เกินเข้า "ดูเพิ่มเติม", เว้นวรรคไทย/อังกฤษให้อ่านง่าย และระบุหน่วย/ตัวส่วนของเปอร์เซ็นต์ให้ชัด (เช่น แถวที่กระทบ vs เซลล์ที่แก้) | Executive HTML reports + grouped insight cards, clearer Thai/English text & denominators |
| **Performance & Robustness** | word cloud และ pipeline ทำความสะอาดเร็วขึ้นมากบนข้อมูลข้อความขนาดใหญ่ (เช่น `run(make_charts=True)` บนข้อมูลภาษาไทย ~24k แถว เร็วขึ้นจาก ~68s เหลือ ~14s) และ degrade อย่างสุภาพเมื่อ backend ตัดคำ/NER ไม่พร้อมใช้งาน (ออฟไลน์/ติดตั้งไม่ครบ) แทนการล้มทั้ง pipeline | Faster on large text data; graceful offline degradation |
| **Schema** | ตรวจหาคีย์หลักและคีย์นอก (PK/FK) เชื่อมโยงข้ามไฟล์ พร้อมสร้างแผนภาพ Mermaid diagram | Multi-file schema discovery |
| **LLM** | สรุปวิเคราะห์ด้วยโมเดลภาษาขนาดใหญ่ พร้อมโหมดรักษาความเป็นส่วนตัวของข้อมูล 5 ระดับ (รองรับ OpenAI, Anthropic และ Ollama) | Privacy-preserving optional LLM summaries |

---

## ติดตั้ง / Installation

ต้องการ **Python 3.10+**

```bash
pip install thaieda
```

ตั้งแต่เวอร์ชัน `v2.2.0` เป็นต้นไป แพ็กเกจและ dependencies ทั้งหมดจะถูกติดตั้งพร้อมใช้งานโดยอัตโนมัติ:

| แพ็กเกจหลัก (Core Packages) | คำอธิบายการใช้งาน |
|-------------------------|----------------|
| `pandas`, `numpy`, `matplotlib`, `Jinja2` | การวิเคราะห์พื้นฐาน, การคำนวณโครงสร้างข้อมูล และรายงาน HTML |
| `pythainlp`, `nlpo3`, `attacut` | การตัดคำภาษาไทย (ตัดคำธรรมดา, ตัดคำความเร็วสูงด้วย Rust และ Deep Learning) |
| `plotly`, `wordcloud` | การสร้างแผนภูมิโต้ตอบได้และ word cloud |
| `ftfy` | การแก้ไขปัญหาตัวอักษรไทยผิดเพี้ยน (mojibake) |
| `scikit-learn` | การเติมค่าว่างและการตรวจจับค่าผิดปกติด้วย Machine Learning |
| `scipy` | การวิเคราะห์ทางสถิติและความสัมพันธ์ของตัวแปรเป้าหมาย |
| `statsmodels` | การแยกองค์ประกอบของอนุกรมเวลา (Timeseries Decomposition) |
| `python-crfsuite` | ระบบจดจำเอนทิตีภาษาไทย (Thai Named Entity Recognition) |
| `rapidfuzz` | การค้นหาข้อมูลกลุ่มที่ใกล้เคียงกันด้วย Fuzzy Matching |
| `chardet` | การตรวจสอบรหัสอักขระ (encoding) ของไฟล์โดยอัตโนมัติ |
| `openpyxl`, `pyarrow` | การอ่านและเขียนไฟล์ Excel และ Parquet |
| `litellm` | การสรุปวิเคราะห์และประมวลผลข้อมูลร่วมกับ LLM |

### แพ็กเกจสำหรับการพัฒนา / Development Dependencies

สำหรับการทดสอบและพัฒนาโครงการ สามารถติดตั้งเพิ่มเติมได้ด้วย:

```bash
pip install "thaieda[dev]"
```

---

## เริ่มต้นเร็ว / Quickstart

### One-liner CLI

เมื่อติดตั้งเรียบร้อยแล้ว สามารถเริ่มรันวิเคราะห์ได้ทันทีด้วยคำสั่งเดียว ซึ่งจะทำการอ่านข้อมูล → ทำความสะอาด → และสร้างรายงานผลลัพธ์แบบ HTML blueprint ให้โดยอัตโนมัติ:

```bash
pip install thaieda
thaieda mydata.csv
# → สร้าง mydata-report.html ในโฟลเดอร์เดียวกับไฟล์
```

พารามิเตอร์และตัวเลือกที่ใช้งานบ่อย:

```bash
thaieda data.csv -o report.html          # ระบุพาธ output
thaieda data.csv --target clicked        # คอลัมน์เป้าหมาย (หรือเดาอัตโนมัติจากชื่อ)
thaieda data.csv -t clicked              # ย่อ — เหมือน --target
thaieda data.csv --columns               # ดูคอลัมน์ + target ที่น่าจะเป็น (ไม่รัน EDA)
thaieda data.csv -y                      # ข้าม prompt เลือก target (CI/batch)
thaieda data.csv --explore               # รายงาน EDA แบบเต็ม (แทน blueprint)
thaieda data.csv --lang en --no-clean    # ภาษาอังกฤษ, ไม่ทำความสะอาด
```

**Target column:** หากไม่กำหนดตัวเลือก `-t` หรือ `--target` ระบบ ThaiEDA จะวิเคราะห์และคาดเดาจากชื่อคอลัมน์โดยอัตโนมัติ (เช่น `target`, `label`, `clicked`, `churn`, `survived`, `y`, `class`, `outcome`, `income`, `quality`, `response`, `default`, `fraud`, `exit` และชื่ออื่น ๆ ที่มีคำเหล่านี้ประกอบอยู่) ในกรณีที่รันผ่านหน้าจอ Terminal (TTY) ระบบจะแสดงรายการคอลัมน์เพื่อให้คุณเลือก (สามารถกด Enter เพื่อข้าม) และหากต้องการรันแบบอัตโนมัติโดยไม่แสดงคำถาม ให้ใช้ตัวเลือก `-y` หรือ `--no-interactive`

#### เลือก target อย่างไร?

1. **ดูคอลัมน์ก่อน** — เพื่อตรวจสอบเบื้องต้นโดยไม่ต้องวิเคราะห์ข้อมูลเต็มรูปแบบ:
   ```bash
   thaieda data.csv --columns
   # หรือ
   thaieda data.csv --preview
   ```
   แสดงตารางข้อมูลประกอบด้วย ลำดับที่ (`#`), ชื่อคอลัมน์, ชนิดข้อมูล (dtype), จำนวนค่าที่ไม่ซ้ำ (unique values) และเครื่องหมายดาว (★) หน้าคอลัมน์ที่ระบบคาดเดาว่าเป็นตัวแปรเป้าหมาย (target)

2. **กำหนดด้วยตนเอง** — ระบุพารามิเตอร์โดยตรง เช่น `thaieda data.csv -t clicked` หรือ `--target churn`

3. **ไม่กำหนดตัวเลือก** — ระบบจะตรวจหาโดยอัตโนมัติจากชื่อคอลัมน์ หรือสอบถามในหน้าจอ Terminal (หากรองรับระบบตอบรับโต้ตอบ)

4. **หลังเสร็จสิ้นกระบวนการ** — หน้าจอ CLI จะแสดงผล `Target: clicked (auto)` / `(จาก --target)` / `(ไม่ระบุ — ไม่มีแผนสร้างโมเดล)`

**การประมวลผลแบบกลุ่ม (Batch) ในโฟลเดอร์:** ทำการประมวลผลข้อมูลทุกไฟล์ที่พบในโฟลเดอร์ และบันทึกรายงานแยกเป็นรายไฟล์ในรูปแบบ `{stem}-report.html` (ซึ่งแตกต่างจากการสั่งงานด้วย `thaieda dataset` ที่จะเป็นการเชื่อมโยงโครงสร้างข้อมูล (schema) ของหลาย ๆ ตารางเข้าด้วยกัน)

```bash
cd ~/my-data/
thaieda .                                # batch ทุก CSV/TSV/JSON/Excel/Parquet
thaieda /path/to/folder --output-dir reports/
thaieda folder/ -t clicked -y            # target เดียวกันทุกไฟล์, ไม่ถาม interactive
```

รองรับไฟล์ข้อมูลหลากหลายรูปแบบ เช่น CSV, TSV, JSON, Excel, Parquet โดยระบบจะตรวจสอบรูปแบบไฟล์ (format) และการเข้ารหัสอักขระ (encoding) ให้โดยอัตโนมัติ

### Python API

```python
import pandas as pd
import thaieda

# ข้อมูลจำลองที่มีปัญหาพบบ่อยในข้อมูลไทย
data = {
    "name": ["สมชาย\u200bรักไทย", "สมหญิง   ใจดี", "นายดำ ๐๑"],
    "birth_year": [2530, 2532, 2528],           # ปี พ.ศ.
    "sales": ["฿1,200", "฿3,500", "฿10,000"],
    "phone": ["081-234-5678", "+66898765432", "๐๒-๓๔๕-๖๗๘๙"],
}
df = pd.DataFrame(data)

# EDA ครบวงจร (ค่าเริ่มต้น lang="th" → รายงานภาษาไทย)
result = thaieda.run(df, clean=True, lang="th")
result.to_html("quickstart-report.html")

print(result.cleaned_df)
print(result.quality_score)   # คะแนนคุณภาพ 0–100
```

---

## One-liner API: `run()` / `EDA()`

ฟังก์ชัน `thaieda.run(df)` และ `thaieda.EDA(df)` ทำงานในลักษณะเดียวกัน โดยจะส่งคืนผลลัพธ์เป็นออบเจกต์ `EDAResult`

```python
result = thaieda.run(
    df,
    clean=True,                  # ทำความสะอาดก่อนวิเคราะห์ (default: True)
    handle_missing="flag",         # flag | median | mode | drop | unknown | ml
    remove_duplicates=True,
    downcast=True,
    lang="th",                     # "th" | "en" — ภาษารายงาน
    report_mode="explore",         # "explore" | "blueprint"
    target_column=None,            # คอลัมน์เป้าหมาย (สำหรับ ML / blueprint)
    make_charts=True,
    timeseries=True,
    insights_engine=True,
    insights_top=8,
    narrative=True,                # บทสรุป offline ไม่ต้องใช้ LLM
    llm=False,                     # เปิด LLM analysis
    privacy="insight_only",        # insight_only | synthetic | anonymized | dp_noise | full
    provider="openai",             # openai | anthropic | ollama
)
```

### `EDAResult` — ผลลัพธ์หลัก

| Property / Method | คำอธิบาย |
|-------------------|----------|
| `.report` | รายงานสรุปผลวิเคราะห์ข้อมูล (`ProfileReport`) ฉบับสมบูรณ์ |
| `.cleaned_df` | ข้อมูลในรูปแบบ DataFrame ที่ผ่านการทำความสะอาดแล้ว |
| `.overview` | ข้อมูลภาพรวมเบื้องต้น เช่น จำนวนแถว จำนวนคอลัมน์ และประเภทของข้อมูล |
| `.quality_issues` | รายการปัญหาด้านคุณภาพของข้อมูลที่ตรวจพบ |
| `.quality_score` | คะแนนประเมินคุณภาพข้อมูลระหว่าง 0–100 พร้อมระดับเกรด |
| `.quality_comparison` | ตารางเปรียบเทียบคุณภาพของข้อมูลระหว่างก่อนและหลังทำความสะอาด |
| `.cleaning_report` | รายงานประวัติขั้นตอนการทำความสะอาดข้อมูลอย่างละเอียด (Audit Trail) |
| `.insights` | รายการข้อมูลเชิงลึกและข้อค้นพบที่ตรวจพบโดยอัตโนมัติ |
| `.anomalies` | รายการข้อมูลที่ผิดปกติ (outliers) และความผิดปกติในรูปแบบข้อความ |
| `.narrative` | บทสรุปผู้บริหารในรูปแบบคำบรรยายตามโครงสร้างมาตรฐาน |
| `.llm_response` | คำสรุปวิเคราะห์จากระบบโมเดลภาษาขนาดใหญ่ (เมื่อเปิดใช้งาน `llm=True`) |
| `.notes` | รายการคำเตือนหรือข้อสังเกตต่าง ๆ ที่เกิดขึ้นระหว่างการทำงาน |
| `.to_html(path)` | คำสั่งสำหรับบันทึกรายงานออกมาเป็นไฟล์ HTML |
| `.to_dict()` | ส่งออกข้อมูลผลลัพธ์ในรูปแบบ Dictionary |
| `.to_json(path)` | ส่งออกข้อมูลผลลัพธ์และบันทึกเป็นไฟล์ JSON |

หากรันบน Jupyter Notebook: เพียงแค่พิมพ์ตัวแปร `result` ระบบจะแสดงผลรายงาน HTML ให้โดยอัตโนมัติผ่านเมธอด `_repr_html_`

---

## Blueprint mode — รายงานสำหรับงาน ML

การกำหนด `report_mode="blueprint"` จะเป็นการสร้างรายงานขนาดกระชับที่เน้นคำแนะนำการดำเนินการในขั้นตอนถัดไป เหมาะสำหรับเตรียมชุดข้อมูลตารางเพื่อสร้างโมเดล Machine Learning ที่มีตัวแปรเป้าหมาย (เช่น การจำแนกประเภท (classification), อัตราการคลิก (CTR) หรือการยกเลิกบริการ (churn))

```python
result = thaieda.run(
    df,
    report_mode="blueprint",
    target_column="clicked",   # หรือ churn, income, quality, …
    lang="th",
    clean=True,
)
result.to_html("modeling-blueprint.html")
```

**สิ่งที่ระบบวิเคราะห์ในโหมด Blueprint:**

- **วิเคราะห์จัดกลุ่มข้อมูลอัตโนมัติ** — ตรวจสอบและจำแนกประเภทโมเดล (`ml_tabular`) เมื่อพบตัวแปรเป้าหมายและคอลัมน์คุณลักษณะ (feature columns) หลายคอลัมน์
- **ข้ามการวิเคราะห์อนุกรมเวลา** — ข้ามกระบวนการวิเคราะห์อนุกรมเวลา (timeseries) ในข้อมูลประเภทเหตุการณ์ เช่น ข้อมูลการบันทึก (event log/impression) ที่แต่ละแถวไม่ใช่ลำดับเวลาต่อเนื่อง
- **ลดจำนวนแผนภูมิลง** — ตัดการแสดงผลแผนภูมิที่มีความซับซ้อนและขนาดใหญ่ เช่น scatter matrix และ violin chart ออกไป
- **ค้นหาข้อมูลเชิงลึกได้กว้างขึ้น** — เพิ่มการแสดงผลข้อมูลเชิงลึก (insights) ได้สูงสุดถึง 12 รายการ
- **วางโครงร่างพิมพ์เขียวสำหรับทำโมเดล (Modeling Blueprint)** (เมื่อมีการกำหนดตัวแปรเป้าหมาย `target_column`):
  - **ค่าสถิติเริ่มต้นของตัวแปรเป้าหมาย (Target baseline)** — ตรวจสอบสัดส่วนของคลาส (class balance) และอัตราส่วนค่าที่เป็นบวก (positive rate)
  - **การตรวจจับข้อมูลรั่วไหล (Leakage detection)** — ค้นหาปัญหาข้อมูลรั่วไหลแบ่งตามความรุนแรง ได้แก่ Tier A (ระดับวิกฤต) และ Tier B (ระดับแจ้งเตือน)
  - **คุณลักษณะที่มีความสัมพันธ์สูง (Strong features)** — ระบุคอลัมน์ที่มีผลต่อตัวแปรเป้าหมายอย่างมีนัยสำคัญทางสถิติ
  - **คอลัมน์ที่ควรคัดออก (Columns to drop)** — แนะนำคอลัมน์ประเภทตัวบ่งชี้เฉพาะ (ID) หรือคอลัมน์ที่ไม่มีประโยชน์ต่อการสร้างแบบจำลอง
  - **ขั้นตอนการดำเนินงานถัดไป (Next steps)** — รายการตรวจสอบ (checklist) สิ่งที่ควรทำก่อนเริ่มฝึกสอนโมเดล (training model)

### Target leakage tiers

| Tier | ความรุนแรง | ตัวอย่าง heuristic |
|------|------------|-------------------|
| **A — critical** | ห้ามใช้คุณลักษณะนี้ในการสร้างแบบจำลองโดยเด็ดขาด | ข้อมูลคอลัมน์ซ้ำกับตัวแปรเป้าหมาย, ค่าสัมประสิทธิ์สหสัมพันธ์ \|corr\| ≥ 0.98, มีความสัมพันธ์เชื่อมโยงกันอย่างแน่นอนทางคณิตศาสตร์ (deterministic mapping) หรือสามารถจำแนกกลุ่มเป้าหมายได้เกือบสมบูรณ์ (near-perfect separation) |
| **B — warning** | ควรตรวจสอบความถูกต้องก่อนตัดสินใจใช้งาน | ชื่อคอลัมน์บ่งบอกถึงความเป็นตัวแทนทางอ้อม (proxy) เช่น (`_ctr`, `_rate`, `historical_` และอื่น ๆ) หรือมีความเกี่ยวข้องสัมพันธ์กันในระดับสูง |

---

## สูตรใช้งาน / Core Recipes

### 1. ทำความสะอาดแยก (`clean`)

```python
cleaned_df, report = thaieda.clean(
    df,
    handle_missing="ml",       # หรือ flag, median, mode, drop, unknown
    remove_duplicates=True,
    fix_dates=True,
    fix_numerals=True,
    fix_encoding=True,
    downcast=True,
)
report.to_json("cleaning-audit.json")
```

ตัวเลือก `skip_id_like=True` (ค่าเริ่มต้น) จะช่วยป้องกันไม่ให้ข้อมูลรหัสสินค้า (product ID) หรือหมายเลขซีเรียล (serial number) สูญเสียรูปแบบเดิมจากการทำความสะอาดอักขระที่ซ้ำซ้อนกัน

### 2. เปรียบเทียบชุดข้อมูล (`compare`)

```python
from thaieda import compare

diff = compare(train_df, prod_df, labels=("train", "prod"))
print(diff["schema_diff"])
print(diff["distribution_drift"])
print(diff["categorical_drift"])
```

### 3. Schema หลายไฟล์

```python
from thaieda import DatasetReport, profile_dataset

dataset = profile_dataset("data/warehouse", validate_values=True)
DatasetReport(dataset, lang="th").to_html("schema-report.html")
print(dataset.to_mermaid())
```

### 4. วิเคราะห์ทั้งโฟลเดอร์ (`run_folder`)

```python
folder = thaieda.run_folder(
    "data/",
    recursive=True,
    lang="th",
    report_mode="blueprint",
    target_column="label",
)
folder.to_master_html("reports/index.html")
print(folder.summary())
```

รองรับไฟล์ข้อมูลหลากหลายนามสกุล ได้แก่ `.csv`, `.tsv`, `.json`, `.jsonl`, `.xlsx`, `.xls` และ `.parquet`

### 5. LLM analysis (optional)

```python
result = thaieda.run(
    df,
    llm=True,
    privacy="insight_only",   # ปลอดภัยที่สุด — ไม่ส่ง raw rows
    provider="openai",
    lang="th",
)
print(result.llm_response)
```

| Privacy mode | LLM เห็นอะไร | เหมาะกับ |
|--------------|-------------|----------|
| `insight_only` | ส่งเฉพาะค่าสถิติสรุปและข้อมูลอินไซต์เท่านั้น โดยไม่ส่งข้อมูลดิบเป็นรายบรรทัด | เหมาะสำหรับข้อมูลที่มีความอ่อนไหวสูง (ค่าเริ่มต้น) |
| `synthetic` | สร้างและส่งข้อมูลจำลองอิงตามการกระจายตัวจริงของข้อมูล | เหมาะสำหรับแชร์โครงสร้างของข้อมูลโดยไม่เปิดเผยค่าจริง |
| `anonymized` | ทำการตรวจหาและทดแทนข้อมูลระบุตัวบุคคล (PII) ด้วยค่าสมมติ | เหมาะสำหรับชุดข้อมูลที่มีเบอร์โทรศัพท์ ชื่อ หรือบัตรประชาชน |
| `dp_noise` | ส่งข้อมูลสถิติสรุปรวมที่มีการผสมสัญญาณรบกวน (Differential Privacy Noise) | เหมาะสำหรับสถิติภาพรวมที่ต้องการการปกป้องความเป็นส่วนตัวขั้นสูง |
| `full` | ส่งข้อมูลดิบทั้งหมดโดยไม่มีการปิดบังใด ๆ | เหมาะสำหรับข้อมูลที่เปิดเผยเป็นสาธารณะอยู่แล้วเท่านั้น |

นอกจากนี้ยังสามารถเรียกใช้ฟังก์ชันนี้แยกเฉพาะได้ผ่าน: `thaieda.llm.analyze_with_llm(...)`

---

## Command Line Interface (CLI)

```bash
thaieda --version

# One-liner (ค่าเริ่มต้น — blueprint report)
thaieda data.csv
thaieda data.csv -o report.html --target clicked

# Batch โฟลเดอร์ — รายงานต่อไฟล์ (ไม่ใช่ schema discovery)
thaieda .
thaieda data/ --output-dir reports/ -y

# EDA + clean + HTML report (subcommand แบบเดิม)
thaieda run data.csv -o report.html --lang th --target clicked

# Profile แบบละเอียด
thaieda profile data.xlsx -o profile.html --clean --lang en

# Clean อย่างเดียว
thaieda clean inputs.csv -o cleaned.csv

# Schema หลายตาราง
thaieda dataset data/warehouse/ -o schema-report.html --lang th
```

| Command | คำอธิบาย |
|---------|----------|
| `thaieda <file>` | **one-liner** — วิเคราะห์คำสั่งเดียว ดำเนินการทำความสะอาดข้อมูล และสร้างรายงาน blueprint ในชื่อ `<stem>-report.html` |
| `thaieda <folder>` / `thaieda .` | **batch one-liner** — สั่งประมวลผลข้อมูลแบบกลุ่มเพื่อสร้างรายงาน HTML แยกเป็นรายไฟล์ภายในโฟลเดอร์ |
| `thaieda run` | รันกระบวนการทำความสะอาด วิเคราะห์ข้อมูล และสร้างรายงานผลลัพธ์แบบ HTML (พร้อมตัวเลือก `--cleaned-output` สำหรับบันทึกไฟล์ CSV ที่ล้างแล้ว) |
| `thaieda profile` | สร้างรายงานวิเคราะห์อย่างละเอียด (รองรับตัวเลือก `--no-charts` เพื่อปิดการสร้างกราฟ หรือ `--sample N` เพื่อสุ่มตัวอย่างแถวข้อมูล) |
| `thaieda clean` | ดำเนินการทำความสะอาดข้อมูลและส่งออก (export) เป็นไฟล์ข้อมูลใหม่ |
| `thaieda dataset` | ค้นหาคีย์หลัก คีย์นอก (PK/FK) และสร้างแผนผังความสัมพันธ์เชื่อมโยงข้ามไฟล์ข้อมูล |

> **Blueprint mode:** โหมดเริ่มต้นของการทำงานผ่าน CLI และ Python API (`report_mode="blueprint"`) หากต้องการสร้างรายงานวิเคราะห์ข้อมูล (EDA) แบบจัดเต็มและครบถ้วน ให้ใช้ตัวเลือก `--explore` หรือตั้งค่า `report_mode="explore"`

---

## โครงสร้างโมดูล / Module Overview

```
src/thaieda/
├── __init__.py          run(), EDA(), EDAResult, run_folder(), compare()
├── cli.py               Command-line interface
├── detect/              Column type detection + Thai month names
├── clean/               Smart cleaning pipeline + ML imputation
├── quality/             Thai data quality checks + scoring
├── anomaly/             Numeric, text, and Thai-specific anomalies
├── insight/             Auto insight summary + distributions
├── insight_engine/      Cross-column patterns + leakage detection
├── report/              HTML reports, blueprint mode, DatasetReport
├── narrative/           Offline executive narrative (no LLM)
├── viz/                 Charts + Thai font support
├── io/                  Auto format/encoding detection, downcast
├── tokenize/            Thai tokenizer adapters (pythainlp/nlpo3/attacut)
├── text/                Text metrics
├── analysis/            Target variable analysis
├── timeseries/          Timeseries decomposition
├── schema/              Multi-file PK/FK discovery
├── ner/                 Thai Named Entity Recognition
├── llm/                 Privacy-preserving LLM analysis
├── i18n/                TH/EN labels
└── compare.py           Dataset drift & schema comparison
```

---

## เปรียบเทียบกับเครื่องมือทั่วไป

| ความสามารถ | Profiling ทั่วไป | ThaiEDA |
|------------|------------------|---------|
| สถิติพื้นฐาน | ✅ มีความละเอียดสูง | ✅ มีขนาดเบาและเน้นคำแนะนำที่นำไปใช้ได้จริง (Actionable) |
| ตรวจคุณภาพข้อมูลไทย | ❌ ต้องเขียนโค้ดคัดกรองเองทั้งหมด | ✅ รองรับการตรวจจับปี พ.ศ., เลขไทย และค่าว่างชั่วคราว (Placeholder) โดยอัตโนมัติ |
| ระบบทำความสะอาดอัตโนมัติ | ❌ ไม่มีในตัวรายงาน | ✅ ทำงานทันทีผ่านตัวเลือก `run(clean=True)` |
| ฟอนต์ภาษาไทยในแผนภูมิ | ❌ มักเจอปัญหาแสดงผลเป็นรูปสี่เหลี่ยมอ่านไม่ได้ | ✅ แก้ไขด้วยระบบเลือกฟอนต์สำรอง (Fallback) ให้อัตโนมัติ |
| การวิเคราะห์เชิงลึกข้ามคอลัมน์ | ⚠️ ทำได้ในระดับพื้นฐานเท่านั้น | ✅ ตรวจจับการรั่วไหลของข้อมูล (Leakage) และความขัดแย้งของซิมป์สัน (Simpson's paradox) ได้ |
| การเตรียมข้อมูลก่อนทำ ML (Blueprint) | ❌ ไม่มีฟังก์ชันรองรับ | ✅ มีระบบวางพิมพ์เขียวโมเดล (Modeling Blueprint) และตรวจจับระดับข้อมูลรั่วไหล |
| เชื่อมโยงโครงสร้างข้ามไฟล์ | ❌ ไม่สามารถวิเคราะห์ข้ามไฟล์ได้ | ✅ ตรวจหาความเชื่อมโยงคีย์หลัก/คีย์นอก (PK/FK) พร้อมวาดไดอะแกรมด้วย Mermaid |
| การเปรียบเทียบข้อมูลไหลเบี่ยง (Drift) | ⚠️ รองรับอย่างจำกัด | ✅ วิเคราะห์ทิศทางการเปลี่ยนไปของโครงสร้าง (Schema) และการแจกแจงของข้อมูล (Distribution Drift) |
| สรุปผลด้วย LLM แบบรักษาความเป็นส่วนตัว | ❌ ไม่มีการป้องกันข้อมูลรั่วไหล | ✅ ปลอดภัยด้วยโหมดรักษาความเป็นส่วนตัวในการส่งข้อมูล 5 ระดับ |

---

## คำถามที่พบบ่อย / FAQ

**เมื่อพบปัญหาแผนภูมิแสดงผลเป็นรูปสี่เหลี่ยมแทนตัวอักษรภาษาไทย?**
สามารถแก้ไขได้โดยตรวจสอบว่าระบบปฏิบัติการของคุณติดตั้งฟอนต์ภาษาไทยเรียบร้อยแล้ว โดยตัวระบบ ThaiEDA จะกำหนดฟอนต์สำรอง (Fallback font) เพื่อแสดงผลภาษาไทยให้อัตโนมัติ

**ข้อมูลในชุดข้อมูลจะถูกอัปโหลดหรือส่งออกไปภายนอกเครื่องคอมพิวเตอร์หรือไม่?**
ไม่เลย — การทำงานทุกขั้นตอนประมวลผลภายในเครื่องคอมพิวเตอร์ของคุณเอง (Local) ยกเว้นในกรณีที่คุณสั่งเปิดการใช้งานโมเดลภาษาขนาดใหญ่โดยตั้งค่าพารามิเตอร์ `llm=True` โดยตรงเท่านั้น และสามารถเลือกใช้โหมด `privacy="insight_only"` หรือเชื่อมต่อกับ Ollama ที่รันแบบ Local เพื่อรักษาความปลอดภัยและความเป็นส่วนตัวของข้อมูลขั้นสูงสุดได้

**โหมดวิเคราะห์ข้อมูลแบบ Blueprint แตกต่างกับ Explore อย่างไร?**
โหมด `explore` จะแสดงผลรายงานวิเคราะห์ข้อมูล (EDA) แบบเต็มพิกัดพร้อมแผนภูมิทุกประเภท ในขณะที่โหมด `blueprint` จะสรุปเนื้อหาให้สั้นกระชับพร้อมรายการตรวจสอบสำหรับการเตรียมข้อมูล และตรวจสอบการรั่วไหลของข้อมูล (Leakage detection) ซึ่งเหมาะเป็นขั้นตอนก่อนการนำไปสอนโมเดล Machine Learning

---

## การทดสอบ / Testing

```bash
python -m pytest tests/ -q
```

ชุดทดสอบภายในระบบมีมากกว่า **1,000 เคสทดสอบ (tests)** ครอบคลุมทั้งขั้นตอนการทำความสะอาดข้อมูล (Cleaning Pipeline), รูปแบบการค้นพบข้อค้นพบเชิงลึก (Insight Patterns), โหมดวิเคราะห์พิมพ์เขียว (Blueprint Mode), การตรวจจับข้อมูลรั่วไหล (Leakage Detection), การกันผลบวกลวง (false-positive regression tests) ตลอดจนการทดสอบกับชุดข้อมูลจริงที่มีจุดผิดพลาดในโครงสร้างใน `tests/fixtures/dirty_datasets/`

---

## พัฒนา / Development

```bash
python -m pytest tests/
ruff check src/ tests/
ruff format src/ tests/
```

ศึกษาข้อมูลเพิ่มเติมได้จากแนวทางของโครงการที่ [CONTRIBUTING.md](CONTRIBUTING.md) และข้อตกลงการมีส่วนร่วม [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

---

## License

ThaiEDA is released under the [Apache-2.0 License](LICENSE).
