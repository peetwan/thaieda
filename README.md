# ThaiEDA

> **AutoEDA และทำความสะอาดข้อมูลอัจฉริยะสำหรับข้อมูลภาษาไทยและข้อมูลผสมภาษา**
>
> **One-line Exploratory Data Analysis and Smart Data Cleaning for Thai and Mixed-Language Datasets.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions/workflows/ci.yml)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

**ThaiEDA** ตอบคำถามเดียว: *"ชุดข้อมูลนี้เชื่อถือได้แค่ไหน และควรสำรวจอะไรก่อน?"*

เครื่องมือ profiling ทั่วไปมักนับ missing และวาดกราฟมาตรฐานได้ดี แต่พลาดปัญหาเฉพาะของข้อมูลไทย — ปี พ.ศ., เลขไทย, zero-width space, mojibake, เบอร์โทร/บัตรประชาชน, และฟอนต์กราฟที่อ่านไม่ออก ThaiEDA จัดการเรื่องเหล่านี้เป็นขั้นตอนปกติของ pipeline ไม่ต้อง preprocess เองทีละคอลัมน์

ThaiEDA answers one simple question: **"Can I trust this dataset, and what should I explore first?"** It treats Thai-specific data issues — Buddhist Era dates, Thai numerals, zero-width spaces, encoding errors, local phone formats, and chart fonts — as normal problems to fix automatically.

- **Repository:** [github.com/peetwan/thaieda](https://github.com/peetwan/thaieda)
- **PyPI:** [pypi.org/project/thaieda](https://pypi.org/project/thaieda/)
- **License:** Apache-2.0
- **Current version:** v2.1.0

---

## คุณสมบัติหลัก / Key Features

| หมวด | ไทย | English |
|------|-----|---------|
| **Detection** | ตรวจประเภทคอลัมน์ไทย/อังกฤษ, ตัวเลขที่ซ่อนในข้อความ, ปี พ.ศ., เบอร์/บัตร ปชช., ข้อมูลผสมภาษา | Smart column & type detection for Thai/mixed data |
| **One-liner API** | `run()` / `EDA()` — detect → clean → quality → insights → viz → HTML report ในบรรทัดเดียว | Full AutoEDA pipeline in one function call |
| **Blueprint mode** | รายงานสั้น เน้น actionable — เหมาะกับงาน ML/tabular เมื่อระบุ `target_column` | Shorter actionable reports with modeling blueprint |
| **Cleaning** | Unicode, zero-width, เลขไทย, พ.ศ.→ค.ศ., สกุลเงิน, ซ้ำ, missing (รวม ML imputation), downcast | Thai-aware cleaning with audit trail |
| **Insights** | Cross-column engine — correlation, outliers, trends, Simpson's paradox, **target leakage (Tier A/B)** | Statistical insight discovery with BH correction |
| **Reports** | HTML ภาษาไทย/อังกฤษ, offline narrative, Jupyter rich display | Executive HTML reports + template narratives |
| **Schema** | ค้นหา PK/FK ข้ามหลายไฟล์ + Mermaid diagram | Multi-file schema discovery |
| **LLM** | 5 โหมดความเป็นส่วนตัว (OpenAI / Anthropic / Ollama) | Privacy-preserving optional LLM summaries |

---

## ติดตั้ง / Installation

ต้องการ **Python 3.10+**

```bash
pip install thaieda
```

Core น้ำหนักเบา: `pandas`, `numpy`, `matplotlib`, `Jinja2`

ติดตั้งส่วนเสริมตามความต้องการ:

```bash
# ตัดคำไทย + กราฟ + Excel/Parquet
pip install "thaieda[thai,viz,excel,parquet]"

# สถิติเต็มรูปแบบ (p-values, ANOVA, chi-square)
pip install "thaieda[stats]"

# ทุกอย่างในคำสั่งเดียว
pip install "thaieda[all]"
```

### Optional dependencies / ส่วนเสริม

| Extra | Packages | ใช้เมื่อ |
|-------|----------|----------|
| `thai` | pythainlp | ตัดคำภาษาไทย (แนะนำ) |
| `fast` | nlpo3 | ตัดคำเร็ว (Rust) |
| `dl` | attacut | ตัดคำ deep learning |
| `viz` | plotly, wordcloud | กราฟ interactive + word cloud |
| `fix` | ftfy | ซ่อม mojibake (`clean.normalize_encoding`) |
| `ml` | scikit-learn | anomaly ML + missing imputation |
| `stats` | scipy | p-values สำหรับ target analysis |
| `timeseries` | statsmodels | STL decomposition |
| `ner` | pythainlp, python-crfsuite | Thai NER |
| `fuzzy` | rapidfuzz | จับคู่ categorical ใกล้เคียง |
| `detect` | chardet | ตรวจ encoding อัตโนมัติ (`read_data`) |
| `excel` | openpyxl | อ่าน/เขียน `.xlsx` |
| `parquet` | pyarrow | อ่าน/เขียน Parquet |
| `llm` | litellm | วิเคราะห์ด้วย LLM |
| `all` | ทุกอย่างด้านบน | ติดตั้งครบ |
| `dev` | pytest, ruff, mypy, … | พัฒนา / รันเทส |

---

## เริ่มต้นเร็ว / Quickstart

### One-liner CLI

ติดตั้งแล้วรันคำสั่งเดียว — อ่านไฟล์ → ทำความสะอาด → สร้างรายงาน HTML blueprint อัตโนมัติ:

```bash
pip install thaieda
thaieda mydata.csv
# → สร้าง mydata-report.html ในโฟลเดอร์เดียวกับไฟล์
```

ตัวเลือกที่ใช้บ่อย:

```bash
thaieda data.csv -o report.html          # ระบุพาธ output
thaieda data.csv --target clicked        # คอลัมน์เป้าหมาย (หรือเดาอัตโนมัติจากชื่อ)
thaieda data.csv -t clicked              # ย่อ — เหมือน --target
thaieda data.csv --columns               # ดูคอลัมน์ + target ที่น่าจะเป็น (ไม่รัน EDA)
thaieda data.csv -y                      # ข้าม prompt เลือก target (CI/batch)
thaieda data.csv --explore               # รายงาน EDA แบบเต็ม (แทน blueprint)
thaieda data.csv --lang en --no-clean    # ภาษาอังกฤษ, ไม่ทำความสะอาด
```

**Target column:** ถ้าไม่ระบุ `-t`/`--target` ThaiEDA จะเดาจากชื่อคอลัมน์ (`target`, `label`, `clicked`, `churn`, `survived`, `y`, `class`, `outcome`, `income`, `quality`, `response`, `default`, `fraud`, `exit` และชื่อที่มีคำเหล่านี้) บน TTY จะถามให้เลือกจากรายการคอลัมน์ (Enter = ข้าม) ใช้ `-y`/`--no-interactive` เพื่อข้าม prompt

#### เลือก target อย่างไร?

1. **ดูคอลัมน์ก่อน** — ไม่ต้องรัน EDA เต็มรูปแบบ:
   ```bash
   thaieda data.csv --columns
   # หรือ
   thaieda data.csv --preview
   ```
   แสดงตาราง `#`, ชื่อคอลัมน์, dtype, จำนวน unique และคอลัมน์ที่น่าจะเป็น target (★)

2. **ระบุเอง** — `thaieda data.csv -t clicked` หรือ `--target churn`

3. **ไม่ใส่** — auto-detect จากชื่อ หรือถามใน terminal (ถ้าเป็น TTY)

4. **หลังรันเสร็จ** — CLI พิมพ์ `Target: clicked (auto)` / `(จาก --target)` / `(ไม่ระบุ — ไม่มีแผนสร้างโมเดล)`

**Batch โฟลเดอร์:** ประมวลผลทุกไฟล์ในโฟลเดอร์ → `{stem}-report.html` ต่อไฟล์ (ต่างจาก `thaieda dataset` ที่เป็น schema หลายตาราง)

```bash
cd ~/my-data/
thaieda .                                # batch ทุก CSV/TSV/JSON/Excel/Parquet
thaieda /path/to/folder --output-dir reports/
thaieda folder/ -t clicked -y            # target เดียวกันทุกไฟล์, ไม่ถาม interactive
```

รองรับ CSV, TSV, JSON, Excel, Parquet — ตรวจ format/encoding อัตโนมัติ

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

`thaieda.run(df)` และ `thaieda.EDA(df)` ทำงานเหมือนกัน — คืน `EDAResult`

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
| `.report` | `ProfileReport` เต็มรูปแบบ |
| `.cleaned_df` | DataFrame หลัง clean |
| `.overview` | แถว, คอลัมน์, ประเภท |
| `.quality_issues` | ปัญหาคุณภาพที่พบ |
| `.quality_score` | คะแนน 0–100 + grade |
| `.quality_comparison` | เปรียบเทียบก่อน/หลัง clean |
| `.cleaning_report` | audit trail การทำความสะอาด |
| `.insights` | สรุปข้อค้นพบอัตโนมัติ |
| `.anomalies` | outliers / text anomalies |
| `.narrative` | บทสรุป executive แบบ template |
| `.llm_response` | คำตอบ LLM (เมื่อ `llm=True`) |
| `.notes` | คำเตือนระหว่างรัน |
| `.to_html(path)` | บันทึกรายงาน HTML |
| `.to_dict()` | export เป็น dict |
| `.to_json(path)` | export เป็น JSON |

ใน Jupyter: พิมพ์ `result` แล้วแสดง HTML report อัตโนมัติ (`_repr_html_`)

---

## Blueprint mode — รายงานสำหรับงาน ML

`report_mode="blueprint"` สร้างรายงานสั้น เน้นสิ่งที่ต้องทำต่อ เหมาะกับข้อมูลตารางที่มี target (classification, CTR, churn ฯลฯ)

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

**Blueprint ทำอะไรบ้าง:**

- **จำแนกประเภทข้อมูลอัตโนมัติ** — รวม `ml_tabular` เมื่อมี target + ฟีเจอร์หลายคอลัมน์
- **ข้าม timeseries** บนข้อมูล event/impression (แถวไม่ใช่อนุกรมเวลา)
- **กราฟน้อยลง** — scatter matrix / violin ถูกตัดออก
- **ข้อค้นพบมากขึ้น** — แสดง insight ได้ถึง 12 รายการ
- **Modeling Blueprint** (เมื่อมี `target_column`):
  - **Target baseline** — positive rate, class balance
  - **Leakage detection** — Tier A (critical) และ Tier B (warning)
  - **Strong features** — ฟีเจอร์ที่สัมพันธ์กับ target อย่างมีนัยสำคัญ
  - **Columns to drop** — ID columns ที่ควรตัดออก
  - **Next steps** — checklist ก่อนเทรนโมเดล

### Target leakage tiers

| Tier | ความรุนแรง | ตัวอย่าง heuristic |
|------|------------|-------------------|
| **A — critical** | หยุดใช้ฟีเจอร์นี้เทรนทันที | ค่าซ้ำ target, \|corr\| ≥ 0.98, deterministic mapping, near-perfect separation |
| **B — warning** | ตรวจด้วยตาก่อนใช้ | ชื่อคอลัมน์บ่ง proxy (`_ctr`, `_rate`, `historical_`, …) + association สูง |

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

`skip_id_like=True` (ค่าเริ่มต้น) ป้องกัน product ID / serial number จากการถูกยุบ repeated characters

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

รองรับ `.csv`, `.tsv`, `.json`, `.jsonl`, `.xlsx`, `.xls`, `.parquet`

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
| `insight_only` | สถิติสรุป + insights เท่านั้น | ข้อมูลอ่อนไหว (default) |
| `synthetic` | แถวจำลองจาก distribution จริง | แชร์โครงสร้างโดยไม่มีค่าจริง |
| `anonymized` | PII ถูกแทนที่ | ข้อมูลที่มีเบอร์/ชื่อ/บัตร |
| `dp_noise` | สรุปรวม + noise differential privacy | สถิติรวมที่ต้องปกปิด |
| `full` | ข้อมูลดิบ | ข้อมูลสาธารณะเท่านั้น |

เรียกแยกได้: `thaieda.llm.analyze_with_llm(...)`

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
| `thaieda <file>` | **one-liner** — clean → blueprint report → `<stem>-report.html` |
| `thaieda <folder>` / `thaieda .` | **batch one-liner** — รายงาน HTML ต่อไฟล์ในโฟลเดอร์ |
| `thaieda run` | clean → analyze → HTML (+ `--cleaned-output` CSV) |
| `thaieda profile` | รายงานเต็ม พร้อม `--no-charts`, `--sample N` |
| `thaieda clean` | ทำความสะอาดแล้ว export ไฟล์ |
| `thaieda dataset` | PK/FK discovery ข้ามหลายไฟล์ |

> **Blueprint mode:** ค่าเริ่มต้นของ one-liner CLI และ Python API (`report_mode="blueprint"`) — ใช้ `--explore` หรือ `report_mode="explore"` สำหรับรายงาน EDA แบบเต็ม

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
| สถิติพื้นฐาน | ✅ ละเอียด | ✅ กระเบา เน้น actionable |
| ตรวจคุณภาพข้อมูลไทย | ❌ ต้องทำเอง | ✅ พ.ศ., เลขไทย, placeholder |
| Auto-clean ในรายงาน | ❌ | ✅ `run(clean=True)` |
| ฟอนต์ไทยในกราฟ | ❌ มักเป็นสี่เหลี่ยม | ✅ fallback อัตโนมัติ |
| Cross-column insights | ⚠️ พื้นฐาน | ✅ leakage, Simpson's paradox |
| Blueprint / ML prep | ❌ | ✅ modeling blueprint + leakage tiers |
| Schema หลายไฟล์ | ❌ | ✅ PK/FK + Mermaid |
| Dataset drift | ⚠️ จำกัด | ✅ schema + distribution drift |
| LLM แบบปกปิดข้อมูล | ❌ | ✅ 5 privacy modes |

---

## คำถามที่พบบ่อย / FAQ

**กราฟแสดงสี่เหลี่ยมแทนตัวอักษรไทย?**
ติดตั้ง `pip install "thaieda[viz]"` และตรวจว่า OS มีฟอนต์ไทย ThaiEDA ตั้ง fallback ให้อัตโนมัติ

**ข้อมูลถูกส่งออกไปนอกเครื่องไหม?**
ไม่ — ทุกอย่างรัน local ยกเว้นเมื่อเปิด `llm=True` โดยเจตนา ใช้ `privacy="insight_only"` หรือ Ollama local เพื่อความปลอดภัยสูงสุด

**Blueprint กับ Explore ต่างกันอย่างไร?**
`explore` = รายงาน EDA เต็มรูปแบบพร้อมกราฟ `blueprint` = สรุปสั้น + modeling checklist + leakage — เหมาะก่อนเทรนโมเดล

---

## การทดสอบ / Testing

```bash
python -m pytest tests/ -q
```

ชุดทดสอบมี **950+ tests** ครอบคลุม cleaning pipeline, insight patterns, blueprint mode, leakage detection, และ golden dirty datasets ใน `tests/fixtures/dirty_datasets/`

---

## พัฒนา / Development

```bash
python -m pytest tests/
ruff check src/ tests/
ruff format src/ tests/
```

ดู [CONTRIBUTING.md](CONTRIBUTING.md) และ [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

---

## License

ThaiEDA is released under the [Apache-2.0 License](LICENSE).
