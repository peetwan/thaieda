# ThaiEDA

**Exploratory data analysis that actually understands Thai.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions/workflows/ci.yml)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

---

## มันคืออะไร

ThaiEDA คือ Python library สำหรับทำ Exploratory Data Analysis (EDA) แบบอัตโนมัติ — ใส่ DataFrame เข้าไปหนึ่งบรรทัด ได้ผลลัพธ์ออกมาครบ: ตรวจข้อมูล → ทำความสะอาด → หา anomaly → สร้าง insight → ออกรายงาน HTML

สิ่งที่ทำให้ต่างจาก ydata-profiling, sweetviz, หรือ Evidently คือ **รู้เรื่องภาษาไทย** — Buddhist Era (พ.ศ.), เลขไทย (๑๒๓), zero-width space, mojibake จาก TIS-620, ที่อยู่ภาษาไทย, เลขบัตรประชาชน, บริษัทไทย — ทั้งหมดนี้ทำให้ข้อมูลไทยไม่พังเวลาผ่าน pipeline

---

## ทำอะไรได้บ้าง

### One-liner ครบจบ

```python
import thaieda
import pandas as pd

df = pd.read_csv("data.csv")
result = thaieda.run(df)          # ครบทุกขั้นตอนในบรรทัดเดียว
result.to_html("report.html")     # รายงาน HTML แบบ standalone
```

`run(df)` ทำสิ่งต่อไปนี้ตามลำดับ:

```
DataFrame → thaieda.run(df) → EDAResult

  Step 0  pre-analyze    data type + language detection
  Step 1  detect         column types + Thai months + addresses
  Step 2  clean          smart cleaning (auto-decide what to fix)
  Step 3  quality        language-aware checks + 0–100 score
  Step 4  anomaly        IQR + ML + text anomaly detection
  Step 5  insights       6 cross-column patterns (BH-corrected)
  Step 6  viz            static (matplotlib) + interactive (Plotly)
  Step 7  report         executive HTML narrative
```

### ความสามารถหลัก

| ความสามารถ | รายละเอียด |
|-----------|-----------|
| **Thai text ไม่พัง** | ฟอนต์ไทยในทุก chart (Sarabun auto), เลขไทย → 123, พ.ศ. → CE, ตัด zero-width space, แก้ mojibake |
| **Insight ไม่ใช่แค่ stats** | หา cross-column patterns เช่น "คอลัมน์ A ทำนาย B ได้ดี", "กลุ่มนี้สูงกว่าค่าเฉลี่ย 3 เท่า" พร้อม Benjamini-Hochberg correction |
| **Anomaly detection** | IQR + z-score + Isolation Forest + text anomaly |
| **Quality score 0–100** | ให้คะแนนคุณภาพข้อมูลแบบมี grade |
| **Smart cleaning** | ตัดสินใจเองว่าควรแก้อะไร — encoding, numerals, BE dates, duplicates, missing values |
| **Privacy-first LLM** | 4 โหมดความเป็นส่วนตัว — default ส่งข้อมูลดิบออกจากเครื่องเป็นศูนย์ (PDPA-ready) |
| **Folder mode** | `run_folder("data/")` วิเคราะห์ทุกไฟล์ในโฟลเดอร์ + รวมเป็น master HTML |
| **Compare** | `compare(df1, df2)` เปรียบเทียบสอง dataset พร้อม drift detection |
| **Thai NER** | ดึงชื่อคน/สถานที่/องค์กรจากข้อความไทย |
| **Thai address parsing** | 123 ม.4 ต.บางบัว อ.บางบัว จ.กรุงเทพฯ → structured fields |
| **Thai ID validation** | เช็ค checksum เลขบัตรประชาชน |

### ภาษาไทยที่จับได้

| ปัญหา | ตัวอย่าง | ThaiEDA ทำอะไร |
|---------|---------|----------------|
| Buddhist Era dates | `15/03/2567` | ตรวจ พ.ศ. → แปลงเป็น CE |
| Thai numerals | `๑๒๓` ในคอลัมน์ตัวเลข | แปลงเป็น `123` |
| Zero-width spaces | `สม\u200bชาย` | ตัดอักขระที่มองไม่เห็น + แจ้ง |
| Mojibake encoding | `Ã ¬Â¸Â¡Â¹` | auto-detect TIS-620 → UTF-8 |
| Thai month names | `มกราคม` | parse เป็น ISO date |
| Mixed Thai/English | `อร่อยมาก 5/5 stars` | ตรวจเป็น mixed ไม่ใช่ English/numeric |
| National ID | `1-1234-56789-01-2` | checksum validation |
| Phone numbers | `081-234-5678` | ตรวจ + normalize |
| Thai addresses | `123 ม.4 ต.บางบัว อ.บางบัว จ.กรุงเทพฯ` | parse เป็น structured fields |

### คุณภาพข้อมูลที่ตรวจ

| ปัญหา | ThaiEDA ทำอะไร |
|---------|----------------|
| Placeholder values (`-`, `N/A`, `ไม่มี`) | แจ้งเป็น missing |
| Constant columns | แจ้งเป็นไร้ประโยชน์ |
| High-NA columns (>80%) | แจ้ง `mostly_missing` |
| Smart data type | จำแนก transaction/registry/survey/timeseries/mixed |
| Language-aware | ข้อมูล English ไม่เตือนเรื่อง พ.ศ./เลขไทย |
| ID/FK semantics | `order_id`, `store_id` ไม่นำมาหา category anomaly |
| Keyboard layout guard | `Floyd` ในคอลัมน์ English ไม่ถูกแปลงเป็นไทย |

### Privacy LLM — 4 โหมด

| โหมด | ส่งอะไรไป LLM | ความเสี่ยง | เหมาะกับ |
|------|--------------|----------|---------|
| `insight_only` (default) | สถิติ + insight เท่านั้น | ต่ำ | ภาครัฐ, การแพทย์, PDPA |
| `synthetic` | mock data จาก fitted distributions | ต่ำ | LLM เห็นรูปข้อมูลจริงแต่ไม่ใช่ของจริง |
| `anonymized` | PII → tokens | ปานกลาง | ต้องการ structure ไม่มี PII |
| `dp_noise` | stats + Laplace noise | ต่ำ | dataset เล็กที่ stats รั่วได้ |
| `full` | ข้อมูลดิบ | สูง | ข้อมูลสาธารณะ, demo |

---

## ทำไมถึงใช้

มี ydata-profiling และ sweetviz อยู่แล้ว ทำไมต้องใช้ ThaiEDA?

1. **ข้อมูลไทยไม่พัง** — ydata/sweetviz render ไทยเป็น tofu (□□□) ในทุก chart ไม่เห็น พ.ศ., เลขไทย, zero-width space, mojibake ThaiEDA ตรวจและแก้ทั้งหมดอัตโนมัติ ไม่ต้อง config ฟอนต์ ไม่ต้อง manual cleanup

2. **Insight ไม่ใช่แค่สถิติ** — ydata ให้ distribution + correlation matrix แต่ ThaiEDA หา *actionable* cross-column patterns จัดอันดับตามความน่าสนใจทางสถิติ + มี anomaly detection + quality scoring

3. **ครั้งเดียวจบ** — `run(df)` ทำ pipeline เต็มรูปแบบ ใช้ ydata ต้องมี anomaly detector แยก, Thai font config แยก, cleaner แยก, interpretation แยก

4. **Privacy-first** — ถาม LLM เกี่ยวกับข้อมูลได้โดยไม่ส่ง raw rows ไป cloud 4 โหมด default ส่งข้อมูลดิบออกจากเครื่องเป็นศูนย์

5. **Report เล็ก** — ydata สร้าง 71 MB HTML บน dataset 171 คอลัมน์ ThaiEDA สร้าง 0.48 MB เล็กกว่า 148 เท่า เพราะ cap charts, collapse tables, sample อัจฉริยะบนข้อมูลใหญ่

---

## เปรียบเทียบกับเครื่องมืออื่น

### ความสามารถ

| Feature | ydata-profiling | sweetviz | Evidently | **ThaiEDA** |
|---------|:---:|:---:|:---:|:---:|
| HTML report | ✅ | ✅ | ✅ | ✅ |
| Cross-column insights | ❌ | ❌ | ❌ | ✅ 6 patterns + BH |
| Anomaly detection | ❌ | ❌ | ❌ | ✅ IQR + ML + text |
| Quality score (0–100) | ❌ | ❌ | ❌ | ✅ |
| Language detection | ❌ | ❌ | ❌ | ✅ Thai/English/mixed |
| Thai font in charts | ❌ tofu | ❌ tofu | ❌ tofu | ✅ Sarabun auto |
| Buddhist Era (พ.ศ.) | ❌ | ❌ | ❌ | ✅ → CE |
| Thai numerals (๑๒๓) | ❌ | ❌ | ❌ | ✅ → 123 |
| Zero-width space | ❌ | ❌ | ❌ | ✅ |
| Mojibake repair | ❌ | ❌ | ❌ | ✅ |
| Smart cleaning | ❌ | ❌ | ❌ | ✅ auto-decide |
| Thai NER | ❌ | ❌ | ❌ | ✅ |
| Privacy LLM modes | ❌ | ❌ | ❌ | ✅ 4 modes (PDPA) |
| Folder mode | ❌ | ❌ | ❌ | ✅ `run_folder()` |

### ความเร็วและขนาด report

| Dataset | Rows | Cols | ydata | ydata size | sweetviz | sv size | Evidently | ev size | **ThaiEDA** | **EDA size** |
|---------|-----:|-----:|-------:|-----------:|---------:|--------:|----------:|--------:|------------:|-------------:|
| titanic | 891 | 12 | 5.3s | 1.95 MB | 3.3s | 0.92 MB | — | — | 8.2s | **0.82 MB** |
| superstore | 10,800 | 21 | 9.3s | 5.16 MB | 5.4s | 1.49 MB | — | — | 26.0s | **1.50 MB** |
| adult | 32,561 | 15 | 5.4s | 1.65 MB | 8.0s | 1.26 MB | — | — | 17.2s | **1.05 MB** |
| aps-failure | 16,000 | 171 | 99.8s | **71.2 MB** | 15.8s | 8.2 MB | — | — | 93.0s | **0.48 MB** |
| synthetic | 2,000 | 12 | 45s | 7.2 MB | 3s | 0.9 MB | 1s | 3.7 MB | 16s | **1.5 MB** |

### คุณภาพการตรวจจับ (synthetic dataset 10 known issues)

**General EDA** (6 issues):

| Metric | ydata | sweetviz | Evidently | **ThaiEDA** |
|--------|:---:|:---:|:---:|:---:|
| Ground-Truth Recall | 100% | 83% | 100% | **100%** |
| Issue Type Breadth (11) | 73% | 64% | 91% | **91%** |
| Report Completeness (10) | 70% | 50% | 70% | **100%** |
| HTML size | 7.2 MB | 0.9 MB | 3.7 MB | **1.5 MB** |

**Thai-specific** (4 issues):

| Thai issue | ydata | sweetviz | Evidently | **ThaiEDA** |
|-----------|:---:|:---:|:---:|:---:|
| Buddhist Era (พ.ศ.) | 0% | 0% | 0% | **✅** |
| Thai numerals (๑๒๓) | 0% | 0% | 0% | **✅** |
| Zero-width spaces | 0% | 0% | 0% | ❌ |
| Mojibake (TIS-620) | 0% | 0% | 0% | **✅** |
| **Thai GTR** | **0%** | **0%** | **0%** | **75%** |

### Full pipeline test — 9 public datasets

ทุก dataset รันผ่าน `run(df, lang="en")` ครบทุกขั้นตอน ไม่ crash ไม่ต้องใช้ API key:

| Dataset | Shape | Time | Report | Insights | Issues | Anomalies |
|---------|------:|-----:|-------:|---------:|-------:|----------:|
| titanic | 891×12 | 8.9s | 814 KB | 30 | 3 | 12 |
| superstore | 5,000×21 | 19.9s | 1,635 KB | 30 | 2 | 16 |
| adult | 5,000×15 | 9.4s | 1,007 KB | 29 | 3 | 17 |
| online-shoppers | 5,000×18 | 10.4s | 1,051 KB | 30 | 0 | 18 |
| beijing-pm25 | 5,000×13 | 6.8s | 788 KB | 22 | 1 | 6 |
| telco-churn | 5,000×21 | 10.2s | 861 KB | 11 | 0 | 4 |
| california-housing | 5,000×10 | 11.9s | 946 KB | 30 | 0 | 11 |
| winequality-red | 1,599×12 | 8.7s | 956 KB | 29 | 0 | 16 |
| synthetic-benchmark | 2,000×12 | 10.7s | 1,536 KB | 24 | 5 | 7 |

**9/9 ผ่าน** — เฉลี่ย 30 insights ต่อ dataset report 0.8–1.6 MB

### ความใกล้เคียงของข้อมูลหลัง clean()

ThaiEDA รักษาสถิติข้อมูลเดิม วัด mean/std change ก่อน-หลัง cleaning บน 5 datasets:

| Dataset | Mean change | Std change | NaN fixed | Memory reduction |
|---------|------------:|----------:|----------:|----------------:|
| titanic | 2.84% | 3.02% | 866→0 | 29.6% |
| superstore | 0.00% | 0.00% | 0→0 | 59.6% |
| adult | 0.00% | 0.00% | 0→0 | 89.0% |
| winequality-red | 0.30% | 1.59% | 0→0 | 60.1% |
| california-housing | 0.06% | 0.02% | 11→0 | 57.6% |

**สถิติคงตัว** — mean เปลี่ยน <3%, std <3.2% ลด memory 30–89%

---

## ตัวอย่างการใช้งาน

### One-Line EDA

```python
import thaieda
import pandas as pd

df = pd.read_csv("data.csv")
result = thaieda.run(df)

result.to_html("report.html")
print(result.quality_issues)
print(result.insights)

# ใน Jupyter: แค่แสดง result
result  # renders HTML inline
```

### Folder Mode

```python
import thaieda

results = thaieda.run_folder("data/")
print(results.summary())
results.to_html("reports/")
results.to_master_html("master-report.html")
```

รองรับ: CSV, TSV, Excel, JSON, JSONL, Parquet — `recursive=True` สำหรับ subfolders

### Supported File Formats

| Entry point | Supported formats |
|-------------|-------------------|
| `read_data()` | CSV, TSV, JSON, JSONL/NDJSON, Excel (`.xlsx`/`.xls`), Parquet |
| `run_folder()` | CSV, TSV, JSON, JSONL/NDJSON, Excel (`.xlsx`/`.xls`), Parquet |
| CLI `--format` | `auto`, `csv`, `tsv`, `json`, `jsonl`, `excel`, `parquet` |
| `export_synthetic_data()` | CSV, TSV, JSON, Excel (`.xlsx`), Parquet |

### LLM Analysis (Privacy-Safe)

```python
result = thaieda.run(df, llm=True, privacy="insight_only", provider="ollama")
print(result.llm_response)
# Default: ไม่มีข้อมูลดิบออกจากเครื่อง
```

### Compare Two Datasets

```python
from thaieda import compare

diff = compare(df_train, df_test, labels=("train", "test"))
print(diff["schema_diff"])
print(diff["drift"]["numeric"])
```

### Thai ID Card Validation

```python
from thaieda.quality import validate_thai_id

validate_thai_id("1-1234-56789-01-2")  # → True/False
```

### Thai Address Parsing

```python
from thaieda.detect import parse_thai_address

addr = parse_thai_address("123 หมู่ 4 ต.บางบัว อ.บางบัว จ.กรุงเทพฯ 10230")
# {'house_number': '123', 'moo': '4', 'subdistrict': 'บางบัว', ...}
```

### Smart Cleaning

```python
from thaieda.clean._smart import plan_cleaning

plan = plan_cleaning(df)
print(plan.actions)   # ['zwspace', 'numerals', 'duplicates']
print(plan.skipped)   # ['encoding', 'whitespace']
```

### Statistical Accuracy

```python
from thaieda.insight_engine import discover_insights  # Spearman + Pearson
from thaieda.analysis import analyze_target            # Cramér's V effect size
from thaieda.quality import detect_missing_mechanism  # MCAR/MAR/MNAR
from thaieda.quality import fit_distributions         # KS goodness-of-fit
```

---

## การจัดการข้อมูลขนาดใหญ่

ทดสอบบน 19 public datasets — 500 ถึง 541K rows, 2 ถึง 171 columns:

- **Insight capping** — แสดง 30 insight ที่สำคัญที่สุด สรุปบอกจำนวนจริง ("679 found, showing top 30")
- **HTML bloat control** — สูงสุด 40 charts, 1.6 MB ตาราง collapse หลัง 50 แถว wide table เป็น summary หลัง 60 คอลัมน์
- **Wide-table fast path** — insight engine sample เมื่อคอลัมน์เกิน 100
- **Tall-table fast path** — anomaly/quality check sample 50K rows เมื่อข้อมูลเกิน 100K
- **dtype downcasting** — int64→int32, float64→float32 ลดขนาด 50-70%

---

## Modules

| Module | ทำอะไร |
|--------|--------|
| `run()` / `EDA()` | One-liner API — full pipeline ในครั้งเดียว |
| `run_folder()` | วิเคราะห์ทุกไฟล์ในโฟลเดอร์ + master HTML |
| `compare()` | เปรียบเทียบสอง dataset + drift detection |
| `io/` | auto-read CSV/TSV/JSON/Excel/Parquet + encoding detection |
| `detect/` | column type detection + Thai months + address + language |
| `clean/` | smart cleaning: encoding, numerals, BE, zwspace, duplicates, missing |
| `quality/` | language-aware quality checks + score 0–100 + Thai ID validation |
| `anomaly/` | statistical + ML + text anomaly detection |
| `ner/` | Thai NER: person/place/organization |
| `insight_engine/` | 6 cross-column insight patterns (BH-corrected) |
| `viz/` | static + interactive charts + colorblind-safe palette |
| `report/` | executive HTML report + smart pre-analysis |
| `llm/` | privacy-preserving LLM analysis (4 modes, 3 providers) |
| `timeseries/` | trend/seasonality/STL/ACF + Thai holiday awareness |
| `schema/` | multi-file PK/FK discovery + relationship matching |

---

## Installation

```bash
pip install thaieda
```

ติดตั้งทุกอย่างในครั้งเดียว — Thai tokenizer, NER, ML, interactive charts, Excel, stats, encoding detection

LLM providers (optional, lazy-imported):

```bash
pip install openai       # OpenAI GPT
pip install anthropic    # Anthropic Claude
pip install ollama       # Ollama local LLM
```

**Requirements:** Python 3.10+

---

## Testing

```bash
pytest tests/ -v                    # run test suite
ruff check src/ tests/              # lint
ruff format src/ tests/             # format
```

---

## License

[Apache-2.0](LICENSE) © Peet Wannasarnmetha
