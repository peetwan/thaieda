# ThaiEDA

**AutoEDA สำหรับข้อมูลภาษาไทย — Exploratory data analysis that speaks Thai**

[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

> เข้าใจข้อมูลภาษาไทยในบรรทัดเดียว

ThaiEDA คือ library สำหรับทำ Exploratory Data Analysis (EDA) โดยเฉพาะข้อมูลที่มีภาษาไทย — ตรวจจับปัญหาข้อมูลที่ tool ทั่วไปมองข้าม เช่น ปีพุทธศักราชผสมคริสต์ศักราช เลขไทยผสมเลขอารบิก อักขระ zero-width space ที่ทำให้ groupby พัง และอื่น ๆ

> 🆕 **v0.4** — วิเคราะห์ **อนุกรมเวลา (timeseries)** อัตโนมัติ: ตรวจหา trend, seasonality, ช่องว่างเวลา, spike พร้อม STL decomposition + ACF — บวกกับ **insight การกระจาย** (skewness, kurtosis, bimodal, correlation, แถวซ้ำ) ทุกอย่างทำงานในคำสั่งเดียวเหมือนเดิม

> **v0.3** — ครบจบในคำสั่งเดียว: `thaieda run data.csv -o report.html` → ทำความสะอาด + หา insight + สร้างรายงานอัตโนมัติ พร้อมรองรับ JSON, ตรวจ encoding อัตโนมัติ และสรุปข้อค้นพบสำคัญเป็นภาษาไทย

---

## ✨ ฟีเจอร์

### 🆕 Timeseries Analysis & Comprehensive Insights (v0.4)

- 📈 **Auto timeseries analysis** — ตรวจหา datetime column แล้ววิเคราะห์ทุกคอลัมน์ตัวเลขเป็นอนุกรมเวลาอัตโนมัติ
- 🔭 **Trend & seasonality** — ตรวจแนวโน้ม (เพิ่ม/ลด/คงที่) และรูปแบบตามฤดูกาล (รอบสัปดาห์/เดือน/ปี) จาก autocorrelation
- 🧩 **STL decomposition** — แยก trend / seasonal / residual (ใช้ statsmodels เมื่อมี `thaieda[timeseries]`, ไม่งั้น fallback แบบ moving-average)
- ⏱️ **Time gaps & spikes** — ตรวจช่วงเวลาที่ขาดหาย และค่าผิดปกติเฉพาะช่วง (spike/level shift)
- 📉 **Timeseries charts** — กราฟเส้น (พร้อม trend line), decomposition 4 แผง, ACF plot
- 📐 **Distribution insights** — ความเบ้ (skewness), หางหนัก (kurtosis), 2 กลุ่ม (bimodal) ของคอลัมน์ตัวเลข
- 🔗 **Correlation & duplicates** — คู่คอลัมน์ที่สหสัมพันธ์สูง (อาจซ้ำซ้อน), แถวซ้ำ, คอลัมน์ที่เก็บตัวเลขเป็นข้อความ

### Single-command Pipeline & Auto Insights (v0.3)

- ⚡ **คำสั่งเดียวจบ** — `thaieda run data.csv` ทำความสะอาด → วิเคราะห์ → สร้างรายงาน + ไฟล์ที่สะอาดแล้ว ในขั้นตอนเดียว
- 💡 **Auto insight summary** — สรุป "อะไรสำคัญ ควรทำอะไรต่อ" เป็นภาษาไทย พร้อมบทสรุปผู้บริหาร (executive summary) ไม่ใช่แค่ทวนตัวเลข
- 📁 **JSON/JSONL input** — อ่าน `.csv`, `.json`, `.jsonl`, `.ndjson` อัตโนมัติ (ระบุ `--format` ได้)
- 🔠 **Auto encoding detection** — เดา encoding เอง (utf-8 → tis-620 → cp874 → cp1252) ไม่ต้องระบุ `--encoding`
- 🔁 **Before/after cleaning diff** — รายงานแสดงตารางก่อน/หลังการทำความสะอาด ว่าแก้อะไรไปกี่เซลล์

### จุดเด่น — Thai Data Quality (สิ่งที่ tool อื่นมองข้าม)

- 🔍 **Buddhist Era detection** — ตรวจจับปี พ.ศ. ผสม ค.ศ. ในคอลัมน์ date/number (เช่น `2567` vs `2024`)
- 🔢 **Thai numeral detection** — ตรวจจับเลขไทย (๐๑๒๓) ผสมเลขอารบิกในคอลัมน์เดียวกัน
- 📞 **Phone number detection** — ตรวจจับและทำความสะอาดเบอร์โทรไทย (เลขไทย→อารบิก, ลบ dash, +66→0, เก็บ leading zero)
- 👻 **Zero-width space detection** — ตรวจจับอักขระ U+200B ที่ทำให้ `groupby`, `join`, และ string equality พังเงียบ ๆ
- 📝 **Script composition** — วิเคราะห์อัตราส่วน Thai/Latin/digit/emoji ต่อคอลัมน์
- ⚠️ **Normalization issues** — ระบุปัญหา combining character, tone mark ซ้ำ, สระซ้ำ

### Thai Text EDA

- 📊 **Length 3 แบบ** — นับความยาวเป็น characters, tokens, และ words (ในภาษาไทยตัวเลขสามตัวนี้ต่างกันมาก)
- 🔤 **Top tokens / word frequency** — พร้อม Thai stopword handling
- 🔗 **N-grams** (bi-gram, tri-gram) — หลังการ tokenize ที่ถูกต้อง
- ☁️ **Word cloud** — พร้อม Thai font ที่ bundle มาให้ (ไม่เป็น tofu boxes □□□)
- 🏷️ **Thai NER** — สกัดชื่อคน/สถานที่/องค์กร จากข้อความไทย (ผ่าน pythainlp)

### Anomaly Detection

- 📈 **Statistical outliers** — z-score, modified z-score (MAD), IQR — เลือกวิธีตาม skew อัตโนมัติ
- 🌲 **ML-based outliers** — Isolation Forest, Local Outlier Factor (optional `thaieda[ml]`)
- 🔗 **Unified API** — `detect_anomalies(df, method="auto")` รวมทุกวิธีในฟังก์ชันเดียว
- 👤 **Text anomalies** — ความยาวผิดปกติ, mojibake, อักขระซ้ำ

### Data Cleaning

- 🔧 **Thai-specific cleaning** — ลบ zero-width space, แปลงเลขไทย→อารบิก, แก้ mojibake (ftfy)
- 📞 **Phone number normalization** — แปลงเบอร์โทรไทยเป็นมาตรฐาน 10 หลัก (เลขไทย→อารบิก, ลบ dash/space, +66→0, เก็บ leading zero ไม่ให้หาย)
- ⌨️ **Keyboard layout fix** — แก้การพิมพ์ผิดแป้นไทย/อังกฤษ (เช่น `l;ylfu` → `สวัสดี`)
- ✨ **PyThaiNLP normalize** — จัดระเบียบข้อความไทยรวมในขั้นตอนเดียว

### Visualization

- 📊 **Auto chart selection** — เลือก chart type อัตโนมัติตาม data type
- 🔥 **Correlation heatmap**, box plot, violin plot, scatter matrix
- 🕳️ **Missing data matrix + heatmap** — เห็น pattern ของค่าว่างแบบ missingno
- 📈 **Timeseries plots** — line + trend, STL decomposition, ACF (v0.4)

### อื่น ๆ

- 📄 **HTML report** — self-contained, ส่งต่อได้
- 🎯 **Target analysis** — ระบุ target column → แสดงความสัมพันธ์ของทุกคอลัมน์ (Pearson/ANOVA/Chi-square)
- 🌐 **Bilingual UI** — ป้ายและคำอธิบายเป็นไทยและอังกฤษ
- 🔌 **Tokenizer adapter** — รองรับ pythainlp, nlpo3, attacut (optional)
- 💬 **LLM Q&A** (v0.3+) — ถามคำถามเกี่ยวกับข้อมูลเป็นภาษาไทย

---

## 📦 การติดตั้ง

```bash
# Core (ไม่มี Thai tokenizer — เหมาะกับทดสอบ)
pip install thaieda

# พร้อม Thai tokenizer (แนะนำ)
pip install "thaieda[thai]"

# พร้อม Thai NER (สกัดชื่อคน/สถานที่)
pip install "thaieda[ner]"

# พร้อม ML anomaly detection (Isolation Forest / LOF)
pip install "thaieda[ml]"

# พร้อม statistical target analysis (p-values)
pip install "thaieda[stats]"

# พร้อม timeseries decomposition (STL ผ่าน statsmodels)
pip install "thaieda[timeseries]"

# พร้อม fast tokenizer (Rust-based)
pip install "thaieda[fast]"

# พร้อมตรวจ encoding อัตโนมัติ (chardet)
pip install "thaieda[detect]"

# ครบทุกอย่าง
pip install "thaieda[thai,ner,viz,ml,stats,timeseries,detect]"
```

---

## 🚀 การใช้งาน

### Python

```python
import pandas as pd
from thaieda import profile, read_data

# อ่านไฟล์อัตโนมัติ (CSV/JSON, เดา encoding ให้เอง)
df = read_data("data.json")

# สร้าง report พร้อมทำความสะอาด + auto insight
report = profile(df, clean=True)
report.to_html("report.html")  # → เปิดใน browser

# ดูสรุปข้อค้นพบสำคัญ (ภาษาไทย)
print(report.insights.executive_summary_th)
for ins in report.insights.insights:
    print(ins.severity, ins.title_th, "→", ins.recommendation_th)

# หรือใน Jupyter
report  # → แสดงผลใน cell
```

### CLI

```bash
# ครบจบในคำสั่งเดียว: ทำความสะอาด → วิเคราะห์ → รายงาน + ไฟล์ที่สะอาดแล้ว
thaieda run data.csv -o report.html --cleaned-output cleaned.csv

# ระบุคอลัมน์เป้าหมาย (target analysis)
thaieda run data.csv -o report.html --target price

# อ่าน JSON (auto-detect)
thaieda run data.json -o report.html

# วิเคราะห์อย่างเดียว (ไม่ทำความสะอาด)
thaieda run data.csv -o report.html --no-clean

# ข้ามการวิเคราะห์อนุกรมเวลา (เร็วขึ้นบนข้อมูลที่ไม่ใช่ timeseries)
thaieda run data.csv -o report.html --no-timeseries

# คำสั่งย่อยอื่น ๆ
thaieda profile data.csv -o report.html --clean   # วิเคราะห์ + ทำความสะอาด
thaieda clean data.csv -o cleaned.csv             # ทำความสะอาดอย่างเดียว
```

### Timeseries (Python)

```python
from thaieda import analyze_timeseries, analyze_dataframe_timeseries

# วิเคราะห์ทั้ง DataFrame อัตโนมัติ (หา datetime column เอง)
results = analyze_dataframe_timeseries(df)
for col, r in results.items():
    print(col, r.frequency_th, r.trend_direction_th, r.has_seasonality, r.seasonal_period)
    print(r.insights)  # ข้อค้นพบเป็นภาษาไทย

# หรือวิเคราะห์ทีละคอลัมน์ (series ที่ index ด้วยเวลา)
r = analyze_timeseries(series, engine="auto")  # "auto" | "statsmodels" | "basic"
```

---

## 🗺️ Roadmap

| Version | ฟีเจอร์ | สถานะ |
|---------|----------|--------|
| **v0.1** | Thai text profiling + data quality + HTML report + CLI | ✅ เสร็จ |
| **v0.2** | Thai NER, pythainlp normalize, auto chart, unified anomaly API, target analysis | ✅ เสร็จ |
| **v0.3** | Single-command pipeline (`run`), JSON input, auto encoding, auto insights, cleaning diff | ✅ เสร็จ |
| **v0.4** | Timeseries analysis (trend/seasonality/STL/ACF/gaps), distribution & correlation insights | ✅ เสร็จ |
| **v0.5** | LLM Q&A (litellm + Ollama local), Thai explanations | 📋 วางแผน |
| **v0.6** | Interactive dashboard (Streamlit/FastAPI), Thai UI | 📋 วางแผน |

---

## 🏗️ Architecture

```
thaieda/
  io/         # อ่าน CSV/JSON อัตโนมัติ + ตรวจ encoding/format (v0.3)
  detect/     # ตรวจจับประเภทคอลัมน์ (Thai text classifier)
  tokenize/   # adapter สำหรับ pythainlp / nlpo3 / attacut
  text/       # วัดค่า text metrics (length, freq, ngrams, TF-IDF)
  quality/    # Thai-specific data quality checks ← จุดเด่น
  anomaly/    # anomaly detection (statistical + ML + text + unified API)
  clean/      # data cleaning (encoding, zwspace, keyboard layout, pythainlp normalize)
  ner/        # Thai NER — สกัดชื่อคน/สถานที่/องค์กร (v0.2)
  analysis/   # target variable analysis — Pearson/ANOVA/Chi-square (v0.2)
  insight/    # auto insight summary — ตีความผลเป็นภาษาไทย (v0.3) + distribution/correlation/timeseries (v0.4)
  timeseries/ # timeseries analysis — trend/seasonality/STL/ACF/gaps (v0.4)
  viz/        # visualization + auto chart selection + Thai font (+ timeseries plots v0.4)
  report/     # สร้าง HTML report (Jinja2)
  i18n/       # ป้ายและคำอธิบาย TH/EN
  llm/        # LLM Q&A (v0.5+)
```

---

## 🤝 ร่วมพัฒนา

- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- เปิด [Issue](https://github.com/peetwan/thaieda/issues) รายงาน bug หรือเสนอ feature
- คุยกับเราได้ที่ [Discussions](https://github.com/peetwan/thaieda/discussions)

---

## 🙏 ขอบคุณ

- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) — Thai NLP library ที่เป็นรากฐาน
- [ydata-profiling](https://github.com/ydata-profiling/ydata-profiling) — แรงบันดาลใจด้าน EDA
- ชุมชน Thai NLP ทุกคน

---

## 📄 License

Apache-2.0 — ดูรายละเอียดใน [LICENSE](LICENSE)