# ThaiEDA

**AutoEDA สำหรับข้อมูลภาษาไทย — Exploratory data analysis that speaks Thai**

[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

> เข้าใจข้อมูลภาษาไทยในบรรทัดเดียว

ThaiEDA คือ library สำหรับทำ Exploratory Data Analysis (EDA) โดยเฉพาะข้อมูลที่มีภาษาไทย — ตรวจจับปัญหาข้อมูลที่ tool ทั่วไปมองข้าม เช่น ปีพุทธศักราชผสมคริสต์ศักราช เลขไทยผสมเลขอารบิก อักขระ zero-width space ที่ทำให้ groupby พัง และอื่น ๆ

---

## ✨ ฟีเจอร์

### จุดเด่น — Thai Data Quality (สิ่งที่ tool อื่นมองข้าม)

- 🔍 **Buddhist Era detection** — ตรวจจับปี พ.ศ. ผสม ค.ศ. ในคอลัมน์ date/number (เช่น `2567` vs `2024`)
- 🔢 **Thai numeral detection** — ตรวจจับเลขไทย (๐๑๒๓) ผสมเลขอารบิกในคอลัมน์เดียวกัน
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
- ⌨️ **Keyboard layout fix** — แก้การพิมพ์ผิดแป้นไทย/อังกฤษ (เช่น `l;ylfu` → `สวัสดี`)
- ✨ **PyThaiNLP normalize** — จัดระเบียบข้อความไทยรวมในขั้นตอนเดียว

### Visualization

- 📊 **Auto chart selection** — เลือก chart type อัตโนมัติตาม data type
- 🔥 **Correlation heatmap**, box plot, violin plot, scatter matrix
- 🕳️ **Missing data matrix + heatmap** — เห็น pattern ของค่าว่างแบบ missingno

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

# พร้อม fast tokenizer (Rust-based)
pip install "thaieda[fast]"

# ครบทุกอย่าง
pip install "thaieda[thai,ner,viz,ml,stats]"
```

---

## 🚀 การใช้งาน

### Python

```python
import pandas as pd
from thaieda import profile

df = pd.read_csv("data.csv")

# สร้าง report
report = profile(df)
report.to_html("report.html")  # → เปิดใน browser

# หรือใน Jupyter
report  # → แสดงผลใน cell
```

### CLI

```bash
thaieda profile data.csv -o report.html
```

---

## 🗺️ Roadmap

| Version | ฟีเจอร์ | สถานะ |
|---------|----------|--------|
| **v0.1** | Thai text profiling + data quality + HTML report + CLI | ✅ เสร็จ |
| **v0.2** | Thai NER, pythainlp normalize, auto chart, unified anomaly API, target analysis | ✅ เสร็จ |
| **v0.3** | LLM Q&A (litellm + Ollama local), Thai explanations | 📋 วางแผน |
| **v0.4** | Interactive dashboard (Streamlit/FastAPI), Thai UI | 📋 วางแผน |

---

## 🏗️ Architecture

```
thaieda/
  detect/     # ตรวจจับประเภทคอลัมน์ (Thai text classifier)
  tokenize/   # adapter สำหรับ pythainlp / nlpo3 / attacut
  text/       # วัดค่า text metrics (length, freq, ngrams, TF-IDF)
  quality/    # Thai-specific data quality checks ← จุดเด่น
  anomaly/    # anomaly detection (statistical + ML + text + unified API)
  clean/      # data cleaning (encoding, zwspace, keyboard layout, pythainlp normalize)
  ner/        # Thai NER — สกัดชื่อคน/สถานที่/องค์กร (v0.2)
  analysis/   # target variable analysis — Pearson/ANOVA/Chi-square (v0.2)
  viz/        # visualization + auto chart selection + Thai font
  report/     # สร้าง HTML report (Jinja2)
  i18n/       # ป้ายและคำอธิบาย TH/EN
  llm/        # LLM Q&A (v0.3+)
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