# ThaiEDA

**Exploratory data analysis that actually understands Thai.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests: 424 passed](https://img.shields.io/badge/tests-424%20passed-brightgreen.svg)]()
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

---

## What is ThaiEDA?

ThaiEDA is a Python library that automates exploratory data analysis for Thai-language datasets. You give it a DataFrame, it gives you back a full report — column types, quality issues, anomalies, cross-column insights, charts, and an HTML report. All in one line.

It handles the things generic EDA tools miss: Buddhist Era dates, Thai numerals, zero-width spaces, mojibake encoding, Thai month names, and PII like phone numbers and national ID cards.

---

## Quick Start

```bash
pip install thaieda
```

```python
import thaieda
import pandas as pd

df = pd.read_csv("data.csv")
result = thaieda.run(df)          # that's it — full EDA in one line
result.to_html("report.html")     # self-contained HTML report
```

`pip install thaieda` ติดตั้งทุกอย่างเลย — Thai tokenizer, NER, ML, Excel, stats, encoding detection ไม่ต้องใส่ extras

---

## Why ThaiEDA?

**Generic tools don't understand Thai data.** Pandas Profiling, ydata-profiling, and Sweetviz are great — until you feed them Thai data. They miss Buddhist Era years (พ.ศ.), Thai numerals (๑๒๓), zero-width spaces that break tokenization, and mojibake from TIS-620 encoding. ThaiEDA catches all of these.

**Privacy-first LLM analysis.** Want to ask an LLM about your data but can't send raw rows to a cloud API? ThaiEDA has 4 privacy modes — the default sends zero raw data off your machine. Just stats and insights. Perfect for government, finance, and medical data under PDPA.

**Insights, not just summaries.** Most EDA tools show you `df.describe()` with nicer formatting. ThaiEDA has a cross-column insight engine that finds non-obvious patterns — "column A strongly predicts column B", "this group is 3× higher than average", "this column has outliers at row 47" — ranked by statistical interestingness with Benjamini-Hochberg correction.

**One line to get everything.** `thaieda.run(df)` chains the full pipeline: type detection → data cleaning → quality checks → anomaly detection → insight discovery → visualization → HTML report. No config needed.

---

## How It Works

```
DataFrame
    │
    ▼
┌─────────────────────────────────────────┐
│  thaieda.run(df)                        │
│                                         │
│  1. detect    → column types             │
│  2. clean     → fix encoding/numerals/BE │
│  3. quality   → nulls, placeholders, BE │
│  4. anomaly   → statistical + text      │
│  5. insights  → 6 cross-column patterns │
│  6. viz       → auto charts (Thai font) │
│  7. report    → self-contained HTML     │
│                                         │
│  + optional: LLM analysis (4 modes)     │
└─────────────────────────────────────────┘
    │
    ▼
EDAResult
  .to_html()      → report.html
  .to_dict()      → Python dict
  .to_json()      → JSON string
  .insights       → insight cards
  .cleaned_df     → cleaned DataFrame
  .quality_issues → list of issues
  .anomalies      → anomaly findings
  .llm_response   → LLM analysis (if enabled)
```

---

## Examples

### One-Line EDA

```python
import thaieda
import pandas as pd

df = pd.read_csv("data.csv")

# Full pipeline: detect → clean → quality → insights → viz → report
result = thaieda.run(df)

# Access results
result.to_html("report.html")
print(result.insights)           # cross-column insight cards
print(result.quality_issues)     # data quality findings
print(result.notes)              # pipeline notes/warnings

# Alias works too
result = thaieda.EDA(df)
```

### With LLM Analysis (Privacy-Safe)

```python
import thaieda

# Default: zero raw data leaves your machine
result = thaieda.run(df, llm=True, privacy="insight_only", provider="ollama")
print(result.llm_response)

# Or use OpenAI/Anthropic — still safe with insight_only
result = thaieda.run(df, llm=True, privacy="insight_only", provider="openai")
```

### Privacy Modes

Control exactly what data leaves your machine:

| Mode | What Leaves | Guarantee | When to Use |
|------|------------|-----------|-------------|
| `insight_only` (default) | Stats + insights only | Raw data never leaves | Government, medical, PDPA data |
| `anonymized` | Data with PII → tokens | Names/phones/ID cards masked | Need structure without raw PII |
| `dp_noise` | Stats + Laplace noise | Prevents re-identification | Small datasets where stats leak |
| `full` | Everything | None — you accept the risk | Public data, demos |

```python
from thaieda.llm import analyze_with_llm

# Each mode as a standalone call
answer = analyze_with_llm(df, privacy="insight_only", provider="ollama")
answer = analyze_with_llm(df, privacy="anonymized", provider="openai")
answer = analyze_with_llm(df, privacy="dp_noise", provider="anthropic", epsilon=0.5)
```

### Manual Pipeline (Full Control)

```python
from thaieda import profile, discover_insights
from thaieda.detect import detect_all

# Step-by-step if you want control
report = profile(df, clean=True)
report.to_html("report.html")

result = discover_insights(df, detect_all(df), top_n=8)
for card in result.cards:
    print(f"[{card.pattern}] {card.description_th}")
    print(f"  → {card.recommendation_th}")
```

---

## What ThaiEDA Catches

| Problem | Example | What Happens |
|---------|---------|-------------|
| Buddhist Era dates | `15/03/2567` | Auto-detects พ.ศ. → converts to CE |
| Thai numerals | `๑๒๓` in numeric column | Converts to `123` |
| Zero-width spaces | `สม\u200bชาย` | Strips invisible chars |
| Mojibake encoding | `Ã ¬Â¸Â¡Â¹` | Auto-detects TIS-620 → UTF-8 |
| Thai month names | `มกราคม` | Parses to ISO date |
| Phone numbers | `081-234-5678` | Detects + normalizes |
| National ID cards | `1-1234-56789-01-2` | Detects via regex |
| Placeholder values | `-`, `N/A`, `ไม่มี` | Flags as missing |
| Constant columns | All same value | Flags as useless |

---

## Installation

```bash
# ติดตั้งทุกอย่างในคำสั่งเดียว
pip install thaieda
```

ไม่ต้องใส่ extras — `pip install thaieda` ติดตั้งทั้งหมดเลย: Thai tokenizer, NER, ML anomaly detection, timeseries, Excel, stats, encoding detection

LLM providers ยังเป็น optional (lazy-imported — ไม่ต้องติดตั้งถ้าไม่ใช้):

```bash
pip install openai       # OpenAI GPT
pip install anthropic    # Anthropic Claude
pip install ollama       # Ollama local LLM (หรือใช้ HTTP fallback ไม่ต้องติดตั้ง)
```

**Requirements:** Python 3.10+

---

## Modules

| Module | What It Does |
|--------|-------------|
| `run()` / `EDA()` | One-liner API — full pipeline in one call |
| `io/` | Auto-read CSV/JSON/JSONL/Excel + encoding detection |
| `detect/` | Column type detection + Thai month names |
| `clean/` | Encoding fix, numerals, BE→CE, dates, duplicates, missing |
| `quality/` | Thai quality checks + placeholder/constant detection |
| `anomaly/` | Statistical + ML + text anomaly detection |
| `ner/` | Thai NER: person/place/organization |
| `insight_engine/` | 6 cross-column insight patterns (BH-corrected) |
| `viz/` | Auto charts with Thai font support |
| `report/` | Self-contained HTML report (Jinja2) |
| `llm/` | Privacy-preserving LLM analysis (4 modes, 3 providers) |
| `schema/` | Multi-file PK/FK discovery + relationship matching |
| `timeseries/` | Trend/seasonality/STL/ACF/gap detection |

---

## Testing

```bash
pytest tests/ -v              # all tests (424 passed)
pytest tests/test_oneliner.py # one-liner API tests
pytest tests/test_llm.py      # LLM + privacy mode tests
ruff check src/ tests/        # lint
ruff format src/ tests/       # format
```

---

## License

[Apache-2.0](LICENSE) © Peet Wannasarnmetha