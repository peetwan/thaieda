# ThaiEDA

**Exploratory data analysis that actually understands Thai.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests: 631 passed](https://img.shields.io/badge/tests-631%20passed-brightgreen.svg)]()
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Language aware](https://img.shields.io/badge/language-Thai%20%2B%20English%20aware-blueviolet.svg)]()

---

## What is ThaiEDA?

ThaiEDA is a Python library that automates exploratory data analysis for Thai and mixed Thai/English datasets. You give it a DataFrame, it gives you back a full report — smart pre-analysis, language detection, column types, quality issues, anomalies, cross-column insights, charts, and an executive-style HTML report. All in one line.

It handles the things generic EDA tools miss: Buddhist Era dates, Thai numerals, zero-width spaces, Thai vowel/tone marks, mixed Thai/English cells, mojibake encoding, Thai month names, national ID card validation, Thai address parsing, and PII like phone numbers.

---

## Quick Start

```bash
pip install thaieda
```

```python
import thaieda
import pandas as pd

df = pd.read_csv("data.csv")
result = thaieda.run(df)          # full EDA in one line
result.to_html("report.html")     # self-contained HTML report
```

`pip install thaieda` ติดตั้งทุกอย่างเลย — Thai tokenizer, NER, ML, Excel, stats, encoding detection, interactive charts ไม่ต้องใส่ extras

---

## Why ThaiEDA?

**Generic tools don't understand Thai data.** Pandas Profiling, ydata-profiling, and Sweetviz are great — until you feed them Thai data. They miss Buddhist Era years (พ.ศ.), Thai numerals (๑๒๓), zero-width spaces that break tokenization, and mojibake from TIS-620 encoding. ThaiEDA catches all of these.

**Privacy-first LLM analysis.** Want to ask an LLM about your data but can't send raw rows to a cloud API? ThaiEDA has 4 privacy modes — the default sends zero raw data off your machine. Perfect for government, finance, and medical data under PDPA.

**Insights, not just summaries.** A cross-column insight engine finds non-obvious patterns — "column A strongly predicts column B", "this group is 3× higher than average" — ranked by statistical interestingness with Benjamini-Hochberg correction.

**Thai-specific validation.** National ID card checksum validation, Thai address parsing (province/district/subdistrict), Thai holiday awareness for timeseries spike attribution. No other EDA tool does this.

**One line to get everything.** `thaieda.run(df)` chains the full pipeline: type detection → smart cleaning → quality checks → anomaly detection → insight discovery → visualization → HTML report. No config needed.

---

## How It Works

```
DataFrame
    │
    ▼
┌──────────────────────────────────────────────┐
│  thaieda.run(df)                             │
│                                              │
│  0. pre-analyze → data type + language       │
│  1. detect      → column types + Thai months │
│  2. clean       → smart cleaning (auto-decide)│
│  3. quality     → language-aware checks      │
│  4. anomaly     → statistical + ML + text    │
│  5. insights    → 6 cross-column patterns    │
│  6. viz         → interactive + static charts│
│  7. report      → executive HTML narrative   │

│  + optional: LLM analysis (4 privacy modes)  │
│  + optional: compare(df1, df2) side-by-side  │
└──────────────────────────────────────────────┘
    │
    ▼
EDAResult
  .to_html()        → report.html
  .to_dict()        → Python dict
  .to_json()        → JSON string
  .insights         → insight cards
  .cleaned_df       → cleaned DataFrame
  .quality_issues   → list of issues
  .quality_score    → 0-100 score with grade
  .anomalies        → anomaly findings
  .llm_response     → LLM analysis (if enabled)
  ._repr_html_()    → Jupyter rich display

run_folder("data/")  → FolderResult
  .to_html("dir/")      → individual HTML per file
  .to_master_html()     → single master HTML with sidebar
  .summary()            → text summary
  ._repr_html_()        → Jupyter rich display
```

---

## Version Highlights

- **v1.1 — Folder reports:** `run_folder()` analyzes CSV/Excel/JSON/JSONL/TSV folders, `FolderResult.to_master_html()` builds one master HTML with sidebar navigation, and the internal `_Breakdown` result is frozen so report summaries are stable and hash-safe.
- **v1.2 — Report redesign:** HTML reports now read like an executive briefing: executive summary, key findings, business translation, priority actions, plain-language explanations, and a narrative flow from “what happened” to “what to do next”.
- **v1.3 — Smart pre-analysis:** `_detect_language()` detects Thai/English/mixed/numeric data with confidence and per-column detail; `_detect_data_type()` classifies transaction/registry/survey/timeseries/mixed datasets before EDA; quality checks become language-aware so English-only data skips Thai-specific warnings automatically.

---

## Examples

### One-Line EDA

```python
import thaieda
import pandas as pd

df = pd.read_csv("data.csv")

# Full pipeline in one call
result = thaieda.run(df)

# Access results
result.to_html("report.html")
print(result.quality_issues)
print(result.insights)

# In Jupyter: just display the result
result  # renders HTML report inline
```

### Folder Mode — Analyze Every File at Once

```python
import thaieda

# One line — analyzes every CSV/Excel/JSON in the folder
results = thaieda.run_folder("data/")

# Print summary
print(results.summary())
# ThaiEDA FolderResult — data/
#   Files: 5 (✅ 5 / ❌ 0)
#   ✅ customers.csv — 10,000 rows × 8 cols, 15 insights
#   ✅ orders.csv    — 50,000 rows × 12 cols, 28 insights
#   ...

# Save individual HTML reports
results.to_html("reports/")

# Generate a single master HTML with sidebar navigation
results.to_master_html("master-report.html")
```

**`run_folder()` features:**
- Auto-scans for CSV, Excel (.xlsx/.xls), JSON, JSONL, TSV
- `recursive=True` to include subfolders
- `output_dir=` to specify where HTML goes
- Error isolation — one broken file doesn't crash the rest
- `progress=` callback for progress tracking
- All `run()` kwargs supported (`lang`, `clean`, `llm`, etc.)
- **`to_master_html()`** — combines all reports into one page with sidebar nav + summary table

### With LLM Analysis (Privacy-Safe)

```python
import thaieda

# Default: zero raw data leaves your machine
result = thaieda.run(df, llm=True, privacy="insight_only", provider="ollama")
print(result.llm_response)
```

### Compare Two Datasets

```python
from thaieda.compare import compare_datasets

diff = compare_datasets(df_train, df_test, labels=("train", "test"))
print(diff["schema_diff"])      # columns added/removed
print(diff["drift"]["numeric"]) # KS statistic per column
```

### Thai ID Card Validation

```python
from thaieda.quality import validate_thai_id, validate_thai_id_column

# Single ID
validate_thai_id("1-1234-56789-01-2")  # → True/False

# Entire column
result = validate_thai_id_column(df["id_card"])
print(f"Valid: {result['valid_count']}, Invalid: {result['invalid_count']}")
```

### Thai Address Parsing

```python
from thaieda.detect import parse_thai_address

addr = parse_thai_address("123 หมู่ 4 ต.บางบัว อ.บางบัว จ.กรุงเทพฯ 10230")
print(addr)
# {'house_number': '123', 'moo': '4', 'subdistrict': 'บางบัว',
#  'district': 'บางบัว', 'province': 'กรุงเทพฯ', 'postal_code': '10230'}
```

### Language Detection

```python
import pandas as pd
from thaieda.detect import _detect_language

df = pd.DataFrame({
    "product": ["กาแฟ", "ชาไทย", "ขนม"],
    "review": ["อร่อยมาก 5/5 stars", "ดีครับ", "ไม่ดี"],
    "sku": ["SKU001", "SKU002", "SKU003"],
})

info = _detect_language(df)
print(info["language"], info["confidence"])
print(info["columns"])
# thai/mixed/english/numeric + per-column language map
```

**Language Detection v2 features:**
- Unicode Thai block analysis (U+0E00–U+0E7F) including vowels/tone marks (U+0E30–U+0E4D)
- Zero-width-space aware (`\u200b`, BOM, word joiner)
- Mixed-cell detection เช่น `"อร่อยมาก 5/5 stars"`
- Common Thai word hints: `ครับ`, `ค่ะ`, `ไทย`, `อร่อย`, `ดี`, `ไม่`, `มี`, `และ`
- Lazy `pythainlp` tokenizer when installed; regex fallback when unavailable
- Per-column `column_details` + dataset-level `confidence` (0.0–1.0)
- Sample-based scan (first 500 rows/column) for large DataFrames

### Smart Pre-Analysis

ThaiEDA profiles the dataset *before* running the full report, so the narrative and quality checks match the data:

```python
from thaieda.report import _detect_data_type

pre = _detect_data_type(df)
print(pre["label"], pre["language"]["language"])
print(pre["focus"])
```

Smart pre-analysis detects:
- **Transaction data** — orders, payments, revenue, invoices
- **Registry/master data** — customers, products, stores, entity attributes
- **Survey/review data** — ratings, comments, feedback text
- **Timeseries data** — datetime index/columns + numeric measures
- **Mixed data** — conservative fallback when signals overlap
- **Language impact** — Thai/mixed data enables Thai-specific checks; English-only data skips พ.ศ./เลขไทย checks automatically

### Data Quality Score

```python
from thaieda.quality import compute_quality_score

score = compute_quality_score(quality_issues, n_columns=10, n_rows=1000)
print(f"Score: {score.score}/100 ({score.grade})")
# Score: 85/100 (B)
```

### Smart Cleaning

```python
from thaieda.clean._smart import plan_cleaning

plan = plan_cleaning(df)
print(f"Actions: {plan.actions}")    # ['zwspace', 'numerals', 'duplicates']
print(f"Skipped: {plan.skipped}")    # ['encoding', 'whitespace']
```

---

## Privacy Modes

Control exactly what data leaves your machine when using LLM analysis:

| Mode | What Leaves | Guarantee | When to Use |
|------|------------|-----------|-------------|
| `insight_only` (default) | Stats + insights only | Raw data never leaves | Government, medical, PDPA data |
| `anonymized` | Data with PII → tokens | Names/phones/ID cards masked | Need structure without raw PII |
| `dp_noise` | Stats + Laplace noise | Prevents re-identification | Small datasets where stats leak |
| `full` | Everything | None — you accept the risk | Public data, demos |

---

## What ThaiEDA Catches

| Problem | Example | What Happens |
|---------|---------|-------------|
| Buddhist Era dates | `15/03/2567` | Auto-detects พ.ศ. → converts to CE |
| Thai numerals | `๑๒๓` in numeric column | Converts to `123` |
| Zero-width spaces | `สม\u200bชาย` | Strips invisible chars and reports language evidence |
| Thai vowel/tone marks | `อร่อยค่ะ` | Counts U+0E30–U+0E4D for better Thai detection |
| Mixed Thai/English cells | `อร่อยมาก 5/5 stars` | Detects as mixed language instead of English/numeric |
| Thai text in English-heavy tables | Thai product column + English IDs | Column-level language detection preserves Thai checks |
| Common Thai words | `ครับ`, `ค่ะ`, `ไม่ดี` | Boosts confidence for short Thai text |
| Mojibake encoding | `Ã ¬Â¸Â¡Â¹` | Auto-detects TIS-620 → UTF-8 |
| Thai month names | `มกราคม` | Parses to ISO date |
| Phone numbers | `081-234-5678` | Detects + normalizes |
| National ID cards | `1-1234-56789-01-2` | Checksum validation |
| Thai addresses | `123 ม.4 ต.บางบัว อ.บางบัว จ.กรุงเทพฯ` | Parses to structured fields |
| Placeholder values | `-`, `N/A`, `ไม่มี` | Flags as missing |
| Constant columns | All same value | Flags as useless |
| Smart data type | Orders/reviews/timeseries | Pre-classifies transaction/registry/survey/timeseries/mixed |
| Language-aware checks | English-only DataFrame | Skips Thai-specific พ.ศ./เลขไทย warnings automatically |
| Thai holidays | Spike on Dec 5 | Attributes to Father's Day |

---

## Visualization

ThaiEDA generates both static (matplotlib) and interactive (Plotly) charts:

- **Static**: correlation heatmap, distribution, box/violin, missing matrix, scatter matrix, wordcloud, timeseries, pair plot, KDE, QQ plot, sunburst
- **Interactive**: hover tooltips, zoom, pan — using Plotly with Thai font (Sarabun) via Google Fonts
- **Color palette**: Okabe-Ito colorblind-safe (7 colors)
- **Thai font**: auto-detected for matplotlib, CSS-loaded for Plotly

```python
from thaieda.viz._interactive import create_correlation_heatmap_interactive

html_div = create_correlation_heatmap_interactive(df)  # → HTML <div> for reports
```

---

## Installation

```bash
# ติดตั้งทุกอย่างในคำสั่งเดียว
pip install thaieda
```

ไม่ต้องใส่ extras — `pip install thaieda` ติดตั้งทั้งหมด: Thai tokenizer, NER, ML, interactive charts, Excel, stats, encoding detection

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
| `run_folder()` | Analyze every CSV/Excel/JSON in a folder + master HTML |
| `compare()` | Side-by-side dataset comparison with drift detection |
| `io/` | Auto-read CSV/JSON/JSONL/Excel + encoding detection |
| `detect/` | Column type detection + Thai month names + address parsing + language detection v2 |
| `clean/` | Smart cleaning: auto-decide what to fix (encoding, numerals, BE, zwspace) |
| `quality/` | Language-aware quality checks + score 0-100 + Thai ID card validation |
| `anomaly/` | Statistical + ML + text anomaly detection |
| `ner/` | Thai NER: person/place/organization |
| `insight_engine/` | 6 cross-column insight patterns (BH-corrected) |
| `viz/` | Static + interactive charts with colorblind-safe palette |
| `report/` | Executive HTML report + smart pre-analysis (`_detect_data_type`) |
| `llm/` | Privacy-preserving LLM analysis (4 modes, 3 providers) |
| `timeseries/` | Trend/seasonality/STL/ACF + Thai holiday awareness |
| `schema/` | Multi-file PK/FK discovery + relationship matching |

---

## Testing

```bash
pytest tests/ -v                    # all tests (631 passed)
pytest tests/test_language_detection.py  # language detection + language-aware quality
pytest tests/test_thai_id.py        # ID card validation
pytest tests/test_thai_address.py   # address parsing
pytest tests/test_compare.py        # dataset comparison
pytest tests/test_run_folder.py     # folder mode + master HTML
pytest tests/test_llm.py            # LLM + privacy modes
ruff check src/ tests/              # lint
ruff format src/ tests/             # format
```

---

## License

[Apache-2.0](LICENSE) © Peet Wannasarnmetha