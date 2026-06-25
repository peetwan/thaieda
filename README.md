<div align="center">

# ThaiEDA

**AutoEDA for Thai-language data — Exploratory data analysis that speaks Thai**

[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Tests: 356](https://img.shields.io/badge/tests-356%20passed-brightgreen.svg)]()

<p align="center">
  <strong>เข้าใจข้อมูลภาษาไทยในบรรทัดเดียว</strong><br>
  Understand your Thai data in a single command
</p>

---

</div>

## Table of Contents

- [Overview](#overview)
- [Why ThaiEDA?](#why-thaieda)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

ThaiEDA is a Python library for **Exploratory Data Analysis (EDA)** built specifically for datasets containing Thai-language text. It catches data quality issues that general-purpose tools miss — Buddhist Era dates mixed with CE years, Thai numerals mixed with Arabic numerals, zero-width spaces that silently break `groupby`, and more.

```bash
# One command — clean, analyze, and generate a report
thaieda run data.csv -o report.html
```

### What's New

| Version | Highlights |
|---------|-----------|
| **v0.8** | Clean data + actionable insights — Thai numeral→numeric, BE→CE conversion, date standardization, duplicate/missing handling, correlation & outlier patterns, placeholder/constant column detection, Excel support, adaptive thresholds |
| **v0.7** | Insight visualization — auto-generated charts for each cross-column finding (bar, donut, box plot, trend line) |
| **v0.6** | Cross-column insight engine — discovers outstanding / attribution / comparison / trend findings (group-by + statistical scoring, BH-corrected) |
| **v0.5** | Multi-file schema discovery, ER diagram, relationship validation, orphan detection |
| **v0.4** | Timeseries analysis (trend/seasonality/STL/ACF/gaps), distribution & correlation insights |
| **v0.3** | Single-command pipeline, JSON input, auto encoding detection, auto insights, cleaning diff |
| **v0.2** | Thai NER, pythainlp normalize, auto chart selection, unified anomaly API, target analysis |
| **v0.1** | Thai text profiling, data quality checks, HTML report, CLI |

> See [CHANGELOG.md](CHANGELOG.md) for the full history.

---

## Why ThaiEDA?

General-purpose EDA tools (pandas-profiling, ydata-profiling) treat Thai text as opaque strings. ThaiEDA understands Thai-specific data problems:

| Problem | Example | What ThaiEDA does |
|---------|---------|-------------------|
| **Buddhist Era vs CE** | `2567` mixed with `2024` in date columns | Detects, flags, and **converts** BE→CE |
| **Thai numerals** | `๐๑๒๓` mixed with `0123` in the same column | Detects, converts to Arabic, and **coerces to numeric dtype** |
| **Zero-width spaces** | `กข\u200bค` — invisible characters that break `groupby`/`join` | Detects and removes them |
| **Phone numbers** | `+66 ๘๘-๙๖๓-๒๑๐` | Normalizes to `0889632100` |
| **Keyboard layout typos** | `l;ylfu` typed instead of `สวัสดี` | Fixes wrong keyboard layout |
| **Mojibake** | Text with broken encoding from TIS-620/CP874 | Repairs with ftfy |
| **Placeholder values** | `-`, `N/A`, `ไม่มี` used instead of NaN | Detects and flags as quality issue |
| **Thai month names** | `15 มกราคม 2567` in date columns | Detects as datetime and **standardizes to ISO** |
| **Duplicate rows** | Same record entered multiple times | Detects and offers removal |
| **Constant columns** | Column with only one value (zero variance) | Flags as useless for analysis |

### When to use ThaiEDA?

- **Use ThaiEDA when** your data contains Thai text, Buddhist Era dates, Thai numerals, zero-width characters, or multiple related files that need relationship discovery
- **Use ydata-profiling when** you want rich interactive charts on clean English numeric data
- **Use both** — ThaiEDA to clean and profile Thai-specific issues → ydata-profiling to explore the cleaned result

### Eval Results

ThaiEDA includes a reproducible eval framework (`eval/`) with hand-authored ground-truth manifests. Run it:

```bash
PYTHONPATH="src" python eval/run_eval.py
```

| Capability | Metric | Result | Target |
|------------|--------|--------|--------|
| **Relationship discovery** | Precision / Recall / F1 | **1.00 / 1.00 / 1.00** | ≥0.90 |
| **Insight honesty** | False discoveries on noise | **0** (BH correction works) | ≤2 |
| **Insight honesty** | Determinism (repeat runs) | **identical** | identical |
| **Thai quality** | Detection precision | **1.00** | 1.00 |
| **Thai quality** | Silent corruption caught | **city 5→4 groups** (zero-width merge) | — |

> See [eval/results/REPORT.md](eval/results/REPORT.md) for the full report with findings and limitations.

---

## Quick Start

```bash
pip install "thaieda[thai]"

# Generate a full report from CSV/JSON/Excel (auto encoding detection)
thaieda run data.csv -o report.html
```

```python
import pandas as pd
from thaieda import profile, read_data

# Read any format — auto-detects CSV/JSON/JSONL/Excel and encoding
df = read_data("data.json")

# Profile + clean + auto insights in one call
report = profile(df, clean=True)
report.to_html("report.html")

# Get Thai-language insights
print(report.insights.executive_summary_th)
```

---

## Installation

ThaiEDA uses optional dependencies — install only what you need:

```bash
# Core (no Thai tokenizer — for quick testing)
pip install thaieda

# With Thai tokenizer (recommended)
pip install "thaieda[thai]"

# With Thai NER (person/place/organization extraction)
pip install "thaieda[ner]"

# With ML anomaly detection (Isolation Forest / LOF)
pip install "thaieda[ml]"

# With statistical target analysis (p-values)
pip install "thaieda[stats]"

# With timeseries decomposition (STL via statsmodels)
pip install "thaieda[timeseries]"

# With auto encoding detection (chardet)
pip install "thaieda[detect]"

# With Excel support (openpyxl)
pip install "thaieda[excel]"

# Everything
pip install "thaieda[thai,ner,viz,ml,stats,timeseries,detect,excel]"
```

**Requirements:** Python 3.10+, pandas, numpy, matplotlib, Jinja2

---

## Features

### Thai Data Quality

- **Buddhist Era detection + conversion** — catches BE/CE year mixing (`2567` vs `2024`) and converts BE→CE (v0.8)
- **Thai numeral detection + coercion** — catches Thai digits (`๐๑๒๓`) and converts to numeric dtype, not NaN (v0.8)
- **Phone number normalization** — converts Thai phone numbers to standard 10-digit format
- **Zero-width space detection** — catches U+200B characters that silently break `groupby`, `join`, and string equality
- **Placeholder value detection** — flags `-`, `N/A`, `ไม่มี` used instead of NaN (v0.8)
- **Constant column detection** — flags columns with zero variance as useless for analysis (v0.8)
- **Script composition** — analyzes Thai/Latin/digit/emoji ratio per column
- **Normalization issues** — flags combining characters, duplicate tone marks, duplicate vowels

### Data Cleaning

- **Thai-specific cleaning** — remove zero-width spaces, convert Thai→Arabic numerals, fix mojibake (ftfy)
- **Numeric coercion** — convert string columns with Thai numerals to proper numeric dtype (v0.8)
- **Buddhist Era conversion** — convert พ.ศ. → ค.ศ. in numeric and date-string columns (v0.8)
- **Date standardization** — Thai month names → ISO format + พ.ศ. → ค.ศ. (v0.8)
- **Duplicate row removal** — detect and remove fully duplicated rows (v0.8)
- **Missing value handling** — 5 strategies: flag, drop, median, mode, unknown (v0.8)
- **Phone number normalization** — standardize Thai phone numbers to 10-digit format
- **Keyboard layout fix** — correct Thai/English keyboard typos (`l;ylfu` → `สวัสดี`)
- **PyThaiNLP normalize** — consolidate text normalization in one step

### Cross-Column Insight Engine (v0.6 + v0.8)

- **Discovers, not just interprets** — combines columns (group-by + aggregate + statistical scoring) to surface non-obvious findings, ranked by an interestingness pipeline
- **6 patterns**:
  - *outstanding* — one segment dominates
  - *attribution* — one segment is a large share of a total
  - *comparison* — a segment differs significantly from the rest (ANOVA/Kruskal + JSD)
  - *trend* — monotonic movement over time (Mann-Kendall)
  - *correlation* — strong pairwise correlation between numeric columns (|r| ≥ 0.7) (v0.8)
  - *outlier* — row-level statistical outliers (z-score ≥ 3) (v0.8)
- **Statistically honest** — Benjamini-Hochberg correction across all candidate tests (FDR control for hundreds of comparisons); degrades gracefully without scipy
- **Cross-pattern dedup** — when multiple patterns point to the same breakdown × measure × segment, only the highest-ranked keeps full score (v0.8)
- **Adaptive thresholds** — `min_segment` auto-adjusts for small datasets (v0.8)
- **Domain-agnostic** — zero column-name logic, no overfitting; driven entirely by `ColumnType` + cardinality + value ranges
- **Scales to 1M+ rows** — two-phase sampling (score on a ~100k sample, recompute exact numbers on the full data for the top-N only)
- **Full-data recompute** — correlation and outlier evidence recomputed on full dataset, not just sample (v0.8)
- **Thai-aware** — category keys normalized before group-by; every finding written in Thai with evidence

### Insight Visualization (v0.7)

- **Auto-chart per finding** — each cross-column insight card includes a visualization matched to its pattern:
  - **Outstanding** → horizontal bar chart (top segment highlighted in green)
  - **Attribution** → donut chart with share % in the center
  - **Comparison** → box plot by group (top segment highlighted in red)
  - **Trend** → line chart with full bucket series, direction arrow, and τ value (v0.8 fix)
- **Seamless integration** — charts generated automatically during `profile()` and embedded in the HTML report
- **Dark-themed** — all charts match the report's dark theme
- **Graceful degradation** — if a chart can't be generated, the card shows with text + evidence table only

### Thai Text EDA

- **Length in 3 dimensions** — characters, tokens, and words (these differ significantly in Thai)
- **Top tokens / word frequency** — with Thai stopword handling
- **N-grams** — bi-gram, tri-gram after proper tokenization
- **Word cloud** — bundled Thai font (no tofu boxes □□□)
- **Thai NER** — extract person/place/organization names via pythainlp

### Anomaly Detection

- **Statistical outliers** — z-score, modified z-score (MAD), IQR — auto-selects method by skewness
- **ML-based outliers** — Isolation Forest, Local Outlier Factor (optional `thaieda[ml]`)
- **Unified API** — `detect_anomalies(df, method="auto")` for all methods in one call
- **Text anomalies** — abnormal length, mojibake, character repetition

### Timeseries Analysis (v0.4)

- **Auto timeseries analysis** — detects datetime columns and analyzes all numeric columns automatically
- **Trend & seasonality** — detects direction and seasonal patterns via autocorrelation
- **STL decomposition** — separates trend / seasonal / residual (uses statsmodels, falls back to moving-average)
- **Time gaps & spikes** — detects missing time intervals and period-specific outliers
- **Timeseries charts** — line + trend line, 4-panel decomposition, ACF plot
- **Distribution insights** — skewness, kurtosis, bimodal detection for numeric columns
- **Correlation & duplicates** — highly correlated column pairs, duplicate rows, numbers stored as text

### Multi-File Schema Discovery (v0.5)

- **Auto relationship detection** — scan a directory of CSV/JSON files and discover how tables connect via primary/foreign keys
- **Value validation** — confirms relationships with real data overlap; detects orphan records
- **ER diagram** — generates a Mermaid.js ER diagram showing all tables, columns, and relationships
- **Thai-aware key normalization** — normalizes Thai numerals, zero-width spaces, and float artifacts before comparing keys
- **Combined report** — single HTML report covering all files + relationships + orphan findings

### Visualization

- **Auto chart selection** — picks chart type based on data type automatically
- **Correlation heatmap**, box plot, violin plot, scatter matrix
- **Missing data matrix + heatmap** — visualize missing-value patterns
- **Timeseries plots** — line + trend, STL decomposition, ACF
- **Insight charts** — pattern-matched charts per finding (v0.7)

### Pipeline & Reporting

- **Single-command pipeline** — `thaieda run data.csv` does clean → analyze → report + cleaned file output
- **Auto insight summary** — Thai-language executive summary with actionable recommendations
- **Multi-format input** — auto-reads `.csv`, `.tsv`, `.json`, `.jsonl`, `.ndjson`, `.xlsx`, `.xls` (v0.8)
- **Auto encoding detection** — auto-guesses encoding (utf-8 → tis-620 → cp874 → cp1252)
- **Before/after cleaning diff** — visual report showing what was fixed and how many cells changed
- **HTML report** — self-contained, shareable, responsive, print-friendly
- **Bilingual UI** — labels and descriptions in both Thai and English

---

## Usage

### CLI

```bash
# Full pipeline: clean → analyze → report + cleaned file
thaieda run data.csv -o report.html --cleaned-output cleaned.csv

# Read Excel file (v0.8 — requires pip install thaieda[excel])
thaieda run data.xlsx -o report.html

# Multi-file: analyze a whole directory (auto-discovers relationships)
thaieda dataset data-folder/ -o schema-report.html

# With target column analysis
thaieda run data.csv -o report.html --target price

# Cross-column insights are on by default; tune or disable them
thaieda profile data.csv -o report.html --insights-top 12
thaieda profile data.csv -o report.html --no-insights

# Sample N rows before analysis (for large files)
thaieda run data.csv -o report.html --sample 5000

# Clean only
thaieda clean data.csv -o cleaned.csv
```

### Python

```python
import pandas as pd
from thaieda import profile, read_data

# Auto-read any format (CSV/JSON/JSONL/Excel, auto encoding detection)
df = read_data("data.xlsx")

# Profile with cleaning + auto insights
report = profile(df, clean=True)
report.to_html("report.html")

# Thai-language insights
print(report.insights.executive_summary_th)
for ins in report.insights.insights:
    print(ins.severity, ins.title_th, "→", ins.recommendation_th)

# In Jupyter — display directly in cell
report
```

### Cross-Column Insights

```python
from thaieda import discover_insights, profile
from thaieda.detect import detect_all

# Standalone — discover insights from any DataFrame
result = discover_insights(df, detect_all(df), top_n=8)
for card in result.cards:
    p = card.perspective
    print(f"[{card.pattern}] score={card.score:.2f}  {p.breakdown} × {p.measure} ({p.agg})")
    print(f"  {card.description_th}")
    print(f"  → {card.recommendation_th}")

# Or via the full report (on by default; top 3 feed the executive summary)
report = profile(df)
report.to_html("report.html")
print(report.insight_engine.total, "cross-column insights")
```

### Data Cleaning (v0.8)

```python
from thaieda.clean import (
    coerce_numeric_column,
    convert_buddhist_era,
    normalize_dates,
    remove_duplicate_rows,
    handle_missing_values,
    clean_thai_text,
)

# Convert Thai numeral strings to numeric dtype
cleaned, result = coerce_numeric_column(df["ยอดขาย"])

# Convert Buddhist Era → CE
cleaned, result = convert_buddhist_era(df["ปีเกิด"])

# Standardize Thai month names to ISO + convert BE→CE
cleaned, result = normalize_dates(df["วันที่สั่งซื้อ"])

# Remove duplicate rows
df_clean, result = remove_duplicate_rows(df)

# Handle missing values (5 strategies)
cleaned, result = handle_missing_values(df["อายุ"], strategy="median")

# Full Thai text cleaning pipeline
cleaned, results = clean_thai_text(df["ชื่อลูกค้า"])
```

### Multi-File Schema Discovery

```python
from thaieda import profile_dataset, DatasetReport

# Analyze a directory of related files — discovers relationships automatically
dataset = profile_dataset("data-folder/")
print(f"Tables: {len(dataset.tables)}, Relationships: {len(dataset.relationships)}")

# View discovered relationships
for rel in dataset.relationships:
    print(f"  {rel.from_table}.{rel.from_column} → {rel.to_table}.{rel.to_column}  "
          f"[{rel.cardinality}]  overlap={rel.overlap_ratio:.1%}  orphans={rel.orphan_count}")

# Generate ER diagram (Mermaid text)
print(dataset.to_mermaid())

# Render combined HTML report with ER diagram
report = DatasetReport(dataset, lang="th")
report.to_html("schema-report.html")
```

---

## Architecture

```
thaieda/
  io/             # Auto-read CSV/JSON/JSONL/Excel + encoding detection
  detect/         # Column type detection + Thai month name detection (v0.8)
  tokenize/       # Tokenizer adapter: pythainlp / nlpo3 / attacut
  text/           # Text metrics: length, frequency, n-grams, TF-IDF
  quality/        # Thai quality checks + placeholder/constant detection (v0.8) + vectorized BE check
  anomaly/        # Anomaly detection: statistical + ML + text + unified API
  clean/          # Data cleaning: encoding, zwspace, numerals, BE→CE, dates, duplicates, missing (v0.8)
  ner/            # Thai NER: person/place/organization extraction
  analysis/       # Target variable analysis: Pearson/ANOVA/Chi-square
  insight/        # Auto insight summary in Thai (interpreter)
  insight_engine/ # Cross-column insight discovery: 6 patterns + BH correction + correlation + outlier (v0.8)
  timeseries/     # Timeseries analysis: trend/seasonality/STL/ACF/gaps
  schema/         # Multi-file schema discovery: PK/FK detection + relationship matching
  viz/            # Visualization + auto chart + Thai font + insight charts (v0.7)
  report/         # HTML report generation (Jinja2) + DatasetReport
  i18n/           # Bilingual labels (Thai/English)
  llm/            # LLM Q&A (planned)
```

**Design principles:**

- **Lazy imports** — core library works without heavy optional dependencies
- **No silent fallbacks** — missing tokenizer fails loudly with a helpful message
- **matplotlib Agg backend** — no GUI dependencies, works headless
- **Vectorized operations** — quality checks and cleaning use pandas `.str` accessors for performance
- **Adaptive thresholds** — `min_segment` auto-adjusts for small datasets (v0.8)
- **Full-data recompute** — insight evidence computed on full dataset, not just sample (v0.8)

---

## Roadmap

| Version | Feature | Status |
|---------|---------|--------|
| **v0.1** | Thai text profiling, data quality, HTML report, CLI | ✅ Done |
| **v0.2** | Thai NER, pythainlp normalize, auto chart, unified anomaly API, target analysis | ✅ Done |
| **v0.3** | Single-command pipeline, JSON input, auto encoding, auto insights, cleaning diff | ✅ Done |
| **v0.4** | Timeseries analysis, distribution & correlation insights | ✅ Done |
| **v0.5** | Multi-file schema discovery, ER diagram, relationship validation, orphan detection | ✅ Done |
| **v0.6** | Cross-column insight engine (outstanding / attribution / comparison / trend, BH-corrected) | ✅ Done |
| **v0.7** | Insight visualization — auto charts for each cross-column finding | ✅ Done |
| **v0.8** | Clean data + actionable insights — BE→CE, numeric coercion, date standardization, correlation/outlier patterns, placeholder/constant detection, Excel support, adaptive thresholds | ✅ Done |
| **v0.9** | LLM Q&A (litellm + Ollama local), Thai explanations | 📋 Planned |
| **v1.0** | Interactive dashboard (Streamlit/FastAPI), Thai UI | 📋 Planned |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

```bash
# Development setup
pip install -e ".[dev,thai,viz]"
pre-commit install

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

- Open an [Issue](https://github.com/peetwan/thaieda/issues) to report bugs or suggest features
- Start a [Discussion](https://github.com/peetwan/thaieda/discussions) for questions or ideas
- Please read our [Code of Conduct](CODE_OF_CONDUCT.md)

---

## Acknowledgments

- [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp) — the foundational Thai NLP library
- [ydata-profiling](https://github.com/ydata-profiling/ydata-profiling) — inspiration for EDA design
- The Thai NLP community

---

## License

[Apache-2.0](LICENSE) © Peet Wannasarnmetha