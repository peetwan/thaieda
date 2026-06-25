<div align="center">

# ThaiEDA

**AutoEDA for Thai-language data — Exploratory data analysis that speaks Thai**

[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

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
| **Buddhist Era vs CE** | `2567` mixed with `2024` in date columns | Detects and flags the mismatch |
| **Thai numerals** | `๐๑๒๓` mixed with `0123` in the same column | Detects and converts to Arabic digits |
| **Zero-width spaces** | `กข\u200bค` — invisible characters that break `groupby`/`join` | Detects and removes them |
| **Phone numbers** | `+66 ๘๘-๙๖๓-๒๑๐` | Normalizes to `0889632100` |
| **Keyboard layout typos** | `l;ylfu` typed instead of `สวัสดี` | Fixes wrong keyboard layout |
| **Mojibake** | Text with broken encoding from TIS-620/CP874 | Repairs with ftfy |

---

## Quick Start

```bash
pip install "thaieda[thai]"

# Generate a full report from CSV/JSON (auto encoding detection)
thaieda run data.csv -o report.html
```

```python
import pandas as pd
from thaieda import profile, read_data

# Read any format — auto-detects CSV/JSON/JSONL and encoding
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

# Everything
pip install "thaieda[thai,ner,viz,ml,stats,timeseries,detect]"
```

**Requirements:** Python 3.10+, pandas, numpy, matplotlib, Jinja2

---

## Features

### Thai Data Quality

- **Buddhist Era detection** — catches BE/CE year mixing (`2567` vs `2024`) in date/number columns
- **Thai numeral detection** — catches Thai digits (`๐๑๒๓`) mixed with Arabic numerals
- **Phone number normalization** — converts Thai phone numbers to standard 10-digit format (Thai→Arabic, strip dashes, +66→0, preserve leading zero)
- **Zero-width space detection** — catches U+200B characters that silently break `groupby`, `join`, and string equality
- **Script composition** — analyzes Thai/Latin/digit/emoji ratio per column
- **Normalization issues** — flags combining characters, duplicate tone marks, duplicate vowels

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

### Data Cleaning

- **Thai-specific cleaning** — remove zero-width spaces, convert Thai→Arabic numerals, fix mojibake (ftfy)
- **Phone number normalization** — standardize Thai phone numbers to 10-digit format
- **Keyboard layout fix** — correct Thai/English keyboard typos (`l;ylfu` → `สวัสดี`)
- **PyThaiNLP normalize** — consolidate text normalization in one step

### Timeseries Analysis (v0.4)

- **Auto timeseries analysis** — detects datetime columns and analyzes all numeric columns automatically
- **Trend & seasonality** — detects direction (increasing/decreasing/steady) and seasonal patterns (weekly/monthly/yearly) via autocorrelation
- **STL decomposition** — separates trend / seasonal / residual (uses statsmodels with `thaieda[timeseries]`, falls back to moving-average)
- **Time gaps & spikes** — detects missing time intervals and period-specific outliers
- **Timeseries charts** — line + trend line, 4-panel decomposition, ACF plot
- **Distribution insights** — skewness, kurtosis, bimodal detection for numeric columns
- **Correlation & duplicates** — highly correlated column pairs, duplicate rows, numbers stored as text

### Visualization

- **Auto chart selection** — picks chart type based on data type automatically
- **Correlation heatmap**, box plot, violin plot, scatter matrix
- **Missing data matrix + heatmap** — visualize missing-value patterns
- **Timeseries plots** — line + trend, STL decomposition, ACF (v0.4)

### Pipeline & Reporting

- **Single-command pipeline** — `thaieda run data.csv` does clean → analyze → report + cleaned file output
- **Auto insight summary** — Thai-language executive summary with actionable recommendations
- **JSON/JSONL input** — auto-reads `.csv`, `.json`, `.jsonl`, `.ndjson`
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

# With target column analysis
thaieda run data.csv -o report.html --target price

# Read JSON (auto-detected)
thaieda run data.json -o report.html

# Analyze only (no cleaning)
thaieda run data.csv -o report.html --no-clean

# Skip timeseries analysis (faster on non-timeseries data)
thaieda run data.csv -o report.html --no-timeseries

# Sample N rows before analysis (for large files)
thaieda run data.csv -o report.html --sample 5000

# Quiet mode (minimal output)
thaieda run data.csv -o report.html --quiet

# Other subcommands
thaieda profile data.csv -o report.html --clean   # Profile + clean
thaieda clean data.csv -o cleaned.csv              # Clean only
```

### Python

```python
import pandas as pd
from thaieda import profile, read_data

# Auto-read any format (CSV/JSON, auto encoding detection)
df = read_data("data.json")

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

### Timeseries

```python
from thaieda import analyze_timeseries, analyze_dataframe_timeseries

# Analyze entire DataFrame (auto-detects datetime column)
results = analyze_dataframe_timeseries(df)
for col, r in results.items():
    print(col, r.frequency_th, r.trend_direction_th, r.has_seasonality, r.seasonal_period)
    print(r.insights)  # Thai-language findings

# Or analyze a single time-indexed Series
r = analyze_timeseries(series, engine="auto")  # "auto" | "statsmodels" | "basic"
```

---

## Architecture

```
thaieda/
  io/          # Auto-read CSV/JSON + encoding detection (v0.3)
  detect/      # Column type detection (Thai text classifier)
  tokenize/    # Tokenizer adapter: pythainlp / nlpo3 / attacut
  text/        # Text metrics: length, frequency, n-grams, TF-IDF
  quality/     # Thai-specific data quality checks ← core differentiator
  anomaly/     # Anomaly detection: statistical + ML + text + unified API
  clean/       # Data cleaning: encoding, zwspace, keyboard layout, normalize
  ner/         # Thai NER: person/place/organization extraction (v0.2)
  analysis/    # Target variable analysis: Pearson/ANOVA/Chi-square (v0.2)
  insight/     # Auto insight summary in Thai (v0.3) + distribution/correlation (v0.4)
  timeseries/  # Timeseries analysis: trend/seasonality/STL/ACF/gaps (v0.4)
  viz/         # Visualization + auto chart + Thai font (+ timeseries plots v0.4)
  report/      # HTML report generation (Jinja2)
  i18n/        # Bilingual labels (Thai/English)
  llm/         # LLM Q&A (v0.5+)
```

**Design principles:**

- **Lazy imports** — core library works without heavy optional dependencies
- **No silent fallbacks** — missing tokenizer fails loudly with a helpful message
- **matplotlib Agg backend** — no GUI dependencies, works headless

---

## Roadmap

| Version | Feature | Status |
|---------|---------|--------|
| **v0.1** | Thai text profiling, data quality, HTML report, CLI | ✅ Done |
| **v0.2** | Thai NER, pythainlp normalize, auto chart, unified anomaly API, target analysis | ✅ Done |
| **v0.3** | Single-command pipeline, JSON input, auto encoding, auto insights, cleaning diff | ✅ Done |
| **v0.4** | Timeseries analysis, distribution & correlation insights | ✅ Done |
| **v0.5** | LLM Q&A (litellm + Ollama local), Thai explanations | 📋 Planned |
| **v0.6** | Interactive dashboard (Streamlit/FastAPI), Thai UI | 📋 Planned |

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