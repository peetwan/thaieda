# ThaiEDA

**AutoEDA for Thai and mixed-language data.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions/workflows/ci.yml)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

ThaiEDA is an open-source Python library for exploratory data analysis on Thai,
English, and mixed-language datasets. It turns a pandas DataFrame, a file, or a
folder of related tables into a clean, executive-ready EDA report with Thai-aware
data cleaning, quality checks, anomaly detection, cross-column insights,
visualizations, and optional privacy-preserving LLM analysis.

```python
import pandas as pd
import thaieda

df = pd.read_csv("orders.csv")

result = thaieda.run(df, lang="en")
result.to_html("orders.thaieda.html")

print(result.overview)
print(result.quality_issues)
print(result.insights)
```

## Table of Contents

- [Why ThaiEDA Exists](#why-thaieda-exists)
- [What It Solves](#what-it-solves)
- [Why It Is Different](#why-it-is-different)
- [Quickstart](#quickstart)
- [Core Capabilities](#core-capabilities)
- [Common Workflows](#common-workflows)
- [LLM Privacy Modes](#llm-privacy-modes)
- [CLI](#cli)
- [Project Design](#project-design)
- [Development](#development)

## Why ThaiEDA Exists

Most EDA tools are excellent at general-purpose profiling, but they often assume
the data is already clean, Western-formatted, and English-first. Thai datasets
usually are not like that.

Real Thai business, public-sector, and research data often contains Buddhist Era
years, Thai numerals, legacy encodings, zero-width spaces, mixed Thai/English
text, Thai phone numbers, national ID formats, address fragments, and chart
rendering issues caused by missing Thai fonts. These issues silently distort
profiles, break parsing, split categories, hide quality problems, and make
reports harder to trust.

ThaiEDA was created to make that first EDA pass reliable for Thai data:

- Detect the shape and language characteristics of a dataset.
- Clean common Thai-specific data issues before they corrupt analysis.
- Surface quality risks in plain language.
- Find non-obvious relationships across columns.
- Produce compact HTML reports that are useful to both analysts and decision
  makers.
- Let users ask an LLM for help without sending raw sensitive rows by default.

## What It Solves

| Problem in real datasets | Why it matters | ThaiEDA response |
| --- | --- | --- |
| Buddhist Era dates such as `2567` | Years can be misread as future CE dates or numeric values | Detects BE patterns and converts to CE where appropriate |
| Thai numerals in numeric columns | Numeric summaries become strings or NaN-heavy columns | Normalizes Thai digits to Arabic digits and coerces numeric columns |
| Zero-width spaces and BOM characters | Duplicate categories and joins fail silently | Removes invisible characters and reports affected rows |
| TIS-620/CP874/mojibake text | Thai text becomes unreadable and downstream NLP breaks | Detects legacy encodings and repairs common mojibake patterns |
| Thai phone, ID, and address formats | PII, identifiers, and locations are missed by generic checks | Detects and normalizes Thai-specific formats |
| Missing Thai fonts in charts | Reports show tofu boxes instead of readable labels | Uses Thai-aware chart rendering with bundled font support |
| Mixed Thai/English columns | Language-specific rules trigger on the wrong columns | Detects Thai, English, mixed text, IDs, categories, dates, and numeric fields |
| Large and wide tables | HTML reports become too large to open comfortably | Caps charts, collapses large tables, samples expensive stages, and downcasts dtypes |

## Why It Is Different

ThaiEDA is not just a prettier profile report. It is a Thai-aware EDA pipeline:

| Capability | General EDA profilers | ThaiEDA |
| --- | --- | --- |
| Descriptive profile report | Yes | Yes |
| Thai-specific data cleaning | Usually manual | Built in |
| Buddhist Era, Thai digits, zero-width spaces | Usually missed | First-class checks |
| Thai chart/font handling | Often manual | Built in |
| Cross-column insight discovery | Limited or separate | Built in with statistical scoring |
| Anomaly detection | Separate workflow | Numeric, text, and Thai-specific checks |
| Dataset quality scoring | Limited | Built in |
| Multi-file schema discovery | Separate workflow | PK/FK discovery and relationship report |
| Privacy-preserving LLM analysis | Usually raw prompt work | Five explicit privacy modes |
| One-line end-to-end workflow | Partial | `thaieda.run(df)` |

Use ThaiEDA when the dataset is Thai, mixed-language, messy, operational,
privacy-sensitive, or intended for a business-facing EDA report. Use it alongside
specialized modeling, monitoring, and BI tools when you need deeper downstream
analysis.

## Quickstart

### Install

```bash
pip install thaieda
```

Install the full optional stack:

```bash
pip install "thaieda[all]"
```

Optional LLM provider clients are lazy-loaded. Install only the provider you use:

```bash
pip install openai
pip install anthropic
pip install ollama
```

### One-Line EDA

```python
import pandas as pd
import thaieda

df = pd.read_csv("data.csv")

result = thaieda.run(
    df,
    lang="en",
    clean=True,
    make_charts=True,
    insights_engine=True,
)

result.to_html("report.html")
result.to_json("report.json")

cleaned_df = result.cleaned_df
```

`thaieda.EDA(df)` is an alias for `thaieda.run(df)`.

### Pipeline

```text
DataFrame / file / folder
  -> read and detect
  -> smart cleaning
  -> quality checks
  -> anomaly detection
  -> cross-column insights
  -> visualizations
  -> executive narrative
  -> HTML / JSON report
  -> optional privacy-preserving LLM analysis
```

## Core Capabilities

### Smart Cleaning

The v2 cleaning pipeline works on a copy of the DataFrame and returns an audit
report describing what changed.

```python
cleaned, cleaning_report = thaieda.clean(
    df,
    handle_missing="ml",
    remove_duplicates=True,
    fix_dates=True,
    downcast=True,
)

print(cleaning_report.to_dict())
```

It covers:

- Encoding repair and Unicode normalization.
- Zero-width space and whitespace cleanup.
- Thai numeral normalization.
- Buddhist Era to Common Era conversion.
- Date and Thai month normalization.
- Currency normalization.
- Duplicate removal.
- Missing-value handling, including ML imputation.
- dtype downcasting for lower memory use.

### Data Quality

ThaiEDA checks for practical issues that often matter before modeling or
reporting:

- Missing values and mostly-missing columns.
- Placeholder values such as `-`, `N/A`, `NULL`, and Thai equivalents.
- Constant and low-information columns.
- Buddhist Era and date-format problems.
- Thai numerals inside numeric-like fields.
- Zero-width characters and encoding artifacts.
- Thai national ID checksum validation.
- Phone number normalization.
- Language-aware column interpretation.

### Insight Discovery

The insight engine searches across column combinations, not only individual
columns. It can surface:

- Strong numeric correlations.
- Outlier-driven findings.
- Segment comparisons.
- Segment contribution and attribution patterns.
- Trend patterns over ordered or date-like fields.
- Statistically ranked insight cards with false-discovery control.

### Executive Reports

Reports are designed to be readable, compact, and shareable:

- Standalone HTML output.
- English or Thai labels via `lang="en"` or `lang="th"`.
- Executive narrative generated offline, without an LLM.
- Chart caps and collapsed tables for large/wide datasets.
- JSON export for pipelines and downstream tooling.

### Multi-File Schema Discovery

ThaiEDA can profile a folder of related files as one dataset, infer key
candidates, discover likely PK/FK relationships, and report orphan key values.

```python
from thaieda import DatasetReport, profile_dataset

dataset = profile_dataset("data/warehouse", validate_values=True)
DatasetReport(dataset, lang="en").to_html("schema-report.html")

print(dataset.to_mermaid())
```

### Dataset Comparison

Compare two DataFrames for schema differences, row-count changes, numeric drift,
missing-value changes, and categorical shifts.

```python
from thaieda import compare

diff = compare(train_df, current_df, labels=("train", "current"))

print(diff["schema_diff"])
print(diff["distribution_drift"])
print(diff["categorical_drift"])
```

## Common Workflows

### Analyze a Folder

```python
import thaieda

folder_result = thaieda.run_folder(
    "data",
    recursive=True,
    output_dir="reports",
    lang="en",
)

print(folder_result.summary())
folder_result.to_master_html("reports/index.html")
```

Supported input formats include CSV, TSV, JSON, JSONL/NDJSON, Excel, and
Parquet.

### Read Data With Auto Detection

```python
from thaieda import read_data

df = read_data("orders.xlsx")
df = read_data("legacy-thai.csv", encoding="auto")
df = read_data("events.parquet", downcast=True)
```

### Privacy-Aware LLM Analysis

```python
result = thaieda.run(
    df,
    lang="en",
    llm=True,
    provider="ollama",
    privacy="insight_only",
)

print(result.llm_response)
```

You can also use the LLM module directly:

```python
from thaieda.llm import analyze_with_llm

response = analyze_with_llm(
    df,
    privacy="synthetic",
    provider="openai",
    language="en",
)
```

### Generate Synthetic Data for Safer Sharing

```python
from thaieda.llm import export_synthetic_data

summary = export_synthetic_data(
    df,
    "synthetic-orders.parquet",
    n_rows=5000,
    include_audit=True,
)

print(summary)
```

## LLM Privacy Modes

LLM analysis is optional and off by default. When enabled, ThaiEDA prepares data
according to an explicit privacy mode before building the prompt.

| Mode | What the LLM sees | Risk level | Typical use |
| --- | --- | --- | --- |
| `insight_only` | Summary stats and insight cards only | Low | Default for sensitive data |
| `synthetic` | Generated rows with similar statistical properties | Low | Need row-like shape without real records |
| `anonymized` | Data with PII replaced by tokens | Medium | Need structure while masking obvious PII |
| `dp_noise` | Summary stats with Laplace noise | Low | Small or sensitive datasets |
| `full` | Raw rows | High | Public data or explicit risk acceptance |

Providers:

| Provider | Default model | Environment |
| --- | --- | --- |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| `ollama` | `llama3.1` | `OLLAMA_HOST` or local default |

## CLI

ThaiEDA includes a command-line interface for reports, cleaning, and multi-file
dataset profiling.

```bash
thaieda --version

thaieda profile data.csv -o report.html --lang en
thaieda run data.csv -o report.html --cleaned-output cleaned.csv --lang en
thaieda clean data.csv -o cleaned.csv
thaieda dataset data/warehouse -o schema-report.html --lang en
```

Useful flags:

```bash
--format auto|csv|tsv|json|jsonl|excel|parquet
--encoding auto
--target COLUMN
--no-charts
--no-insights
--no-timeseries
--sample N
--quiet
--json output.json
```

## API Overview

| API | Purpose |
| --- | --- |
| `thaieda.run(df)` / `thaieda.EDA(df)` | Full one-line EDA pipeline |
| `EDAResult.to_html()` / `.to_json()` / `.to_dict()` | Export analysis results |
| `thaieda.run_folder(path)` | Analyze every supported file in a folder |
| `thaieda.clean(df)` | DataFrame-level smart cleaning pipeline |
| `thaieda.read_data(path)` | Auto-detect format and encoding |
| `thaieda.compare(df1, df2)` | Compare schema, stats, and drift |
| `thaieda.profile_dataset(path)` | Discover multi-file schema relationships |
| `thaieda.DatasetReport(dataset)` | Render multi-file schema report |
| `thaieda.llm.analyze_with_llm()` | LLM analysis with privacy controls |
| `thaieda.llm.export_synthetic_data()` | Export generated synthetic data plus audit |

## Project Design

ThaiEDA follows a few practical engineering rules:

- Prefer vectorized pandas operations over row-by-row loops.
- Keep optional heavy dependencies lazy-loaded.
- Fail loudly with helpful messages instead of silently degrading behavior.
- Use a non-GUI matplotlib backend for headless environments.
- Keep reports compact enough to share and open.
- Treat Thai data problems as first-class EDA concerns, not edge cases.

## Development

```bash
python -m pytest
ruff check src tests
ruff format src tests
```

The source package lives under `src/thaieda/`, with tests under `tests/`.

## Project Status

- Current version: `2.0.0`
- Python: `3.10+`
- License: Apache-2.0
- Repository: <https://github.com/peetwan/thaieda>
- PyPI: <https://pypi.org/project/thaieda/>

## License

ThaiEDA is released under the [Apache-2.0 License](LICENSE).
