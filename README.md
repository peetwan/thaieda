# ThaiEDA

**One-line exploratory data analysis for Thai and mixed-language data.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions/workflows/ci.yml)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

ThaiEDA helps you answer one simple question:

> "Can I trust this dataset, and what should I look at first?"

If you are new to data work, think of ThaiEDA as a **data checkup**. You give it
a pandas DataFrame, file, or folder. It gives you a readable report about what
is inside, what looks suspicious, what may need cleaning, and what patterns are
worth exploring next.

It is built for real Thai data: Buddhist Era years, Thai numerals, hidden
zero-width spaces, old encodings, Thai phone numbers, Thai national IDs, Thai
addresses, mixed Thai/English text, and charts that need Thai font handling.

## Start Here

### Install

```bash
pip install thaieda
```

You can also install the historical all-in-one extra:

```bash
pip install "thaieda[all]"
```

ThaiEDA is an open-source Python package for Python 3.10+.

### The One-Liner

If you already have a pandas DataFrame named `df`, this is the main workflow:

```python
import thaieda

thaieda.run(df, lang="en").to_html("report.html")
```

That one line runs the EDA pipeline and saves a standalone HTML report.

For a notebook or script, you will usually keep the result object:

```python
import pandas as pd
import thaieda

df = pd.read_csv("orders.csv")

result = thaieda.run(df, lang="en")
result.to_html("orders-report.html")
```

What each line means:

| Line | Meaning |
| --- | --- |
| `pd.read_csv("orders.csv")` | Load your data into a pandas DataFrame. |
| `thaieda.run(df, lang="en")` | Analyze the DataFrame and return an `EDAResult`. |
| `result.to_html("orders-report.html")` | Save the report so you can open it in a browser. |

`thaieda.EDA(df)` is the same as `thaieda.run(df)`.

Default report labels are Thai (`lang="th"`). Use `lang="en"` for English.

## What You Get Back

`run()` returns an `EDAResult` object. You can inspect it in Python or export it.

```python
result = thaieda.run(df, lang="en")

print(result.overview)
print(result.quality_issues)
print(result.anomalies)
print(result.insights)

result.to_html("report.html")
result.to_json("report.json")
```

| Property or method | What it is for |
| --- | --- |
| `result.cleaned_df` | The DataFrame used by the report after report-level cleaning. |
| `result.overview` | Basic dataset shape, column summary, and high-level profile data. |
| `result.quality_issues` | Practical data problems ThaiEDA found. |
| `result.anomalies` | Numeric, categorical, text, and Thai-specific anomalies. |
| `result.insights` | Automatically discovered patterns and notable findings. |
| `result.narrative` | Offline executive summary generated without an LLM. |
| `result.llm_response` | Optional LLM analysis when `llm=True`. |
| `result.notes` | Useful warnings or fallback notes. |
| `result.to_html(path)` | Save or return the HTML report. |
| `result.to_json(path)` | Save or return JSON for pipelines. |
| `result.to_dict()` | Get a Python dictionary version of the result. |

## What `run(df)` Actually Does

`run()` is the beginner-friendly path. It is designed for the first pass over a
dataset.

```python
result = thaieda.run(
    df,
    lang="en",
    clean=True,
    make_charts=True,
    insights_engine=True,
    timeseries=True,
)
```

Inside `run(df)`, ThaiEDA:

1. Detects column types, including Thai text, numeric-like text, dates,
   categories, IDs, and mixed-language fields.
2. Applies report-level cleaning when `clean=True`:
   duplicate rows, missing-value flags, and common Thai text cleanup.
3. Checks data quality issues such as missing values, placeholders, constant
   columns, Thai numerals, Buddhist Era years, invisible characters, and
   suspicious text encoding.
4. Looks for anomalies in numeric, categorical, and text columns.
5. Builds text metrics for Thai/mixed-language text and uses available Thai NLP
   tools where relevant.
6. Runs target analysis when you pass `target_column="..."`.
7. Runs time-series analysis when date-like and numeric columns are present.
8. Searches for cross-column insights when `insights_engine=True`.
9. Creates charts when `make_charts=True`.
10. Generates an offline executive narrative and an HTML report.
11. Optionally calls an LLM only when you set `llm=True`.

Important: `run(clean=True)` is meant to make the report more useful. If your
main goal is to create a cleaned dataset for later use, use `thaieda.clean(df)`.

## Full Cleaning Pipeline

Use `thaieda.clean()` when you want a cleaned DataFrame plus an audit report.
It works on a copy, so your original DataFrame is not modified.

```python
import thaieda

cleaned_df, cleaning_report = thaieda.clean(
    df,
    handle_missing="ml",
    remove_duplicates=True,
    fix_dates=True,
    fix_numerals=True,
    fix_encoding=True,
    downcast=True,
)

cleaning_report.to_json("cleaning-report.json")
```

The full cleaner can handle:

- encoding repair and Unicode cleanup
- zero-width spaces and whitespace normalization
- Thai numeral normalization
- Buddhist Era to Common Era date conversion
- date normalization, including Thai month names
- currency normalization
- duplicate row removal
- missing-value handling: `flag`, `median`, `mode`, `drop`, `unknown`, or `ml`
- dtype downcasting for lower memory use

## Why ThaiEDA Exists

Many EDA tools are good at general profiling. They count rows, show missing
values, plot distributions, and summarize columns.

Thai datasets often need more than that.

| Real-world issue | What can go wrong | What ThaiEDA does |
| --- | --- | --- |
| Buddhist Era years like `2567` | Dates may look like impossible future years. | Detects BE patterns and can convert them to CE dates. |
| Thai digits written in Thai numeral characters | Numbers may be treated as text. | Normalizes Thai digits to Arabic digits. |
| Hidden zero-width spaces | Categories that look equal may not match. | Detects and removes invisible characters. |
| Old Thai encodings or mojibake | Text becomes unreadable and NLP breaks. | Repairs common encoding problems when possible. |
| Thai phone numbers and IDs | Generic tools may miss local formats. | Provides Thai-specific detection and validation helpers. |
| Thai addresses | Location text is hard to split manually. | Includes Thai address parsing helpers. |
| Mixed Thai/English text | One-language assumptions create bad summaries. | Uses Thai-aware detection and tokenization adapters. |
| Thai labels in charts | Charts can show unreadable square boxes. | Uses Thai-aware visualization and font handling. |

In short: ThaiEDA treats Thai data problems as normal data problems, not edge
cases you must fix by hand before analysis.

## Why It Is Different

ThaiEDA is not trying to replace pandas, notebooks, BI dashboards, or machine
learning tools. It is the fast first step before those tools.

| Need | Generic profiling tool | ThaiEDA |
| --- | --- | --- |
| Quick DataFrame profile | Yes | Yes |
| Thai-specific checks | Usually manual | Built in |
| One-line HTML report | Yes in some tools | `thaieda.run(df).to_html(...)` |
| Report-level cleaning | Limited | Built into `run(clean=True)` |
| Full Thai-aware cleaning pipeline | Usually separate work | `thaieda.clean(df)` |
| Cross-column insight discovery | Often limited | Built in |
| Text anomaly detection | Usually separate work | Built in |
| Multi-file schema discovery | Usually separate work | `profile_dataset(...)` |
| Dataset drift comparison | Usually separate work | `compare(df1, df2)` |
| Optional privacy-aware LLM analysis | Usually custom prompting | Built in privacy modes |

## Common Recipes

### Analyze a CSV and Save English Outputs

```python
import pandas as pd
import thaieda

df = pd.read_csv("orders.csv")

result = thaieda.run(df, lang="en")
result.to_html("orders-report.html")
result.to_json("orders-report.json")
```

### Read Files With Format Detection

```python
from thaieda import read_data, run

df = read_data("orders.xlsx")
result = run(df, lang="en")
result.to_html("orders-report.html")
```

`read_data()` supports CSV, TSV, JSON, JSONL/NDJSON, Excel, and Parquet.

### Analyze Every File in a Folder

```python
import thaieda

folder = thaieda.run_folder(
    "data",
    recursive=True,
    output_dir="reports",
    lang="en",
)

print(folder.summary())
folder.to_master_html("reports/index.html")
```

`run_folder()` scans CSV, TSV, JSON, JSONL, Excel, and Parquet files.

### Compare Two Datasets

Use this when you have "before vs after", "train vs current", or "last month vs
this month".

```python
from thaieda import compare

diff = compare(train_df, current_df, labels=("train", "current"))

print(diff["schema_diff"])
print(diff["missing_diff"])
print(diff["distribution_drift"])
print(diff["categorical_drift"])
```

### Discover Relationships Across Many Files

Use this when a folder contains related tables and you want to find likely
primary keys, foreign keys, and orphan values.

```python
from thaieda import DatasetReport, profile_dataset

dataset = profile_dataset("data/warehouse", validate_values=True)
DatasetReport(dataset, lang="en").to_html("schema-report.html")

print(dataset.to_mermaid())
```

Folder schema discovery is best for CSV, TSV, JSON, JSONL, and NDJSON files.

### Add LLM Analysis

LLM analysis is optional and off by default.

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

The Python API supports OpenAI, Anthropic, and Ollama providers. The CLI does
not currently expose LLM flags.

Provider setup:

| Provider | Default model | Setup |
| --- | --- | --- |
| `openai` | `gpt-4o-mini` | Install `openai` and set `OPENAI_API_KEY`. |
| `anthropic` | `claude-3-5-sonnet-20241022` | Install `anthropic` and set `ANTHROPIC_API_KEY`. |
| `ollama` | `llama3.1` | Run Ollama locally, or set `OLLAMA_HOST`. |

## LLM Privacy Modes

When `llm=True`, ThaiEDA prepares data according to a privacy mode before
building the prompt.

| Mode | What the LLM sees | Good for |
| --- | --- | --- |
| `insight_only` | Summary information and discovered insights only. | Sensitive datasets and default safe analysis. |
| `synthetic` | Generated rows with similar patterns, not original rows. | Sharing row-like examples with lower risk. |
| `anonymized` | Data with detected PII replaced by tokens. | Preserving structure while masking obvious PII. |
| `dp_noise` | Summary stats with differential-privacy style noise. | Extra protection for sensitive summaries. |
| `full` | Raw data. | Public data or cases where you accept the risk. |

You can also call the LLM module directly:

```python
from thaieda.llm import analyze_with_llm

response = analyze_with_llm(
    df,
    privacy="synthetic",
    provider="openai",
    language="en",
)
```

Synthetic data export is available too:

```python
from thaieda.llm import export_synthetic_data

summary = export_synthetic_data(
    df,
    "synthetic-orders.parquet",
    n_rows=5000,
    include_audit=True,
)
```

## Command Line

ThaiEDA includes a CLI for reports, cleaning, and folder schema profiling.

```bash
thaieda --version

thaieda run data.csv -o report.html --lang en
thaieda profile data.csv -o profile.html --clean --lang en
thaieda clean data.csv -o cleaned.csv
thaieda dataset data/warehouse -o schema-report.html --lang en
```

How to choose a command:

| Command | Use it when |
| --- | --- |
| `thaieda run` | You want the one-line style report workflow from a file. |
| `thaieda profile` | You want a profile report and explicit control over `--clean`. |
| `thaieda clean` | You only want a cleaned file. |
| `thaieda dataset` | You want schema discovery across multiple files. |

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

## Supported Inputs and Outputs

| Feature | Supported formats |
| --- | --- |
| `read_data(path)` | CSV, TSV, JSON, JSONL/NDJSON, Excel, Parquet |
| `run_folder(path)` | CSV, TSV, JSON, JSONL, Excel, Parquet |
| CLI `run`, `profile`, `clean` | CSV, TSV, JSON, JSONL/NDJSON via auto detection, Excel, Parquet |
| `profile_dataset(path)` folder scan | CSV, TSV, JSON, JSONL, NDJSON |
| `export_synthetic_data(...)` | CSV, TSV, XLSX, JSON, Parquet |

## API Map

| API | Purpose |
| --- | --- |
| `thaieda.run(df)` | Main one-line EDA pipeline for a DataFrame. |
| `thaieda.EDA(df)` | Alias for `thaieda.run(df)`. |
| `EDAResult.to_html()` | Export a standalone HTML report. |
| `EDAResult.to_json()` | Export report data as JSON. |
| `EDAResult.to_dict()` | Use report data inside Python. |
| `thaieda.clean(df)` | Full Thai-aware cleaning pipeline. |
| `thaieda.read_data(path)` | Read a file with format and encoding detection. |
| `thaieda.run_folder(path)` | Run EDA over many files and create a master report. |
| `thaieda.compare(df1, df2)` | Compare schema, missingness, stats, and drift. |
| `thaieda.profile_dataset(path)` | Discover multi-file table relationships. |
| `thaieda.DatasetReport(...)` | Render schema discovery as HTML. |
| `thaieda.llm.analyze_with_llm(...)` | Ask an LLM with explicit privacy controls. |
| `thaieda.llm.export_synthetic_data(...)` | Export generated synthetic data with an audit. |

## When To Use ThaiEDA

Use ThaiEDA when:

- you have Thai or mixed Thai/English data
- you want a first report before deeper analysis
- you need to explain data quality to non-technical people
- you need Thai-specific checks before modeling
- you want an HTML report you can share
- you want optional LLM help without sending raw rows by default

Use other tools alongside ThaiEDA when:

- you need a full BI dashboard
- you need production data monitoring
- you are building a custom machine learning model
- you already know the exact analysis you want to write by hand

## Development

```bash
python -m pytest
ruff check src tests
ruff format src tests
```

The package source lives in `src/thaieda/`. Tests live in `tests/`.

## Project Status

- Current version: `2.0.0`
- Python: `3.10+`
- License: Apache-2.0
- Repository: <https://github.com/peetwan/thaieda>
- PyPI: <https://pypi.org/project/thaieda/>

## License

ThaiEDA is released under the [Apache-2.0 License](LICENSE).
