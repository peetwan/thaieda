# ThaiEDA

> **One-line Exploratory Data Analysis and Smart Data Cleaning for Thai and Mixed-Language Datasets.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/peetwan/thaieda/actions/workflows/ci.yml/badge.svg)](https://github.com/peetwan/thaieda/actions/workflows/ci.yml)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

ThaiEDA answers one simple question: **"Can I trust this dataset, and what should I explore first?"**

While generic profiling tools count missing values and draw standard charts, they often fail when processing Thai text and mixed-language data. ThaiEDA treats Thai-specific data complexities—such as Buddhist Era (BE) dates, Thai numerals, invisible zero-width spaces, encoding errors (mojibake), local phone formats, and Thai fonts in charts—as normal data problems, eliminating the need for tedious manual preprocessing.

---

## 🚀 Key Features

*   **Smart Column & Type Detection**: Identifies Thai/English text, numbers masquerading as text, Buddhist Era years, Thai phone numbers, Thai national IDs, and mixed-language columns.
*   **One-Line AutoEDA (`run`)**: A complete pipeline that auto-detects, cleans, checks quality, finds anomalies, performs time-series/target analysis, runs a cross-column insight engine, generates charts, builds offline executive narratives, and generates HTML reports.
*   **Thai-Aware Cleaning Pipeline (`clean`)**: Easily cleans and normalizes Unicode formats, fixes zero-width/invisible spaces, normalizes currency/numbers, converts Buddhist Era to Common Era (CE), corrects keyboard layout mistakes (e.g., `l;ylfu` ➔ `สวัสดี`), protects product IDs/codes using heuristic likeness scores, and performs machine-learning (ML) missing value imputation.
*   **Cross-Column Insight Engine**: Automatically discovers complex relationships, outlier influences, trend evidence, Simpson's paradox, and target leakage with statistical scoring.
*   **Multi-File Schema Discovery**: Scans folders of files (`profile_dataset`) to discover primary/foreign key candidates and orphans, then renders schema relationships as interactive reports and Mermaid diagrams.
*   **Privacy-Preserving LLM Integration**: Generates secure LLM summaries with 5 privacy modes (`insight_only`, `synthetic`, `anonymized`, `dp_noise`, and `full`) to safely analyze data without risking raw PII exposure.

---

## 📦 Requirements & Installation

ThaiEDA requires **Python 3.10+**. 

Install the lightweight core package (contains pandas, numpy, matplotlib, and Jinja2):
```bash
pip install thaieda
```

For advanced features, install extras for Thai NLP (tokenizers), visual enhancements, and Excel/Parquet I/O:
```bash
pip install "thaieda[thai,viz,excel,parquet]"
```

Or install all optional backends and dependencies at once:
```bash
pip install "thaieda[all]"
```

---

## ⚡ Quickstart

Here is a fully reproducible example. Copy and run this script to see ThaiEDA in action:

```python
import pandas as pd
import thaieda

# 1. Create a messy DataFrame simulating real Thai data issues
data = {
    "name": ["สมชาย\u200bรักไทย", "สมหญิง   ใจดี", "นายดำ ๐๑"],  # Has zero-width space and multiple spaces
    "birth_year": [2530, 2532, 2528],                          # Buddhist Era (BE) years
    "sales": ["฿1,200", "฿3,500", "฿10,000"],                  # Currency formatting as text
    "phone": ["081-234-5678", "+66898765432", "๐๒-๓๔๕-๖๗๘๙"]     # Phone formats
}
df = pd.DataFrame(data)

# 2. Run the full EDA pipeline with cleaning enabled
# By default, lang="th" produces Thai reports. Set lang="en" for English.
result = thaieda.run(df, clean=True, lang="en")

# 3. Save the interactive report
result.to_html("quickstart-report.html")

# 4. Extract the clean DataFrame
cleaned_df = result.cleaned_df
print(cleaned_df)
```

---

## 💡 Core Recipes

### 1. One-Line EDA & Result Inspection
The `run()` function (also aliased as `EDA()`) performs the analysis and returns an `EDAResult` object:

```python
result = thaieda.run(df, lang="en")

# Access details in Python
print(result.overview)          # Dataset metadata
print(result.quality_issues)    # Quality flags (e.g., constant columns, BE years)
print(result.anomalies)         # Outliers and text anomalies
print(result.insights)          # Discovered statistical insights
print(result.narrative)         # Offline, rule-based executive summary
print(result.llm_response)      # Optional LLM analysis response (if llm=True)

# Export options
result.to_html("report.html")   # Save HTML report
result.to_json("report.json")   # Export structured report metadata
result.to_dict()                # Convert result to Python dict
```

### 2. Standalone Cleaning Pipeline
Use `clean()` to sanitize a DataFrame on a copy, returning both the clean DataFrame and a structured cleaning report.

```python
cleaned_df, report = thaieda.clean(
    df,
    handle_missing="ml",        # Imputation strategy: flag, median, mode, drop, unknown, or ml
    remove_duplicates=True,
    fix_dates=True,             # Converts BE to CE, normalizes formats
    fix_numerals=True,          # Normalizes Thai digits to Arabic
    fix_encoding=True,          # Repairs mojibake and spacing
    downcast=True               # Optimizes data types for memory efficiency
)

# Export the audit trail of modifications
report.to_json("cleaning-audit.json")
```

#### 🧼 Smart Cleaning & ID Protection
When normalizing text (such as using `fix_repeated_chars` to collapse excessive characters), standard rule-based approaches might unintentionally mangle product codes, serial numbers, or model names. ThaiEDA solves this with an intelligent heuristic protection system.

*   **`skip_id_like` Parameter** (default `True`): Under `fix_repeated_chars`, setting this to `True` protects strings and sub-tokens that look like identifier codes from being modified.
*   **Token-Level Protection**: Instead of analyzing the entire text block globally, ThaiEDA splits text into individual tokens and applies the safeguard locally. This ensures a product ID embedded within a chat or review text remains completely untouched, while surrounding natural text is cleaned.
*   **`_id_likeness_score` Heuristic**: Determines if a token is an ID using a 7-criteria scoring algorithm (0.0 to 1.0):
    1.  `digit_ratio`: The ratio of numeric digits to length (IDs generally have $\ge 0.3$).
    2.  `upper_ratio`: The ratio of uppercase characters to all alphabet characters (IDs generally have $> 0.5$).
    3.  `separator`: Presence of symbols like hyphens, underscores, or dots in the middle.
    4.  `length`: Usually short strings ($\le 20$ characters).
    5.  `no_spaces`: No spaces within the token.
    6.  `alnum_mix`: Mixtures of both letters and digits.
    7.  `entropy`: Lower character entropy due to repeated structures.

**Examples in Action:**
*   `'55555'` ➔ `'555'` (digits in laughter are normalized)
*   `'มากกกกกก'` ➔ `'มากกก'` (exaggerated Thai text is collapsed)
*   `'SKU-AAA111'` ➔ `'SKU-AAA111'` (safely kept intact as an ID)

### 3. Comparing Two Datasets (Drift & Schema)
Use `compare()` to detect schema changes and statistical distribution drift between two datasets (e.g., training vs. production data).

```python
from thaieda import compare

diff = compare(train_df, current_df, labels=("train", "current"))

print(diff["schema_diff"])          # Mismatched column names or types
print(diff["missing_diff"])         # Changes in missing value rates
print(diff["distribution_drift"])    # Numerical distribution shift (using statistical tests)
print(diff["categorical_drift"])     # Categorical frequency drift
```

### 4. Folder Schema Discovery (Multi-File)
If your folder contains multiple tables with relationships, `profile_dataset()` identifies key connections and outputs Mermaid schemas.

```python
from thaieda import DatasetReport, profile_dataset

# Scan directory for CSV/JSON/TSV/Excel/Parquet tables
dataset = profile_dataset("data/warehouse", validate_values=True)

# Export interactive multi-table relationship report
DatasetReport(dataset, lang="en").to_html("schema-report.html")

# Output Mermaid diagram representing PK/FK relationships
print(dataset.to_mermaid())
```

### 5. Multi-File Batch EDA
Analyze every file in a directory and compile them into a unified master report with a navigation sidebar.

```python
import thaieda

# Scans supported file formats in the folder
folder = thaieda.run_folder(
    "data/",
    recursive=True,
    output_dir="reports",
    lang="en"
)

# Generate master index report containing all summaries
folder.to_master_html("reports/index.html")
print(folder.summary())
```

---

## 🔒 Privacy-Preserving LLM Analysis

ThaiEDA offers privacy-first, local-first LLM analysis. When `llm=True`, the data is processed according to a specified `privacy` mode to ensure sensitive information never leaves your environment.

```python
result = thaieda.run(
    df,
    llm=True,
    provider="openai",        # openai, anthropic, or ollama
    privacy="insight_only",    # insight_only, synthetic, anonymized, dp_noise, or full
    lang="en"
)
print(result.llm_response)
```

### Privacy Modes Overview

| Privacy Mode | What the LLM Sees | Best Used For |
| :--- | :--- | :--- |
| `insight_only` | Summary stats and statistical insights only. **No raw rows.** | Highly sensitive datasets; default safe setting. |
| `synthetic` | Generative synthetic rows with identical patterns. | Sharing realistic dataset structure without raw records. |
| `anonymized` | Data with PII (phone, ID, name) replaced by placeholders. | Masking obvious personal identifier columns. |
| `dp_noise` | Aggregated summaries with Differential Privacy noise. | Protecting aggregated statistical distributions. |
| `full` | Original raw dataset. | Non-sensitive, public datasets. |

*Note: The LLM module can also be invoked independently using `thaieda.llm.analyze_with_llm(...)`.*

---

## 💻 Command Line Interface (CLI)

ThaiEDA comes with a powerful command line tool.

```bash
# Get version info
thaieda --version

# Run AutoEDA report on a file
thaieda run data.csv -o report.html --lang en

# Generate report with explicit cleaning
thaieda profile data.xlsx -o profile.html --clean --lang en

# Clean data and output clean file
thaieda clean inputs.csv -o cleaned.csv

# Multi-file schema profiling
thaieda dataset data/warehouse/ -o schema-report.html --lang en
```

### Command Reference

| Command | Usage |
| :--- | :--- |
| `thaieda run` | Generates a quick HTML report from a file (includes default cleaning). |
| `thaieda profile` | Generates a full profile report with granular `--clean` options. |
| `thaieda clean` | Performs data cleaning and outputs a sanitized data file. |
| `thaieda dataset` | Discovers primary/foreign keys and relationships across folders. |

---

## ⚖️ How ThaiEDA Compares

ThaiEDA does not replace generic profiling packages; it complements them by handling the unique nuances of Thai data.

| Capability | Generic Profiling Tools (e.g., YData-Profiling) | ThaiEDA |
| :--- | :--- | :--- |
| **Basic Data Profiling** | ✅ Excellent, detailed standard statistics | ✅ Lightweight statistics and distributions |
| **Thai-Specific Quality Checks** | ❌ Manual (treats BE years as outliers) | ✅ Out-of-the-box (detects BE, Thai numerals) |
| **Report-Level Cleaning** | ❌ None or minimal | ✅ Auto-cleaning embedded in `run(clean=True)` |
| **Interactive Viz Thai Font Support**| ❌ Shows unreadable squares (`[]`) | ✅ Pre-configured Thai font fallback |
| **Cross-Column Insights** | ⚠️ Basic correlations | ✅ Scoring engine (leakage, Simpson's paradox) |
| **Multi-File Schema Discovery** | ❌ Single-file focus | ✅ Automatic PK/FK detection & Mermaid schemas |
| **Dataset Comparison & Drift** | ⚠️ Basic comparison | ✅ Detailed statistical schema & drift comparison |
| **Privacy-First LLM Summaries** | ❌ None | ✅ 5 levels of privacy-preserving modes |

---

## 💬 FAQ

### Q: Why do the charts in my report show empty boxes instead of Thai characters?
**A:** This happens if your OS lacks standard Thai fonts or if matplotlib cannot locate them. ThaiEDA configures fallback fonts automatically. If the issue persists, install the visualization extra (`pip install "thaieda[viz]"`) which packages appropriate open-source Thai fonts.

### Q: Does ThaiEDA send my data to external servers for LLM analysis?
**A:** No, unless you explicitly enable `llm=True`. By default, all operations run 100% locally. When LLM is enabled, data is aggressively aggregated or anonymized (based on your chosen `privacy` mode) before being sent to the provider. You can also run Ollama locally to keep 100% of LLM processing local.

---

## 📊 Quality & Regression Tests

Regression coverage lives in the pytest suite (`tests/`, **933+ tests**), including golden dirty datasets under `tests/fixtures/dirty_datasets/` that verify cleaning and quality scoring end-to-end.

```bash
python -m pytest tests/ -q
```

---

## 🛠️ Development & Contributing

To run the test suite and verify formatting, use the following commands:

```bash
# Run tests
python -m pytest tests/

# Code quality checks
ruff check src/ tests/
ruff format src/ tests/
```

For guidelines on coding style, checkout [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## 📄 License

ThaiEDA is released under the [Apache-2.0 License](LICENSE).
