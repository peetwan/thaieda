# ThaiEDA

**AutoEDA for Thai-language data — exploratory data analysis that understands Thai.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests: 424 passed](https://img.shields.io/badge/tests-424%20passed-brightgreen.svg)]()
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)

---

## One-Liner EDA (v1.0)

```python
import thaieda

result = thaieda.run(df)          # detect → clean → quality → insights → viz → report
result.to_html("report.html")     # self-contained HTML report
print(result.insights)            # cross-column insight cards
print(result.llm_response)        # optional LLM analysis (if enabled)
```

**Install from PyPI:** `pip install thaieda`

---

## Quick Start

```bash
pip install thaieda            # basic install
pip install "thaieda[all]"     # everything (Thai tokenizer + NER + ML + timeseries + Excel + stats + LLM deps)
```

```python
import pandas as pd
from thaieda import profile
from thaieda.llm import analyze_with_llm

df = pd.read_csv("data.csv")
report = profile(df, clean=True)
report.to_html("report.html")

# Ask an LLM about the data — privacy-safe by default
answer = analyze_with_llm(df, privacy="insight_only", provider="ollama")
print(answer)
```

---

## Why ThaiEDA?

- **Thai-specific** — catches Buddhist Era dates, Thai numerals, zero-width spaces, mojibake, and Thai month names that generic tools miss.
- **Privacy-first** — LLM analysis with 4 privacy modes; the default sends zero raw data off your machine.
- **Auto insights** — a cross-column insight engine surfaces non-obvious findings, ranked by statistical interestingness (BH-corrected).
- **One-liner API** — `thaieda.run(df)` does detect → clean → quality → insights → viz → report → optional LLM in one call.
- **No lock-in** — generates a self-contained HTML report; works as a library or CLI; all LLM providers are optional and lazy-imported.

---

## Features by Version

| Version | Feature | Description |
|---------|---------|-------------|
| **v1.0** | One-liner API + PyPI publish | `thaieda.run(df)` / `thaieda.EDA(df)`, `pip install thaieda[all]`, EDAResult dataclass |
| **v0.9** | Privacy-preserving LLM analysis | 4 privacy modes + 3 LLM providers (OpenAI / Anthropic / Ollama) |
| **v0.8** | Data cleaning + actionable insights | Thai numeral→numeric, BE→CE, date standardization, correlation/outlier patterns, Excel support |
| **v0.7** | Insight visualization | Auto-generated charts for each cross-column finding (bar, donut, box plot, trend line) |
| **v0.6** | Cross-column insight engine | 6 patterns: outstanding / attribution / comparison / trend / correlation / outlier (BH-corrected) |
| **v0.5** | Multi-file schema discovery | PK/FK matching, ER diagram, relationship validation, orphan detection |
| **v0.4** | Timeseries analysis | Trend/seasonality/STL/ACF/gaps + distribution & correlation insights |
| **v0.3** | Single-command pipeline | JSON input, auto encoding detection, auto insights, cleaning diff |
| **v0.2** | Thai NER + target analysis | pythainlp normalize, auto chart selection, unified anomaly API |
| **v0.1** | Thai text profiling | Column type detection, quality checks, HTML report, CLI |

---

## Privacy Modes (v0.9)

Control exactly what data leaves your machine when calling `analyze_with_llm()`:

| Mode | What Leaves Machine | Privacy Guarantee | Use Case |
|------|---------------------|-------------------|----------|
| `insight_only` (default) | Summary statistics + insight cards only | Raw data never leaves | Regulated / PDPA data, cautious users |
| `anonymized` | Data with PII replaced by tokens (`[PHONE_1]`, `[NAME_1]`) | Names/phones/ID cards masked; `token_map` returned for local reversal | LLM needs to see structure without raw PII |
| `dp_noise` | Statistics with Laplace noise (configurable ε) | DP noise prevents re-identification from small stats | Small datasets where stats alone may leak identity |
| `full` | All raw data sent | None — user accepts tradeoff | Public data, dev/demo workflows |

---

## Examples

### One-Liner EDA (v1.0)

```python
import thaieda
import pandas as pd

df = pd.read_csv("data.csv")

# One call: detect → clean → quality → insights → viz → report
result = thaieda.run(df)

# EDAResult dataclass — access everything
result.to_html("report.html")          # self-contained HTML report
result.to_dict()                       # full result as dict
result.to_json()                        # full result as JSON
print(result.insights)                 # cross-column insight cards
print(result.llm_response)             # LLM analysis (if enabled)
print(result.notes)                     # pipeline notes/warnings

# Alias — same thing
result = thaieda.EDA(df)
```

### Basic EDA

```python
import pandas as pd
from thaieda import profile, read_data

# Auto-reads CSV/JSON/JSONL/Excel with auto encoding detection
df = read_data("data.xlsx")

# Profile + clean + auto insights in one call
report = profile(df, clean=True)
report.to_html("report.html")

# Get Thai-language executive summary
print(report.insights.executive_summary_th)
```

### Insight Discovery

```python
from thaieda import discover_insights
from thaieda.detect import detect_all

result = discover_insights(df, detect_all(df), top_n=8)
for card in result.cards:
    print(f"[{card.pattern}] score={card.score:.2f}  {card.description_th}")
    print(f"  → {card.recommendation_th}")
```

### LLM Analysis — All 4 Privacy Modes

```python
from thaieda.llm import analyze_with_llm

# Mode 1: safest — only stats leave, no raw data (default)
answer = analyze_with_llm(df, privacy="insight_only", provider="ollama")

# Mode 2: anonymized — PII replaced with tokens before sending
answer = analyze_with_llm(df, privacy="anonymized", provider="openai", model="gpt-4o-mini")

# Mode 3: differential privacy — Laplace noise on stats
answer = analyze_with_llm(df, privacy="dp_noise", provider="anthropic", epsilon=0.5)

# Mode 4: full raw data (user accepts risk)
answer = analyze_with_llm(df, privacy="full", provider="ollama", language="en")
```

---

## Architecture

```
src/thaieda/
  __init__.py     # run(df) / EDA(df) one-liner API + EDAResult (v1.0)
  io/             # Auto-read CSV/JSON/JSONL/Excel + encoding detection
  detect/         # Column type detection + Thai month name detection
  tokenize/       # Tokenizer adapter: pythainlp / nlpo3 / attacut
  text/           # Text metrics: length, frequency, n-grams, TF-IDF
  quality/        # Thai quality checks + placeholder/constant detection
  anomaly/        # Anomaly detection: statistical + ML + text + unified API
  clean/          # Data cleaning: encoding, zwspace, numerals, BE→CE, dates, duplicates, missing
  ner/            # Thai NER: person/place/organization extraction
  analysis/       # Target variable analysis: Pearson/ANOVA/Chi-square
  insight/        # Auto insight summary in Thai (interpreter)
  insight_engine/ # Cross-column insight discovery: 6 patterns + BH correction
  timeseries/     # Timeseries analysis: trend/seasonality/STL/ACF/gaps
  schema/         # Multi-file schema discovery: PK/FK detection + relationship matching
  viz/            # Visualization + auto chart + Thai font + insight charts
  report/         # HTML report generation (Jinja2) + DatasetReport
  i18n/           # Bilingual labels (Thai/English)
  llm/            # Privacy-preserving LLM analysis (4 modes, 3 providers) — v0.9
```

---

## Installation

**Install from PyPI:** `pip install thaieda`

```bash
# Core library (no Thai tokenizer)
pip install thaieda

# Everything — single command (v1.0)
pip install "thaieda[all]"

# Individual extras (all lazy-imported)
pip install "thaieda[thai]"          # Thai tokenizer (pythainlp)
pip install "thaieda[ner]"           # pythainlp NER
pip install "thaieda[ml]"            # Isolation Forest / LOF anomaly detection
pip install "thaieda[timeseries]"     # STL decomposition (statsmodels)
pip install "thaieda[excel]"         # Excel support (openpyxl)
pip install "thaieda[stats]"         # p-values (scipy)
pip install "thaieda[detect]"        # auto encoding detection (chardet)

# LLM providers (v0.9 — all optional)
pip install openai                 # OpenAI GPT
pip install anthropic              # Anthropic Claude
pip install ollama                 # Ollama local server (or use built-in HTTP fallback)
```

**Requirements:** Python 3.10+, pandas, numpy, matplotlib, Jinja2

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only one-liner API tests (v1.0)
pytest tests/test_oneliner.py -v

# Run only LLM module tests (v0.9)
pytest tests/test_llm.py -v

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

---

## License

[Apache-2.0](LICENSE) © Peet Wannasarnmetha