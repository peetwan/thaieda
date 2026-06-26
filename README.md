# ThaiEDA

**Exploratory data analysis that actually understands Thai.**

[![PyPI](https://img.shields.io/pypi/v/thaieda.svg)](https://pypi.org/project/thaieda/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests: 691 passed](https://img.shields.io/badge/tests-691%20passed-brightgreen.svg)]()
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Language aware](https://img.shields.io/badge/language-Thai%20%2B%20English%20aware-blueviolet.svg)]()

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

That's it. `pip install thaieda` ติดตั้งทุกอย่างเลย — Thai tokenizer, NER, ML, Excel, stats, encoding detection, interactive charts ไม่ต้องใส่ extras

---

## Why ThaiEDA?

You already have ydata-profiling and sweetviz. Here's why you'd reach for ThaiEDA instead:

**1. Thai text doesn't break.** Generic tools render Thai as tofu boxes (□□□) in every chart. They miss Buddhist Era dates (พ.ศ. 2567), Thai numerals (๑๒๓), zero-width spaces, and mojibake from TIS-620 encoding. ThaiEDA detects and fixes all of these automatically — no font config, no manual cleanup.

**2. Insights, not just stats.** ydata-profiling gives you distributions and correlation matrices. ThaiEDA finds *actionable* cross-column patterns — "column A strongly predicts column B", "this group is 3× higher than average" — ranked by statistical interestingness with Benjamini-Hochberg correction. Plus anomaly detection, quality scoring, and data type classification.

**3. One call, everything done.** `run(df)` chains the full pipeline: type detection → smart cleaning → quality checks → anomaly detection → insight discovery → visualization → HTML report. With ydata you'd still need a separate anomaly detector, a Thai font config, a cleaner, and manual interpretation.

**4. Privacy-first LLM.** Ask an LLM about your data without sending raw rows to a cloud API. 4 privacy modes — the default sends zero raw data. PDPA-ready.

**5. Smaller reports on big data.** ydata-profiling produces a 71 MB HTML on a 171-column dataset. ThaiEDA produces 0.48 MB — 148× smaller — because it caps charts, collapses tables, and samples intelligently on wide/tall data.

---

## How It Works

```
DataFrame → thaieda.run(df) → EDAResult

  Step 0  pre-analyze    data type + language detection
  Step 1  detect         column types + Thai months + addresses
  Step 2  clean          smart cleaning (auto-decide what to fix)
  Step 3  quality        language-aware checks + 0–100 score
  Step 4  anomaly        IQR + ML + text anomaly detection
  Step 5  insights       6 cross-column patterns (BH-corrected)
  Step 6  viz            static (matplotlib) + interactive (Plotly)
  Step 7  report         executive HTML narrative

  + optional: LLM analysis (4 privacy modes)
  + optional: run_folder("data/") → multi-file master HTML
  + optional: compare(df1, df2) → drift detection
```

```python
result = thaieda.run(df)

result.to_html()         # → report.html (self-contained)
result.to_dict()          # → Python dict
result.to_json()          # → JSON string
result.insights           # → insight cards
result.cleaned_df         # → cleaned DataFrame
result.quality_issues     # → list of issues
result.quality_score      # → 0–100 score with grade
result.anomalies          # → anomaly findings
result.llm_response       # → LLM analysis (if enabled)
result                    # → Jupyter rich display
```

---

## Benchmarks — ThaiEDA vs ydata-profiling vs sweetviz

We ran all three on **6 representative datasets** (small/large/wide, Thai + non-Thai):

### Capability comparison

| Feature | ydata-profiling | sweetviz | **ThaiEDA** |
|---------|:-------------:|:--------:|:-----------:|
| Standalone HTML report | ✅ | ✅ | ✅ |
| Cross-column insights | ❌ | ❌ | ✅ 6 patterns + BH correction |
| Anomaly detection | ❌ | ❌ | ✅ IQR + ML + text |
| Quality score (0–100) | ❌ | ❌ | ✅ |
| Language detection | ❌ | ❌ | ✅ Thai/English/mixed |
| Thai font in charts | ❌ tofu | ❌ tofu | ✅ Sarabun auto |
| Buddhist Era (พ.ศ.) | ❌ | ❌ | ✅ → CE |
| Thai numerals (๑๒๓) | ❌ | ❌ | ✅ → 123 |
| Zero-width space fix | ❌ | ❌ | ✅ |
| Mojibake repair | ❌ | ❌ | ✅ |
| Smart cleaning | ❌ | ❌ | ✅ auto-decide |
| Thai NER | ❌ | ❌ | ✅ |
| Privacy LLM modes | ❌ | ❌ | ✅ 4 modes (PDPA) |
| Folder mode | ❌ | ❌ | ✅ `run_folder()` |

### Speed & report size

| Dataset | Rows | Cols | ydata | ydata size | sweetviz | sv size | Evidently | ev size | **ThaiEDA** | **EDA size** |
|---------|-----:|-----:|-------:|-----------:|---------:|--------:|----------:|--------:|------------:|-------------:|
| titanic | 891 | 12 | 5.3s | 1.95 MB | 3.3s | 0.92 MB | — | — | 8.2s | **0.82 MB** |
| superstore | 10,800 | 21 | 9.3s | 5.16 MB | 5.4s | 1.49 MB | — | — | 26.0s | **1.50 MB** |
| adult | 32,561 | 15 | 5.4s | 1.65 MB | 8.0s | 1.26 MB | — | — | 17.2s | **1.05 MB** |
| dirty-thai-retail | 500 | 8 | 3.1s | 0.90 MB | 2.1s | 0.68 MB | — | — | 2.1s | **0.53 MB** |
| wisesight | 26,737 | 2 | 2.6s | 0.68 MB | 0.8s | 0.50 MB | — | — | 18.8s | **0.42 MB** |
| aps-failure | 16,000 | 171 | 99.8s | **71.2 MB** | 15.8s | 8.2 MB | — | — | 93.0s | **0.48 MB** |
| synthetic | 2,000 | 12 | 45s | 7.2 MB | 3s | 0.9 MB | 1s | 3.7 MB | 16s | **1.5 MB** |

### Quality benchmark — 4 tools on synthetic dataset (10 known issues)

We injected 10 known defects into a 2,000-row synthetic dataset and measured detection. All tools processed identically: HTML output stripped to plain text, same keyword detection applied uniformly.

**Table A — General EDA quality** (6 issues all tools can detect)

| Metric | ydata (default) | sweetviz | Evidently | **ThaiEDA** |
|--------|:---:|:---:|:---:|:---:|
| **GTR** — Ground-Truth Recall | 100% | 83% | 100% | **100%** |
| **ITB** — Issue Type Breadth (11) | 73% | 64% | 91% | **91%** |
| **RC** — Report Completeness (10) | 70% | 50% | 70% | **100%** |
| Time | 45s | 3s | 1s | 16s |
| HTML size | 7.2 MB | 0.9 MB | 3.7 MB | **1.5 MB** |

On general EDA, ThaiEDA and Evidently both achieve 100% recall and 91% breadth. ThaiEDA wins on report completeness (100% vs 70%) while producing a 2× smaller report than Evidently and 5× smaller than ydata.

**Table B — Thai-specific detection** (4 Thai issues — competitors don't claim Thai support)

| Thai issue | ydata | sweetviz | Evidently | **ThaiEDA** |
|-----------|:------:|:--------:|:---------:|:-----------:|
| Buddhist Era dates (พ.ศ.) | ✅\* | 0% | ✅\* | **✅** |
| Thai numerals (๑๒๓) | 0% | 0% | ✅\* | **✅** |
| Zero-width spaces | 0% | 0% | 0% | **❌** |
| Mojibake (TIS-620) | 0% | 0% | ✅\* | **✅** |
| **Thai GTR** | **25%** | **0%** | **100%\*** | **75%** |

\* Evidently scores high on Thai keywords via generic matches ("encoding" appears in its CSS/JS framework), not Thai-specific recognition. ydata detected BE dates via generic encoding keywords. These are keyword-matching artifacts, not genuine Thai detection — competitors do not claim Thai language support.

ThaiEDA's 75% (3/4) reflects purpose-built Thai detection. The one miss (zero-width space in `category_text`) is a known gap. Competitors score 0% on genuine Thai detection by design.

---

## What ThaiEDA Catches

### Thai-specific problems

| Problem | Example | What ThaiEDA does |
|---------|---------|-------------------|
| Buddhist Era dates | `15/03/2567` | Detects พ.ศ. → converts to CE |
| Thai numerals | `๑๒๓` in numeric column | Converts to `123` |
| Zero-width spaces | `สม\u200bชาย` | Strips invisible chars + reports |
| Thai vowel/tone marks | `อร่อยค่ะ` | Counts U+0E30–U+0E4D for detection |
| Mixed Thai/English cells | `อร่อยมาก 5/5 stars` | Detects as mixed, not English/numeric |
| Thai month names | `มกราคม` | Parses to ISO date |
| Mojibake encoding | `Ã ¬Â¸Â¡Â¹` | Auto-detects TIS-620 → UTF-8 |
| National ID cards | `1-1234-56789-01-2` | Checksum validation |
| Thai addresses | `123 ม.4 ต.บางบัว อ.บางบัว จ.กรุงเทพฯ` | Parses to structured fields |
| Phone numbers | `081-234-5678` | Detects + normalizes |
| Thai holidays | Spike on Dec 5 | Attributes to Father's Day |

### Data quality & intelligence

| Problem | What ThaiEDA does |
|---------|-------------------|
| Placeholder values (`-`, `N/A`, `ไม่มี`) | Flags as missing |
| Constant columns | Flags as useless |
| High-NA columns (>80%) | Flags `mostly_missing`, preserves NaN |
| Missing % per column | Severity threshold (warning >5%, info 1–5%) |
| Smart data type | Pre-classifies transaction/registry/survey/timeseries/mixed |
| Language-aware checks | English-only skips Thai พ.ศ./เลขไทย warnings |
| ID/FK semantics | `order_id`, `store_id` excluded from category anomaly |
| Numeric string preservation | `1.00005` left alone — not "spam" |
| Keyboard layout guard | `Floyd` in English column not converted to Thai |
| Index artifacts | `Unnamed: 0` ignored + flagged |
| CSV delimiter mismatch | `;`-delimited file warns to re-read |

### Opt-in operations (not in default pipeline)

| Operation | Function | Effect |
|-----------|----------|--------|
| Abbreviation expansion | `expand_abbreviations()` | กทม. → กรุงเทพมหานคร |
| Spell correction | `spell_correct()` | ขอบคุน → ขอบคุณ |
| NFKC normalization | `normalize_nfkc()` | Ａ→A, ９→9 |
| Fast tokenizer | `engine="auto-fast"` | nlpo3 (Rust, 3–4× faster) |
| Quality tokenizer | `engine="auto-quality"` | AttaCut (neural, better for OOV) |
| Keyboard layout anomaly | report-only | Detects suspicious Latin/Thai mixing |
| Grapheme validation | report-only | Detects abnormal stacked tone marks |

---

## v1.8 — Statistical Accuracy Improvements

Five new techniques that improve detection accuracy across different data patterns:

### 1. Spearman rank correlation (non-linear relationships)

Previously only Pearson (linear) correlation was computed. Now also computes Spearman ρ to catch **monotonic non-linear** relationships that Pearson misses (e.g., y = x⁵). The method with the highest |coefficient| is reported automatically.

```python
from thaieda.insight_engine import discover_insights
# Now detects both linear AND non-linear strong correlations
```

### 2. Cramér's V effect size (categorical association)

Chi-square test only tells you *if* two categorical variables are associated (p-value). Cramér's V tells you *how strongly* — a 0–1 effect size with bias correction:

| V range | Strength |
|---------|----------|
| < 0.3 | เบาบาง (weak) |
| 0.3–0.5 | ปานกลาง (moderate) |
| > 0.5 | ชัดเจน (strong) |

```python
from thaieda.analysis import analyze_target
results = analyze_target(df, "category_column")
# Each chi_square result now includes effect_size (Cramér's V)
```

### 3. Generalized ESD test (multiple outlier detection)

The existing z-score/IQR/MAD methods detect outliers one at a time, suffering from **masking** (outliers hide each other). The Generalized Extreme Studentized Deviate (Rosner 1983) test detects multiple outliers simultaneously with controlled Type I error:

- Automatically selected when data is approximately normal (skew < 0.5, n ≥ 25)
- Falls back to z-score/IQR/MAD for skewed or small datasets
- Detects up to 10 outliers in one pass

### 4. Missing data mechanism detection (MCAR / MAR / MNAR)

Beyond counting missing values, ThaiEDA now classifies the **missing data mechanism**:

| Mechanism | Meaning | Implication |
|-----------|---------|-------------|
| MCAR | Missing Completely at Random | Safe to drop or impute simply |
| MAR_likely | Missing at Random | Imputation should use observed predictors |
| MNAR_likely | Missing Not at Random | Missing depends on unobserved values — needs domain model |

```python
from thaieda.quality import detect_missing_mechanism
result = detect_missing_mechanism(df)
print(result.mechanism)  # "MCAR", "MAR_likely", or "MNAR_likely"
```

### 5. Distribution fitting + Kolmogorov-Smirnov test

Automatically fits 4 distributions (normal, lognormal, exponential, uniform) to each numeric column and reports the best fit via KS goodness-of-fit test:

```python
from thaieda.quality import fit_distributions
result = fit_distributions(df["column"], "column")
# result.best_fit → "normal", result.p_value, result.parameters
```

---

## Scale & Performance

Tested across 19 public datasets — 500 to 541K rows, 2 to 171 columns:

- **Insight capping** — surfaces the 30 most important findings. Executive summary shows the true count ("679 found, showing top 30").
- **HTML bloat control** — 40 charts max, 1.6 MB max. Quality/anomaly tables collapse after 50 rows. Wide tables switch to summary view past 60 columns.
- **Wide-table fast path** — insight engine samples when columns exceed 100. Heatmaps and scatter matrices skip on very wide data.
- **Tall-table fast path** — anomaly/quality/outlier checks sample 50K rows when data exceeds 100K. Timeseries decomposition skips past 200K rows.
- **High-NA handling** — columns >80% missing flagged as `mostly_missing`. >40% gets a warning. <40% unchanged.
- **Smarter type detection** — Thai low-cardinality text → categorical, not free text. `review`/`feedback` stay text.
- **Cleaning safeguards** — numeric strings untouched. Keyboard conversion only when Thai chars present.

---

## Examples

### One-Line EDA

```python
import thaieda
import pandas as pd

df = pd.read_csv("data.csv")
result = thaieda.run(df)

result.to_html("report.html")
print(result.quality_issues)
print(result.insights)

# In Jupyter: just display the result
result  # renders HTML inline
```

### Folder Mode — Analyze Every File at Once

```python
import thaieda

results = thaieda.run_folder("data/")

print(results.summary())
# ThaiEDA FolderResult — data/
#   Files: 5 (✅ 5 / ❌ 0)
#   ✅ customers.csv — 10,000 rows × 8 cols, 15 insights
#   ...

results.to_html("reports/")
results.to_master_html("master-report.html")  # single HTML with sidebar
```

Supported formats: CSV, Excel (.xlsx/.xls), JSON, JSONL, TSV. `recursive=True` for subfolders. Error isolation — one broken file doesn't crash the rest.

### LLM Analysis (Privacy-Safe)

```python
result = thaieda.run(df, llm=True, privacy="insight_only", provider="ollama")
print(result.llm_response)
# Default: zero raw data leaves your machine
```

| Mode | What Leaves | When to Use |
|------|------------|-------------|
| `insight_only` (default) | Stats + insights only | Government, medical, PDPA |
| `anonymized` | PII → tokens | Need structure without raw data |
| `dp_noise` | Stats + Laplace noise | Small datasets where stats leak |
| `full` | Everything | Public data, demos |

### Compare Two Datasets

```python
from thaieda.compare import compare_datasets

diff = compare_datasets(df_train, df_test, labels=("train", "test"))
print(diff["schema_diff"])       # columns added/removed
print(diff["drift"]["numeric"])  # KS statistic per column
```

### Thai ID Card Validation

```python
from thaieda.quality import validate_thai_id, validate_thai_id_column

validate_thai_id("1-1234-56789-01-2")           # → True/False
result = validate_thai_id_column(df["id_card"]) # entire column
```

### Thai Address Parsing

```python
from thaieda.detect import parse_thai_address

addr = parse_thai_address("123 หมู่ 4 ต.บางบัว อ.บางบัว จ.กรุงเทพฯ 10230")
# {'house_number': '123', 'moo': '4', 'subdistrict': 'บางบัว',
#  'district': 'บางบัว', 'province': 'กรุงเทพฯ', 'postal_code': '10230'}
```

### Language Detection

```python
from thaieda.detect import _detect_language

df = pd.DataFrame({
    "product": ["กาแฟ", "ชาไทย", "ขนม"],
    "review": ["อร่อยมาก 5/5 stars", "ดีครับ", "ไม่ดี"],
    "sku": ["SKU001", "SKU002", "SKU003"],
})

info = _detect_language(df)
print(info["language"], info["confidence"])
# thai/mixed/english/numeric + per-column language map
```

Features: Unicode Thai block analysis (U+0E00–U+0E7F), zero-width-space aware, mixed-cell detection, common Thai word hints, per-column `column_details` + dataset-level `confidence` (0.0–1.0), sample-based scan for large DataFrames.

### Smart Pre-Analysis

```python
from thaieda.report import _detect_data_type

pre = _detect_data_type(df)
print(pre["label"], pre["language"]["language"])
# Detects: transaction, registry, survey, timeseries, or mixed
```

### Data Quality Score

```python
from thaieda.quality import compute_quality_score

score = compute_quality_score(quality_issues, n_columns=10, n_rows=1000)
print(f"Score: {score.score}/100 ({score.grade})")  # Score: 85/100 (B)
```

### Smart Cleaning

```python
from thaieda.clean._smart import plan_cleaning

plan = plan_cleaning(df)
print(plan.actions)   # ['zwspace', 'numerals', 'duplicates']
print(plan.skipped)   # ['encoding', 'whitespace']
```

---

## Visualization

Both static and interactive charts, all with Thai font support:

- **Static** (matplotlib): correlation heatmap, distribution, box/violin, missing matrix, scatter matrix, wordcloud, timeseries, pair plot, KDE, QQ plot, sunburst
- **Interactive** (Plotly): hover tooltips, zoom, pan — Thai font (Sarabun) via Google Fonts
- **Color palette**: Okabe-Ito colorblind-safe (7 colors)

```python
from thaieda.viz._interactive import create_correlation_heatmap_interactive

html_div = create_correlation_heatmap_interactive(df)  # → HTML <div>
```

---

## Installation

```bash
pip install thaieda
```

ไม่ต้องใส่ extras — ติดตั้งทั้งหมด: Thai tokenizer, NER, ML, interactive charts, Excel, stats, encoding detection

LLM providers (optional, lazy-imported):

```bash
pip install openai       # OpenAI GPT
pip install anthropic    # Anthropic Claude
pip install ollama       # Ollama local LLM (หรือใช้ HTTP fallback)
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
| `detect/` | Column type detection + Thai months + address parsing + language detection |
| `clean/` | Smart cleaning: auto-decide what to fix (encoding, numerals, BE, zwspace) |
| `quality/` | Language-aware quality checks + score 0–100 + Thai ID card validation |
| `anomaly/` | Statistical + ML + text anomaly detection |
| `ner/` | Thai NER: person/place/organization |
| `insight_engine/` | 6 cross-column insight patterns (BH-corrected) |
| `viz/` | Static + interactive charts with colorblind-safe palette |
| `report/` | Executive HTML report + smart pre-analysis |
| `llm/` | Privacy-preserving LLM analysis (4 modes, 3 providers) |
| `timeseries/` | Trend/seasonality/STL/ACF + Thai holiday awareness |
| `schema/` | Multi-file PK/FK discovery + relationship matching |

---

## Testing

```bash
pytest tests/ -v                    # all tests (691 passed)
ruff check src/ tests/              # lint
ruff format src/ tests/             # format
```

---

## License

[Apache-2.0](LICENSE) © Peet Wannasarnmetha