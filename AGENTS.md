# AGENTS.md — ThaiEDA

## Project
ThaiEDA คือ open-source AutoEDA library สำหรับข้อมูลภาษาไทย
- Repo: https://github.com/peetwan/thaieda
- PyPI: https://pypi.org/project/thaieda/
- License: Apache-2.0
- Python 3.10+, pandas + matplotlib + Jinja2 + pythainlp (optional)
- Current version: v1.0.0 (commit bc8c8d4)
- Tests: 424 passed, 24 skipped, ruff clean
- Install: `pip install thaieda` or `pip install "thaieda[all]"`

## Structure
```
src/thaieda/
  __init__.py     run(df) / EDA(df) one-liner API + EDAResult dataclass (v1.0)
  detect/         column type detection + Thai month name detection (v0.8)
  tokenize/       tokenizer adapter (pythainlp/nlpo3/attacut)
  text/           text metrics
  quality/        Thai data quality checks + placeholder/constant detection (v0.8) + vectorized BE
  anomaly/        anomaly detection (numeric + text + Thai-specific)
  clean/          data cleaning: encoding, zwspace, numerals, BE→CE, dates, duplicates, missing (v0.8)
  ner/            Thai NER (v0.2)
  analysis/       target variable analysis (v0.2)
  insight/        auto insight summary — interpreter (v0.3) + distribution/correlation (v0.4)
  insight_engine/ cross-column insight discovery — 6 patterns + BH correction (v0.6+v0.8)
  timeseries/     timeseries analysis (v0.4)
  schema/         multi-file schema discovery — PK/FK + relationship matching (v0.5)
  viz/            visualization + Thai font + insight charts (v0.7)
  report/         HTML report (Jinja2) + DatasetReport (v0.5)
  i18n/           TH/EN labels
  llm/            privacy-preserving LLM analysis — 4 modes (insight_only, anonymized, dp_noise, full),
                  3 providers (OpenAI/Anthropic/Ollama), lazy import (v0.9)
tests/            pytest (424 tests, 24 skipped)
research/         research notes (cron-generated, NOT source code)
```

## One-Liner API (v1.0)
- `run(df)` / `EDA(df)` — one-line EDA: detect → clean → quality → insights → viz → report → optional LLM
- Returns `EDAResult` dataclass with `.to_html()`, `.to_dict()`, `.to_json()`, `.llm_response`, `.insights`, `.notes`

## v1.0 additions
- `__init__.py`: `run(df)` / `EDA(df)` one-liner API + `EDAResult` dataclass (to_html, to_dict, to_json, llm_response, notes)
- `pyproject.toml`: `[all]` meta-extra for single-command install of all optional deps
- Published to PyPI: https://pypi.org/project/thaieda/1.0.0/
- 32 new tests in `test_oneliner.py`
- Fixed: `test_comparison_significant` scipy dependency
- Fixed: ruff E501 line-length violation

## v0.9 additions
- `llm/`: privacy-preserving LLM analysis — 4 modes (insight_only, anonymized, dp_noise, full),
  3 providers (OpenAI/Anthropic/Ollama), lazy import
- `llm/__init__.py`: `analyze_with_llm(df, privacy, provider, model, language)` public API
- `llm/_prepare.py`: `prepare_for_llm()` — prepares data per privacy mode
- `llm/_anonymize.py`: `anonymize_dataframe()` — PII detection (phone, ID card, NER)
- `llm/_prompt.py`: `build_prompt()` — Thai/English prompt builder
- `llm/_provider.py`: `call_llm()` — lazy import OpenAI/Anthropic/Ollama
- 59 tests in `test_llm.py`

## v0.8 additions
- clean/: coerce_numeric_column, convert_buddhist_era, normalize_dates, remove_duplicate_rows, handle_missing_values
- insight_engine/: _detect_strong_correlations, _detect_outlier_insights, cross-pattern novelty, adaptive min_segment, all_buckets in trend evidence
- quality/: check_placeholder_values, check_constant_column, vectorized check_buddhist_era
- detect/: Thai month name detection in _looks_like_datetime
- io/: Excel (.xlsx/.xls) support via openpyxl

## Cron job rules
- Cron jobs แก้ไขเฉพาะ `research/` directory เท่านั้น
- ห้ามแก้ src/, tests/, pyproject.toml, หรือ config อื่น ๆ
- ทุก finding ต้องมี source URL
- Git commit message: `docs: research update - <topic> (YYYY-MM-DD)`
- Push to origin main

## Coding conventions
- Python 3.10+ with type hints
- Thai docstrings/comments
- Lazy imports for optional deps
- No silent fallbacks — fail loudly with helpful message
- matplotlib Agg backend (no GUI)
- Vectorized operations preferred (pandas .str accessors over row-by-row loops)
- Use contextlib.suppress instead of try/except/pass