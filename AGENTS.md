# AGENTS.md — ThaiEDA

## Project
ThaiEDA คือ open-source AutoEDA library สำหรับข้อมูลภาษาไทย
- Repo: https://github.com/peetwan/thaieda
- License: Apache-2.0
- Python 3.10+, pandas + matplotlib + Jinja2 + pythainlp (optional)
- Current version: v0.8 (commit 3351e9c)
- Tests: 356 passed, 1 skipped, ruff clean

## Structure
```
src/thaieda/
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
  llm/            placeholder (v0.9+)
tests/            pytest (356 tests)
research/         research notes (cron-generated, NOT source code)
```

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