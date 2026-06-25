# AGENTS.md — ThaiEDA

## Project
ThaiEDA คือ open-source AutoEDA library สำหรับข้อมูลภาษาไทย
- Repo: https://github.com/peetwan/thaieda
- License: Apache-2.0
- Python 3.10+, pandas + matplotlib + Jinja2 + pythainlp (optional)

## Structure
```
src/thaieda/
  detect/         column type detection
  tokenize/       tokenizer adapter (pythainlp/nlpo3/attacut)
  text/           text metrics
  quality/        Thai data quality checks
  anomaly/        anomaly detection (numeric + text + Thai-specific)
  clean/          data cleaning functions
  ner/            Thai NER (v0.2)
  analysis/       target variable analysis (v0.2)
  insight/        auto insight summary — interpreter (v0.3) + distribution/correlation (v0.4)
  insight_engine/ cross-column insight discovery — discoverer: 4 patterns + BH correction (v0.6)
  insight_viz/     auto charts for insight cards: bar/donut/box/line (v0.7, inside viz/)
  timeseries/     timeseries analysis (v0.4)
  schema/         multi-file schema discovery — PK/FK + relationship matching (v0.5)
  viz/            visualization + Thai font
  report/         HTML report (Jinja2) + DatasetReport (v0.5)
  i18n/           TH/EN labels
  llm/            placeholder (v0.7+)
tests/            pytest
research/         research notes (cron-generated, NOT source code)
```

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