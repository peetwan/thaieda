# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-06-25

Headline feature: **multi-file schema discovery** — analyze a whole folder of related
files as one dataset, automatically infer how the tables connect, and render a combined
report with an interactive ER diagram.

### Added
- **`thaieda.schema` module** — discovers relationships between multiple data files:
  - `discover_keys(df, table_name)` — finds primary/foreign key candidate columns
    (name hints via `_name_hints_id` + uniqueness ≥ 95%, excluding boolean/constant columns).
  - `match_relationships(tables, profiles, validate_values=, sample_size=)` — matches
    columns across tables by normalized name, infers direction (unique side = parent/PK,
    non-unique side = child/FK), and confirms with real value overlap. Both-non-unique
    pairs are never linked (prevents `date ↔ date` false positives).
  - `profile_dataset(paths, ...)` — reads a directory or list of files (`.csv/.json/.jsonl/.ndjson`),
    profiles each table, and returns a `DatasetProfile`.
  - Dataclasses `KeyCandidate`, `Relationship`, `TableProfile`, `DatasetProfile`
    (all with `to_dict()`); `DatasetProfile.to_mermaid()` / `to_json()`.
  - Key values are normalized before comparison (Thai numerals → Arabic, zero-width
    characters stripped, trailing `.0` float artifacts removed) — vectorized for million-row tables.
  - Orphan detection: foreign-key values with no matching primary key are reported in Thai.
- **`DatasetReport`** (`thaieda.report._dataset`) — self-contained HTML report with a schema
  overview, a Mermaid.js ER diagram (loaded via CDN), per-table summaries with key columns,
  a relationships table (overlap %, orphans, cardinality, confidence), and orphan findings.
- **CLI `thaieda dataset <dir|files...>`** — multi-file analysis to an HTML report
  (`--no-validate`, `--json`, `--lang`, `--quiet`). `thaieda run`/`thaieda profile` now
  auto-route to dataset mode when given a directory containing ≥ 2 supported files.
- New `thaieda` top-level exports: `profile_dataset`, `DatasetProfile`, `Relationship`,
  `KeyCandidate`, `TableProfile`, `DatasetReport`.

## [0.4.1] - 2026-06-25

Performance fixes from real-world testing on large datasets (1M+ rows), plus UX/visual polish.

### Performance
- **Vectorized text cleaning** (`clean._apply_str_transform`): replaced the per-row Python
  loop with vectorized pandas operations (`.str` accessors / bulk assignment). A single
  cleaning operation on 1M rows now runs in ~1s (was 100s+). Encoding repair (ftfy) now
  pre-filters rows so it only runs on likely-mojibake cells (~14.5s → ~0.5s on 1M rows).
- **Sampled ML anomaly detection**: Isolation Forest / LOF now sample down to 10,000 rows
  on large columns (statistical z-score/MAD/IQR still run on full data). LOF on 1.3M rows
  went from >180s to <1s. The result notes when sampling was used.
- **LOF on duplicate-heavy data**: skips LOF when >50% of values are duplicates, bumps
  `n_neighbors` otherwise, and suppresses the noisy sklearn duplicate-distance warning.

### Fixed
- Cleaning suggestions no longer fail on numeric/ID columns (e.g. `int64` `customer_id`) —
  suggestions now run only on text columns (THAI_TEXT / MIXED_TEXT / ENGLISH_TEXT /
  CATEGORICAL), and the cleaning machinery returns object dtype safely.

### Added
- CLI `--sample N` flag (`run`, `profile`) — randomly sample N rows before analysis.
- CLI `--quiet` flag — minimal output (just result file paths).
- Progress feedback during processing (`อ่านไฟล์...`, `ตรวจจับประเภทคอลัมน์...`, …),
  row count after reading large (>10MB) files, and a "large file" speed hint.
- `ProfileReport(progress=...)` callback hook for step-by-step progress.
- HTML report redesign: sticky section navigation, severity emoji + colored borders on
  insight cards, type-colored column badges, a timeseries trend/seasonality banner, a
  visual before→after cleaning diff (red strikethrough → green), responsive/mobile layout,
  and print-friendly CSS.
- Better error messages: encoding failures list the encodings tried; cleaning failures
  include the column dtype.

## [Unreleased]

### Added
- Initial project structure
- Thai text column detection (script-ratio based)
- Tier-1 data quality checks (Buddhist era, Thai numerals, zero-width spaces, script composition)
- Thai text metrics (length in chars/tokens/words, top tokens, n-grams)
- Word cloud with bundled Thai font
- HTML report generation (Jinja2)
- CLI interface (`thaieda profile data.csv`)
- Bilingual UI labels (Thai/English)
- Tokenizer adapter interface (pythainlp / nlpo3 / attacut)

## [0.1.0] - Unreleased

Initial alpha release.