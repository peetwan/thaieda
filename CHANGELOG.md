# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-06-25

Headline feature: **insight visualization** — every cross-column insight card now
includes an auto-generated chart matched to its pattern, embedded in the HTML report.

### Added
- **`thaieda.viz` insight chart functions**:
  - `create_insight_outstanding_chart(segments, top_segment, title, font_path)` —
    horizontal bar chart; top segment highlighted in green, others in muted gray.
  - `create_insight_attribution_chart(segments, top_segment, share, title, font_path)` —
    donut chart with the dominant segment's share % in the center.
  - `create_insight_comparison_chart(df, breakdown, measure, top_segment, title, font_path)` —
    box plot by group; top segment highlighted in red, groups beyond top-9 collapsed into "อื่น ๆ".
  - `create_insight_trend_chart(segments, direction, tau, title, font_path)` —
    line chart with filled area, direction arrow annotation, and τ value in the title.
  - `create_insight_chart(card_dict, df, font_path)` — dispatcher that picks the right
    chart based on the card's `pattern` field.
- **`ProfileReport._build_insight_charts()`** — generates a chart for each insight card
  during `profile()` and embeds it in the HTML report alongside the text + evidence table.
- **HTML template** — insight cards now render an `<img>` (base64 PNG) when a chart is
  available; evidence table is hidden for trend cards (the line chart replaces it).

### Changed
- Roadmap shifted: v0.8 = LLM Q&A, v0.9 = interactive dashboard (was v0.7/v0.8).

### Fixed
- Trend chart annotation: replaced Unicode arrows (↗↘) with ASCII text ("Trend UP"/"Trend DOWN")
  to avoid missing-glyph warnings on Thai fonts.

---

## [0.6.0] - 2026-06-25

Headline feature: **cross-column insight engine** — a *discoverer* (vs. the existing
`insight` *interpreter*) that combines columns (group-by + aggregate + statistical scoring)
to surface non-obvious business findings in Thai, ranked by an interestingness pipeline.
Works on **any** dataset — zero column-name logic, no domain overfitting; everything is
driven by `ColumnType` + cardinality + value ranges.

### Added
- **`thaieda.insight_engine` module** — discovers insights from column combinations:
  - `discover_insights(df, column_types, *, top_n=8, sample_size=100_000, min_segment=30, progress=)`
    — builds perspectives (breakdown × measure × agg) and detects **4 patterns**:
    - **outstanding** — one segment dominates (`top/second ≥ 1.5`),
    - **attribution** — one segment is a large share (`≥ 50%`) of a total (`≥ 3` segments),
    - **comparison** — top segment differs significantly from the rest (ANOVA/Kruskal + JSD),
    - **trend** — monotonic movement over an ordered (datetime-bucketed) axis (Mann-Kendall).
  - Dataclasses `Perspective`, `InsightCard`, `InsightEngineResult` (all with `to_dict()`).
  - **Interestingness pipeline**: `gate → score → penalize → rank` —
    `final = gate × (0.5·pattern_score + 0.5·effect_size) × novelty × (1 − triviality)`.
  - **Benjamini-Hochberg correction** across all candidate significance tests (FDR control
    for hundreds of comparisons). Mann-Kendall p-values are computed with `math.erf`, so trend
    significance works even without scipy; ANOVA p-values degrade to effect-size-only + a note.
  - **Two-phase**: scores candidates on a stratified sample (~100k rows), then recomputes exact
    numbers on the full data for the top-N only — handles 1M+ rows (804k rows in ~5s).
  - **Triviality / non-additive guards**: excludes ID / near-unique / single-group breakdowns;
    skips `sum` for measures bounded in `[0,1]` (or `[0,100]` percentage-like floats).
  - Category keys are normalized before group-by (Thai numerals → Arabic, zero-width stripped,
    trailing `.0` removed) via the shared `schema._normalize_key_series` — prevents split groups.
- **`ProfileReport(insights_engine=True, insights_top=8)`** — new analysis stage (after target
  analysis, before timeseries), wrapped in try/except → `notes`. Top 3 cards feed the existing
  `InsightSummary` (new `business` category) so they appear in the executive summary.
- **Dedicated HTML report section** "ข้อค้นพบจากการวิเคราะห์คอลัมน์ผสม" (Cross-Column Insights)
  with pattern badges, per-card evidence mini-tables (top segments, share, lift, p-value, τ).
- **CLI** `--no-insights` and `--insights-top N` flags on `profile` and `run` (on by default).
- New `thaieda` top-level exports: `discover_insights`, `InsightCard`, `InsightEngineResult`,
  `Perspective`.

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