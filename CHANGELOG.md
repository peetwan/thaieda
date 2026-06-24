# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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