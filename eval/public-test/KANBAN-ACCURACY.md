# Kanban — ThaiEDA Accuracy Improvement (Research → Develop)

## Research Summary

Tools available beyond current usage:
- **pythainlp.spell** — spell correction (correct, correct_sent) — NOT used, low cost (already a dependency)
- **pythainlp.util.abbreviation_to_full_text** — Thai abbreviation expansion — NOT used, low cost
- **nlpo3** — Rust tokenizer 3-4x faster — adapter exists but auto-order picks pythainlp first
- **AttaCut** — neural tokenizer — adapter exists but lowest priority
- **NFKC** — full-width character normalization — NOT used in clean, low cost (stdlib unicodedata)

## Key Decisions (from Think phase)

1. All new features = **opt-in** (not default) — avoid false positives on clean data
2. Keep existing behavior unchanged — add new operations/modes only
3. Use existing dependencies only — no new packages needed
4. Report-first, fix-second for risky operations (spell correction, keyboard layout)

---

## 🔴 P0 — High Value, Low Cost

### AC-1: Thai Abbreviation Expansion (opt-in operation)
- **What**: Add `expand_abbreviations` cleaning operation using `pythainlp.util.abbreviation_to_full_text`
- **Why**: `กทม.` vs `กรุงเทพมหานคร` counted separately in metrics/TF-IDF/NER
- **File**: `src/thaieda/clean/__init__.py` — add to operation registry (lines 933-947)
- **Design**: NOT in DEFAULT_OPERATIONS — opt-in only
- **Tests**: Verify กทม. → กรุงเทพมหานคร, บจ. → บริษัทจำกัด

### AC-2: Thai Spell Correction (opt-in operation)
- **What**: Add `spell_correct` cleaning operation using `pythainlp.spell.correct_sent`
- **Why**: `ขอบคุน` vs `ขอบคุณ`, `คับ` vs `ครับ` counted separately
- **File**: `src/thaieda/clean/__init__.py` — add to operation registry
- **Design**: NOT in DEFAULT_OPERATIONS — opt-in only, report-only first
- **Tests**: Verify common Thai typos are corrected

### AC-3: NFKC Full-Width Cleaning
- **What**: Add `normalize_nfkc` cleaning operation using `unicodedata.normalize("NFKC", text)`
- **Why**: Quality detects full-width chars but clean only uses NFC — gap
- **File**: `src/thaieda/clean/__init__.py` — add operation
- **Design**: Opt-in, NOT default (NFKC changes semantics — e.g. Ａ→A)
- **Tests**: Verify fullwidth → halfwidth conversion

---

## 🟠 P1 — Medium Value, Low Cost

### AC-4: Tokenizer Selection Modes
- **What**: Add `engine="auto-fast"` (prefer nlpo3) and `engine="auto-quality"` (prefer attacut)
- **Why**: auto always picks pythainlp even when nlpo3/attacut installed
- **File**: `src/thaieda/tokenize/__init__.py` lines 15-16, 111-140
- **Tests**: Verify auto-fast picks nlpo3, auto-quality picks attacut

### AC-5: Keyboard Layout Anomaly Detection
- **What**: Add quality/anomaly check for suspected keyboard layout errors BEFORE cleaning
- **Why**: Current cleaner misses mixed cells (Thai+Latin), whole-column, reverse layout
- **File**: `src/thaieda/quality/__init__.py` or `src/thaieda/anomaly/__init__.py`
- **Design**: Report-only (flag suspicious rows), don't auto-correct
- **Tests**: Verify detection of "สวัสดี l;ylfu" type mixed cells

### AC-6: Thai Grapheme Validation (stronger normalization)
- **What**: Detect invalid Thai grapheme clusters (multiple tone marks, incompatible vowel stacks)
- **Why**: Current check only catches identical repeated tone marks (่่), not ก่้
- **File**: `src/thaieda/quality/__init__.py` — extend normalization check
- **Design**: Report-only, don't auto-fix (too risky)
- **Tests**: Verify ก่้, invalid vowel stacks are detected

---

## 🟡 P2 — Enhancement (if time permits)

### AC-7: Text Metrics Preprocessing Option
- **What**: Add `normalize_for_metrics=True` option to text_metrics
- **Why**: Top tokens fragmented by typos/abbreviations/repeats
- **File**: `src/thaieda/text/__init__.py` lines 120-190
- **Design**: Opt-in, preserves original data

### AC-8: OOV Ratio Anomaly Detection
- **What**: Use tokenizer to compute out-of-vocabulary ratio per row
- **Why**: Rows full of Thai typos but valid Unicode not flagged
- **File**: `src/thaieda/anomaly/__init__.py` — new check
- **Design**: Report-only, uses pythainlp corpus

---

## Execution Plan

1. **Batch 3** → Claude Code: AC-1 + AC-2 + AC-3 (new clean operations)
2. **Batch 4** → Claude Code: AC-4 + AC-5 + AC-6 (tokenizer + quality)
3. **Verify** → Run tests + pipeline
4. **Update README** → Document new features
5. **Commit + Push** → to main