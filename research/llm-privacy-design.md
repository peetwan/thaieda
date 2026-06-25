# ThaiEDA privacy-preserving LLM analysis design

Date: 2026-06-25
Repo context checked: `src/thaieda/llm/__init__.py`, `src/thaieda/clean/__init__.py`, `src/thaieda/ner/__init__.py`, `src/thaieda/detect/__init__.py`, `src/thaieda/quality/__init__.py`, `pyproject.toml`.

## Executive recommendation

Build ThaiEDA v0.9 LLM analysis as **privacy-first by default**, with four modes:

1. **`prompt_only` mode** — generate a ready-to-copy Thai/English LLM prompt from ThaiEDA's computed findings only. No network call. This is the safest MVP and works even without LLM dependencies.
2. **`local` mode** — call a local LLM through LiteLLM, with Ollama as the primary supported backend. Recommended default for government/finance/medical data.
3. **`anonymized_cloud` mode** — send only a sanitized profile/insight package to cloud LLMs after explicit opt-in, never raw rows by default.
4. **`hybrid` mode** — automatically choose local for sensitive datasets and allow cloud only when the privacy scanner reports low risk and the user explicitly allows it.

The core architectural rule should be:

> ThaiEDA LLMs should analyze **profiles, quality issues, anomalies, aggregate distributions, and redacted examples**, not raw dataframes.

For sensitive users, this provides useful LLM-powered explanation while keeping raw records, direct identifiers, and unredacted free text on the user's machine.

---

## Findings from current ThaiEDA codebase

### Existing LLM module

`src/thaieda/llm/__init__.py` is only an 8-line placeholder:

- says LLM Q&A over computed profiles is planned;
- mentions LiteLLM and Ollama for local/private data;
- exports nothing.

### Existing modules that can be reused

- `clean.normalize_phone_numbers(series)` already normalizes Thai phone numbers and can be extended into masking/redaction.
- `detect.ColumnType` already includes `PHONE_NUMBER`, `ID`, `THAI_TEXT`, `MIXED_TEXT`, `ENGLISH_TEXT`, etc.
- `detect.is_phone_number`, `detect.clean_phone_string`, `_THAI_PHONE_RE` already provide Thai phone recognition utilities.
- `ner.extract_entities(series)` already extracts Thai PERSON / LOCATION / ORGANIZATION via PyThaiNLP NER engines.
- `quality` can be extended with privacy/sensitivity findings.
- `insight` already turns computed analysis into Thai-readable summaries, which is the right input for LLM prompts.
- `pyproject.toml` already has optional extra `llm = ["litellm>=1.0"]` and `ner = ["pythainlp>=5.0", "python-crfsuite>=0.9"]`.

---

## 1. Local LLM options

### Recommendation ranking for ThaiEDA

| Rank | Backend | Best use | Thai support | Setup complexity | ThaiEDA recommendation |
|---:|---|---|---|---|---|
| 1 | **Ollama** | Individual users, notebooks, privacy-first default | Good when paired with Thai/multilingual models such as OpenThaiGPT, Typhoon, Qwen2.5 | Low | Primary v0.9 local backend |
| 2 | **LM Studio** | Desktop analysts who prefer GUI | Good if user loads the right model | Low-medium | Support via OpenAI-compatible local endpoint, but not first-class docs initially |
| 3 | **llama.cpp** | Advanced users, CPU/GGUF control, offline deployment | Good with OpenThaiGPT/Typhoon/Qwen GGUF | Medium-high | Support indirectly via Ollama or OpenAI-compatible llama-server |
| 4 | **vLLM** | Enterprise GPU servers, shared internal model API | Good for HF transformer models; excellent throughput | High | Support via OpenAI-compatible endpoint for organizations, not MVP |

### Ollama

**Why it fits ThaiEDA:**

- Simple install and mental model: install Ollama, pull/run model, local HTTP API at `http://localhost:11434`.
- LiteLLM supports Ollama using calls like `completion(model="ollama/qwen2.5", api_base="http://localhost:11434", ...)`.
- Works cross-platform and is easiest for non-ML users.
- Supports GGUF/model library ecosystem; OpenThaiGPT 1.5 14B has an Ollama model page, and Qwen2.5 is in the Ollama library.

**Suggested Thai/local models:**

- **OpenThaiGPT 1.5 7B/14B instruct** — Thai-focused, based on Qwen2.5; GGUF quantizations exist and an Ollama 14B model page exists. Best Thai-specific practical choice if hardware allows.
- **Typhoon / Llama-3-Typhoon family** — Thai-focused model family from SCB10X/SeaX; strong Thai benchmark positioning. Use where GGUF/Ollama/vLLM deployment is available.
- **Qwen2.5 7B/14B/32B instruct** — strong multilingual model, available in Ollama, long context. Good fallback when Thai-specific models are unavailable.
- **Llama 3.1/3.2 multilingual** — usable fallback, but Thai-specific models or Qwen2.5 are preferable for Thai explanations.

**Setup complexity:** low.

Example user-facing setup:

```bash
ollama pull qwen2.5:7b
# or a Thai-specific model, when available in their Ollama registry
ollama run qwen2.5:7b
```

ThaiEDA call:

```python
from thaieda.llm import ask_profile

answer = ask_profile(
    profile,
    question="สรุปความเสี่ยงด้านคุณภาพข้อมูลและความเป็นส่วนตัว",
    mode="local",
    model="ollama/qwen2.5:7b",
)
```

### llama.cpp

**Why it matters:**

- Runs GGUF models directly, including quantized Thai-focused models.
- Good for fully offline, minimal-dependency, CPU-friendly deployments.
- More transparent/control-oriented than Ollama.

**Tradeoffs:**

- Users must choose/download GGUF, run `llama-server`, tune context/GPU layers.
- More docs burden.

**ThaiEDA integration:** do not build a dedicated client first. Support it through:

- LiteLLM if compatible;
- OpenAI-compatible local endpoint, e.g. `api_base="http://localhost:8080/v1"`.

### LM Studio

**Why it matters:**

- Very approachable GUI for analysts.
- Can run GGUF models and expose an OpenAI-compatible local server.
- Good for Windows desktop users.

**Tradeoffs:**

- GUI-driven; less reproducible for server environments.
- Model management outside Python.

**ThaiEDA integration:** support as `provider="openai_compatible"` with `api_base="http://localhost:1234/v1"`.

### vLLM

**Why it matters:**

- Best for enterprise/internal GPU server deployments.
- OpenAI-compatible server API.
- High throughput and batching.

**Tradeoffs:**

- Linux/GPU-oriented, high operational complexity.
- Not a fit for most notebook users.

**ThaiEDA integration:** support through OpenAI-compatible endpoint; document as enterprise deployment pattern.

---

## 2. Data anonymization before LLM

ThaiEDA should distinguish **redaction for prompt safety** from **statistical privacy guarantees**.

### PII types to detect for Thai data

Direct identifiers:

- Thai phone numbers: `0812345678`, `08-1234-5678`, `+66812345678`, Thai digits.
- Thai national ID card numbers: 13 digits, often formatted `1-2345-67890-12-3`; should validate checksum where possible.
- Names/person entities from Thai NER.
- Addresses and locations.
- Organizations if sensitive.
- Emails, URLs, LINE IDs/user handles.
- Bank/account/card-like numbers where feasible.
- Dates of birth or exact dates in medical/finance records.
- Free-text fields containing embedded identifiers.

Quasi-identifiers:

- Age, postcode/province/amphoe/tambon, occupation, small groups, exact dates, rare categories.
- These may not identify alone but can identify when combined.

### Recommended anonymization techniques

#### A. Pattern-based recognizers

Use vectorized regex and validators for structured PII:

- phone: reuse `detect.is_phone_number` / `_clean_phone_str` logic;
- Thai national ID: normalize Thai digits, remove separators, validate 13-digit pattern and checksum;
- email: standard regex;
- URL/domain;
- credit card/account-like number: Luhn for cards, conservative pattern for account numbers;
- dates: exact DOB patterns or column-name hints.

Recommended masking:

- phone: `<PHONE>` or preserve last 2 digits only if needed: `08******78`;
- Thai national ID: `<THAI_NATIONAL_ID>` or `1-****-*****-**-*`;
- email: `<EMAIL>` or hashed domain-level info only;
- free text spans: replace with typed placeholders.

#### B. Thai NER-based recognizers

Use existing `thaieda.ner.extract_entities` for:

- PERSON → `<PERSON>`;
- LOCATION → `<LOCATION>`;
- ORGANIZATION → `<ORG>`.

Important caveat: NER is probabilistic and should be treated as best-effort. It is useful for reducing risk, not a formal guarantee.

#### C. Column-level suppression/generalization

Use `detect.detect_all(df)` plus column-name heuristics:

- suppress entire columns that are likely identifiers: `id`, `user_id`, `citizen_id`, `เลขบัตร`, `hn`, `patient_id`, `account_no`;
- convert exact dates to month/quarter/year before prompt generation;
- bucket ages/incomes/counts;
- replace rare categories with `<OTHER_RARE>`;
- omit high-cardinality categorical values unless whitelisted.

#### D. Prompt package minimization

Prefer sending this to LLM:

```json
{
  "dataset_shape": {"rows": 100000, "columns": 24},
  "column_types": {"province": "categorical", "complaint": "thai_text"},
  "quality_issues": [...],
  "anomalies": [...],
  "aggregate_stats": {...},
  "privacy_report": {...},
  "redacted_examples": {"complaint": ["<PERSON> ร้องเรียนเรื่อง..."]}
}
```

Avoid sending:

- raw rows;
- unredacted free text;
- top values for ID-like columns;
- rare category labels in small groups;
- direct identifiers even if normalized.

---

## 3. Privacy-preserving patterns

### Pattern 1: Local-only LLM

Guarantee: no prompt leaves the machine unless the local endpoint is remote/internal by user configuration.

Best for:

- government;
- finance;
- medical;
- air-gapped/offline environments.

Limitations:

- model quality/hardware-dependent;
- a local HTTP endpoint can still be on a remote LAN if configured so ThaiEDA must label it clearly.

### Pattern 2: Prompt-only mode

ThaiEDA generates a prompt and returns it as text/Markdown. User decides what to do.

Guarantee:

- ThaiEDA makes no network call;
- safest default for compliance-sensitive workflows;
- still useful because the prompt is built from computed findings.

### Pattern 3: Anonymize-then-LLM

ThaiEDA redacts PII and sends only sanitized summaries. This is practical but not absolute privacy.

Guarantee:

- no raw dataframe sent;
- direct identifiers removed/masked on a best-effort basis;
- privacy report tells the user what was removed and what residual risks remain.

Limitations:

- NER/regex can miss PII;
- aggregate stats can leak if small groups are included;
- top category labels can be sensitive.

### Pattern 4: Differential privacy for aggregate stats

Add optional noise to counts/means before cloud prompts.

Practical first version:

- support `privacy_budget_epsilon` for selected aggregate counts/histograms;
- suppress groups with `n < k` before adding noise;
- include a warning that DP only applies to generated aggregates, not raw text.

Recommendation:

- do **not** make DP the v0.9 core feature; implement simple `k`-threshold suppression first.
- DP can be v1.0+ for cloud-safe aggregate exports.

### Pattern 5: Synthetic data generation for prompts

Generate schema-preserving synthetic sample rows and use them as examples for the LLM.

Recommendation:

- use only after direct identifiers are removed;
- prefer synthetic examples derived from distributions, not nearest-neighbor records;
- label clearly as synthetic.

MVP alternative:

- no synthetic rows; use ThaiEDA-generated aggregate summaries and redacted examples.

### Pattern 6: Federated analysis

Each site computes local profiles; only profiles/aggregates are combined centrally.

Recommendation:

- useful later for multi-agency/branch data;
- not needed in v0.9 implementation;
- design profile/prompt package serialization now so federated mode is possible later.

---

## 4. How other tools handle local LLM / privacy

### ydata-profiling

- Strong profiling/reporting tool.
- Traditionally privacy comes from local report generation and configuration to control samples/correlations, not from LLM privacy modes.
- No primary built-in local LLM Q&A mode found in current research.
- Lesson for ThaiEDA: make LLM optional and keep full EDA useful without sending data anywhere.

### PandasAI

- Has local LLM usage patterns, including `LocalLLM` and integrations with Ollama/LM Studio-like local endpoints.
- Community discussions explicitly mention privacy by running local LLMs instead of cloud APIs.
- However, PandasAI often works by letting the LLM write/execute dataframe code, so raw dataframe access may remain in-process.
- Lesson for ThaiEDA: support local LLMs, but avoid giving the LLM raw dataframe rows by default. Use ThaiEDA profiles/insights as the context.

### General EDA/LLM trend

- Many tools support local/OpenAI-compatible endpoints.
- Fewer tools offer strong privacy guarantees beyond “run locally”.
- ThaiEDA can differentiate by shipping a built-in Thai PII scanner, redacted prompt package, and explicit privacy modes.

---

## 5. Proposed ThaiEDA architecture

### Text diagram

```text
User DataFrame
    |
    v
ThaiEDA existing analysis pipeline
(detect, quality, anomaly, text, insight, report)
    |
    v
Profile / Findings object
    |
    +-----------------------------+
    | Privacy scanner             |
    | - column PII detection      |
    | - regex validators          |
    | - Thai NER spans            |
    | - risk scoring              |
    +-----------------------------+
    |
    v
Privacy policy decision
    |
    +--------------------+----------------------+-----------------------+
    |                    |                      |                       |
    v                    v                      v                       v
prompt_only          local                  anonymized_cloud          hybrid
(no network)         Ollama/LM Studio       sanitized package         auto local/cloud
                     llama.cpp/vLLM         + explicit opt-in        by risk + opt-in
    |                    |                      |                       |
    v                    v                      v                       v
PromptBuilder       LiteLLMClient           LiteLLMClient             Router
    |                    |                      |                       |
    v                    v                      v                       v
Markdown prompt     LLM Thai explanation    LLM Thai explanation      LLM Thai explanation
```

### Key design rules

1. **No raw dataframe by default.**
2. **No network by default.** Default mode should be `prompt_only` or `local` depending API design.
3. **Cloud requires explicit opt-in.** For example `allow_external=True`.
4. **Privacy report is always returned** with LLM answers.
5. **Small groups are suppressed** before prompt generation.
6. **Examples are redacted and capped**.
7. **All optional heavy dependencies are lazy-imported**, matching project conventions.

---

## Recommended public API

### High-level API

```python
from thaieda.llm import analyze_with_llm, generate_analysis_prompt

# Safest: no LLM call, no network
prompt = generate_analysis_prompt(report_or_profile, language="th")

# Local-only call via Ollama/LiteLLM
result = analyze_with_llm(
    report_or_profile,
    question="ช่วยสรุปคุณภาพข้อมูลและข้อควรระวัง",
    mode="local",
    model="ollama/qwen2.5:7b",
    language="th",
)

# Cloud only with explicit opt-in and sanitized context
result = analyze_with_llm(
    report_or_profile,
    question="What are the main data quality risks?",
    mode="anonymized_cloud",
    model="gpt-4o-mini",
    allow_external=True,
    privacy_level="strict",
)
```

### Return object

```python
@dataclass
class LLMAnalysisResult:
    answer: str
    mode: Literal["prompt_only", "local", "anonymized_cloud", "hybrid"]
    model: str | None
    prompt: str
    privacy_report: PrivacyReport
    data_sent_summary: dict[str, object]
    warnings: list[str]
```

---

## New modules/functions needed

### `src/thaieda/llm/__init__.py`

Export public API:

```python
__all__ = [
    "LLMConfig",
    "LLMAnalysisResult",
    "generate_analysis_prompt",
    "analyze_with_llm",
    "ask_profile",
]
```

### `src/thaieda/llm/config.py`

```python
@dataclass
class LLMConfig:
    mode: Literal["prompt_only", "local", "anonymized_cloud", "hybrid"] = "prompt_only"
    model: str = "ollama/qwen2.5:7b"
    api_base: str | None = "http://localhost:11434"
    language: Literal["th", "en"] = "th"
    privacy_level: Literal["off", "basic", "strict"] = "strict"
    allow_external: bool = False
    include_examples: bool = False
    max_examples_per_column: int = 3
    min_group_size: int = 10
    redact_entities: bool = True
    timeout: float = 60.0
```

### `src/thaieda/llm/prompt.py`

Functions:

```python
def build_profile_summary(profile: object, *, privacy_report: PrivacyReport) -> dict: ...
def build_analysis_prompt(summary: dict, question: str | None, language: str = "th") -> str: ...
def generate_analysis_prompt(profile: object, question: str | None = None, **kwargs) -> str: ...
```

Prompt should include:

- dataset shape;
- detected column types;
- quality issues;
- anomalies;
- notable correlations/segments if available;
- privacy scan summary;
- explicit instruction: “Do not infer identities; do not ask for raw data; mention privacy limitations.”

### `src/thaieda/llm/client.py`

LiteLLM wrapper with lazy import:

```python
def call_llm(prompt: str, config: LLMConfig) -> str:
    try:
        from litellm import completion
    except ImportError as exc:
        raise ImportError("Install pip install thaieda[llm] to use LLM calls.") from exc
```

Enforce safety:

- if mode is cloud and `allow_external=False`, raise `ValueError`;
- if model starts with known cloud providers and privacy report is high-risk, require explicit override;
- log/return `data_sent_summary`.

### `src/thaieda/llm/router.py`

```python
def choose_mode(privacy_report: PrivacyReport, requested_mode: str, allow_external: bool) -> str:
    if requested_mode == "hybrid":
        if privacy_report.risk_level in {"high", "critical"}:
            return "local"
        return "anonymized_cloud" if allow_external else "prompt_only"
```

### `src/thaieda/privacy/__init__.py` or `src/thaieda/clean/privacy.py`

Prefer a new `privacy` module because this is broader than cleaning.

Data structures:

```python
@dataclass
class PIIFinding:
    column: str
    pii_type: str
    count: int
    percentage: float
    detection_method: str
    examples_redacted: list[str]
    severity: Literal["low", "medium", "high", "critical"]

@dataclass
class PrivacyReport:
    risk_level: Literal["low", "medium", "high", "critical"]
    findings: list[PIIFinding]
    columns_suppressed: list[str]
    cells_redacted: int
    policy: str
    residual_risk: list[str]
```

Functions:

```python
def detect_pii(df: pd.DataFrame, *, use_ner: bool = False, max_sample: int = 1000) -> PrivacyReport: ...
def redact_text(text: str, *, use_ner: bool = False) -> tuple[str, list[PIIFinding]]: ...
def redact_series(series: pd.Series, *, pii_types: set[str] | None = None) -> tuple[pd.Series, PrivacyReport]: ...
def redact_dataframe(df: pd.DataFrame, *, policy: PrivacyPolicy) -> tuple[pd.DataFrame, PrivacyReport]: ...
def build_safe_profile(profile: object, privacy_report: PrivacyReport, policy: PrivacyPolicy) -> dict: ...
```

### `src/thaieda/privacy/patterns.py`

Implement Thai-specific recognizers:

- `is_thai_phone(value)` — reuse existing phone logic;
- `is_thai_national_id(value)` — normalize Thai digits/separators + checksum;
- `find_emails(text)`;
- `find_urls(text)`;
- `find_bank_account_like(text)`;
- `find_line_id_like(text)`.

Thai national ID checksum:

```python
def is_valid_thai_national_id(value: str) -> bool:
    digits = normalize_thai_digits(value)
    digits = re.sub(r"\D", "", digits)
    if not re.fullmatch(r"[1-8]\d{12}", digits):
        return False
    checksum = (11 - sum(int(digits[i]) * (13 - i) for i in range(12)) % 11) % 10
    return checksum == int(digits[12])
```

### `src/thaieda/privacy/policy.py`

```python
@dataclass
class PrivacyPolicy:
    level: Literal["basic", "strict"] = "strict"
    suppress_direct_identifier_columns: bool = True
    redact_free_text: bool = True
    use_ner: bool = False
    min_group_size: int = 10
    include_redacted_examples: bool = False
    max_examples_per_column: int = 3
    date_granularity: Literal["day", "month", "quarter", "year"] = "month"
```

---

## Privacy guarantees by mode

| Mode | What leaves machine | What stays local | Guarantee | Residual risk |
|---|---|---|---|---|
| `prompt_only` | Nothing from ThaiEDA | Everything | No ThaiEDA network call | User may manually paste elsewhere |
| `local` | Prompt to configured local/internal endpoint | Raw dataframe; redaction process; profile generation | No external API if endpoint is local | Endpoint may be remote if user configures remote URL |
| `anonymized_cloud` | Sanitized profile package and redacted examples only | Raw dataframe; direct identifiers; unredacted free text | No raw rows by default; best-effort PII removal | Missed PII, small-group leakage, sensitive aggregate labels |
| `hybrid` | Depends on risk and opt-in | Same as selected mode | Sensitive data routes local/prompt-only by default | Misclassification if privacy scan misses sensitivity |

Add a runtime warning for cloud modes:

> Privacy redaction is best-effort and does not guarantee de-identification under all laws/regulations. Review the generated prompt before sending externally.

---

## Thai language considerations

### Models

Recommended default docs order:

1. **OpenThaiGPT 1.5 7B/14B instruct** — Thai-focused, Qwen2.5-based; GGUF/Ollama options exist. Good for Thai explanations on consumer hardware if quantized.
2. **Typhoon / Llama-3-Typhoon** — Thai-focused with public Thai benchmark work; use for stronger Thai if available locally or on internal vLLM.
3. **Qwen2.5 7B/14B** — strong multilingual fallback, easy Ollama setup, long context.
4. **Llama 3.1/3.2 multilingual** — acceptable fallback but should not be the primary Thai recommendation.

### Prompt language

- Default output language should be Thai (`language="th"`).
- Keep technical terms bilingual where useful: `outlier (ค่าผิดปกติ)`, `missing value (ค่าว่าง)`.
- Ask the model to avoid overconfident causal claims.
- Ask the model to cite specific ThaiEDA findings, not invent data facts.

### Thai PII detection

- Thai has no whitespace word boundary, so names in free text need NER/tokenization.
- Thai digits must be normalized before detecting phone/ID numbers.
- Thai date formats and Buddhist Era years can reveal DOB/age; exact DOB should be generalized.
- Honorifics (`นาย`, `นาง`, `น.ส.`, `ดร.`, `คุณ`) are useful context for name detection.
- Addresses often include markers: `บ้านเลขที่`, `หมู่`, `ต.`, `อ.`, `จ.`, `แขวง`, `เขต`, `จังหวัด`, postcode.

---

## Implementation phases

### Phase 0: API design and tests only

- Replace placeholder `llm/__init__.py` exports.
- Add dataclasses: `LLMConfig`, `LLMAnalysisResult`.
- Add tests for prompt-only mode.
- No new runtime dependency required.

### Phase 1: Prompt-only MVP

Build:

- `thaieda.llm.generate_analysis_prompt(...)`;
- profile/insight summarizer;
- Thai prompt template;
- safety statements.

Value:

- users immediately get LLM-ready analysis without any data transfer;
- easiest to ship and test.

### Phase 2: Basic privacy scanner and redactor

Build:

- `thaieda.privacy.detect_pii(df)`;
- regex/validator PII detection for Thai phone, Thai national ID, email, URL;
- column suppression based on `detect.ColumnType.ID`, `PHONE_NUMBER`, name hints;
- `PrivacyReport`;
- tests with Thai digits and formatted phone/ID.

Extend existing clean:

- either add `clean.mask_phone_numbers(series)` or put masking in `privacy.redact_series` and reuse `clean.normalize_phone_numbers` internals.

### Phase 3: Local LLM via LiteLLM + Ollama

Build:

- `thaieda.llm.client.call_llm` lazy-importing LiteLLM;
- `analyze_with_llm(..., mode="local")`;
- docs for Ollama + Qwen2.5/OpenThaiGPT/Typhoon;
- health check for local endpoint.

Safety:

- default model `ollama/qwen2.5:7b` or `ollama/qwen2.5`;
- warn if `api_base` is not localhost/private IP unless user confirms.

### Phase 4: NER-assisted redaction

Build:

- integrate `thaieda.ner.extract_entities` into privacy redaction;
- optional `use_ner=True` requiring `thaieda[ner]`;
- redact PERSON/LOCATION/ORG spans in redacted examples.

Caveat:

- fail loudly if NER requested but not installed, following ThaiEDA conventions.

### Phase 5: Anonymized cloud and hybrid routing

Build:

- `mode="anonymized_cloud"` with `allow_external=True` required;
- `mode="hybrid"` router;
- `data_sent_summary` returned to user;
- prompt review hook: return prompt before sending if `review=True`.

### Phase 6: Advanced privacy

Build later:

- `min_group_size` suppression for aggregate stats;
- optional DP noise for histograms/counts;
- synthetic profile examples;
- federated profile merge API.

---

## Dependencies / optional extras

Current `pyproject.toml` already includes:

```toml
llm = ["litellm>=1.0"]
ner = ["pythainlp>=5.0", "python-crfsuite>=0.9"]
```

Recommended additions:

```toml
privacy = [
    "pythainlp>=5.0",
    "python-crfsuite>=0.9",
]

# Optional only if ThaiEDA later chooses Presidio integration.
privacy-presidio = [
    "presidio-analyzer>=2.2",
    "presidio-anonymizer>=2.2",
]
```

But for v0.9, avoid making Presidio a required dependency. Presidio is powerful and extensible, but its default recognizers are not Thai-specific enough; ThaiEDA should first implement lightweight Thai recognizers and optionally allow Presidio later.

No Python package dependency is needed for Ollama itself because LiteLLM calls the local HTTP API. Users install Ollama outside Python.

---

## Source notes from research

- LiteLLM Ollama provider docs show using models such as `ollama/mistral`, `ollama/llama3`, etc. with `api_base="http://localhost:11434"`: https://docs.litellm.ai/docs/providers/ollama
- Ollama Qwen2.5 model library notes multilingual support and long context: https://ollama.com/library/qwen2.5
- OpenThaiGPT 1.5 14B Ollama model page states it is Thai-language chat/instruct and based on Qwen2.5: https://ollama.com/promptnow/openthaigpt1.5-14b-instruct-q4_k_m
- OpenThaiGPT 1.5 7B GGUF page describes quantized GGUF availability: https://huggingface.co/QuantFactory/openthaigpt1.5-7b-instruct-GGUF
- Typhoon Thai LLM / benchmark references: https://huggingface.co/papers/2312.13951 and https://crfm.stanford.edu/2024/09/04/thaiexam.html
- Microsoft Presidio supported entities/custom recognizer approach: https://microsoft.github.io/presidio/supported_entities
- PandasAI privacy/local LLM discussion: https://github.com/sinaptik-ai/pandas-ai/discussions/566
- PandasAI + Ollama/local analyst examples found in web research: https://dataskillblog.com/pandasai-and-ollama-local-ai-data-analyst
- Thai national ID format reference found in DLP docs: https://docs.trellix.com/bundle/data-loss-prevention-11.10.x-classification-definitions-reference-guide/page/UUID-3661faf4-4350-b1f5-51ad-f6a4f4466f07.html

---

## Practical MVP acceptance criteria

1. `generate_analysis_prompt(profile, language="th")` returns a Thai prompt and performs no network call.
2. `detect_pii(df)` detects Thai phone numbers, Thai national IDs, emails, ID-like columns, and returns a `PrivacyReport`.
3. `analyze_with_llm(..., mode="local", model="ollama/qwen2.5:7b")` works when Ollama is running.
4. Cloud mode raises unless `allow_external=True`.
5. Returned `LLMAnalysisResult` includes `privacy_report` and `data_sent_summary`.
6. Tests cover Thai digits, phone formats, national ID checksum, ID column suppression, and prompt-only no-network behavior.
