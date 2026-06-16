---
title: ESG Document Intelligence
emoji: 📋
colorFrom: green
colorTo: green
sdk: docker
pinned: false
license: mit
---

# ESG Document Classification & Anonymization Microservice

Upload a BRSR (Business Responsibility and Sustainability Report) or
Sustainability Report PDF. The pipeline anonymizes PII, classifies the
document, scores it across 16 ESG sub-metrics, and produces a downloadable
PDF report, JSON export, and anonymised text file.

---

## Pipeline overview

```
PDF/TXT upload
     |
     v
1. Ingestion         pdfplumber extracts raw text page-by-page
     |
     v
2. Anonymization     Pass A (NER) + Pass B (regex) -> scrubbed text
     |
     v
3. Classification    Keras Text-CNN -> SEBI_BRSR / SUSTAINABILITY_REPORT / INVALID
     |
     v
4. ESG Scoring       Negation-aware keyword coverage -> 16 metrics -> 3 pillars -> overall score
     |
     v
JSON / PDF / anonymised .txt download
```

Classification and scoring both run on the **anonymised** text, never the
raw upload — PII is removed before any further processing happens.

---

## PII Anonymization — detailed logic

The anonymizer (`src/components/data_transformation.py`) is a **two-pass
pipeline**. Pass A always runs first, because it depends on seeing intact
words and phrases — if Pass B's structural regex ran first, it could
fragment a name or organisation string before Pass A gets a chance to
recognise it as a whole unit.

### Why two passes instead of one

Person names and organisation names don't follow a fixed character
pattern — they're recognised by their *surrounding context* (an honorific,
a job title, a corporate suffix). Structural identifiers like CIN, PAN,
emails, and phone numbers, on the other hand, **do** follow strict,
predictable character formats. Mixing both kinds of detection into one
pass invites exactly the bug this project hit during testing: a regex
built for one purpose ends up partially matching text meant for the
other, corrupting words mid-string (e.g. "Care" becoming
`[REDACTED_ORG]re`). Separating them into Pass A (context-based) and Pass
B (format-based), in that order, avoids this entirely.

### Pass A — Linguistic NER (no machine learning model required)

| Rule | Logic | Example |
|---|---|---|
| **Honorific-anchored names** | Match `Mr./Mrs./Dr./Shri./Prof./CA./CS./CEO/...` followed by 1-4 capitalised words | `Dr. Sanjay Kirloskar` -> `[REDACTED_PERSON]` |
| **Designation-anchored names** | Match a capitalised name (1-4 words) immediately followed by a dash or comma and then a job-title keyword (Director, CEO, Manager, Chairman, Trustee, Auditor, etc.) | `Abhijeet Shinde - Assistant General Manager` -> `[REDACTED_PERSON]` |
| **Corporate-suffix organisations** | Match 1-6 capitalised words ending in a recognised corporate suffix (`Limited`, `Ltd.`, `Pvt. Ltd.`, `LLP`, `Inc.`, `Corp.`, `Group`, `Foundation`, etc.) | `Kirloskar Brothers Limited` -> `[REDACTED_ORG]` |
| **Named institutions** | Match known regulator/exchange names (BSE, NSE, SEBI, RBI, Ministry of X, Registrar of Companies), absorbing a trailing corporate suffix if present | `BSE Limited` -> `[REDACTED_ORG]` |

Every Pass A rule is anchored with `\b` word boundaries on both sides of
the match. This is what prevents mid-word corruption — the regex engine
will not match a corporate suffix that is itself embedded inside a larger
unrelated word.

### Pass B — Deterministic regex (structural identifiers)

| Tag | Pattern logic | Notes |
|---|---|---|
| `[REDACTED_CIN]` | `[LU]` + 5 digits + 2 letters + 4 digits + 3 letters + 6 digits (MCA Corporate Identity Number format) | Case-insensitive, for OCR variants |
| `[REDACTED_GSTIN]` | 15-character GST Identification Number | Checked **before** PAN — a GSTIN contains a 10-character PAN-like substring, so matching PAN first would leave fragments behind |
| `[REDACTED_PAN]` | 5 letters + 4 digits + 1 letter (Permanent Account Number format) | Case-insensitive |
| `[REDACTED_IFSC]` | 4 letters + literal `0` + 6 alphanumeric characters (bank branch code) | |
| `[REDACTED_ACCOUNT]` | IBAN format, or any standalone 9-18 digit number | Covers both international and domestic bank account numbers |
| `[REDACTED_AADHAAR]` | 12 digits in `XXXX XXXX XXXX` grouping | |
| `[REDACTED_FINANCIALS]` | A currency symbol (`₹`, `Rs.`, `` ` `` as an OCR artifact for ₹, `INR`, `$`, `€`, `£`) followed by a number, or a number followed by `crores/lakhs/millions/billions` | The backtick is included specifically because OCR engines sometimes misread ₹ as a backtick character |
| `[REDACTED_EMAIL]` | Standard `local@domain.tld` pattern | |
| `[REDACTED_WEBSITE]` | `www.*` or `http(s)://*` URLs | |
| `[REDACTED_PHONE]` | Indian formats (`+91`, STD codes, toll-free `1800`) and generic international formats | |
| `[REDACTED_ADDRESS]` | Four sub-rules merged under one tag: (1) a quoted building/property name immediately before an address block, (2) `Survey No./Plot No./Flat No./Door No.` blocks, (3) a gazetteer of ~30 Indian state and city names, (4) standalone 6-digit PIN codes | See ordering note below |
| `[REDACTED_DATE]` | `DD/MM/YYYY` or `DD-MM-YYYY` | For date-of-birth or date-of-joining fields |

### Two orderings that are load-bearing

**GSTIN before PAN.** A GSTIN embeds a PAN-like 10-character substring
within its 15 characters. If the PAN pattern ran first, it would match
that embedded substring and redact only part of the GSTIN, leaving the
remaining digits exposed as orphaned text.

**Quoted building name before the Survey No. pattern.** A real example
from testing: `"Yamuna", Survey No. 98 / (3 to 7), Plot No. 3, Baner,
Pune`. The building-name rule detects a quoted word by looking ahead for
the literal text "Survey" immediately after the closing quote. If the
Survey No. pattern ran first, it would consume the text `, Survey No.
98...` as part of its own match — and the building-name rule, running
second, would have nothing left after the quote to look ahead at, so
`"Yamuna"` would leak through unredacted. Running the building-name rule
first preserves the lookahead anchor.

### Performance and safety details

- Every pattern uses `pattern.subn(tag, text)` rather than separate
  `findall()` + `sub()` calls — this returns `(new_text, count)` in a
  single pass over the string, roughly halving regex processing time on
  large documents compared to scanning twice.
- `re.IGNORECASE` is applied per-pattern (not globally), so it only
  affects the specific identifiers that need it (CIN, GSTIN, PAN,
  financials) without accidentally loosening patterns that should stay
  case-sensitive.
- If the input text is empty or whitespace-only, `anonymize()` returns
  immediately with an empty result, skipping all sixteen-plus regex
  evaluations entirely.

### Output

`anonymize()` returns an `AnonymizationResult` containing:
- `scrubbed_text` — the fully anonymised document; this is what
  classification and scoring operate on
- `redaction_counts` — e.g. `{"[REDACTED_ADDRESS]": 109, "[REDACTED_FINANCIALS]": 55, ...}`
- `total_redactions` — sum across all categories
- `saved_to` — path to the scrubbed `.txt` file under `data/processed/`

The full anonymised text is downloadable from the UI via "Download
Anonymised Text" (`POST /anonymized`), which prepends a header
summarising the classification result and the full redaction breakdown.

---

## ESG Scoring Engine — detailed logic

The scoring engine (`src/components/scoring_engine.py`) runs on the
**anonymised** text. It is fundamentally a **disclosure-coverage**
measurement: it counts how many of the expected vocabulary terms for each
ESG topic actually appear in the document. It does not, and cannot,
verify whether the underlying claims are true, nor does it measure the
company's real-world environmental, social, or governance performance.

### The 16 sub-metrics and their keyword banks

| Pillar | Weight | Sub-metrics | Keywords | Target |
|---|---|---|---|---|
| **Environment** | 40% | Carbon Emissions | 17 | 10 |
| | | Energy Management | 16 | 9 |
| | | Water Management | 14 | 8 |
| | | Waste Management | 15 | 9 |
| | | Biodiversity | 15 | 9 |
| **Social** | 35% | Labour Rights | 14 | 8 |
| | | Employee Wellbeing | 15 | 9 |
| | | Health & Safety | 15 | 9 |
| | | Human Rights | 13 | 7 |
| | | Diversity & Inclusion | 14 | 8 |
| | | Community & CSR | 14 | 8 |
| **Governance** | 25% | Board Governance | 12 | 7 |
| | | Ethics & Integrity | 12 | 7 |
| | | Compliance | 14 | 8 |
| | | Risk Management | 12 | 7 |
| | | Disclosure & Transparency | 13 | 7 |

For example, the Carbon Emissions keyword bank includes `scope 1`,
`scope 2`, `scope 3`, `ghg`, `net zero`, `carbon footprint`,
`decarbonisation`, `paris agreement`, `induction furnace`, and others.

### Step 1 — Dynamic per-metric target

Rather than a single fixed threshold applied to all 16 metrics, each
metric's target scales with the size of its own keyword bank:

```python
target = max(5, min(12, round(len(keywords) * 0.6)))
```

A metric with a richer vocabulary (Carbon Emissions, 17 keywords, target
10) requires proportionally more hits for a perfect score than a sparser
metric (Human Rights, 13 keywords, target 7) — neither is artificially
easier or harder relative to how much vocabulary actually exists for that
topic. The target is clamped between 5 and 12 so that no metric becomes
trivially easy (target too low) or unreasonably strict (target too high).

### Step 2 — Negation-aware keyword matching

This is the most important correctness mechanism in the scoring engine,
and the reason a document that says "we have **no** carbon emissions
policy" does not score the same as one that says "we have a strong carbon
emissions policy" — even though both contain the keyword "carbon
emissions."

For every keyword match found in the text, the engine looks at the six
words immediately preceding that occurrence and checks them against a
negation word list (`no`, `not`, `none`, `never`, `lack of`, `absence of`,
`without`, `fails to`, `does not`, `insufficient`, `discontinued`, and
similar). If **every** occurrence of a keyword across the document is
preceded by a negation word within that window, the keyword is excluded
from the positive hit count entirely. If even one occurrence appears
without negation nearby, the keyword counts as a genuine positive hit.

```python
hits = 0
for kw in keywords:
    if keyword found in text AND not all_occurrences_negated(kw):
        hits += 1
```

This is implemented as a lightweight heuristic — a fixed-size word window
— rather than true dependency parsing. It correctly handles the common
patterns real reports use ("we have no formal grievance mechanism", "lack
of board diversity"), but it is not immune to harder cases like double
negatives or negation that occurs more than six words away from the
keyword. The window size (6 words) was chosen as a reasonable default and
has not been calibrated against a large corpus of real negated ESG
disclosures.

### Step 3 — Numeric value extraction (evidence, not scoring input)

Alongside hit counting, the engine separately scans the ~80 characters
immediately following each matched keyword for a nearby number, optionally
with a unit (`%`, `crores`, `lakhs`, `tonnes`, `kWh`, `GJ`, `kl`, etc.).
This is captured as supporting evidence — for example, finding "35%" near
a "paris agreement" mention, or "254" near "water withdrawal" — and is
exposed through the `/json` export and the evidence fields on the result
object (`e_evidence`, `s_evidence`, `g_evidence`).

Important: **this extracted number does not currently feed back into the
score calculation.** A document claiming "emissions reduced by 2%" scores
identically to one claiming "emissions reduced by 80%", because both
contain the same keyword hit. The numeric extraction exists purely to
give a human reviewer more context about what the document actually
claims, not to adjust the disclosure-coverage score itself.

### Step 4 — Sub-metric score

```python
score = min(hits / target, 1.0) * 100
```

A metric with target 10 that records 9 non-negated keyword hits scores
`9/10 x 100 = 90`. Hitting the target or exceeding it caps the score at
100 — there is no bonus for going beyond the target.

### Step 5 — Pillar score

Each pillar's score is the unweighted mean of its own sub-metric scores:

```python
environment_score = mean(5 Environmental sub-metric scores)
social_score       = mean(6 Social sub-metric scores)
governance_score   = mean(5 Governance sub-metric scores)
```

### Step 6 — Overall score

```python
overall = environment_score * 0.40
        + social_score      * 0.35
        + governance_score  * 0.25
```

### Step 7 — Letter grade

| Overall score | Grade |
|---|---|
| >= 85 | A+ |
| >= 75 | A |
| >= 65 | B+ |
| >= 55 | B |
| >= 45 | C |
| < 45 | D |

### Low-confidence flag

A separate, independent flag is set when a document's results may not be
reliable simply due to insufficient content to score against:

```python
low_confidence = (word_count < 200) or (total_keyword_hits_across_all_pillars < 6)
```

When `low_confidence` is `True`, the frontend displays a warning banner
and the PDF report's executive summary appends an explicit caveat
recommending manual review. This does not change any score — it is purely
an honesty signal that the scoring engine had very little text to work
with.

### Immutability and edge cases

`compute_scores()` returns an `ESGScoreResult`, a `frozen=True` dataclass
— once constructed, no downstream component (the API layer, the PDF
generator) can accidentally mutate the computed scores. If the input text
is empty or whitespace-only, the function short-circuits immediately and
returns a zeroed result (`overall_score=0`, `grade="D"`, empty metric and
evidence dictionaries, `low_confidence=True`) without running any regex
matching.

### What this score deliberately does and does not measure

**Does measure**: whether a document's language covers the topic areas
and uses the vocabulary that frameworks like SEBI BRSR, GRI, and TCFD
expect a complete disclosure to include, while correctly excluding
keyword matches that appear in a negative framing.

**Does not measure**: the company's actual real-world environmental,
social, or governance performance. A company with genuinely poor
emissions outcomes but a thorough, jargon-complete, non-negated report
can score highly — because the score reflects what the document discusses
and how it's framed, not what the company has actually achieved. This is
why the PDF report's recommendations section suggests pursuing
third-party assurance as a next step for high-scoring documents, rather
than treating the score itself as a performance certification.

### Why anonymization doesn't affect scoring

The 16 keyword banks consist entirely of generic ESG and compliance
vocabulary — terms like `scope 1`, `brsr`, `ltifr`, `csr`, `posh` — none
of which overlaps with the categories anonymization removes (names,
addresses, financial figures, ID numbers). This was verified directly:
running the full pipeline before and after a major anonymization rewrite
that added six new redaction categories produced identical ESG scores
(Overall 97.8/100, Grade A+) on the same test document, because none of
the newly-redacted text ever contained scoring-relevant keywords.

---

## Real-time pipeline progress

The frontend's progress indicator (Ingesting -> Anonymizing -> Classifying
-> Scoring) is driven by genuine Server-Sent Events, not a simulated
client-side timer. The flow is: the uploaded file is first staged via
`POST /analyse-stream/start`, which returns a `job_id`; the frontend then
opens `GET /analyse-stream/{job_id}`, which runs the real pipeline in a
background thread and pushes an event the instant each actual stage
begins. Each "stage started" event corresponds to a real function call
beginning in `predict_pipeline.run()`, not a guessed delay.

---

## Project structure

```
esg_app/
├── main.py                          FastAPI app
├── static/index.html                Single-page frontend (Chart.js)
├── src/
│   ├── logger.py                    Rich console + file logging
│   ├── exception.py                 ESGException with file/line capture
│   ├── components/
│   │   ├── data_ingestion.py        PDF/TXT -> raw text (pdfplumber)
│   │   ├── data_transformation.py   Pass A (NER) + Pass B (regex) anonymizer
│   │   ├── model_trainer.py         Multi-scale Text-CNN training
│   │   ├── model_inference.py       Loads .keras model + tokenizer
│   │   ├── scoring_engine.py        16-metric ESG scorecard with negation detection
│   │   ├── model_evaluation.py      Accuracy/loss + confusion matrix generation
│   │   └── pdf_reporter.py          ReportLab + Matplotlib PDF generation
│   └── pipeline/
│       └── predict_pipeline.py      Orchestrates all 4 stages, supports progress callbacks
├── models/                           esg_classifier.keras + tokenizer.pkl
├── evaluate_model.py                 Standalone script: accuracy/loss curves + confusion matrix
├── requirements.txt
├── Dockerfile                        HuggingFace Spaces-ready
└── data/
    ├── raw/                          Extracted raw text (gitignored)
    └── processed/                    Scrubbed text (gitignored)
```

## API routes

| Route | Method | Returns |
|---|---|---|
| `/` | GET | The single-page frontend |
| `/health` | GET | `{"status": "ok", "model_ready": bool}` |
| `/analyse` | POST | JSON — classification, ESG scores, evidence, PII counts, scrubbed preview |
| `/analyse-stream/start` | POST | `{"job_id": "..."}` — stages a file for streamed analysis |
| `/analyse-stream/{job_id}` | GET | Server-Sent Events stream of real pipeline stage progress, ending in the full result |
| `/json` | POST | Full analysis (scores, evidence, PII breakdown, metadata) as a downloadable `.json` |
| `/anonymized` | POST | Full anonymised text as a downloadable `.txt` |
| `/pdf` | POST | Full PDF report as a downloadable file |

## Model architecture

A multi-scale Text-CNN: embedding -> `SpatialDropout1D` -> three parallel
`Conv1D` branches at 2/3/4-gram widths, each with `GlobalMaxPooling1D` +
`GlobalAveragePooling1D` (hybrid pooling) -> concatenation ->
`BatchNormalization` -> two dense layers with L2 regularization and
dropout -> 3-class softmax (`SEBI_BRSR`, `SUSTAINABILITY_REPORT`,
`INVALID_DOCUMENT`).

Trained on 900 synthetic samples (300/class) with disjoint train/validation
sentence pools, typically reaching 100% validation accuracy within
7-8 epochs.

Run `python evaluate_model.py` to generate `accuracy_loss.png` and
`confusion_matrix.png` locally. Note: the saved `.keras` model does not
retain its original training history, so generating these performs a
short warm-start re-fit on the same dataset rather than an instant lookup.
