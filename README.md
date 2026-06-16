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
Sustainability Report PDF. The pipeline:

1. **Ingests** PDF or TXT files
2. **Anonymizes** PII — person names, organisations, financials, CIN, PAN,
   GSTIN, IFSC, Aadhaar, bank accounts, emails, phone numbers, websites,
   addresses, and dates
3. **Classifies** the document (`SEBI_BRSR` / `SUSTAINABILITY_REPORT` / `INVALID_DOCUMENT`)
4. **Scores** ESG performance across 16 sub-metrics
   (Environment × 40% + Social × 35% + Governance × 25%)
5. Generates a **PDF report** with charts and written analysis, and a
   downloadable **anonymised text** file

---

## Quick start

```bash
pip install -r requirements.txt
python -m src.components.model_trainer   # train once (~60s)
uvicorn main:app --reload --port 7860
```

Open http://localhost:7860

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
4. ESG Scoring       Keyword coverage -> 16 sub-metrics -> 3 pillars -> overall score
     |
     v
JSON response, PDF report, anonymised .txt download
```

Each stage operates on the **output of the previous stage** — notably,
classification and scoring run on the *anonymised* text, not the raw text.

---

## PII Anonymization — how it works

The anonymizer (`src/components/data_transformation.py`) runs a **two-pass
pipeline**. Pass A handles linguistic patterns (names, organisations);
Pass B handles deterministic structural patterns (ID numbers, contact
details, addresses). Pass A always runs first — running structural regex
first would risk fragmenting names and organisation strings before the
linguistic rules get a chance to see them whole.

### Pass A — Linguistic NER (rule-based, no ML model required)

| Rule | What it catches | Example |
|---|---|---|
| **Honorific-anchored names** | `Mr./Mrs./Dr./Shri./Prof./CA./CEO` + capitalised name (1-4 words) | `Dr. Sanjay Kirloskar` -> `[REDACTED_PERSON]` |
| **Designation-anchored names** | A capitalised name immediately followed by a dash/comma and a job title (Director, CEO, Manager, Chairman, Trustee, Auditor, etc.) | `Abhijeet Shinde - Assistant General Manager` -> `[REDACTED_PERSON]` |
| **Corporate-suffix organisations** | 1-6 capitalised words ending in `Limited/Ltd./LLP/Pvt. Ltd./Inc./Corp./Group/Foundation/...` | `Kirloskar Brothers Limited` -> `[REDACTED_ORG]` |
| **Named institutions** | BSE, NSE, SEBI, RBI, Ministry of X, Registrar of Companies (with optional trailing "Limited") | `BSE Limited` -> `[REDACTED_ORG]` |

These rules use `subn()` for atomic count-and-replace, and are anchored
with `\b` word boundaries on both sides — this prevents the classic
mid-word corruption bug where "Care" inside a sentence gets partially
swallowed into `[REDACTED_ORG]re`.

### Pass B — Deterministic regex (structural identifiers)

| Tag | What it catches | Notes |
|---|---|---|
| `[REDACTED_CIN]` | Corporate Identity Number (e.g. `L29113PN1920PLC000670`) | Case-insensitive for OCR variants |
| `[REDACTED_GSTIN]` | 15-character GST Identification Number | Checked before PAN (GSTIN embeds a PAN-like substring) |
| `[REDACTED_PAN]` | Permanent Account Number (e.g. `ABCDE1234F`) | Case-insensitive |
| `[REDACTED_IFSC]` | Bank branch IFSC code | 4 letters + `0` + 6 alphanumeric |
| `[REDACTED_ACCOUNT]` | IBAN, and standalone 9-18 digit bank account numbers | |
| `[REDACTED_AADHAAR]` | 12-digit Aadhaar number in `XXXX XXXX XXXX` format | |
| `[REDACTED_FINANCIALS]` | Monetary figures — `₹`, `Rs.`, `` ` `` (OCR artifact for ₹), `INR`, `$`, `€`, `£`, crores, lakhs, millions, billions | |
| `[REDACTED_EMAIL]` | Standard email addresses | |
| `[REDACTED_WEBSITE]` | `www.*` and `http(s)://*` URLs | |
| `[REDACTED_PHONE]` | Indian (`+91`, STD codes, toll-free `1800`) and international phone formats | |
| `[REDACTED_ADDRESS]` | Quoted building names before an address (e.g. `"Yamuna"`), `Survey No./Plot No./Flat No.` blocks, Indian state/city names, and standalone 6-digit PIN codes | Four sub-rules under one tag |
| `[REDACTED_DATE]` | `DD/MM/YYYY` or `DD-MM-YYYY` dates | For DOB / date-of-joining fields |

### Ordering matters

Two specific orderings are load-bearing:

1. **GSTIN before PAN** — a GSTIN contains a 10-character PAN-like
   substring; matching PAN first would leave fragments of the GSTIN behind.
2. **Quoted building name before "Survey No." address block** — the
   pattern `"Yamuna", Survey No. 98...` requires a lookahead for "Survey"
   immediately after the closing quote. If the Survey No. rule ran first,
   it would consume `, Survey No. 98...` and the building-name rule would
   have nothing left to look ahead at.

### Safety guards

- **Empty input**: if the text is empty or whitespace-only, `anonymize()`
  returns immediately with an empty result — no regex runs on nothing.
- **Atomic counting**: every pattern uses `pattern.subn(tag, text)`, which
  returns `(new_text, count)` in one pass — no separate `findall()` +
  `sub()` calls that could double-scan large documents.

### Output

`anonymize()` returns an `AnonymizationResult` with:
- `scrubbed_text` — the fully anonymised document, used by classification
  and scoring
- `redaction_counts` — a dict like `{"[REDACTED_ADDRESS]": 109, "[REDACTED_FINANCIALS]": 55, ...}`
- `total_redactions` — sum of all counts
- `saved_to` — path to the saved scrubbed `.txt` file in `data/processed/`

The full anonymised text can be downloaded via the **"Download Anonymised
Text"** button in the UI (`POST /anonymized`), which also includes a header
summarising the classification and redaction breakdown.

---

## ESG Scoring Engine — how it works

The scoring engine (`src/components/scoring_engine.py`) runs on the
**anonymised** text and is a **disclosure-coverage** model: it measures
*how many of the expected vocabulary terms for each topic appear in the
document*, not the underlying real-world performance those terms describe.

### The 16 sub-metrics

| Pillar | Weight | Sub-metrics |
|---|---|---|
| **Environment** | 40% | Carbon Emissions, Energy Management, Water Management, Waste Management, Biodiversity |
| **Social** | 35% | Labour Rights, Employee Wellbeing, Health & Safety, Human Rights, Diversity & Inclusion, Community & CSR |
| **Governance** | 25% | Board Governance, Ethics & Integrity, Compliance, Risk Management, Disclosure & Transparency |

Each sub-metric has a hand-curated keyword bank of 12-17 terms. For
example, **Carbon Emissions** (17 keywords) includes `scope 1`, `scope 2`,
`scope 3`, `ghg`, `net zero`, `carbon footprint`, `decarbonisation`,
`paris agreement`, `induction furnace`, and more.

### Step 1 — Dynamic target per metric

Rather than one fixed threshold for all 16 metrics, each metric's target
is derived from the size of its own keyword bank:

```python
target = max(5, min(12, round(len(keywords) * 0.6)))
```

| Metric | Keywords | Target |
|---|---|---|
| Carbon Emissions | 17 | 10 |
| Energy Management | 16 | 9 |
| Water Management | 14 | 8 |
| Waste Management | 15 | 9 |
| Biodiversity | 15 | 9 |
| Labour Rights | 14 | 8 |
| Employee Wellbeing | 15 | 9 |
| Health & Safety | 15 | 9 |
| Human Rights | 13 | 7 |
| Diversity & Inclusion | 14 | 8 |
| Community & CSR | 14 | 8 |
| Board Governance | 12 | 7 |
| Ethics & Integrity | 12 | 7 |
| Compliance | 14 | 8 |
| Risk Management | 12 | 7 |
| Disclosure & Transparency | 13 | 7 |

A metric with a larger vocabulary (e.g. Carbon Emissions, target 10) needs
proportionally more hits for a perfect score than a metric with a smaller
vocabulary (e.g. Human Rights, target 7) — both are achievable, but neither
is artificially easier or harder relative to its own bank size.

### Step 2 — Count keyword hits

The scrubbed text is lowercased and each keyword is searched with a
word-boundary regex, case-insensitive:

```python
hits = sum(
    1 for kw in keywords
    if re.search(r"\b" + re.escape(kw) + r"\b", text_lower, re.IGNORECASE)
)
```

Each keyword counts **once** if present anywhere — frequency doesn't
matter, only presence/absence.

### Step 3 — Sub-metric score

```python
score = min(hits / target, 1.0) * 100
```

If a document hits 9 of the 10-target keywords for Carbon Emissions, the
score is `9/10 x 100 = 90`. Hitting the target or more caps at 100 — there
is no bonus beyond the target.

### Step 4 — Pillar scores

Each pillar score is the unweighted average of its sub-metric scores:

```python
environment_score = mean(5 E-metric scores)
social_score       = mean(6 S-metric scores)
governance_score   = mean(5 G-metric scores)
```

### Step 5 — Overall score

```python
overall = environment_score * 0.40
        + social_score      * 0.35
        + governance_score  * 0.25
```

### Step 6 — Letter grade

| Overall score | Grade |
|---|---|
| >= 85 | A+ |
| >= 75 | A |
| >= 65 | B+ |
| >= 55 | B |
| >= 45 | C |
| < 45 | D |

### Output and immutability

`compute_scores()` returns an `ESGScoreResult` — a `frozen=True` dataclass
(read-only once constructed) containing `overall_score`, `environment_score`,
`social_score`, `governance_score`, `grade`, and the three per-pillar
metric dictionaries (`e_metrics`, `s_metrics`, `g_metrics`).

If the input text is empty or whitespace-only, the function short-circuits
and returns a zeroed result (`overall_score=0`, `grade="D"`, empty metric
dicts) — avoiding division-by-zero on empty searches.

### What this score does and does not measure

**Measures**: whether the document uses the vocabulary and covers the
topic areas that regulatory frameworks (SEBI BRSR, GRI, TCFD) expect to
see in a complete disclosure.

**Does not measure**: whether the company's actual environmental, social,
or governance performance is good — a company with poor real-world
emissions but a thorough, jargon-complete report can score highly, because
the score reflects *what the document talks about*, not *what the company
actually does*. The PDF report's recommendations section reflects this —
high scores prompt a suggestion to "pursue third-party assurance" as the
next credibility step, rather than treating the score itself as a
performance certification.

### Why anonymization doesn't affect scoring

The scoring engine's 16 keyword banks consist entirely of generic ESG /
compliance vocabulary (`scope 1`, `brsr`, `ltifr`, `csr`, `posh`, etc.).
None of this vocabulary overlaps with the categories anonymization removes
(names, addresses, financial figures, ID numbers). Verified empirically:
running the pipeline before and after a major anonymization rewrite (which
added 6 new redaction categories) produced **identical ESG scores**
(Overall 97.8/100, Grade A+) on the same test document.

---

## Project structure

```
esg_app/
├── main.py                          FastAPI app - 4 routes
├── static/index.html                Single-page frontend (Chart.js)
├── src/
│   ├── logger.py                    Rich console + file logging
│   ├── exception.py                 ESGException with file/line capture
│   ├── components/
│   │   ├── data_ingestion.py        PDF/TXT -> raw text (pdfplumber)
│   │   ├── data_transformation.py   Pass A (NER) + Pass B (regex) anonymizer
│   │   ├── model_trainer.py         Multi-scale Text-CNN training
│   │   ├── model_inference.py       Loads .keras model + tokenizer
│   │   ├── scoring_engine.py        16-metric ESG scorecard
│   │   └── pdf_reporter.py          ReportLab + Matplotlib PDF generation
│   └── pipeline/
│       └── predict_pipeline.py      Orchestrates all 4 stages
├── models/                           esg_classifier.keras + tokenizer.pkl
├── evaluate_model.py                 Accuracy/loss curves + confusion matrix
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
| `/analyse` | POST | JSON - classification, ESG scores, PII counts, scrubbed preview |
| `/anonymized` | POST | Full anonymised text as a downloadable `.txt` |
| `/pdf` | POST | Full PDF report as a downloadable file |

## Model architecture

A multi-scale Text-CNN: embedding -> `SpatialDropout1D` -> three parallel
`Conv1D` branches at 2/3/4-gram widths, each with `GlobalMaxPooling1D` +
`GlobalAveragePooling1D` (hybrid pooling) -> concatenation ->
`BatchNormalization` -> two dense layers with L2 regularization and dropout
-> 3-class softmax (`SEBI_BRSR`, `SUSTAINABILITY_REPORT`, `INVALID_DOCUMENT`).

Trained on 900 synthetic samples (300/class) with disjoint train/validation
sentence pools, typically reaching 100% validation accuracy within
7-8 epochs.

Run `python evaluate_model.py` to generate `accuracy_loss.png` and
`confusion_matrix.png`.
