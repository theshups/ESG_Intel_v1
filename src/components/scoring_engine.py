"""
src/components/scoring_engine.py
=================================
Optimizations applied:
- frozen=True on ESGScoreResult (immutability protection)
- Dynamic per-metric targets (replaces global hardcoded _ENV_TARGET=8)
- re.IGNORECASE in keyword matching
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List

_ENV: Dict[str, List[str]] = {
    "Carbon Emissions": [
        "scope 1","scope 2","scope 3","ghg","carbon","co2","emissions","net zero",
        "carbon neutral","decarbonisation","greenhouse gas","emission intensity",
        "climate","paris agreement","induction furnace","cupola","carbon footprint",
    ],
    "Energy Management": [
        "renewable energy","solar","wind","energy consumption","energy intensity",
        "energy efficiency","non-renewable","electricity","gigajoule","power purchase",
        "ppa","open access","encon","energy management","fuel consumption","kwh",
    ],
    "Water Management": [
        "water withdrawal","water consumption","water intensity","kilolitre","groundwater",
        "surface water","zero liquid discharge","effluent","sewage treatment",
        "water neutrality","water stress","watershed","water reuse","rainwater",
    ],
    "Waste Management": [
        "waste","recycle","recycled","reuse","landfill","hazardous","e-waste",
        "plastic waste","zero waste","co-processing","incineration","reduce reuse recycle",
        "battery waste","metal scrap","waste diversion",
    ],
    "Biodiversity": [
        "biodiversity","ecosystem","plantation","afforestation","flora","fauna",
        "ecologically sensitive","national park","wildlife","greenpro","greenco",
        "lca","life cycle","miyawaki","nature positive",
    ],
}
_SOC: Dict[str, List[str]] = {
    "Labour Rights": [
        "minimum wage","wages","equal pay","pay gap","fair wage","living wage",
        "collective bargaining","union","workers","permanent employees","contract workers",
        "labour rights","overtime","employment contract",
    ],
    "Employee Wellbeing": [
        "health insurance","accident insurance","maternity","paternity","day care",
        "wellbeing","wellness","employee engagement","pf","gratuity","esi","pension",
        "retirement","skill development","employee assistance",
    ],
    "Health & Safety": [
        "iso 45001","ohs","ltifr","lost time injury","fatality","safety committee",
        "hira","hazard","near miss","fire drill","safety audit","mock drill",
        "safety training","safety yellow tag","suraksha mitra",
    ],
    "Human Rights": [
        "human rights","child labour","forced labour","posh","sexual harassment",
        "discrimination","whistle blower","grievance","conflict of interest",
        "due diligence","modern slavery","anti-trafficking","code of conduct",
    ],
    "Diversity & Inclusion": [
        "diversity","inclusion","women","female","gender","differently abled",
        "disability","marginalised","equal opportunity","dei","representation",
        "board diversity","lgbtq","gender pay",
    ],
    "Community & CSR": [
        "csr","community","social impact","beneficiaries","healthcare","education",
        "rural development","livelihood","ngo","skill development","rehabilitation",
        "aspirational","social investment","community engagement",
    ],
}
_GOV: Dict[str, List[str]] = {
    "Board Governance": [
        "board of directors","committee","chairman","managing director",
        "independent director","audit committee","nomination remuneration",
        "board oversight","governance","board diversity","quorum","board meetings",
    ],
    "Ethics & Integrity": [
        "anti-corruption","anti-bribery","code of conduct","ethics","integrity",
        "whistle blower","conflict of interest","disciplinary action",
        "zero tolerance","transparency","accountability","speak up",
    ],
    "Compliance": [
        "compliance","statutory","sebi","companies act","regulation",
        "non-compliance","adjudication","penalty","fine","compounding",
        "legal","secretarial audit","statutory audit","legatrix",
    ],
    "Risk Management": [
        "risk management","enterprise risk","erm","material issues","risk appetite",
        "risk mitigation","business continuity","cyber security","data privacy",
        "resilience","risk register","scenario analysis",
    ],
    "Disclosure & Transparency": [
        "disclosure","transparency","reporting","brsr","annual report",
        "integrated report","assurance","external audit","third party",
        "verified","material disclosures","sebi notification","gri",
    ],
}

# Dynamic per-metric targets based on keyword bank size
def _target(keywords: List[str]) -> int:
    """Dynamic target: 60% of keyword bank size, min 5, max 12."""
    return max(5, min(12, int(len(keywords) * 0.6)))


# ── Negation detection ─────────────────────────────────────────────────────────
# Words that, when found within a short window BEFORE a keyword, flip a
# positive disclosure ("we have a policy") into a negative one
# ("we do NOT have a policy") — such hits should not count toward the score.
_NEGATION_WORDS = (
    r"no|not|none|never|lack(?:s|ing)?|absence(?:\s+of)?|"
    r"without|fails?\s+to|failed\s+to|unable\s+to|"
    r"does\s+not|do\s+not|did\s+not|doesn't|don't|didn't|"
    r"insufficient|inadequate|zero\s+progress|"
    r"discontinued|ceased|stopped"
)
_NEGATION_WINDOW_WORDS = 6   # how many words before the keyword we scan

def _is_negated(text_l: str, keyword: str) -> bool:
    """
    Check whether `keyword`'s occurrences in `text_l` are preceded by a
    negation word within a short window. If EVERY occurrence is negated,
    the keyword is treated as a negative disclosure and excluded from
    the positive hit count.

    This is a lightweight heuristic, not full NLP negation scope parsing —
    it catches the common patterns reports actually use:
        "we have NO carbon emissions policy"      -> negated
        "lack of a formal grievance mechanism"     -> negated
        "we have a strong carbon emissions policy" -> NOT negated
    """
    kw_pattern = re.escape(keyword)
    # Find every occurrence of the keyword
    occurrences = list(re.finditer(r"\b" + kw_pattern + r"\b", text_l))
    if not occurrences:
        return False

    neg_re = re.compile(r"\b(?:" + _NEGATION_WORDS + r")\b")

    for m in occurrences:
        # Look at a window of words immediately before this occurrence
        window_start = max(0, m.start() - 60)   # ~60 chars back as a proxy
        window_text  = text_l[window_start:m.start()]
        window_words = window_text.split()[-_NEGATION_WINDOW_WORDS:]
        window_str   = " ".join(window_words)

        if not neg_re.search(window_str):
            # Found at least one NON-negated occurrence -> count it as positive
            return False

    # Every single occurrence was negated
    return True


# ── Numeric value extraction ───────────────────────────────────────────────────
# Pulls out a representative number/percentage near a matched keyword, so the
# report can show "emissions -35%" instead of just "emissions: keyword found".
_NUMERIC_RE = re.compile(
    r"(\d{1,3}(?:[.,]\d+)?)\s*(%|percent|per\s*cent|"
    r"crores?|lakhs?|million|billion|tonnes?|mt|kl|kwh|gj)?",
    re.IGNORECASE,
)

def _nearby_numeric(text_l: str, keyword: str, window_chars: int = 80) -> str | None:
    """
    Return the first numeric value (with unit, if any) found within
    `window_chars` characters AFTER the first occurrence of `keyword`.
    Returns None if no number is nearby.
    """
    m = re.search(r"\b" + re.escape(keyword) + r"\b", text_l)
    if not m:
        return None
    window = text_l[m.end(): m.end() + window_chars]
    num_match = _NUMERIC_RE.search(window)
    if not num_match:
        return None
    value, unit = num_match.group(1), num_match.group(2) or ""
    return f"{value}{unit}".strip()


def _score_metrics(text: str, kw_dict: Dict[str, List[str]]) -> Dict[str, float]:
    text_l = text.lower()
    scores = {}
    for metric, keywords in kw_dict.items():
        target = _target(keywords)
        hits = 0
        for kw in keywords:
            found = re.search(r"\b" + re.escape(kw) + r"\b", text_l, re.IGNORECASE)
            if not found:
                continue
            if _is_negated(text_l, kw):
                continue   # negated disclosure -> does not count as a positive hit
            hits += 1
        scores[metric] = round(min(hits / target, 1.0) * 100, 1)
    return scores


def _metric_evidence(text: str, kw_dict: Dict[str, List[str]]) -> Dict[str, dict]:
    """
    For each metric, return supporting evidence: which keywords matched
    (excluding negated ones), which were negated, and any nearby numeric
    values found. This powers richer reporting without changing the
    underlying score calculation.
    """
    text_l = text.lower()
    evidence = {}
    for metric, keywords in kw_dict.items():
        matched, negated, numerics = [], [], {}
        for kw in keywords:
            found = re.search(r"\b" + re.escape(kw) + r"\b", text_l, re.IGNORECASE)
            if not found:
                continue
            if _is_negated(text_l, kw):
                negated.append(kw)
                continue
            matched.append(kw)
            num = _nearby_numeric(text_l, kw)
            if num:
                numerics[kw] = num
        evidence[metric] = {
            "matched_keywords": matched,
            "negated_keywords": negated,
            "numeric_values":   numerics,
        }
    return evidence


def _grade(s: float) -> str:
    if s >= 85: return "A+"
    if s >= 75: return "A"
    if s >= 65: return "B+"
    if s >= 55: return "B"
    if s >= 45: return "C"
    return "D"

@dataclass(frozen=True)          # immutability protection (optimization)
class ESGScoreResult:
    overall_score:     float
    environment_score: float
    social_score:      float
    governance_score:  float
    e_metrics:         dict
    s_metrics:         dict
    g_metrics:         dict
    grade:             str
    e_evidence:         dict = None
    s_evidence:         dict = None
    g_evidence:         dict = None
    low_confidence:      bool = False

def compute_scores(text: str) -> ESGScoreResult:
    if not text or not text.strip():
        return ESGScoreResult(0, 0, 0, 0, {}, {}, {}, "D", {}, {}, {}, True)

    e = _score_metrics(text, _ENV)
    s = _score_metrics(text, _SOC)
    g = _score_metrics(text, _GOV)

    e_ev = _metric_evidence(text, _ENV)
    s_ev = _metric_evidence(text, _SOC)
    g_ev = _metric_evidence(text, _GOV)

    e_sc = round(sum(e.values()) / len(e), 1)
    s_sc = round(sum(s.values()) / len(s), 1)
    g_sc = round(sum(g.values()) / len(g), 1)
    ov   = round(e_sc * 0.40 + s_sc * 0.35 + g_sc * 0.25, 1)

    # Flag very short / sparse documents as low-confidence scores —
    # a handful of keyword hits on a short text shouldn't be presented
    # with the same confidence as a hit-rate computed over a full report.
    word_count     = len(text.split())
    total_hits     = sum(len(v["matched_keywords"]) for v in {**e_ev, **s_ev, **g_ev}.values())
    low_confidence = word_count < 200 or total_hits < 6

    return ESGScoreResult(
        ov, e_sc, s_sc, g_sc, e, s, g, _grade(ov),
        e_ev, s_ev, g_ev, low_confidence,
    )

