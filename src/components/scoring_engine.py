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

def _score_metrics(text: str, kw_dict: Dict[str, List[str]]) -> Dict[str, float]:
    text_l = text.lower()
    scores = {}
    for metric, keywords in kw_dict.items():
        target = _target(keywords)
        hits   = sum(1 for kw in keywords
                     if re.search(r"\b" + re.escape(kw) + r"\b", text_l,
                                  re.IGNORECASE))
        scores[metric] = round(min(hits / target, 1.0) * 100, 1)
    return scores

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

def compute_scores(text: str) -> ESGScoreResult:
    if not text or not text.strip():
        return ESGScoreResult(0,0,0,0,{},{},{},"D")
    e = _score_metrics(text, _ENV)
    s = _score_metrics(text, _SOC)
    g = _score_metrics(text, _GOV)
    e_sc = round(sum(e.values()) / len(e), 1)
    s_sc = round(sum(s.values()) / len(s), 1)
    g_sc = round(sum(g.values()) / len(g), 1)
    ov   = round(e_sc * 0.40 + s_sc * 0.35 + g_sc * 0.25, 1)
    return ESGScoreResult(ov, e_sc, s_sc, g_sc, e, s, g, _grade(ov))
