"""
src/components/data_transformation.py

from __future__ import annotations
import os, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from src.exception import ESGException
from src.logger import get_logger

log = get_logger(__name__)
_ROOT         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# PASS A — Linguistic NER
# ══════════════════════════════════════════════════════════════════════════════

# ── A1: Honorific-anchored person names ───────────────────────────────────────
_HON = (r"(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?|Shri\.?|Smt\.?|"
        r"Sh\.?|Er\.?|Adv\.?|CA\.?|CS\.?|Capt\.?|Col\.?|Sir|Madam)")
_NW  = r"[A-Z][a-zA-Z'\-]{1,30}"
_PERSON_HONORIFIC_RE = re.compile(
    rf"\b{_HON}\.?\s+{_NW}(?:\s+[A-Z]\.?)?(?:\s+{_NW}){{0,3}}",
    re.UNICODE,
)

# ── A2: Designation-anchored person names ──────────────────────────────────────
# Catches "Abhijeet Shinde – Assistant General Manager" or
# "Sanjay Kirloskar, Chairman and Managing Director"
_DESIGNATIONS = (
    r"(?:Chairman|Chairperson|Managing Director|Director|CEO|CFO|COO|CTO|"
    r"CMD|President|Vice\s+President|Manager|General\s+Manager|"
    r"Company\s+Secretary|Chief\s+\w+\s+Officer|Head\s+of\s+\w+|"
    r"Executive|Officer|Whole[- ]time\s+Director|Trustee|Auditor|"
    r"Assistant\s+General\s+Manager)"
)
# Name (2-4 capitalised words), then a dash/comma, then a designation
_PERSON_DESIGNATION_RE = re.compile(
    rf"\b(?:{_NW}(?:\s+{_NW}){{1,3}})"
    rf"(?=\s*[–—\-,]\s*(?:{_DESIGNATIONS}))",
    re.UNICODE,
)

# ── A3: Corporate-suffix anchored ORG names ────────────────────────────────────
_ORG_SFX = (r"(?:Limited|Ltd\.?|Pvt\.?\s*Ltd\.?|Private\s+Limited|LLP|LLC|"
            r"Inc\.?|Corp\.?|Corporation|PLC|B\.?V\.?|GmbH|Associates|Partners|"
            r"Consultants|Advisors|Industries|Enterprises|Holdings|Group|"
            r"Solutions|Services|Foundation|Trust|Society|Institute)")
# Require at least one capitalised word boundary before the suffix,
# and stop at sentence-ending punctuation / lowercase-run boundaries
_ORG_RE = re.compile(
    rf"\b(?:[A-Z][A-Za-z&.\-]*(?:\s+(?:of|and|&|the)\s+|\s+)){{1,6}}"
    rf"{_ORG_SFX}\b(?:\s+{_ORG_SFX}\b)?",
    re.UNICODE,
)

# ── A4: Named stock exchanges / regulators / institutions ─────────────────────
# Includes optional trailing "Limited"/"Ltd." so "BSE Limited" is fully masked.
_INSTITUTION_RE = re.compile(
    r"\b(?:BSE|NSE|National\s+Stock\s+Exchange(?:\s+of\s+India)?|"
    r"Bombay\s+Stock\s+Exchange|SEBI|Ministry\s+of\s+\w+(?:\s+\w+)?|"
    r"Reserve\s+Bank\s+of\s+India|Registrar\s+of\s+Companies)"
    rf"\b(?:\s+{_ORG_SFX}\b)?",
    re.UNICODE,
)


# ══════════════════════════════════════════════════════════════════════════════
# PASS B — Deterministic Regex
# ══════════════════════════════════════════════════════════════════════════════

# Indian states / major cities — used to redact location mentions in addresses
_INDIAN_PLACES = (
    r"(?:Maharashtra|Karnataka|Tamil\s+Nadu|Kerala|Gujarat|Rajasthan|Punjab|"
    r"Haryana|Uttar\s+Pradesh|Madhya\s+Pradesh|West\s+Bengal|Bihar|Odisha|"
    r"Telangana|Andhra\s+Pradesh|Assam|Jharkhand|Chhattisgarh|Goa|"
    r"Pune|Mumbai|Bengaluru|Bangalore|Chennai|Hyderabad|Kolkata|Delhi|"
    r"Ahmedabad|Surat|Jaipur|Lucknow|Kanpur|Nagpur|Indore|Thane|Bhopal|"
    r"Visakhapatnam|Patna|Vadodara|Ghaziabad|Coimbatore|Kochi|Chandigarh|"
    r"Gurugram|Gurgaon|Noida|Baner)"
)

# Each entry: (tag, pattern_string, extra_flags)
_RAW: List[Tuple[str, str, int]] = [

    # 1 — CIN (Corporate Identity Number) — case-insensitive for OCR variants
    ("[REDACTED_CIN]",
     r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b",
     re.IGNORECASE),

    # 2 — GSTIN (15-char: 2-digit state + PAN + entity code + Z + checksum)
    ("[REDACTED_GSTIN]",
     r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b",
     re.IGNORECASE),

    # 3 — PAN (Permanent Account Number) — must come after GSTIN check
    ("[REDACTED_PAN]",
     r"\b[A-Z]{5}\d{4}[A-Z]\b",
     re.IGNORECASE),

    # 4 — IFSC code (bank branch identifier)
    ("[REDACTED_IFSC]",
     r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
     re.IGNORECASE),

    # 5 — IBAN
    ("[REDACTED_ACCOUNT]",
     r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
     0),

    # 6 — Aadhaar number (12 digits, often space/dash separated in groups of 4)
    ("[REDACTED_AADHAAR]",
     r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}\b",
     0),

    # 7 — Bank account: 9-18 digit standalone numbers (not already matched)
    ("[REDACTED_ACCOUNT]",
     r"(?<!\d)\d{9,18}(?!\d)",
     0),

    # 8 — Financial figures (₹ / Rs / ` (OCR artifact for ₹) / INR / $ / crores / lakhs)
    ("[REDACTED_FINANCIALS]",
     r"(?:(?:₹|Rs\.?|INR|USD|GBP|EUR|\$|€|£|`)[\s,]*\d[\d,\.]*"
     r"(?:\s*(?:crores?|lakhs?|millions?|billions?|mn|bn|cr|k))?)"
     r"|(?:\d[\d,\.]{3,}\s*(?:crores?|lakhs?|millions?|billions?))",
     re.IGNORECASE),

    # 9 — Email addresses
    ("[REDACTED_EMAIL]",
     r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
     0),

    # 10 — Websites / URLs
    ("[REDACTED_WEBSITE]",
     r"\b(?:https?://)?(?:www\.)[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/\S*)?\b",
     re.IGNORECASE),

    # 11 — Phone numbers (Indian + international)
    ("[REDACTED_PHONE]",
     r"(?:\+91[\s\-]?(?:\(\d{2,4}\)[\s\-]?)?\d{4}[\s\-]?\d{4,6})"
     r"|(?:0\d{2,4}[\s\-]\d{6,8})"
     r"|(?:1800[\s\-]\d{3}[\s\-]\d{4})"
     r"|(?:\+\d{1,3}[\s\-]\d{3}[\s\-]\d{3}[\s\-]\d{4})",
     0),

    # 12 — Quoted building/property name immediately preceding an address
    # (e.g. "Yamuna", Survey No. 98 ... -> catches the quoted name itself)
    # Must run BEFORE the Survey/Plot No. pattern, which would otherwise
    # consume the comma+"Survey No" text that this lookahead depends on.
    ("[REDACTED_ADDRESS]",
     r"[\u201c\"']([A-Z][a-zA-Z]+)[\u201d\"'](?=\s*,\s*(?:Survey|Plot|Flat|Door|Block|S\.?\s*No))",
     0),

    # 13 — Address blocks anchored on Survey/Plot/Door/Flat/Block/Unit No.
    ("[REDACTED_ADDRESS]",
     r"(?:(?:Survey\s+No\.?|Plot\s+No\.?|Flat\s+No\.?|"
     r"Door\s+No\.?|Block\s+No\.?|S\.?\s*No\.?|Unit\s+No\.?|Wing\s*[A-Z]?)"
     r"[\s\w/,\(\)\-\.]{3,80})",
     re.IGNORECASE),

    # 14 — Indian city / state names (location component of addresses)
    ("[REDACTED_ADDRESS]",
     rf"\b{_INDIAN_PLACES}\b",
     0),

    # 15 — PIN code (standalone 6-digit Indian postal code)
    ("[REDACTED_ADDRESS]",
     r"(?<!\d)\d{6}(?!\d)",
     0),

    # 16 — Date of birth / joining (DD/MM/YYYY or DD-MM-YYYY)
    ("[REDACTED_DATE]",
     r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\b",
     0),
]

_COMPILED: List[Tuple[str, re.Pattern]] = [
    (tag, re.compile(pat, re.UNICODE | flags))
    for tag, pat, flags in _RAW
]


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnonymizationResult:
    scrubbed_text:    str
    redaction_counts: dict = field(default_factory=dict)
    total_redactions: int  = 0
    saved_to:         str  = ""


# ══════════════════════════════════════════════════════════════════════════════
# Anonymize
# ══════════════════════════════════════════════════════════════════════════════

def anonymize(raw_text: str, source_stem: str = "document") -> AnonymizationResult:
    if not raw_text or not raw_text.strip():
        log.warning("Empty input — returning empty AnonymizationResult")
        return AnonymizationResult("", {}, 0, "")

    log.info("Anonymization started (len=%d)", len(raw_text))
    try:
        counts: dict = {}
        text = raw_text

        # ── Pass A: Linguistic NER ──────────────────────────────────────────
        # Order matters: designation-anchored names first (longer match),
        # then honorific names, then institutions, then ORGs.
        text, n = _PERSON_DESIGNATION_RE.subn("[REDACTED_PERSON]", text)
        if n: counts["[REDACTED_PERSON]"] = counts.get("[REDACTED_PERSON]", 0) + n

        text, n = _PERSON_HONORIFIC_RE.subn("[REDACTED_PERSON]", text)
        if n: counts["[REDACTED_PERSON]"] = counts.get("[REDACTED_PERSON]", 0) + n

        text, n = _INSTITUTION_RE.subn("[REDACTED_ORG]", text)
        if n: counts["[REDACTED_ORG]"] = counts.get("[REDACTED_ORG]", 0) + n

        text, n = _ORG_RE.subn("[REDACTED_ORG]", text)
        if n: counts["[REDACTED_ORG]"] = counts.get("[REDACTED_ORG]", 0) + n

        # ── Pass B: Deterministic regex ─────────────────────────────────────
        for tag, pat in _COMPILED:
            text, n = pat.subn(tag, text)
            if n:
                counts[tag] = counts.get(tag, 0) + n

        total = sum(counts.values())
        out   = os.path.join(PROCESSED_DIR, f"{source_stem}_scrubbed.txt")
        Path(out).write_text(text, encoding="utf-8")

        log.info("Anonymization done — %d total redactions across %d categories → %s",
                 total, len(counts), out)
        return AnonymizationResult(text, counts, total, out)
    except Exception as e:
        raise ESGException(e) from e
