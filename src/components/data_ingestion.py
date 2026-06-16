"""src/components/data_ingestion.py — PDF / TXT / string ingestion."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import pdfplumber
from src.exception import ESGException
from src.logger import get_logger

log = get_logger(__name__)
_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

@dataclass
class IngestionResult:
    raw_text:   str
    source:     str
    char_count: int
    page_count: int
    saved_to:   str

def ingest(source: str | Path) -> IngestionResult:
    log.info("Ingestion started")
    try:
        src = str(source)
        if os.path.isfile(src):
            p, ext = Path(src), Path(src).suffix.lower()
            if ext == ".pdf":
                pages = []
                with pdfplumber.open(src) as pdf:
                    n = len(pdf.pages)
                    for pg in pdf.pages:
                        pages.append(pg.extract_text() or "")
                text = "\n".join(pages).strip()
                if not text:
                    raise ValueError(f"No text layer in '{p.name}'. May be scanned.")
            elif ext == ".txt":
                text = Path(src).read_text(encoding="utf-8", errors="ignore")
                n    = 0
            else:
                raise ValueError(f"Unsupported file type '{ext}'.")
            stem = p.stem
        else:
            if len(src) < 20:
                raise ValueError("Input too short or file not found.")
            text, n, stem = src, 0, "raw_input"

        saved = os.path.join(RAW_DIR, f"{stem}_raw.txt")
        Path(saved).write_text(text, encoding="utf-8")
        log.info("Ingested %d chars, %d pages", len(text), n)
        return IngestionResult(text, stem, len(text), n, saved)
    except Exception as e:
        raise ESGException(e) from e
