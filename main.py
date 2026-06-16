"""
main.py — FastAPI backend for ESG Document Intelligence
Run locally:  uvicorn main:app --reload --port 7860
HuggingFace:  runs on port 7860 automatically
"""
from __future__ import annotations
import os, sys, tempfile, logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

app = FastAPI(title="ESG Document Intelligence")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── serve static files ────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(Path(_HERE) / "static")), name="static")

# ── lazy-load pipeline ────────────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline.predict_pipeline import run
        _pipeline = run
    return _pipeline

# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(_HERE) / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": (Path(_HERE) / "models" / "esg_classifier.keras").exists(),
    }

@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(400, f"Only .pdf and .txt files are supported. Got '{ext}'.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = get_pipeline()(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")
    finally:
        try: os.unlink(tmp_path)
        except: pass

    sc   = result.scores
    clf  = result.inference
    anon = result.anonymization
    ing  = result.ingestion

    # Scrubbed preview — first 2000 chars, safe for JSON
    scrub_preview = anon.scrubbed_text[:2000]
    if len(anon.scrubbed_text) > 2000:
        scrub_preview += "\n… [truncated]"

    return JSONResponse({
        "source":              file.filename or "upload",
        "classification":      clf.predicted_class,
        "confidence":          round(clf.confidence, 4),
        "all_probs":           clf.all_probabilities,
        "esg_scores": {
            "overall":         sc.overall_score,
            "grade":           sc.grade,
            "environment":     sc.environment_score,
            "social":          sc.social_score,
            "governance":      sc.governance_score,
        },
        "environment_metrics": sc.e_metrics,
        "social_metrics":      sc.s_metrics,
        "governance_metrics":  sc.g_metrics,
        "pii_redactions":      anon.redaction_counts,
        "total_redactions":    anon.total_redactions,
        "scrubbed_preview":    scrub_preview,
        "char_count":          ing.char_count,
        "page_count":          ing.page_count or 0,
        "duration_ms":         result.duration_ms,
    })

@app.post("/anonymized")
async def download_anonymized(file: UploadFile = File(...)):
    """Return the FULL anonymised text as a downloadable .txt file."""
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(400, "Only .pdf and .txt files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = get_pipeline()(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")
    finally:
        try: os.unlink(tmp_path)
        except: pass

    anon = result.anonymization
    stem = Path(file.filename or "document").stem

    header = (
        f"ESG ANONYMISED DOCUMENT\n"
        f"Source file        : {file.filename}\n"
        f"Document class     : {result.inference.predicted_class}\n"
        f"Total redactions   : {anon.total_redactions}\n"
        f"Redaction breakdown: {dict(sorted(anon.redaction_counts.items(), key=lambda x: -x[1]))}\n"
        f"{'=' * 70}\n\n"
    )
    content = header + anon.scrubbed_text

    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="anonymized_{stem}.txt"',
        },
    )


@app.post("/pdf")
async def generate_pdf_report(file: UploadFile = File(...)):
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(400, "Only .pdf and .txt files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = get_pipeline()(tmp_path)
        from src.components.pdf_reporter import generate_pdf
        pdf_bytes = generate_pdf(result)
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")
    finally:
        try: os.unlink(tmp_path)
        except: pass

    stem = Path(file.filename or "report").stem
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="esg_report_{stem}.pdf"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
