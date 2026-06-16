"""
main.py — FastAPI backend for ESG Document Intelligence
Run locally:  uvicorn main:app --reload --port 7860
HuggingFace:  runs on port 7860 automatically
"""
from __future__ import annotations
import os, sys, tempfile, logging, json, queue, threading, uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
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

# ── Real-time progress via Server-Sent Events ─────────────────────────────────
#
# Why this exists: a plain request/response round-trip gives the frontend no
# visibility into which pipeline stage is currently running. Instead of
# faking a timer client-side, the upload is staged in two steps:
#
#   1. POST /analyse-stream/start  -> saves the file, returns a job_id
#   2. GET  /analyse-stream/{job_id} -> opens an SSE connection; a background
#      thread runs the REAL pipeline and pushes an event the instant each
#      real stage starts (via the `on_progress` callback added to
#      predict_pipeline.run). The final event carries the full JSON result.
#
# This replaces the earlier client-side setInterval() timer entirely —
# every "stage started" event below corresponds to an actual function call
# beginning in the real pipeline, not a guessed delay.

_UPLOAD_DIR = Path(tempfile.gettempdir()) / "esg_uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)
_jobs: dict[str, dict] = {}   # job_id -> {"path": str, "filename": str}

@app.post("/analyse-stream/start")
async def analyse_stream_start(file: UploadFile = File(...)):
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in (".pdf", ".txt"):
        raise HTTPException(400, f"Only .pdf and .txt files are supported. Got '{ext}'.")

    job_id = uuid.uuid4().hex
    dest   = _UPLOAD_DIR / f"{job_id}{ext}"
    dest.write_bytes(await file.read())
    _jobs[job_id] = {"path": str(dest), "filename": file.filename or "upload"}
    return {"job_id": job_id}


@app.get("/analyse-stream/{job_id}")
async def analyse_stream(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown or expired job_id.")

    def event_stream():
        q: queue.Queue = queue.Queue()

        def on_progress(stage: str):
            # Called synchronously by the pipeline the instant that real
            # stage begins — pushed straight onto the SSE queue.
            q.put({"event": "stage", "stage": stage})

        def worker():
            try:
                result = get_pipeline()(job["path"], on_progress=on_progress)
                sc, clf, anon, ing = result.scores, result.inference, result.anonymization, result.ingestion
                scrub_preview = anon.scrubbed_text[:2000]
                if len(anon.scrubbed_text) > 2000:
                    scrub_preview += "\n… [truncated]"
                payload = {
                    "source":              job["filename"],
                    "classification":      clf.predicted_class,
                    "confidence":          round(clf.confidence, 4),
                    "all_probs":           clf.all_probabilities,
                    "esg_scores": {
                        "overall":     sc.overall_score, "grade": sc.grade,
                        "environment": sc.environment_score,
                        "social":      sc.social_score,
                        "governance":  sc.governance_score,
                    },
                    "environment_metrics": sc.e_metrics,
                    "social_metrics":      sc.s_metrics,
                    "governance_metrics":  sc.g_metrics,
                    "low_confidence":      sc.low_confidence,
                    "pii_redactions":      anon.redaction_counts,
                    "total_redactions":    anon.total_redactions,
                    "scrubbed_preview":    scrub_preview,
                    "char_count":          ing.char_count,
                    "page_count":          ing.page_count or 0,
                    "duration_ms":         result.duration_ms,
                }
                q.put({"event": "done", "data": payload})
            except Exception as e:
                q.put({"event": "error", "message": str(e)})
            finally:
                q.put(None)   # sentinel: stop the stream
                try: os.unlink(job["path"])
                except Exception: pass
                _jobs.pop(job_id, None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


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
        "low_confidence":      sc.low_confidence,
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


@app.post("/json")
async def download_json_report(file: UploadFile = File(...)):
    """Return the FULL analysis (scores + evidence + PII + metadata) as a downloadable .json file."""
    import json as _json
    from datetime import datetime as _dt

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

    sc, clf, anon, ing = result.scores, result.inference, result.anonymization, result.ingestion
    stem = Path(file.filename or "report").stem

    payload = {
        "report_generated_at": _dt.now().isoformat(),
        "source_file":          file.filename,
        "classification": {
            "predicted_class": clf.predicted_class,
            "confidence":       round(clf.confidence, 4),
            "all_probabilities": clf.all_probabilities,
        },
        "esg_scores": {
            "overall":     sc.overall_score,
            "grade":       sc.grade,
            "environment": sc.environment_score,
            "social":      sc.social_score,
            "governance":  sc.governance_score,
            "low_confidence_flag": sc.low_confidence,
        },
        "environment_metrics": sc.e_metrics,
        "social_metrics":      sc.s_metrics,
        "governance_metrics":  sc.g_metrics,
        "evidence": {
            "environment": sc.e_evidence,
            "social":      sc.s_evidence,
            "governance":  sc.g_evidence,
        },
        "pii_anonymization": {
            "total_redactions":   anon.total_redactions,
            "redaction_breakdown": anon.redaction_counts,
        },
        "document_metadata": {
            "char_count": ing.char_count,
            "page_count": ing.page_count or 0,
        },
        "pipeline_duration_ms": result.duration_ms,
    }

    body = _json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(
        content=body.encode("utf-8"),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="esg_report_{stem}.json"',
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

# ── Model evaluation charts ───────────────────────────────────────────────────
# These don't depend on any uploaded document — they evaluate the trained
# classifier itself against its training dataset. Generation takes ~10-20s
# because the saved .keras model doesn't retain its original training
# history, so a short warm-start re-fit is required to produce real curves
# (see src/components/model_evaluation.py for the full explanation).

@app.get("/model-evaluation/accuracy-loss")
async def get_accuracy_loss_chart():
    try:
        from src.components.model_evaluation import generate_evaluation_charts
        acc_loss_png, _, _ = generate_evaluation_charts()
    except Exception as e:
        raise HTTPException(500, f"Evaluation error: {e}")

    return Response(
        content=acc_loss_png,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="accuracy_loss.png"'},
    )

@app.get("/model-evaluation/confusion-matrix")
async def get_confusion_matrix_chart():
    try:
        from src.components.model_evaluation import generate_evaluation_charts
        _, cm_png, _ = generate_evaluation_charts()
    except Exception as e:
        raise HTTPException(500, f"Evaluation error: {e}")

    return Response(
        content=cm_png,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="confusion_matrix.png"'},
    )

@app.get("/model-evaluation/both")
async def get_both_evaluation_charts():
    """Returns both charts plus accuracy as a single JSON with base64 PNGs,
    so the frontend can preview them inline before offering separate downloads."""
    import base64
    try:
        from src.components.model_evaluation import generate_evaluation_charts
        acc_loss_png, cm_png, accuracy = generate_evaluation_charts()
    except Exception as e:
        raise HTTPException(500, f"Evaluation error: {e}")

    return JSONResponse({
        "accuracy_loss_png_b64":    base64.b64encode(acc_loss_png).decode("ascii"),
        "confusion_matrix_png_b64": base64.b64encode(cm_png).decode("ascii"),
        "full_dataset_accuracy":    round(accuracy, 4),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
