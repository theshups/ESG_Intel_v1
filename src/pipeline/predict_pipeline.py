"""src/pipeline/predict_pipeline.py — End-to-end pipeline orchestrator."""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from src.components.data_ingestion     import ingest,         IngestionResult
from src.components.data_transformation import anonymize,     AnonymizationResult
from src.components.model_inference    import predict,        InferenceResult
from src.components.scoring_engine     import compute_scores, ESGScoreResult
from src.exception import ESGException
from src.logger    import get_logger

log = get_logger(__name__)

@dataclass
class PipelineResult:
    ingestion:     IngestionResult
    anonymization: AnonymizationResult
    inference:     InferenceResult
    scores:        ESGScoreResult
    duration_ms:   float

# Type alias for a progress callback: called with a stage name as each
# REAL pipeline stage actually starts — not simulated/timed.
ProgressCallback = Callable[[str], None]

def run(source: str | Path, on_progress: Optional[ProgressCallback] = None) -> PipelineResult:
    """
    Run the full pipeline. If `on_progress` is provided, it is invoked
    synchronously right before each stage begins, with one of:
        "ingest", "anonymize", "classify", "score"
    This reflects the actual execution order — there is no simulated
    timing here; each callback fires exactly when that stage starts.
    """
    log.info("Pipeline started")
    t0 = time.perf_counter()
    try:
        if on_progress: on_progress("ingest")
        ing   = ingest(source)

        if on_progress: on_progress("anonymize")
        anon  = anonymize(ing.raw_text, ing.source)

        if on_progress: on_progress("classify")
        clf   = predict(anon.scrubbed_text)

        if on_progress: on_progress("score")
        sc    = compute_scores(anon.scrubbed_text)

        dur   = round((time.perf_counter() - t0) * 1000, 1)
        log.info("Pipeline done — class=%s  score=%.1f  %.0fms",
                 clf.predicted_class, sc.overall_score, dur)
        return PipelineResult(ing, anon, clf, sc, dur)
    except Exception as e:
        raise ESGException(e) from e
