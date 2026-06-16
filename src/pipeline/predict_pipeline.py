"""src/pipeline/predict_pipeline.py — End-to-end pipeline orchestrator."""
from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
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

def run(source: str | Path) -> PipelineResult:
    log.info("Pipeline started")
    t0 = time.perf_counter()
    try:
        ing   = ingest(source)
        anon  = anonymize(ing.raw_text, ing.source)
        clf   = predict(anon.scrubbed_text)
        sc    = compute_scores(anon.scrubbed_text)
        dur   = round((time.perf_counter() - t0) * 1000, 1)
        log.info("Pipeline done — class=%s  score=%.1f  %.0fms",
                 clf.predicted_class, sc.overall_score, dur)
        return PipelineResult(ing, anon, clf, sc, dur)
    except Exception as e:
        raise ESGException(e) from e
