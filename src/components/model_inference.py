"""src/components/model_inference.py — Inference with .keras format."""
from __future__ import annotations
import os, pickle
from dataclasses import dataclass
import numpy as np
from src.exception import ESGException
from src.logger import get_logger

log = get_logger(__name__)

_ROOT          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH     = os.path.join(_ROOT, "models", "esg_classifier.keras")
TOKENIZER_PATH = os.path.join(_ROOT, "models", "tokenizer.pkl")
MAX_SEQ_LEN    = 350
LABEL_MAP      = {0: "SEBI_BRSR", 1: "SUSTAINABILITY_REPORT", 2: "INVALID_DOCUMENT"}

_model = _tokenizer = None

def _load():
    global _model, _tokenizer
    if _model and _tokenizer: return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: '{MODEL_PATH}'. Run model_trainer first.")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found: '{TOKENIZER_PATH}'.")
    log.info("Loading model → %s", MODEL_PATH)
    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)
    log.info("Artifacts loaded (vocab=%d)", len(_tokenizer.word_index))

@dataclass
class InferenceResult:
    predicted_class:   str
    class_index:       int
    confidence:        float
    all_probabilities: dict

def predict(text: str) -> InferenceResult:
    if not text or not text.strip():
        return InferenceResult("INVALID_DOCUMENT", 2, 1.0,
                               {"SEBI_BRSR":0.0,"SUSTAINABILITY_REPORT":0.0,
                                "INVALID_DOCUMENT":1.0})
    try:
        _load()
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq    = _tokenizer.texts_to_sequences([text.lower()])
        padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
        probs  = _model.predict(padded, verbose=0)[0]
        idx    = int(np.argmax(probs))
        return InferenceResult(
            LABEL_MAP[idx], idx, float(probs[idx]),
            {LABEL_MAP[i]: round(float(probs[i]), 6) for i in range(3)},
        )
    except Exception as e:
        raise ESGException(e) from e
