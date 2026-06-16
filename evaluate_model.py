"""
evaluate_model.py
==================
Generates two PNG files:
  - accuracy_loss.png   : training/validation accuracy & loss curves
  - confusion_matrix.png: confusion matrix on the validation split

Run from the esg_app/ directory:
    python evaluate_model.py
"""
from __future__ import annotations
import os, sys, pickle
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.components.model_trainer import (
    _generate, MAX_SEQ_LEN, NUM_CLASSES, SEED,
    MODEL_PATH, TOKENIZER_PATH, DATASET_PATH,
)

LABELS = ["SEBI_BRSR", "SUSTAINABILITY_REPORT", "INVALID_DOCUMENT"]


def main():
    # ── Load or regenerate dataset ────────────────────────────────────────────
    if os.path.exists(DATASET_PATH):
        df = pl.read_csv(DATASET_PATH)
    else:
        df = _generate()

    texts  = df["text"].to_list()
    labels = df["label"].to_list()
    splits = df["split"].to_list() if "split" in df.columns else None

    # ── Load tokenizer + model ────────────────────────────────────────────────
    with open(TOKENIZER_PATH, "rb") as f:
        tok = pickle.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)

    seqs = tok.texts_to_sequences(texts)
    X    = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    y    = np.array(labels)

    # ── Re-run a short training pass purely to capture history for plotting ──
    # (If you already have a `hist` object from training, skip this and reuse it.)
    print("Re-fitting briefly to capture training curves (uses saved weights as warm start)...")
    y_onehot = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    tf.keras.utils.set_random_seed(SEED)

    hist = model.fit(
        X, y_onehot,
        epochs=10,
        batch_size=32,
        validation_split=0.15,
        verbose=1,
    )

    # ══ 1. Accuracy / Loss curves ══════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(hist.history["accuracy"],     label="Train Accuracy", color="#16A34A", linewidth=2)
    axes[0].plot(hist.history["val_accuracy"], label="Val Accuracy",   color="#2563EB", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(alpha=0.3, linestyle="--")

    axes[1].plot(hist.history["loss"],     label="Train Loss", color="#DC2626", linewidth=2)
    axes[1].plot(hist.history["val_loss"], label="Val Loss",   color="#D97706", linewidth=2, linestyle="--")
    axes[1].set_title("Loss", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3, linestyle="--")

    fig.suptitle("ESG Classifier — Training History", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("accuracy_loss.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved accuracy_loss.png")

    # ══ 2. Confusion matrix ═══════════════════════════════════════════════════
    preds      = model.predict(X, verbose=0)
    pred_labels= np.argmax(preds, axis=1)

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(y, pred_labels):
        cm[true, pred] += 1

    fig2, ax2 = plt.subplots(figsize=(6, 5.2))
    im = ax2.imshow(cm, cmap="Greens")

    ax2.set_xticks(range(NUM_CLASSES))
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=9)
    ax2.set_yticklabels(LABELS, fontsize=9)
    ax2.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    ax2.set_ylabel("True Label", fontsize=10, fontweight="bold")
    ax2.set_title("Confusion Matrix — Full Dataset", fontsize=12, fontweight="bold")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val   = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax2.text(j, i, str(val), ha="center", va="center",
                     color=color, fontsize=12, fontweight="bold")

    fig2.colorbar(im, ax=ax2, shrink=0.8, label="Count")
    fig2.tight_layout()
    fig2.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print("Saved confusion_matrix.png")

    # Accuracy summary
    acc = (pred_labels == y).mean()
    print(f"\nOverall accuracy on full dataset: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
