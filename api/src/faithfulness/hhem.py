"""HHEM-2.1-open faithfulness scorer.
Returns a 0..1 score per (premise, hypothesis) pair where higher = more faithful.
Model is loaded once at first call and cached.

Inputs are batched and per-side character-truncated. HHEM's encoder has a
512-token cap and very long RAGTruth contexts otherwise OOM the container
or trip indexing errors after the tokenizer truncation warning.
"""
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification

MODEL_ID = "vectara/hallucination_evaluation_model"
CHAR_CAP = 1500
BATCH_SIZE = 16


@lru_cache(maxsize=1)
def _load():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model.eval()
    return model


def score(pairs: list[tuple[str, str]]) -> list[float]:
    """Score (premise, hypothesis) pairs. Returns float in [0, 1] per pair."""
    if not pairs:
        return []
    model = _load()

    safe = [
        ((p or "")[:CHAR_CAP], (h or "")[:CHAR_CAP])
        for p, h in pairs
    ]

    out: list[float] = []
    for i in range(0, len(safe), BATCH_SIZE):
        batch = safe[i:i + BATCH_SIZE]
        with torch.inference_mode():
            scores = model.predict(batch)
        if isinstance(scores, torch.Tensor):
            scores = scores.tolist()
        out.extend(float(s) for s in scores)
        if (i // BATCH_SIZE) % 25 == 0:
            print(f"hhem: {min(i + BATCH_SIZE, len(safe))}/{len(safe)}", flush=True)
    return out
