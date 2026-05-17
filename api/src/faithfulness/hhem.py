"""HHEM-2.1-open faithfulness scorer.
Returns a 0..1 score per (premise, hypothesis) pair where higher = more faithful.
Model is loaded once at first call and cached.
"""
from functools import lru_cache
import torch
from transformers import AutoModelForSequenceClassification

MODEL_ID = "vectara/hallucination_evaluation_model"


@lru_cache(maxsize=1)
def _load():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    model.eval()
    return model


def score(pairs: list[tuple[str, str]]) -> list[float]:
    """Score (premise, hypothesis) pairs. Returns float in [0, 1] per pair.

    HHEM-2.1-open ships a `predict()` helper on the model itself that
    handles tokenization and the prompt-based input format internally.
    """
    if not pairs:
        return []
    model = _load()
    scores = model.predict(pairs)
    if isinstance(scores, torch.Tensor):
        scores = scores.tolist()
    return [float(s) for s in scores]