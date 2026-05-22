"""IR metrics + deterministic answer-correctness scoring.

For retrieval: precision@k and recall@k against ground-truth chunk IDs.
For answers: `contains_match` (normalized containment) is the PRIMARY metric — it
matches the MultiHop-RAG paper (Tang & Yang 2024), whose gold answers are short
factoids (yes/no, entity, before/after) scored by presence in the response.
`exact_match` is a stricter SECONDARY variant; the CRAG LLM-as-judge lives in judge.py.
"""
import re
import string


def precision_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for r in top if r in relevant_set) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for r in relevant if r in top) / len(relevant)


def reciprocal_rank_at_k(retrieved: list[str], relevant: list[str], k: int = 10) -> float:
    """1 / (rank of the first relevant item) within top-k; 0 if none. Mean → MRR@k."""
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant_set:
            return 1.0 / i
    return 0.0


def average_precision_at_k(retrieved: list[str], relevant: list[str], k: int = 10) -> float:
    """Average precision over the relevant items in top-k; mean → MAP@k.

    Normalised by min(#relevant, k) — the common IR convention; check this matches
    the target paper's MAP definition before comparing absolute values.
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    score = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant_set:
            hits += 1
            score += hits / i
    return score / min(len(relevant_set), k)


def hit_at_k(retrieved: list[str], relevant: list[str], k: int = 10) -> float:
    """Hit rate: 1.0 if at least one relevant item is in top-k, else 0.0. Mean → Hits@k."""
    relevant_set = set(relevant)
    return 1.0 if any(r in relevant_set for r in retrieved[:k]) else 0.0


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s


def exact_match(predicted: str, gold: str) -> bool:
    """Normalized full-string equality. Stricter SECONDARY metric (the paper uses containment)."""
    if not predicted or not gold:
        return False
    return _normalize(predicted) == _normalize(gold)


def contains_match(predicted: str, gold: str) -> bool:
    """Normalized containment: does the gold answer appear inside the prediction?

    PRIMARY correctness metric — the MultiHop-RAG paper (Tang & Yang 2024) scores
    a short factoid gold (yes/no, entity, before/after) by presence in the response.
    """
    if not predicted or not gold:
        return False
    return _normalize(gold) in _normalize(predicted)
