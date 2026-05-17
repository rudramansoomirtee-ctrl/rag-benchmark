"""IR metrics + answer-correctness scoring. Deterministic, no LLM-as-judge.

For retrieval: precision@k and recall@k against ground-truth chunk IDs.
For answer correctness: normalized exact-match (case-insensitive, whitespace-collapsed,
punctuation-stripped). This is the standard scoring approach for MultiHop-RAG.
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


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s


def exact_match(predicted: str, gold: str) -> bool:
    """Normalized exact-match. Use this for MultiHop QA."""
    if not predicted or not gold:
        return False
    return _normalize(predicted) == _normalize(gold)


def contains_match(predicted: str, gold: str) -> bool:
    """Looser: does the normalized gold appear inside the normalized prediction?

    Useful when the gold is a short factoid and the model wraps it in a sentence.
    """
    if not predicted or not gold:
        return False
    return _normalize(gold) in _normalize(predicted)
