"""Cross-encoder reranker.

Wraps sentence-transformers' CrossEncoder for scoring (query, chunk) pairs.
Used after the hybrid first-stage to reorder a candidate pool down to top-k.
Standard production RAG pattern — gives ~5-15% P@k uplift on MS-MARCO-style
news QA over single-stage retrieval.

Model is loaded once at first call and cached. CPU-friendly: MiniLM at ~80MB.
"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import settings


@lru_cache(maxsize=1)
def _load() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model, max_length=512)


def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """Re-score (query, candidate.text) with a cross-encoder and return top_k."""
    top_k = top_k or settings.top_k
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    model = _load()
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(candidates, scores),
        key=lambda cs: float(cs[1]),
        reverse=True,
    )
    return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]
