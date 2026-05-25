"""Reranker — local cross-encoder by default, optional Bedrock Cohere Rerank 3.5.

The first-stage hybrid (BM25+dense+RRF) returns a `retrieval_pool` of candidates;
the reranker reorders them and trims to `top_k`. Standard production RAG pattern.

Provider is selected via `settings.rerank_provider`:
  - "local"          → sentence-transformers CrossEncoder (CPU, free, default)
  - "bedrock-cohere" → Cohere Rerank 3.5 via Bedrock Agent Runtime

Bedrock failures (auth/quota/model-not-enabled) log a warning and fall back to
the local cross-encoder so a long experiment never aborts on a transient API
error or misconfiguration.
"""
import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import settings

logger = logging.getLogger("rag.reranker")

# Per-doc text cap for Bedrock Rerank — Cohere Rerank 3.5 accepts up to ~4096
# tokens per doc, and MultiHop articles can exceed that. Truncating at ~16K
# chars keeps us safely inside the limit without touching short chunks.
_BEDROCK_DOC_CHAR_CAP = 16000


@lru_cache(maxsize=1)
def _load_local() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model, max_length=512)


@lru_cache(maxsize=1)
def _bedrock_client():
    import boto3
    return boto3.client("bedrock-agent-runtime", region_name=settings.aws_region)


def _rerank_local(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    model = _load_local()
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda cs: float(cs[1]), reverse=True)
    return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]


def _rerank_bedrock_cohere(
    query: str, candidates: list[dict], top_k: int
) -> list[dict]:
    """Cohere Rerank 3.5 via Bedrock Agent Runtime."""
    client = _bedrock_client()
    model_arn = (
        f"arn:aws:bedrock:{settings.aws_region}::foundation-model/"
        f"{settings.bedrock_rerank_model_id}"
    )
    sources = [
        {
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {
                    "text": (c.get("text") or "")[:_BEDROCK_DOC_CHAR_CAP]
                },
            },
        }
        for c in candidates
    ]
    resp = client.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        sources=sources,
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {"modelArn": model_arn},
                "numberOfResults": top_k,
            },
        },
    )
    results = resp.get("results", [])
    out = []
    for r in results:
        idx = r["index"]
        if 0 <= idx < len(candidates):
            out.append({**candidates[idx], "rerank_score": float(r["relevanceScore"])})
    return out[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """Re-score (query, candidate.text) pairs and return top_k.

    Selects provider via `settings.rerank_provider`; on Bedrock failure, falls
    back to local cross-encoder and logs a warning.
    """
    top_k = top_k or settings.top_k
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    if settings.rerank_provider == "bedrock-cohere":
        try:
            return _rerank_bedrock_cohere(query, candidates, top_k)
        except Exception as e:
            logger.warning(
                "Bedrock rerank failed (%s: %s); falling back to local cross-encoder",
                type(e).__name__, e,
            )
            return _rerank_local(query, candidates, top_k)

    return _rerank_local(query, candidates, top_k)
