"""Unified retrieval entry point: hybrid first-stage + cross-encoder rerank.

Systems A, B, and F all call this, so their retrieval pipeline is identical.
Differences between them come from orchestration (single-shot vs. agentic loop
vs. query decomposition), not from how chunks are fetched — which is what lets
A-vs-B and A-vs-F isolate the *reasoning* strategy. System E deliberately uses
its own retriever (OpenRag's RAPTOR forest) instead, so A-vs-E isolates the
*retriever*.

Pipeline per call:
  1. Embed the query (BAAI/llm-embedder)
  2. Hybrid BM25 + dense kNN, fused with RRF, top `retrieval_pool` candidates
  3. Cross-encoder rerank the pool down to `top_k`
"""
from src.config import settings
from src.retrieval.embeddings import embed_one
from src.retrieval.opensearch_client import hybrid_search
from src.retrieval.reranker import rerank


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Hybrid first-stage retrieval followed by cross-encoder rerank."""
    top_k = top_k or settings.top_k
    qvec = embed_one(query)
    pool = hybrid_search(query, qvec, top_k=settings.retrieval_pool)
    return rerank(query, pool, top_k=top_k)
