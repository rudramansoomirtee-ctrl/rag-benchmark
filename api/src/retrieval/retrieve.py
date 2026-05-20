"""Unified retrieval entry point: hybrid first-stage + cross-encoder rerank.

All four systems (A/B/C/D) call this so the retrieval pipeline is constant
across the benchmark. The four systems differ in answer generation and
faithfulness filtering, not in how they retrieve.

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
