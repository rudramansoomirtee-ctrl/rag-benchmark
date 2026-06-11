"""Unified retrieval entry point: hybrid first-stage + cross-encoder rerank.

Systems A/B/F call `retrieve()` so the retrieval pipeline is constant across
the controlled-orchestration comparison; the systems differ only in answer
generation. F-tuned additionally uses `retrieve_filtered()` to scope retrieval
by metadata (e.g. source).

Pipeline per call:
  1. Embed the query (BAAI/llm-embedder)
  2. Hybrid BM25 + dense kNN, fused with RRF, top `retrieval_pool` candidates
  3. Cross-encoder rerank the pool down to `top_k`

Multi-list systems (B across iterations, F across sub-questions) merge their
per-query ranked lists with `rrf_fuse` and answer over the fused top
`FUSED_ANSWER_TOP_K` — shared so the answer-context budget is held constant
across the orchestration comparison.
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


def retrieve_filtered(
    query: str,
    filters: dict | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Hybrid retrieval with a metadata filter applied at the OpenSearch level + rerank.

    `filters` is a dict of {metadata_field: value}; supported fields are those with
    a .keyword sub-mapping on the index (source / category / author / published_at /
    title). F-tuned uses this to scope retrieval to e.g. {"source": "Hacker News"}
    for queries that name a publisher.
    """
    top_k = top_k or settings.top_k
    qvec = embed_one(query)
    pool = hybrid_search(
        query, qvec, top_k=settings.retrieval_pool, filters=filters
    )
    return rerank(query, pool, top_k=top_k)


# Answer-context budget for systems that fuse multiple ranked lists (B, F).
# A single top_k=5 list passes through unchanged (a one-list fusion is the
# identity), so a no-op B/F run degenerates to A's exact context.
FUSED_ANSWER_TOP_K = 8


def rrf_fuse(ranked_lists: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across per-query result lists, deduped by chunk_id.

    A chunk appearing in several lists accumulates score — surfacing evidence
    that stays relevant under different query formulations.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            cid = h["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(cid, h)
    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [chunks[c] for c in ordered]


def format_context(hits: list[dict]) -> str:
    """Format retrieved chunks for the LLM, surfacing dataset metadata.

    MultiHop-RAG queries frequently identify articles by `source` and `title`
    (e.g. "the Hacker News article on The Epoch Times"), but neither field is
    present in chunk text or the URL-keyed chunk_id. Including them in the
    bracket prefix lets the generator match a query's named article to the
    retrieved chunk; without it, comparison queries that name a publisher
    cannot be answered even when the right chunk is retrieved.
    """
    lines = []
    for h in hits:
        meta = h.get("metadata") or {}
        src = meta.get("source") or "?"
        title = meta.get("title") or "?"
        lines.append(f"[{h['chunk_id']}] (source: {src} | title: {title}) {h['text']}")
    return "\n\n".join(lines)
