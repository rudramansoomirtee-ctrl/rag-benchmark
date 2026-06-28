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
from src.retrieval.opensearch_client import hybrid_search, knn_search
from src.retrieval.reranker import rerank
from src.trace import trace_active, trace_event


# Stratification over-fetches this multiple of `retrieval_pool` from the hybrid
# stage, then trims back to a source-diverse pool — so minority-source chunks
# ranked just outside the normal pool still get a chance at the reranker.
STRATIFY_FANOUT = 3


def _stratify_by_source(hits: list[dict], pool_size: int) -> list[dict]:
    """Guarantee each source's top hit enters the rerank pool, then fill by relevance.

    A hybrid pool dominated by one publisher starves minority sources before the
    reranker ever sees them — the root cause behind comparison queries that name
    two publishers (one floods the pool, the other never appears). This pulls the
    top chunk of each source (in order of first appearance, so relevance order is
    respected) as a diversity floor, then fills the remaining slots strictly by
    the original fused relevance order. Source-agnostic: no query parsing, so it
    applies uniformly to every system and every query.
    """
    from collections import OrderedDict

    by_source: "OrderedDict[str, list[dict]]" = OrderedDict()
    for h in hits:
        src = (h.get("metadata") or {}).get("source") or "?"
        by_source.setdefault(src, []).append(h)

    out: list[dict] = []
    taken: set[str] = set()
    for group in by_source.values():
        if len(out) >= pool_size:
            break
        out.append(group[0])
        taken.add(group[0]["chunk_id"])
    for h in hits:
        if len(out) >= pool_size:
            break
        if h["chunk_id"] in taken:
            continue
        out.append(h)
        taken.add(h["chunk_id"])
    return out


def _trace_retrieve(query: str, pool: list[dict], reranked: list[dict], mode: str) -> None:
    """Emit a glass-box trace event for one retrieval. No-op unless capturing
    (so experiment runs are unaffected). Records the hybrid pool, the reranked
    top-k, and each reranked hit's original hybrid rank so the UI can show how
    the cross-encoder reordered the candidates."""
    if not trace_active():
        return

    def _ser(hits: list[dict], score_key: str) -> list[dict]:
        rows = []
        for i, h in enumerate(hits[:12], start=1):
            meta = h.get("metadata") or {}
            rows.append({
                "rank": i,
                "id": h.get("chunk_id"),
                "score": round(float(h.get(score_key) or 0.0), 4),
                "source": meta.get("source"),
                "title": meta.get("title"),
                "text": (h.get("text") or "")[:240],
            })
        return rows

    hybrid_rank = {h.get("chunk_id"): i for i, h in enumerate(pool, start=1)}
    reranked_rows = _ser(reranked, "rerank_score")
    for r in reranked_rows:
        r["hybrid_rank"] = hybrid_rank.get(r["id"])
    trace_event("retrieve", query=query, mode=mode,
                hybrid=_ser(pool, "score"), reranked=reranked_rows)


def retrieve(query: str, top_k: int | None = None, semantic_only: bool = False) -> list[dict]:
    """Hybrid first-stage retrieval followed by cross-encoder rerank.

    `semantic_only=True` (or the global `retrieval_semantic_only` flag)
    short-circuits to a naive dense-kNN-only retriever (no BM25, RRF or rerank).
    The per-call argument lets one system opt into the weakened retriever while
    others keep the full pipeline in the same run — System A-minus uses it to be
    a semantic-search-only counterpart of A. With `retrieval_stratify_sources`
    on, the first-stage pool is over-fetched and re-pooled to span sources (see
    `_stratify_by_source`) before the rerank; off, it is the plain hybrid
    top-`retrieval_pool`.
    """
    top_k = top_k or settings.top_k
    qvec = embed_one(query)
    if semantic_only or settings.retrieval_semantic_only:
        out = knn_search(qvec, top_k=top_k)
        _trace_retrieve(query, out, out, "semantic-only")
        return out
    if settings.retrieval_stratify_sources:
        raw = hybrid_search(query, qvec, top_k=settings.retrieval_pool * STRATIFY_FANOUT)
        pool = _stratify_by_source(raw, settings.retrieval_pool)
    else:
        pool = hybrid_search(query, qvec, top_k=settings.retrieval_pool)
    reranked = rerank(query, pool, top_k=top_k)
    _trace_retrieve(query, pool, reranked, "hybrid+rerank")
    return reranked


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


# Answer-context budget for systems that fuse multiple ranked lists (B, F), read
# from settings so it can be swept (budget-sensitivity ablations). A single list
# passes through unchanged (one-list fusion is the identity), so a no-op B/F run
# degenerates to A's exact context.
FUSED_ANSWER_TOP_K = settings.fused_answer_top_k


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
