"""OpenSearch client and helpers.

The same code runs locally against the Docker container and in AWS against
OpenSearch Service — that's the whole point of choosing OpenSearch over Qdrant
for this dissertation.
"""
from functools import lru_cache

from opensearchpy import OpenSearch

from src.config import settings


@lru_cache(maxsize=1)
def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[settings.opensearch_url],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )


def ensure_index() -> None:
    """Create the chunks index if it doesn't exist. Idempotent."""
    client = get_client()
    if client.indices.exists(index=settings.opensearch_index):
        return

    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "dataset": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": settings.embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                    },
                },
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }
    client.indices.create(index=settings.opensearch_index, body=body)


def bulk_index(docs: list[dict]) -> None:
    """Bulk index. `docs` should contain chunk_id, dataset, text, embedding, metadata."""
    from opensearchpy.helpers import bulk

    actions = [
        {
            "_op_type": "index",
            "_index": settings.opensearch_index,
            "_id": d["chunk_id"],
            "_source": d,
        }
        for d in docs
    ]
    bulk(get_client(), actions)


def knn_search(query_vector: list[float], top_k: int | None = None) -> list[dict]:
    """Dense vector search. Returns hits with text + chunk_id + score."""
    top_k = top_k or settings.top_k
    body = {
        "size": top_k,
        "query": {"knn": {"embedding": {"vector": query_vector, "k": top_k}}},
        "_source": ["chunk_id", "text", "dataset", "metadata"],
    }
    resp = get_client().search(index=settings.opensearch_index, body=body)
    return [
        {**hit["_source"], "score": hit["_score"]}
        for hit in resp["hits"]["hits"]
    ]


def bm25_search(query_text: str, top_k: int | None = None) -> list[dict]:
    """Lexical BM25 search over the chunk text field."""
    top_k = top_k or settings.top_k
    body = {
        "size": top_k,
        "query": {"match": {"text": query_text}},
        "_source": ["chunk_id", "text", "dataset", "metadata"],
    }
    resp = get_client().search(index=settings.opensearch_index, body=body)
    return [
        {**hit["_source"], "score": hit["_score"]}
        for hit in resp["hits"]["hits"]
    ]


def hybrid_search(
    query_text: str,
    query_vector: list[float],
    top_k: int | None = None,
    rrf_k: int = 60,
) -> list[dict]:
    """BM25 + dense kNN combined client-side with Reciprocal Rank Fusion.

    RRF score per chunk = sum over rankings of 1/(rrf_k + rank). This is the
    standard fusion used in production hybrid pipelines (Lin et al. 2024);
    doing it client-side avoids the OpenSearch neural-search plugin entirely
    and keeps scoring independent of either subquery's absolute score scale.
    """
    top_k = top_k or settings.top_k
    pool = max(top_k * 4, settings.retrieval_pool)

    bm25 = bm25_search(query_text, top_k=pool)
    knn = knn_search(query_vector, top_k=pool)

    fused: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for rank, h in enumerate(bm25, start=1):
        cid = h["chunk_id"]
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chunks[cid] = h
    for rank, h in enumerate(knn, start=1):
        cid = h["chunk_id"]
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chunks.setdefault(cid, h)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [{**chunks[cid], "score": score} for cid, score in ordered[:top_k]]
