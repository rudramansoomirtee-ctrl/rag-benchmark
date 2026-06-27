"""Data plane: dataset counts, source links, browse endpoints, and sync ingest/index.

Ingest/index are long-running (seconds to minutes); the request stays open
until they finish. Use the CLI for long-running jobs. The browse endpoints
(`/datasets/{ds}/queries`, `/datasets/{ds}/chunks`) are read-only and paginated
so the SPA can explore what was actually ingested.
"""
from fastapi import APIRouter, HTTPException
from sqlalchemy import func, or_, select

from src.db.models import Chunk, Query
from src.db.session import get_session

router = APIRouter(prefix="/api", tags=["data"])

# Real upstream sources, surfaced in the UI so the dataset is one click away.
DATASET_SOURCES = {
    "multihop": {
        "title": "MultiHop-RAG",
        "hf_id": "yixuantt/MultiHopRAG",
        "url": "https://huggingface.co/datasets/yixuantt/MultiHopRAG",
        "paper": "Tang & Yang 2024",
    },
    "musique": {
        "title": "MuSiQue",
        "hf_id": "dgslibisey/MuSiQue",
        "url": "https://huggingface.co/datasets/dgslibisey/MuSiQue",
        "paper": "Trivedi et al. 2022",
    },
}

KNOWN_DATASETS = set(DATASET_SOURCES)
# Removed datasets whose rows linger in the DB as history (do not surface as active).
HIDDEN_DATASETS = {"ragtruth"}


@router.get("/datasets")
def list_datasets():
    """Row counts + upstream source per dataset in Postgres."""
    session = get_session()
    try:
        q_counts = dict(session.execute(
            select(Query.dataset, func.count(Query.id)).group_by(Query.dataset)
        ).all())
        c_counts = dict(session.execute(
            select(Chunk.dataset, func.count(Chunk.id)).group_by(Chunk.dataset)
        ).all())
        names = sorted((set(q_counts) | set(c_counts) | KNOWN_DATASETS) - HIDDEN_DATASETS)
        return [
            {
                "dataset": n,
                "queries": q_counts.get(n, 0),
                "chunks": c_counts.get(n, 0),
                "source": DATASET_SOURCES.get(n),
            }
            for n in names
        ]
    finally:
        session.close()


@router.get("/datasets/{dataset}/queries")
def list_queries(dataset: str, limit: int = 25, offset: int = 0, question_type: str | None = None):
    """Paginated query rows for a dataset, optionally filtered by question_type
    (e.g. MultiHop 'inference_query', MuSiQue '2hop'/'3hop'/'4hop')."""
    limit = max(1, min(limit, 200))
    session = get_session()
    try:
        where = [Query.dataset == dataset]
        if question_type:
            where.append(Query.query_metadata["question_type"].astext == question_type)

        total = session.execute(
            select(func.count(Query.id)).where(*where)
        ).scalar_one()

        types = [
            t for (t,) in session.execute(
                select(Query.query_metadata["question_type"].astext)
                .where(Query.dataset == dataset)
                .distinct()
            ).all() if t is not None
        ]

        rows = session.scalars(
            select(Query).where(*where).order_by(Query.id).limit(limit).offset(offset)
        ).all()
        items = [
            {
                "id": q.id,
                "external_id": q.external_id,
                "split": q.split,
                "task_type": q.task_type,
                "question_type": (q.query_metadata or {}).get("question_type"),
                "query_text": q.query_text,
                "ground_truth": q.ground_truth,
                "n_relevant": len(q.relevant_chunk_ids or []),
            }
            for q in rows
        ]
        return {"total": total, "limit": limit, "offset": offset,
                "question_types": sorted(types), "items": items}
    finally:
        session.close()


@router.get("/datasets/{dataset}/queries/{query_id}")
def query_detail(dataset: str, query_id: int):
    """A single query plus the text of its gold (relevant) chunks.

    MultiHop gold ids are URL-keyed while chunks are passage-keyed (`<url>#p<i>`),
    so fall back to a prefix match when exact lookup finds nothing.
    """
    session = get_session()
    try:
        q = session.get(Query, query_id)
        if q is None or q.dataset != dataset:
            raise HTTPException(404, f"query {query_id} not found in {dataset}")

        ids = q.relevant_chunk_ids or []
        gold = []
        if ids:
            gold = session.scalars(
                select(Chunk).where(Chunk.dataset == dataset, Chunk.external_id.in_(ids))
            ).all()
            if not gold:
                gold = session.scalars(
                    select(Chunk).where(
                        Chunk.dataset == dataset,
                        or_(*[Chunk.external_id.like(f"{i}#%") for i in ids]),
                    )
                ).all()

        return {
            "id": q.id,
            "external_id": q.external_id,
            "split": q.split,
            "task_type": q.task_type,
            "query_text": q.query_text,
            "ground_truth": q.ground_truth,
            "metadata": q.query_metadata,
            "relevant_chunk_ids": ids,
            "relevant_chunks": [
                {"external_id": c.external_id, "metadata": c.chunk_metadata, "text": c.text}
                for c in gold
            ],
        }
    finally:
        session.close()


@router.get("/datasets/{dataset}/chunks")
def list_chunks(dataset: str, limit: int = 25, offset: int = 0, search: str | None = None):
    """Paginated chunk rows for a dataset; `search` is a case-insensitive substring
    over chunk text."""
    limit = max(1, min(limit, 200))
    session = get_session()
    try:
        where = [Chunk.dataset == dataset]
        if search:
            where.append(Chunk.text.ilike(f"%{search}%"))

        total = session.execute(select(func.count(Chunk.id)).where(*where)).scalar_one()
        rows = session.scalars(
            select(Chunk).where(*where).order_by(Chunk.id).limit(limit).offset(offset)
        ).all()
        items = [
            {
                "id": c.id,
                "external_id": c.external_id,
                "title": (c.chunk_metadata or {}).get("title"),
                "source": (c.chunk_metadata or {}).get("source"),
                "text": c.text,
            }
            for c in rows
        ]
        return {"total": total, "limit": limit, "offset": offset, "items": items}
    finally:
        session.close()


@router.post("/ingest/{dataset}")
def ingest(dataset: str):
    if dataset == "multihop":
        from src.datasets.multihop import ingest as _ing
        n_q, n_c = _ing()
        return {"dataset": dataset, "queries": n_q, "chunks": n_c}
    if dataset == "musique":
        from src.datasets.musique import ingest as _ing
        n_q, n_c = _ing()
        return {"dataset": dataset, "queries": n_q, "chunks": n_c}
    raise HTTPException(400, f"unknown dataset: {dataset}")


@router.post("/index/{dataset}")
def index(dataset: str):
    from src.retrieval.indexer import index_corpus
    n = index_corpus(dataset)
    return {"dataset": dataset, "indexed": n}
