"""Data plane: dataset counts + sync ingest/index endpoints.

Ingest/index are long-running (seconds to minutes); the request stays open
until they finish. Use the CLI for the very long calibration job.
"""
from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from src.db.models import Chunk, Query
from src.db.session import get_session

router = APIRouter(prefix="/api", tags=["data"])


@router.get("/datasets")
def list_datasets():
    """Row counts per dataset in Postgres."""
    session = get_session()
    try:
        q_counts = dict(session.execute(
            select(Query.dataset, func.count(Query.id)).group_by(Query.dataset)
        ).all())
        c_counts = dict(session.execute(
            select(Chunk.dataset, func.count(Chunk.id)).group_by(Chunk.dataset)
        ).all())
        names = sorted(set(q_counts) | set(c_counts) | {"multihop", "ragtruth"})
        return [
            {"dataset": n, "queries": q_counts.get(n, 0), "chunks": c_counts.get(n, 0)}
            for n in names
        ]
    finally:
        session.close()


@router.post("/ingest/{dataset}")
def ingest(dataset: str):
    if dataset == "multihop":
        from src.datasets.multihop import ingest as _ing
        n_q, n_c = _ing()
        return {"dataset": dataset, "queries": n_q, "chunks": n_c}
    if dataset == "ragtruth":
        from src.datasets.ragtruth import ingest as _ing
        n = _ing()
        return {"dataset": dataset, "queries": n}
    raise HTTPException(400, f"unknown dataset: {dataset}")


@router.post("/index/{dataset}")
def index(dataset: str):
    from src.retrieval.indexer import index_corpus
    n = index_corpus(dataset)
    return {"dataset": dataset, "indexed": n}
