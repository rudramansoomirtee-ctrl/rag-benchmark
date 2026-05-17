"""Load MultiHop-RAG (yixuantt/MultiHopRAG) into our Postgres tables.

The HF repo ships two configs:
  - `corpus`      — the news articles (one per row).
  - `MultiHopRAG` — the multi-hop questions with an `evidence_list`
                    pointing back into the corpus by URL.

Chunks are keyed by URL; a query's `relevant_chunk_ids` is the unique URL
list from its evidence.
"""
from datasets import load_dataset
from sqlalchemy import select

from src.db.models import Chunk, Query
from src.db.session import get_session


DATASET_NAME = "multihop"


def ingest(split: str = "eval") -> tuple[int, int]:
    """Ingest queries and corpus chunks. Returns (n_queries, n_chunks).

    Idempotent: re-running skips rows already present (by external_id).
    """
    corpus = load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    queries = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")

    session = get_session()
    try:
        existing_chunks = set(session.scalars(
            select(Chunk.external_id).where(Chunk.dataset == DATASET_NAME)
        ).all())
        existing_queries = set(session.scalars(
            select(Query.external_id).where(Query.dataset == DATASET_NAME)
        ).all())

        n_chunks = 0
        for row in corpus:
            cid = row["url"]
            if cid in existing_chunks:
                continue
            existing_chunks.add(cid)
            session.add(Chunk(
                dataset=DATASET_NAME,
                external_id=cid,
                text=row["body"],
                chunk_metadata={
                    "title": row.get("title"),
                    "author": row.get("author"),
                    "source": row.get("source"),
                    "category": row.get("category"),
                    "published_at": row.get("published_at"),
                },
            ))
            n_chunks += 1

        n_queries = 0
        for i, row in enumerate(queries):
            qid = f"mh-{i}"
            if qid in existing_queries:
                continue
            relevant = list(dict.fromkeys(
                ev["url"] for ev in row.get("evidence_list", []) if ev.get("url")
            ))
            session.add(Query(
                dataset=DATASET_NAME,
                external_id=qid,
                split=split,
                task_type="qa",
                query_text=row["query"],
                ground_truth=row.get("answer"),
                relevant_chunk_ids=relevant,
                query_metadata={"question_type": row.get("question_type")},
            ))
            n_queries += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return n_queries, n_chunks
