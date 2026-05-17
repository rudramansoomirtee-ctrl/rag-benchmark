"""Load RAGTruth into Postgres for HHEM threshold calibration.

Each row is one (source, response) pair. RAGTruth annotates hallucinations
as character-span labels — a non-empty `labels` list means the response
contains at least one hallucinated span, which is what the calibrator
treats as the positive class.
"""
from datasets import load_dataset
from sqlalchemy import select

from src.db.models import Query
from src.db.session import get_session


DATASET_NAME = "ragtruth"


def ingest(split: str = "calibration") -> int:
    """Ingest RAGTruth response rows. Returns n rows ingested. Idempotent."""
    ds = load_dataset("ParticleMedia/RAGTruth", split="train")

    session = get_session()
    try:
        existing = set(session.scalars(
            select(Query.external_id).where(Query.dataset == DATASET_NAME)
        ).all())

        n = 0
        for i, row in enumerate(ds):
            rid = str(row.get("id") or f"rt-{i}")
            if rid in existing:
                continue

            # Non-empty span list => the response contains a hallucination.
            labels = row.get("labels") or []
            session.add(Query(
                dataset=DATASET_NAME,
                external_id=rid,
                split=split,
                task_type=row.get("task_type", "qa"),
                query_text=row.get("source") or row.get("prompt", ""),
                ground_truth=row.get("response", ""),
                relevant_chunk_ids=[],
                query_metadata={
                    "labels": labels,
                    "hallucination": 1 if labels else 0,
                    "model": row.get("model"),
                },
            ))
            n += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return n
