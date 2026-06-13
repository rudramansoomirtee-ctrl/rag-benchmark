"""Load RAGTruth into Postgres.

Sourced from wandb/RAGTruth-processed — the public mirror of the original
ParticleMedia/RAGTruth, which became gated after the original release.
Same rows, parquet format. Hallucination is signalled two ways: a raw
`hallucination_labels` JSON-string of span annotations, and a
`hallucination_labels_processed` struct with evident_conflict + baseless_info
counts. A row is positive iff either source indicates at least one
hallucinated span, matching the original schema's non-empty list rule.
"""
import json

from datasets import load_dataset
from sqlalchemy import select

from src.db.models import Query
from src.db.session import get_session


DATASET_NAME = "ragtruth"


def ingest(split: str = "calibration") -> int:
    """Ingest RAGTruth response rows. Returns n rows ingested. Idempotent."""
    ds = load_dataset("wandb/RAGTruth-processed", split="train")

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

            processed = row.get("hallucination_labels_processed") or {}
            counts = (processed.get("evident_conflict") or 0) + (processed.get("baseless_info") or 0)
            try:
                spans = json.loads(row.get("hallucination_labels") or "[]") or []
            except (TypeError, ValueError):
                spans = []
            hallucinated = counts > 0 or bool(spans)

            session.add(Query(
                dataset=DATASET_NAME,
                external_id=rid,
                split=split,
                task_type=row.get("task_type", "qa"),
                query_text=row.get("context") or row.get("query", ""),
                ground_truth=row.get("output", ""),
                relevant_chunk_ids=[],
                query_metadata={
                    "labels": spans,
                    "hallucination": 1 if hallucinated else 0,
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
