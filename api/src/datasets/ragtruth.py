"""Load RAGTruth (ParticleMedia/RAGTruth) into Postgres.

Used primarily for HHEM threshold calibration: RAGTruth ships hallucination
labels per response that we can use to fit a precision/recall threshold.
"""
from datasets import load_dataset

from src.db.models import Query
from src.db.session import get_session


DATASET_NAME = "ragtruth"


def ingest(split: str = "calibration") -> int:
    """Ingest the RAGTruth response rows as queries. Returns n rows ingested.

    Stub: inspect the dataset and map the real fields. We treat each
    (prompt, response, label) as one row; calibration uses these labels
    against HHEM scores.
    """
    ds = load_dataset("ParticleMedia/RAGTruth", split="train")

    session = get_session()
    n = 0
    try:
        for row in ds:
            session.add(Query(
                dataset=DATASET_NAME,
                external_id=str(row.get("id")),
                split=split,
                task_type=row.get("task_type", "qa"),
                query_text=row.get("prompt", ""),
                ground_truth=row.get("response", ""),
                relevant_chunk_ids=[],  # RAGTruth doesn't expose ground-truth chunk IDs
                query_metadata={
                    "labels": row.get("labels"),
                    "hallucination": row.get("hallucination"),
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
