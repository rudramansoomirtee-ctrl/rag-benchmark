"""Load MultiHop-RAG (yixuantt/MultiHopRAG) into our Postgres tables.

Schema reference: https://huggingface.co/datasets/yixuantt/MultiHopRAG
Keys vary slightly across HF dataset versions — adjust the field names below
once you've inspected a sample row.
"""
from datasets import load_dataset

from src.db.models import Chunk, Query
from src.db.session import get_session


DATASET_NAME = "multihop"


def ingest(split: str = "eval") -> tuple[int, int]:
    """Ingest queries and chunks. Returns (n_queries, n_chunks).

    Stub: inspect the actual dataset schema first, then map fields. The
    structure below is a placeholder.
    """
    ds = load_dataset("yixuantt/MultiHopRAG", split="train")

    session = get_session()
    n_queries = n_chunks = 0
    try:
        # TODO: map your real fields. The dataset has 'query', 'answer',
        # 'evidence_list' (chunks), and supporting metadata.
        seen_chunks: set[str] = set()
        for row in ds:
            # Chunks first
            for ev in row.get("evidence_list", []):
                cid = ev.get("id") or ev.get("title")
                if cid in seen_chunks:
                    continue
                seen_chunks.add(cid)
                session.add(Chunk(
                    dataset=DATASET_NAME,
                    external_id=str(cid),
                    text=ev.get("fact", "") or ev.get("body", ""),
                    chunk_metadata={"title": ev.get("title"), "source": ev.get("source")},
                ))
                n_chunks += 1

            # Then the query
            session.add(Query(
                dataset=DATASET_NAME,
                external_id=str(row.get("_id") or row.get("query")),
                split=split,
                task_type="qa",
                query_text=row["query"],
                ground_truth=row.get("answer"),
                relevant_chunk_ids=[
                    str(ev.get("id") or ev.get("title"))
                    for ev in row.get("evidence_list", [])
                ],
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
