"""Load MuSiQue (multi-hop QA, anti-shortcut) into our Postgres tables.

MuSiQue (Trivedi et al. 2022) composes single-hop questions into 2-4 hop
questions, choosing distractors so that answering genuinely requires combining
evidence across paragraphs (it is built to defeat single-paragraph / shortcut
answering — the exact weakness that makes MultiHop-RAG answerable from partial
evidence). Each example ships ~20 candidate paragraphs (a few `is_supporting`
gold + distractors), a short answer, and a gold `question_decomposition`.

Unlike MultiHop-RAG (one shared news corpus), MuSiQue's paragraphs are
per-question. We pool every ingested question's paragraphs into one corpus
(keyed `<question_id>::p<idx>`) and index it in a SEPARATE OpenSearch index
(run ingest/index/experiments with OPENSEARCH_INDEX=rag-chunks-musique) so
retrieval is scoped to MuSiQue and never mixes with the news corpus. Gold =
the supporting paragraphs' ids; hop count (len of the decomposition) is stored
as `question_type` ("2hop"/"3hop"/"4hop") so the runner can stratify by it.

NOTE: HF id + field names below are verified against the loaded dataset before
ingest; adjust HF_DATASET / field keys if the available mirror differs.
"""
from datasets import load_dataset
from sqlalchemy import select

from src.db.models import Chunk, Query
from src.db.session import get_session

DATASET_NAME = "musique"
HF_DATASET = "dgslibisey/MuSiQue"   # candidate mirror; confirm by loading


def _hop_count(row: dict) -> int:
    """Number of hops = length of the gold decomposition; fall back to the id
    prefix (MuSiQue ids look like '2hop__<a>_<b>')."""
    decomp = row.get("question_decomposition") or []
    if decomp:
        return len(decomp)
    try:
        return int(str(row.get("id", "")).split("hop")[0])
    except (ValueError, TypeError):
        return 0


def ingest(split: str = "eval", limit: int | None = None, seed: int = 42, hf_split: str = "validation") -> tuple[int, int]:
    """Ingest MuSiQue questions + their pooled paragraph corpus. Returns (n_queries, n_chunks).

    `limit` caps the number of questions (pilot runs). The HF split is ordered by
    hop count (all 2-hop first), so we seed-shuffle before capping to get a 2/3/4-hop
    mix. Idempotent via external_id skip.
    """
    ds = load_dataset(HF_DATASET, split=hf_split)
    if limit:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))

    session = get_session()
    try:
        existing_chunks = set(session.scalars(
            select(Chunk.external_id).where(Chunk.dataset == DATASET_NAME)).all())
        existing_queries = set(session.scalars(
            select(Query.external_id).where(Query.dataset == DATASET_NAME)).all())

        n_chunks = 0
        n_queries = 0
        for row in ds:
            qid = row["id"]
            supporting: list[str] = []
            for p in (row.get("paragraphs") or []):
                pid = f"{qid}::p{p.get('idx')}"
                if p.get("is_supporting"):
                    supporting.append(pid)
                if pid in existing_chunks:
                    continue
                text = (p.get("paragraph_text") or "").strip()
                if not text:
                    continue
                existing_chunks.add(pid)
                session.add(Chunk(
                    dataset=DATASET_NAME,
                    external_id=pid,
                    text=text,
                    chunk_metadata={"title": p.get("title"), "question_id": qid},
                ))
                n_chunks += 1

            if qid in existing_queries:
                continue
            existing_queries.add(qid)
            n_hop = _hop_count(row)
            session.add(Query(
                dataset=DATASET_NAME,
                external_id=qid,
                split=split,
                task_type="qa",
                query_text=row["question"],
                ground_truth=row.get("answer"),
                relevant_chunk_ids=supporting,
                query_metadata={
                    "question_type": f"{n_hop}hop",
                    "hop": n_hop,
                    "answer_aliases": row.get("answer_aliases"),
                },
            ))
            n_queries += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return n_queries, n_chunks
