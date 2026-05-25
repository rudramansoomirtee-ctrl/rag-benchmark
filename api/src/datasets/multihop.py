"""Load MultiHop-RAG (yixuantt/MultiHopRAG) into our Postgres tables.

The HF repo ships two configs:
  - `corpus`      — the news articles (one per row).
  - `MultiHopRAG` — the multi-hop questions with an `evidence_list`
                    pointing back into the corpus by URL.

By default, each article is split into overlapping 256-token passages
(matching Tang & Yang 2024's setup), with each passage stored as a separate
chunk row keyed `<url>#p<i>`. To revert to article-granular chunks (the
historical setup used in experiments 1–12), pass `passage_tokens=None`.

Queries' `relevant_chunk_ids` remain URL-keyed; `metrics.py` maps retrieved
passage IDs back to their parent URL (`split('#', 1)[0]`) when scoring, so
the gold data is identical across granularities.
"""
from functools import lru_cache

from datasets import load_dataset
from sqlalchemy import select

from src.db.models import Chunk, Query
from src.db.session import get_session


DATASET_NAME = "multihop"
DEFAULT_PASSAGE_TOKENS = 256
DEFAULT_PASSAGE_STRIDE = 128


@lru_cache(maxsize=1)
def _tokenizer():
    """The embedder's tokenizer — keeps passage token budgets aligned with the 512-cap
    embedding model so a 256-token passage never gets truncated at embed time."""
    from src.retrieval.embeddings import get_model
    return get_model().tokenizer


def _split_into_passages(
    body: str, window: int, stride: int
) -> list[str]:
    """Split text into overlapping token windows; returns the decoded passages.

    Empty bodies return []. If the body fits in one window, returns a single passage.
    """
    if not body or not body.strip():
        return []
    tok = _tokenizer()
    ids = tok.encode(body, add_special_tokens=False)
    if len(ids) <= window:
        return [tok.decode(ids, skip_special_tokens=True).strip()]
    passages = []
    i = 0
    while i < len(ids):
        chunk_ids = ids[i : i + window]
        if not chunk_ids:
            break
        passages.append(tok.decode(chunk_ids, skip_special_tokens=True).strip())
        if i + window >= len(ids):
            break
        i += stride
    return [p for p in passages if p]


def ingest(
    split: str = "eval",
    passage_tokens: int | None = DEFAULT_PASSAGE_TOKENS,
    passage_stride: int = DEFAULT_PASSAGE_STRIDE,
) -> tuple[int, int]:
    """Ingest queries and corpus chunks. Returns (n_queries, n_chunks).

    `passage_tokens=N` → split articles into overlapping N-token passages. Each
    passage gets external_id `<url>#p<i>`, inheriting the parent article's
    metadata. `passage_tokens=None` → store one chunk per article (legacy).

    Idempotent on re-run via external_id skip.
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
            url = row["url"]
            metadata = {
                "title": row.get("title"),
                "author": row.get("author"),
                "source": row.get("source"),
                "category": row.get("category"),
                "published_at": row.get("published_at"),
            }
            if passage_tokens is None:
                if url in existing_chunks:
                    continue
                existing_chunks.add(url)
                session.add(Chunk(
                    dataset=DATASET_NAME,
                    external_id=url,
                    text=row["body"],
                    chunk_metadata=metadata,
                ))
                n_chunks += 1
            else:
                passages = _split_into_passages(
                    row["body"], passage_tokens, passage_stride
                )
                for i, passage_text in enumerate(passages):
                    pid = f"{url}#p{i}"
                    if pid in existing_chunks:
                        continue
                    existing_chunks.add(pid)
                    session.add(Chunk(
                        dataset=DATASET_NAME,
                        external_id=pid,
                        text=passage_text,
                        chunk_metadata=metadata,
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
