"""Corpus indexer — read chunks from Postgres, embed, bulk-load into OpenSearch."""
from sqlalchemy import select

from src.db.models import Chunk
from src.db.session import get_session
from src.retrieval.embeddings import embed
from src.retrieval.opensearch_client import bulk_index, ensure_index


def index_corpus(dataset: str, batch_size: int = 64) -> int:
    """Index all chunks for a given dataset. Returns the number of chunks indexed."""
    ensure_index()
    session = get_session()
    try:
        stmt = select(Chunk).where(Chunk.dataset == dataset).order_by(Chunk.id)
        chunks = session.scalars(stmt).all()
    finally:
        session.close()

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embed([c.text for c in batch])
        docs = [
            {
                "chunk_id": c.external_id,
                "dataset": c.dataset,
                "text": c.text,
                "embedding": emb,
                "metadata": c.chunk_metadata or {},
            }
            for c, emb in zip(batch, embeddings)
        ]
        bulk_index(docs)
        total += len(batch)
    return total
