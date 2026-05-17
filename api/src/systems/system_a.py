"""System A: naive RAG. Embed query -> retrieve top-k -> answer once.

No reformulation, no faithfulness gate. This is the baseline.
"""
import time

from src.config import settings
from src.llm.client import generate
from src.retrieval.embeddings import embed_one
from src.retrieval.opensearch_client import knn_search
from src.systems.base import RunResult


class SystemA:
    name = "A"

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        qvec = embed_one(query)
        hits = knn_search(qvec, top_k=settings.top_k)
        context = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits)

        result = generate(
            messages=[
                {
                    "role": "system",
                    "content": "Answer using only the provided context. Cite chunk IDs in brackets.",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
        )

        return RunResult(
            answer=result["content"],
            retrieved_chunk_ids=[h["chunk_id"] for h in hits],
            hhem_score=None,
            flagged=None,
            n_steps=1,
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=result["cost_usd"],
        )
