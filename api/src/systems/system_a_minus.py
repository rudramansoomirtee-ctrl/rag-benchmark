"""System A-minus: naive RAG over a DELIBERATELY WEAKENED retriever.

Identical to System A — one retrieval, one answer, the same ANSWER_SYSTEM_PROMPT
and the same top_k=10 answer context — except retrieval is **dense-kNN semantic
search only**: no BM25 lexical match, no RRF hybrid fusion, no cross-encoder
rerank. It is A with the retrieval pipeline stripped back to a single vector
nearest-neighbour lookup.

Where A/B/F/F-seq hold the retriever constant and vary orchestration, A-minus
holds orchestration constant (naive single-shot, = A) and varies the RETRIEVER.
A-minus-vs-A therefore isolates the contribution of the hybrid+rerank pipeline
itself — the retrieval-quality axis of the study, orthogonal to the
orchestration axis.
"""
import time

from src.llm.client import generate
from src.retrieval.retrieve import format_context, retrieve
from src.systems.base import RunResult
from src.systems.system_a import ANSWER_SYSTEM_PROMPT


class SystemAMinus:
    name = "A-minus"

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        hits = retrieve(query, semantic_only=True)
        context = format_context(hits)

        result = generate(
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
        )

        return RunResult(
            answer=result["content"],
            retrieved_chunk_ids=[h["chunk_id"] for h in hits],
            n_steps=1,
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=result["cost_usd"],
        )
