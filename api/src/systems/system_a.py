"""System A: naive RAG. Embed query -> hybrid retrieve+rerank -> answer once.

No reformulation. This is the single-shot baseline.
Retrieval uses the same hybrid+rerank pipeline as B/F so that the
A/B/F comparison isolates orchestration strategy, not retrieval quality.
"""
import time

from src.llm.client import generate
from src.retrieval.retrieve import format_context, retrieve
from src.systems.base import RunResult


ANSWER_SYSTEM_PROMPT = (
    "You are answering a multi-hop question over a corpus of news articles.\n"
    "\n"
    "Rules:\n"
    "1. Use ONLY the provided context. If the answer is not in the context, "
    "say 'The provided context does not contain the answer.' Do not invent.\n"
    "2. Multi-hop questions require synthesising facts from MULTIPLE chunks. "
    "Connect the dots explicitly.\n"
    "3. Cite the chunk IDs in square brackets after each claim, e.g. "
    "'X joined the board in 2023 [chunk-7], replacing Y [chunk-12].'\n"
    "4. Be direct and concise. No hedging openers like 'Based on the context'."
)


class SystemA:
    name = "A"

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        hits = retrieve(query)
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
