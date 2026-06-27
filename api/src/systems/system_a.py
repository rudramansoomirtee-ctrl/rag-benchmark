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
    "1. Answer by combining and reasoning over the provided context. A multi-hop "
    "answer is rarely stated verbatim in one chunk — connect facts across chunks "
    "to derive it, and if the chain of evidence supports a conclusion, COMMIT to it.\n"
    "2. Do not invent facts. Reply that the context does not contain the answer "
    "ONLY when a fact the question genuinely requires is absent — never merely "
    "because you are unsure, or because the answer is implied rather than quoted.\n"
    "3. Cite the chunk IDs in square brackets after each claim, e.g. "
    "'X joined the board in 2023 [chunk-7], replacing Y [chunk-12].'\n"
    "4. Be direct and concise. No hedging openers like 'Based on the context'.\n"
    "5. On a new final line, end with 'Final answer: ' followed by the shortest "
    "exact answer only (an entity, number, date, or yes/no) — no citations, no "
    "explanation. If a required fact is truly missing, write 'Final answer: The "
    "provided context does not contain the answer.' Only the text after the last "
    "'Final answer:' is scored."
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
