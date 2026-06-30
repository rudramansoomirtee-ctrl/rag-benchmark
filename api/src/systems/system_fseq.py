"""System F-seq: sequential (self-ask) query decomposition.

Where System F decomposes a multi-hop question into sub-questions and retrieves
for them all in PARALLEL, F-seq resolves them ONE HOP AT A TIME, carrying each
hop's answer forward into the next — the "self-ask" / least-to-most pattern
(Press et al. 2023; Zhou et al. 2023).

This targets F's structural weakness on true multi-hop questions: F's later
sub-questions are bridge hops phrased descriptively ("the spouse of THAT
director"), and "that director" is a dead search query because the bridge
entity is unknown until the prior hop is answered. F-seq closes that loop:

  1. Decompose into an ORDERED list of single-hop sub-questions (shared
     `DECOMPOSE_FEWSHOT_PROMPT` / `Decomposition` with System F).
  2. For each sub-question, in order: substitute the already-resolved bridge
     answers into the retrieval query (so "that director" becomes the real
     name), retrieve, then a small LLM call answers THAT sub-question from its
     own retrieved context. The answer is carried into the next hop.
  3. Answer the original question once over the RRF-fused union of every hop's
     retrieval (top `FUSED_ANSWER_TOP_K`), with the resolved intermediate facts
     supplied as a reasoning scaffold.

Retriever and answer prompt are held constant with A/B/F, so F-seq-vs-F isolates
SEQUENTIAL vs PARALLEL decomposition, and F-seq-vs-B isolates pre-decomposed
self-ask vs free-form iterative reformulation — the three-way decomposition
carve-out. A single-hop question yields no sub-questions, so F-seq reduces to a
single retrieval + answer (System A's context).
"""
import logging
import time

from src.config import settings
from src.llm.client import generate
from src.retrieval.retrieve import (
    FUSED_ANSWER_TOP_K,
    format_context,
    retrieve,
    rrf_fuse,
)
from src.systems.base import RunResult
from src.systems.system_a import ANSWER_SYSTEM_PROMPT
from src.systems.system_f import DECOMPOSE_FEWSHOT_PROMPT, _decompose
from src.trace import trace_event

logger = logging.getLogger("rag.system_fseq")

# Per-hop answer call sees a tight context — it only needs the single bridge fact,
# so a small budget keeps the extra LLM calls cheap and focused.
SUB_HOP_TOP_K = 5

SUBANSWER_SYSTEM_PROMPT = (
    "You are resolving ONE sub-question of a larger multi-hop question, using only "
    "the provided context.\n"
    "Reply with ONLY the answer — the shortest exact phrase (an entity, date, "
    "number, or yes/no), nothing else. If the context does not answer it, reply "
    "exactly 'UNKNOWN'."
)


def _is_resolved(sub_answer: str) -> bool:
    """A usable bridge answer — non-empty and not the 'UNKNOWN' sentinel. Unusable
    answers are not carried forward, so a failed hop can't poison the next query."""
    return bool(sub_answer) and not sub_answer.strip().upper().startswith("UNKNOWN")


class SystemFSeq:
    name = "F-seq"

    def __init__(self, semantic_only: bool = False):
        self.semantic_only = semantic_only

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        subs, tin, tout, cost = _decompose(query)

        ranked_lists: list[list[dict]] = [
            retrieve(query, top_k=settings.top_k, semantic_only=self.semantic_only)
        ]
        resolved: list[tuple[str, str]] = []

        for sub in subs:
            if resolved:
                known = "; ".join(f"{q} -> {a}" for q, a in resolved)
                search_q = f"{sub} (known so far: {known})"
            else:
                search_q = sub
            hits = retrieve(search_q, top_k=settings.top_k, semantic_only=self.semantic_only)
            ranked_lists.append(hits)

            hop = generate(messages=[
                {"role": "system", "content": SUBANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Context:\n{format_context(hits[:SUB_HOP_TOP_K])}\n\n"
                    f"Sub-question: {sub}"
                )},
            ])
            tin += hop["tokens_in"]
            tout += hop["tokens_out"]
            cost += hop["cost_usd"]
            sub_answer = (hop["content"] or "").strip()
            if _is_resolved(sub_answer):
                resolved.append((sub, sub_answer))
            trace_event("self_ask_hop", subquestion=sub, answer=sub_answer, used=_is_resolved(sub_answer))

        fused = rrf_fuse(ranked_lists)
        answer_chunks = fused[:FUSED_ANSWER_TOP_K]
        scaffold = ""
        if resolved:
            scaffold = "\n\nResolved intermediate facts:\n" + "\n".join(
                f"- {q} {a}" for q, a in resolved
            )
        gen = generate(messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context:\n{format_context(answer_chunks)}{scaffold}\n\n"
                f"Question: {query}"
            )},
        ])

        return RunResult(
            answer=gen["content"],
            retrieved_chunk_ids=[h["chunk_id"] for h in answer_chunks],
            all_retrieved_chunk_ids=[h["chunk_id"] for h in fused],
            n_steps=len(ranked_lists),
            tokens_in=tin + gen["tokens_in"],
            tokens_out=tout + gen["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=cost + gen["cost_usd"],
        )
