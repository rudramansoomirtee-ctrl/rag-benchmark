"""System F: query-decomposition multi-hop RAG.

Decomposes a multi-hop question into single-hop sub-questions, retrieves for each
(plus the original) over the SAME hybrid+rerank pipeline as A/B, fuses the
results with Reciprocal Rank Fusion, and answers once. Holding the retriever
constant means F's effect is attributable to decomposition, not retrieval
quality — the point of the comparison study.

Distinct from System B: B reformulates one query iteratively, conditioned on
what it has already seen; F decomposes up front into parallel sub-questions and
fuses. Both share the retriever; they differ only in orchestration.

On a single-hop question the decomposer returns no sub-questions, so F reduces to
a single retrieval over the original query (i.e. System A's retrieval).
"""
import time

import instructor
from litellm import completion

from src.config import settings
from src.llm.client import generate
from src.retrieval.retrieve import retrieve
from src.systems.base import RunResult
from src.systems.schemas import Decomposition
from src.systems.system_a import ANSWER_SYSTEM_PROMPT


DECOMPOSE_SYSTEM_PROMPT = (
    "You break a multi-hop question into the minimal set of standalone, single-hop "
    "sub-questions whose answers, combined, answer the original.\n"
    "\n"
    "Rules:\n"
    "- Each sub-question must stand alone: name the entity, no pronouns referring to "
    "other sub-questions.\n"
    "- Use 2-4 sub-questions for a multi-hop question; return an EMPTY list if it is "
    "already single-hop.\n"
    "- Do not answer the question; only decompose it."
)

MAX_SUBQUESTIONS = 4


def _decompose(query: str) -> tuple[list[str], int, int, float]:
    """Return (subquestions, tokens_in, tokens_out, cost_usd) for the decompose call."""
    client = instructor.from_litellm(completion)
    decomp, raw = client.chat.completions.create_with_completion(
        model=settings.litellm_model,
        response_model=Decomposition,
        aws_region_name=settings.aws_region,
        temperature=0,
        messages=[
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    usage = getattr(raw, "usage", None)
    tin = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    tout = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    cost = float((getattr(raw, "_hidden_params", None) or {}).get("response_cost") or 0.0)
    subs = [s.strip() for s in decomp.subquestions if s and s.strip()][:MAX_SUBQUESTIONS]
    return subs, tin, tout, cost


def _rrf_fuse(ranked_lists: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across per-query result lists, deduped by chunk_id."""
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, start=1):
            cid = h["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(cid, h)
    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [chunks[c] for c in ordered]


class SystemF:
    name = "F"

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        subs, tin, tout, cost = _decompose(query)

        queries = [query, *subs]
        ranked_lists = [retrieve(q, top_k=settings.top_k) for q in queries]
        fused = _rrf_fuse(ranked_lists)

        context = "\n\n".join(
            f"[{h['chunk_id']}] {h['text']}" for h in fused[: settings.top_k]
        )
        gen = generate(
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
        )

        return RunResult(
            answer=gen["content"],
            retrieved_chunk_ids=[h["chunk_id"] for h in fused],
            hhem_score=None,
            flagged=None,
            n_steps=len(queries),
            tokens_in=tin + gen["tokens_in"],
            tokens_out=tout + gen["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=cost + gen["cost_usd"],
        )
