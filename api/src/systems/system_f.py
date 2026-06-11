"""System F: query-decomposition multi-hop RAG.

Decomposes a multi-hop question into single-hop sub-questions, retrieves for
each (plus the original) over the SAME hybrid+rerank pipeline as A/B, fuses
the results with Reciprocal Rank Fusion, and answers once over the fused top
`FUSED_ANSWER_TOP_K` chunks. Holding the retriever constant means F's effect
is attributable to decomposition, not retrieval quality — the point of the
comparison study.

Distinct from System B: B reformulates one query iteratively, conditioned on
what it has already seen; F decomposes up front into parallel sub-questions.
Both merge their per-query lists with the shared `rrf_fuse` and answer over
the same `FUSED_ANSWER_TOP_K` budget, so they differ only in how the extra
queries are produced.

On a single-hop question the decomposer returns no sub-questions, so F reduces
to a single retrieval over the original query — System A's exact context.
"""
import logging
import time

from src.config import settings
from src.llm.client import generate, make_instructor_client
from src.retrieval.retrieve import (
    FUSED_ANSWER_TOP_K,
    format_context,
    retrieve,
    rrf_fuse,
)
from src.systems.base import RunResult
from src.systems.schemas import Decomposition
from src.systems.system_a import ANSWER_SYSTEM_PROMPT

logger = logging.getLogger("rag.system_f")


DECOMPOSE_FEWSHOT_PROMPT = (
    "You break a multi-hop question into the minimal set of standalone, single-hop "
    "sub-questions whose answers, combined, answer the original.\n"
    "\n"
    "Rules:\n"
    "- Each sub-question must stand alone: name the entity explicitly, no pronouns "
    "referring to other sub-questions.\n"
    "- Use 2-4 sub-questions for a multi-hop question; return an EMPTY list if it is "
    "already single-hop.\n"
    "- Do not answer the question; only decompose it.\n"
    "\n"
    "Examples:\n"
    "\n"
    "Q: 'Do the TechCrunch article on software companies and the Hacker News article on "
    "The Epoch Times both report an increase in revenue related to payment and "
    "subscription models, respectively?'\n"
    "Sub-questions:\n"
    "- 'Does the TechCrunch article on software companies report an increase in revenue "
    "related to payment models?'\n"
    "- 'Does the Hacker News article on The Epoch Times report an increase in revenue "
    "related to subscription models?'\n"
    "\n"
    "Q: 'Who is the individual associated with the cryptocurrency industry facing a "
    "criminal trial on fraud and conspiracy charges?'\n"
    "Sub-questions: [] (already single-hop)\n"
    "\n"
    "Q: 'Which article published earlier, the TechCrunch piece on Pixel 8 or The Verge "
    "piece on Spotify, also mentions layoffs?'\n"
    "Sub-questions:\n"
    "- 'When was the TechCrunch article on Pixel 8 published?'\n"
    "- 'When was The Verge article on Spotify published?'\n"
    "- 'Does the TechCrunch article on Pixel 8 mention layoffs?'\n"
    "- 'Does The Verge article on Spotify mention layoffs?'\n"
    "\n"
    "Q: 'According to the Sporting News report and the CBSSports.com report, which team "
    "won the matchup discussed?'\n"
    "Sub-questions:\n"
    "- 'Which team did the Sporting News report say won the matchup?'\n"
    "- 'Which team did the CBSSports.com report say won the matchup?'"
)

MAX_SUBQUESTIONS = 4


def _decompose(query: str) -> tuple[list[str], int, int, float]:
    """Return (subquestions, tokens_in, tokens_out, cost_usd) for the decompose call.

    Graceful degradation: when the instructor call exhausts retries (Nova/Qwen
    JSON-mode can append prose after the JSON object, which Pydantic's strict
    parser rejects), return no sub-questions so F proceeds as a single
    retrieval instead of crashing the run into a stub row.
    """
    client = make_instructor_client()
    try:
        decomp, raw = client.chat.completions.create_with_completion(
            model=settings.litellm_model,
            response_model=Decomposition,
            aws_region_name=settings.aws_region,
            temperature=0,
            messages=[
                {"role": "system", "content": DECOMPOSE_FEWSHOT_PROMPT},
                {"role": "user", "content": query},
            ],
        )
    except Exception as e:
        logger.warning(
            "F decompose failed (%s); falling back to no sub-questions",
            type(e).__name__,
        )
        return [], 0, 0, 0.0
    usage = getattr(raw, "usage", None)
    tin = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    tout = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    cost = float((getattr(raw, "_hidden_params", None) or {}).get("response_cost") or 0.0)
    subs = [s.strip() for s in decomp.subquestions if s and s.strip()][:MAX_SUBQUESTIONS]
    return subs, tin, tout, cost


class SystemF:
    name = "F"

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        subs, tin, tout, cost = _decompose(query)

        queries = [query, *subs]
        ranked_lists = [retrieve(q, top_k=settings.top_k) for q in queries]
        fused = rrf_fuse(ranked_lists)

        answer_chunks = fused[: FUSED_ANSWER_TOP_K]
        context = format_context(answer_chunks)
        gen = generate(
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
        )

        return RunResult(
            answer=gen["content"],
            retrieved_chunk_ids=[h["chunk_id"] for h in answer_chunks],
            all_retrieved_chunk_ids=[h["chunk_id"] for h in fused],
            hhem_score=None,
            flagged=None,
            n_steps=len(queries),
            tokens_in=tin + gen["tokens_in"],
            tokens_out=tout + gen["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=cost + gen["cost_usd"],
        )
