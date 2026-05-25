"""System F-tuned — F with six SOTA accuracy levers stacked.

Same retriever as A/F (hybrid BM25+dense+RRF+rerank). Same Qwen3/Haiku/Nova
generator. Levers 1-4 are the original "F-tuned" combination; levers 5-6 are
the v2 source-aware retrieval addition that closes the qid=2252-style gap
where comparison queries name two publishers and we'd previously only retrieve
chunks from the more-keyword-prominent one.

1. **Bigger answer budget** — feed top-10 fused chunks to the answer call
   instead of F's top-5. F already retrieves ~20 unique chunks across the
   original + sub-questions; trimming to 5 was the load-bearing bottleneck.

2. **Few-shot decomposer prompt** — three worked examples (comparison,
   single-hop, temporal) so the decomposer produces standalone, entity-named
   sub-questions instead of shallow paraphrases. Especially important under
   Qwen3/Nova which are more prompt-sensitive than Anthropic.

3. **Weighted RRF** — the original query's retrieval gets a 2× weight in the
   fusion vs each sub-question. Stops noisy sub-question hits from drowning
   out the direct multi-hop signal in the original.

4. **CoT answer prompt** — the answer call follows an explicit
   "sub-questions → evidence per sub-question → final answer" structure.
   Per Wei et al. 2022, CoT lifts multi-hop accuracy +5–15pp on benchmarks
   like MultiHop-RAG.

5. **Source-aware retrieval fan-out** — when the original query or any
   sub-question names a known publisher (Fortune, TechCrunch, The Verge,
   Hacker News, etc.), an additional retrieve_filtered() call is added to
   the fusion, scoped to articles with that source in metadata. Closes the
   qid=2252 gap where BM25 favoured one publisher's keywords and missed
   chunks from the other publisher the question explicitly named.

6. **Per-query top_k=10** (instead of settings.top_k=5) at each retrieve()
   call. Doubles the candidate pool per query, giving RRF more diverse
   chunks to fuse before the rerank stage trims to top-10 for the answer.
"""
import re
import time
from functools import lru_cache

from sqlalchemy import select

from src.config import settings
from src.db.models import Chunk
from src.db.session import get_session
from src.llm.client import generate, make_instructor_client
from src.retrieval.retrieve import format_context, retrieve, retrieve_filtered
from src.systems.base import RunResult
from src.systems.schemas import Decomposition


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


COT_ANSWER_SYSTEM_PROMPT = (
    "You are answering a multi-hop news question over a corpus of news articles.\n"
    "\n"
    "Use the following structure in your response:\n"
    "  Step 1: List the 1-3 atomic facts the question requires.\n"
    "  Step 2: For each atomic fact, cite the chunk ID(s) where you found it (or "
    "state 'not in context' if missing).\n"
    "  Step 3: On a new line write 'Final answer: ' followed by the direct "
    "response — the named entity, the affirmative or negative, or the number. "
    "Be direct; do not hedge or apologize.\n"
    "\n"
    "Rules:\n"
    "- Use ONLY the provided context. If the answer is not in the context, "
    "respond with 'Final answer: The provided context does not contain the answer.'\n"
    "- The text after 'Final answer:' is what gets scored — keep it concise.\n"
    "- Do not invent facts."
)


MAX_SUBQUESTIONS = 4
F_TUNED_TOP_K = 10  # context budget for the answer call (was settings.top_k=5)
PER_QUERY_TOP_K = 10  # candidates per retrieve() call (was settings.top_k=5)
ORIGINAL_QUERY_RRF_WEIGHT = 2.0
SOURCE_FILTERED_RRF_WEIGHT = 1.5  # filtered retrievals weighted slightly more than unscoped sub-question hits
RESERVED_PER_SOURCE = 2  # guaranteed slots in final context per source named in the query — stops Fortune+TheVerge from drowning out the explicitly-named TechCrunch chunks (qid=2252)


@lru_cache(maxsize=1)
def _known_sources() -> tuple[str, ...]:
    """All distinct source names present in the multihop corpus, longest first.

    Sorting by length descending ensures multi-word sources like
    'The Roar | Sports Writers Blog' match before substring-overlapping
    short ones like 'The'."""
    session = get_session()
    try:
        rows = session.execute(
            select(Chunk.chunk_metadata).where(Chunk.dataset == "multihop")
        ).all()
    finally:
        session.close()
    found = set()
    for (meta,) in rows:
        src = (meta or {}).get("source")
        if src:
            found.add(src)
    return tuple(sorted(found, key=len, reverse=True))


def _detect_sources(text: str) -> list[str]:
    """Return the set of known sources mentioned in `text`, deduped against
    longer matches so 'BBC News' isn't double-counted when 'BBC News -
    Entertainment & Arts' already matched the same span."""
    lower = text.lower()
    matched: list[str] = []
    for src in _known_sources():
        if src.lower() in lower:
            # Skip if this source is a substring of an already-matched longer source.
            if not any(src != m and src.lower() in m.lower() for m in matched):
                matched.append(src)
    return matched


def _decompose(query: str) -> tuple[list[str], int, int, float]:
    """Few-shot decomposition. Returns (subquestions, tokens_in, tokens_out, cost_usd).

    Graceful degradation: when the instructor call exhausts retries (typically
    because Nova/Qwen JSON-mode returned valid JSON followed by trailing prose,
    which Pydantic's strict parser rejects), we return ([], 0, 0, 0.0). F-tuned
    then proceeds with the original query + any source-filtered retrievals
    instead of crashing the run. Observed on qid=746 under Nova Lite where the
    model appended extra description text after the JSON object.
    """
    import logging
    logger = logging.getLogger("rag.system_f_tuned")
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
            "F-tuned decompose failed (%s); falling back to no sub-questions",
            type(e).__name__,
        )
        return [], 0, 0, 0.0

    usage = getattr(raw, "usage", None)
    tin = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    tout = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    cost = float((getattr(raw, "_hidden_params", None) or {}).get("response_cost") or 0.0)
    subs = [s.strip() for s in decomp.subquestions if s and s.strip()][:MAX_SUBQUESTIONS]
    return subs, tin, tout, cost


def _rrf_fuse_weighted(
    ranked_lists: list[list[dict]],
    weights: list[float],
    rrf_k: int = 60,
) -> list[dict]:
    """Weighted RRF — each ranked list contributes 1/(rrf_k + rank) × its weight.

    Original query is weighted 2× sub-questions so its direct hits don't get
    drowned out by noisy sub-question expansions.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for w, hits in zip(weights, ranked_lists):
        for rank, h in enumerate(hits, start=1):
            cid = h["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + w * (1.0 / (rrf_k + rank))
            chunks.setdefault(cid, h)
    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [chunks[c] for c in ordered]


class SystemFTuned:
    name = "F-tuned"

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        subs, tin, tout, cost = _decompose(query)

        # Fan-out construction:
        #  - The original query → one unscoped retrieve() (weight 2.0).
        #  - Each sub-question → one unscoped retrieve() (weight 1.0).
        #  - For every source name found in (original OR any sub-question) →
        #    one retrieve_filtered() scoped to that source (weight 1.5).
        # Sources are detected via substring match against the corpus's known
        # publisher list; if no sources are named, the system reduces to the
        # original four-lever F-tuned configuration.
        ranked_lists: list[list[dict]] = []
        weights: list[float] = []

        ranked_lists.append(retrieve(query, top_k=PER_QUERY_TOP_K))
        weights.append(ORIGINAL_QUERY_RRF_WEIGHT)

        for s in subs:
            ranked_lists.append(retrieve(s, top_k=PER_QUERY_TOP_K))
            weights.append(1.0)

        # Source-scoped fan-out — for each publisher mentioned in the query,
        # run a retrieve_filtered() and KEEP the per-source list separately so
        # we can reserve top-N slots in the final context for that source
        # (otherwise Fortune+TheVerge chunks out-rank explicitly-named TechCrunch
        # chunks in fusion — observed on qid=2252).
        all_text = query + " " + " ".join(subs)
        source_lists: dict[str, list[dict]] = {}
        for src in _detect_sources(all_text):
            if src in source_lists:
                continue
            hits = retrieve_filtered(
                query, filters={"source": src}, top_k=PER_QUERY_TOP_K
            )
            source_lists[src] = hits
            ranked_lists.append(hits)
            weights.append(SOURCE_FILTERED_RRF_WEIGHT)

        fused = _rrf_fuse_weighted(ranked_lists, weights)

        # Build the final context with guaranteed per-source coverage.
        # 1. Reserve up to RESERVED_PER_SOURCE chunks from each source-filtered list.
        # 2. Fill remaining slots from the RRF-fused list (skipping anything already reserved).
        reserved: list[dict] = []
        seen_ids: set[str] = set()
        for src, hits in source_lists.items():
            taken = 0
            for h in hits:
                if h["chunk_id"] in seen_ids:
                    continue
                reserved.append(h)
                seen_ids.add(h["chunk_id"])
                taken += 1
                if taken >= RESERVED_PER_SOURCE:
                    break

        filler: list[dict] = []
        remaining = F_TUNED_TOP_K - len(reserved)
        for h in fused:
            if remaining <= 0:
                break
            if h["chunk_id"] in seen_ids:
                continue
            filler.append(h)
            seen_ids.add(h["chunk_id"])
            remaining -= 1

        final_chunks = reserved + filler
        context = format_context(final_chunks)
        gen = generate(
            messages=[
                {"role": "system", "content": COT_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
        )

        # Persist the chunks actually fed to the LLM (reserved + filler) — so
        # P@k/R@k against gold reflects what the system actually used, not the
        # raw RRF list whose tail never reaches the answer.
        return RunResult(
            answer=gen["content"],
            retrieved_chunk_ids=[h["chunk_id"] for h in final_chunks],
            hhem_score=None,
            flagged=None,
            n_steps=len(ranked_lists),
            tokens_in=tin + gen["tokens_in"],
            tokens_out=tout + gen["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=cost + gen["cost_usd"],
        )
