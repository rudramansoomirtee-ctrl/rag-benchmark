"""System B: agentic RAG with LangGraph.

State machine: RETRIEVE -> ROUTE -> (REFORMULATE -> RETRIEVE) or ANSWER

Each iteration does TWO LLM calls:
  1. A tiny RouteDecision call (one-field schema: action ∈ {reformulate, answer}).
  2. An execute call — either a free-text `generate()` returning the reformulated
     query, or a free-text `generate()` returning the final answer using
     System A's ANSWER_SYSTEM_PROMPT.

The split exists to decouple routing from content generation. The previous
single-call design (one instructor invocation populating action + reformulated_query
+ final_answer conditionally) was reliable on Anthropic but fragile on Bedrock
Nova / Qwen3 — those providers either left fields empty or refused to populate
the multi-field schema at all. With Option-1 two-call routing each LLM call has
a trivial single-job schema, robust across providers. Cost is roughly ~2× the
old design per step but makes the orchestration story portable across model
classes.

Evidence ACCUMULATES across iterations (IRCoT-style union): route, reformulate
and answer all operate on the RRF-fused merge of every iteration's ranked
list, capped at FUSED_ANSWER_TOP_K — the same fusion + budget System F applies
to its sub-question lists. Without this, a chunk found at step 1 is invisible
by step 3 and the agent re-hunts evidence it already holds. A one-step run
fuses a single list, so B degenerates to System A's exact context.

Retrieval shares the hybrid+rerank pipeline used by A/F so the comparison
isolates agent behaviour from retrieval quality.

The persisted `retrieved_chunk_ids` is the fused answering context (what the
final generate() saw); `all_retrieved_chunk_ids` is the raw union of every
iteration's hits.
"""
import logging
import time

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from src.config import settings
from src.llm.client import generate, make_instructor_client
from src.retrieval.retrieve import (
    FUSED_ANSWER_TOP_K,
    format_context,
    retrieve,
    rrf_fuse,
)
from src.systems.base import RunResult
from src.systems.schemas import AgentAction, RouteDecision
from src.systems.system_a import ANSWER_SYSTEM_PROMPT

logger = logging.getLogger("rag.system_b")


ROUTE_SYSTEM_PROMPT = (
    "You are an iterative research assistant answering multi-hop questions over "
    "a news corpus.\n"
    "\n"
    "Each step you receive:\n"
    "- The ORIGINAL multi-hop question.\n"
    "- The EVIDENCE GATHERED SO FAR: the best-ranked chunks fused across every "
    "search you have run so far, not just the latest one.\n"
    "- How many steps you have used out of the budget.\n"
    "\n"
    "Choose exactly ONE action:\n"
    "  - ANSWER: the gathered evidence contains every fact needed.\n"
    "  - REFORMULATE: at least one fact is still missing.\n"
    "\n"
    "Hard rules:\n"
    "- On the final allowed step (n == budget) you MUST ANSWER.\n"
    "- Do NOT produce the answer or query in this call — only the action."
)

REFORMULATE_SYSTEM_PROMPT = (
    "You write a NEW search query that targets the missing piece of evidence.\n"
    "\n"
    "Rules:\n"
    "- Name the specific missing entity, date, or relationship rather than rephrasing.\n"
    "- Use distinct keywords from the original query — avoid trivial paraphrase.\n"
    "- Decompose into ONE sub-question if the original has multiple parts.\n"
    "- Reply with just the query — no preamble, no quotes."
)


class AgentState(TypedDict):
    original_query: str
    current_query: str
    iteration_hits: list[list[dict]]
    answer_chunks: list[dict]
    n_steps: int
    max_agent_steps: int
    final_answer: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float


def _retrieve_node(state: AgentState) -> AgentState:
    hits = retrieve(state["current_query"], top_k=settings.top_k)
    state["iteration_hits"].append(hits)
    state["n_steps"] += 1
    return state


def _route_node(state: AgentState) -> AgentState:
    """Pick ANSWER or REFORMULATE via a tiny one-field instructor call.

    Then execute the chosen route in a follow-up `generate()` call so the
    content (reformulated query or final answer) comes from a focused, free-text
    response rather than a conditional field on a multi-field schema.

    Both calls see the same fused working memory the answer would use — judging
    sufficiency on only the latest batch would re-request facts already held.
    """
    client = make_instructor_client()
    working = rrf_fuse(state["iteration_hits"])[:FUSED_ANSWER_TOP_K]
    context = format_context(working)
    user_prompt = (
        f"Original question: {state['original_query']}\n\n"
        f"Evidence gathered so far:\n{context}\n\n"
        f"Steps taken: {state['n_steps']}/{state['max_agent_steps']}"
    )

    decision, raw = client.chat.completions.create_with_completion(
        model=settings.litellm_model,
        response_model=RouteDecision,
        aws_region_name=settings.aws_region,
        temperature=0,
        messages=[
            {"role": "system", "content": ROUTE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    _accumulate_cost(state, raw)

    forced_answer = state["n_steps"] >= state["max_agent_steps"]
    if decision.action == AgentAction.ANSWER or forced_answer:
        synth = generate(messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['original_query']}"},
        ])
        state["final_answer"] = synth["content"] or "No answer produced."
        state["answer_chunks"] = working
        state["tokens_in"] += synth["tokens_in"]
        state["tokens_out"] += synth["tokens_out"]
        state["cost_usd"] += synth["cost_usd"]
        logger.debug("B step %s/%s -> ANSWER%s", state["n_steps"], state["max_agent_steps"],
                     " (forced)" if forced_answer and decision.action != AgentAction.ANSWER else "")
    else:
        ref = generate(messages=[
            {"role": "system", "content": REFORMULATE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Original question: {state['original_query']}\n\n"
                f"Evidence gathered so far (insufficient):\n{context}\n\n"
                "Write the new search query:"
            )},
        ])
        state["current_query"] = (ref["content"] or "").strip() or state["current_query"]
        state["tokens_in"] += ref["tokens_in"]
        state["tokens_out"] += ref["tokens_out"]
        state["cost_usd"] += ref["cost_usd"]
        logger.debug("B step %s/%s -> REFORMULATE %r", state["n_steps"], state["max_agent_steps"], state["current_query"])

    return state


def _accumulate_cost(state: AgentState, raw) -> None:
    """Pull usage + cost from an instructor `raw` response and accumulate.

    Falls back to litellm's pricing map when the provider's response omits
    response_cost (the same gap that previously left B's $/correct reading 0)."""
    usage = getattr(raw, "usage", None)
    tin = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    tout = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
    state["tokens_in"] += tin
    state["tokens_out"] += tout

    hidden = getattr(raw, "_hidden_params", None) or {}
    cost = float(hidden.get("response_cost") or 0.0)
    if not cost and (tin or tout):
        try:
            from litellm import cost_per_token
            pc, cc = cost_per_token(
                model=settings.litellm_model, prompt_tokens=tin, completion_tokens=tout
            )
            cost = float(pc) + float(cc)
        except Exception:
            cost = 0.0
    state["cost_usd"] += cost


def _route(state: AgentState) -> str:
    if state["final_answer"] is not None:
        return "end"
    return "retrieve"


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("route", _route_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "route")
    g.add_conditional_edges("route", _route, {"retrieve": "retrieve", "end": END})
    return g.compile()


class SystemB:
    name = "B"

    def __init__(self, max_agent_steps: int | None = None):
        self.max_agent_steps = (
            max_agent_steps if max_agent_steps is not None else settings.max_agent_steps
        )
        self._graph = _build_graph()

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        initial: AgentState = {
            "original_query": query,
            "current_query": query,
            "iteration_hits": [],
            "answer_chunks": [],
            "n_steps": 0,
            "max_agent_steps": self.max_agent_steps,
            "final_answer": None,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }
        final = self._graph.invoke(initial)

        answer = final.get("final_answer") or "No answer."
        iteration_hits = final.get("iteration_hits", [])
        answer_chunks = (
            final.get("answer_chunks")
            or rrf_fuse(iteration_hits)[:FUSED_ANSWER_TOP_K]
        )
        all_seen = list(dict.fromkeys(
            h["chunk_id"] for hits in iteration_hits for h in hits
        ))

        return RunResult(
            answer=answer,
            retrieved_chunk_ids=[h["chunk_id"] for h in answer_chunks],
            all_retrieved_chunk_ids=all_seen,
            hhem_score=None,
            flagged=None,
            n_steps=final["n_steps"],
            tokens_in=final["tokens_in"],
            tokens_out=final["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=final["cost_usd"],
        )
