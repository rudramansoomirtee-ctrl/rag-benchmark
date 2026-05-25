"""System G: multi-tool agentic retrieval.

Where System B reformulates the query but always calls the same hybrid+rerank
retriever, System G picks between three retrieval tools on each step:

  - retrieve_semantic — dense kNN only; best when the user query and the
    relevant chunk text share paraphrased meaning rather than exact keywords.
  - retrieve_bm25 — BM25 only; best for named entities, exact strings,
    queries naming a person/place/title whose token appears in the chunk.
  - retrieve_filtered — hybrid (BM25 + dense + RRF) scoped by a metadata
    filter (source, category). Best for "the Hacker News article on…" or
    category-scoped comparisons.

This is the gap Ferrazzi et al. (2026, ACL Industry Track) explicitly call out:
their agentic comparator was single-tool. G is the multi-tool comparator.

Option-1 two-call routing (matches System B's redesign):

    CALL_TOOL -> ROUTE -> (CALL_TOOL)+ or END

Each agent iteration runs TWO LLM calls:
  1. A tiny ToolRouteDecision (one-field schema: action ∈ {semantic, bm25,
     filtered, answer}).
  2. An execute call shaped to the chosen route:
     - retrieve_semantic / retrieve_bm25 → free-text `generate()` for the new query
     - retrieve_filtered → small FilteredToolArgs schema (query + filter_source +
       filter_category)
     - answer → free-text `generate()` returning the final answer

This decouples routing from content generation, so G works on Anthropic,
Qwen3, Nova, and other providers without provider-specific schema gymnastics.
Cost is roughly ~2× the legacy single-call design per step but the
orchestration story is portable across model classes.

Termination: action == ANSWER, or n_steps >= max_agent_steps (per-instance budget,
defaulting to settings.max_agent_steps — same convention as System B).
"""
import logging
import time

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from src.config import settings
from src.llm.client import generate, make_instructor_client
from src.retrieval.retrieve import (
    format_context,
    retrieve_bm25,
    retrieve_filtered,
    retrieve_semantic,
)
from src.systems.base import RunResult
from src.systems.schemas import FilteredToolArgs, ToolAction, ToolRouteDecision
from src.systems.system_a import ANSWER_SYSTEM_PROMPT

logger = logging.getLogger("rag.system_g")


ROUTE_SYSTEM_PROMPT = (
    "You are answering a multi-hop news question by orchestrating retrieval tools.\n"
    "\n"
    "Available tools (pick ONE this step):\n"
    "  - retrieve_semantic: dense vector search. Best for paraphrased questions.\n"
    "  - retrieve_bm25: keyword/BM25 search. Best when the query names a specific\n"
    "    entity, person, or exact phrase.\n"
    "  - retrieve_filtered: hybrid search scoped by a publisher source and/or\n"
    "    category. Best for 'the Hacker News article on X' or topic-scoped queries.\n"
    "  - answer: current context is sufficient — synthesise now.\n"
    "\n"
    "Hard rules:\n"
    "  - On the final allowed step (n == budget) you MUST ANSWER.\n"
    "  - Do NOT produce the query, filters, or answer in this call — only the action."
)

QUERY_SYSTEM_PROMPT = (
    "You write a targeted search query for the retrieval tool chosen this step.\n"
    "\n"
    "Rules:\n"
    "- Name a specific missing entity, date, or relationship.\n"
    "- Use distinct keywords from the original — avoid trivial paraphrase.\n"
    "- Reply with just the query — no preamble, no quotes."
)

FILTER_ARGS_SYSTEM_PROMPT = (
    "You build arguments for retrieve_filtered(query, filter_source, filter_category).\n"
    "\n"
    "Available filter values:\n"
    "  - filter_source (exact publisher name, or null): 'Hacker News', 'TechCrunch',\n"
    "    'The Verge', 'Polygon', 'Sporting News', 'The Guardian', 'Fortune',\n"
    "    'Engadget', 'CBSSports.com', 'The Age', 'Essentially Sports'.\n"
    "  - filter_category (or null): sports | technology | entertainment | business |\n"
    "    science | health.\n"
    "\n"
    "Set ONLY the filters the question explicitly names; leave others null."
)


class AgentStateG(TypedDict):
    original_query: str
    current_query: str
    last_tool: str
    last_filters: dict | None
    retrieved_chunks: list[dict]
    all_retrieved_ids: list[str]
    tool_log: list[str]
    n_steps: int
    max_agent_steps: int
    final_answer: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float


def _accumulate_cost(state: AgentStateG, raw) -> None:
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


def _call_tool(state: AgentStateG) -> AgentStateG:
    """Execute the previously-selected retrieval tool.

    Chunks accumulate across iterations (deduped by chunk_id) — the agent picks
    DIFFERENT tools/filters per step specifically to gather complementary
    evidence, so the union is the right basis for the next routing decision.
    This is the key semantic difference from System B, which overwrites because
    each B iteration is a *replacement* reformulation.
    """
    tool = state["last_tool"]
    q = state["current_query"]
    if tool == ToolAction.SEMANTIC.value:
        hits = retrieve_semantic(q, top_k=settings.top_k)
    elif tool == ToolAction.BM25.value:
        hits = retrieve_bm25(q, top_k=settings.top_k)
    elif tool == ToolAction.FILTERED.value:
        hits = retrieve_filtered(
            q, filters=state.get("last_filters") or {}, top_k=settings.top_k
        )
    else:
        hits = retrieve_semantic(q, top_k=settings.top_k)

    existing = {h["chunk_id"] for h in state["retrieved_chunks"]}
    for h in hits:
        if h["chunk_id"] not in existing:
            state["retrieved_chunks"].append(h)
            existing.add(h["chunk_id"])
        state["all_retrieved_ids"].append(h["chunk_id"])
    state["n_steps"] += 1
    return state


def _route_node(state: AgentStateG) -> AgentStateG:
    """Two-call routing: tiny ToolRouteDecision, then a tool-specific execute call."""
    client = make_instructor_client()
    context = format_context(state["retrieved_chunks"])
    user_prompt = (
        f"Original question: {state['original_query']}\n\n"
        f"Current context (cumulative):\n{context}\n\n"
        f"Steps taken: {state['n_steps']}/{state['max_agent_steps']}"
    )

    decision, raw = client.chat.completions.create_with_completion(
        model=settings.litellm_model,
        response_model=ToolRouteDecision,
        aws_region_name=settings.aws_region,
        temperature=0,
        messages=[
            {"role": "system", "content": ROUTE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    _accumulate_cost(state, raw)

    forced_answer = state["n_steps"] >= state["max_agent_steps"]
    if decision.action == ToolAction.ANSWER or forced_answer:
        synth = generate(messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['original_query']}"},
        ])
        state["final_answer"] = synth["content"] or "No answer produced."
        state["tokens_in"] += synth["tokens_in"]
        state["tokens_out"] += synth["tokens_out"]
        state["cost_usd"] += synth["cost_usd"]
        state["tool_log"].append(
            f"step{state['n_steps']}=ANSWER"
            + (" (forced)" if forced_answer and decision.action != ToolAction.ANSWER else "")
        )
        return state

    # Execute the chosen retrieval tool — build args.
    state["last_tool"] = decision.action.value
    if decision.action == ToolAction.FILTERED:
        args, raw2 = client.chat.completions.create_with_completion(
            model=settings.litellm_model,
            response_model=FilteredToolArgs,
            aws_region_name=settings.aws_region,
            temperature=0,
            messages=[
                {"role": "system", "content": FILTER_ARGS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        _accumulate_cost(state, raw2)
        state["current_query"] = args.query or state["original_query"]
        filters = {}
        if args.filter_source:
            filters["source"] = args.filter_source
        if args.filter_category:
            filters["category"] = args.filter_category
        state["last_filters"] = filters or None
    else:
        # retrieve_semantic / retrieve_bm25 — free-text query call.
        q_call = generate(messages=[
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + "\n\nWrite the new search query:"},
        ])
        state["current_query"] = (q_call["content"] or "").strip() or state["original_query"]
        state["last_filters"] = None
        state["tokens_in"] += q_call["tokens_in"]
        state["tokens_out"] += q_call["tokens_out"]
        state["cost_usd"] += q_call["cost_usd"]

    state["tool_log"].append(
        f"step{state['n_steps']}={decision.action.value}"
        + (f"({state['last_filters']})" if state["last_filters"] else "")
    )
    logger.debug(
        "G step %s/%s action=%s filters=%s query=%r",
        state["n_steps"], state["max_agent_steps"], decision.action.value,
        state["last_filters"], state["current_query"][:80],
    )
    return state


def _route(state: AgentStateG) -> str:
    if state["final_answer"] is not None:
        return "end"
    return "call_tool"


def _build_graph():
    g = StateGraph(AgentStateG)
    g.add_node("call_tool", _call_tool)
    g.add_node("route", _route_node)
    g.set_entry_point("call_tool")
    g.add_edge("call_tool", "route")
    g.add_conditional_edges("route", _route, {"call_tool": "call_tool", "end": END})
    return g.compile()


class SystemG:
    name = "G"

    def __init__(self, max_agent_steps: int | None = None):
        self.max_agent_steps = (
            max_agent_steps if max_agent_steps is not None else settings.max_agent_steps
        )
        self._graph = _build_graph()

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        initial: AgentStateG = {
            "original_query": query,
            "current_query": query,
            # First step: default to hybrid-equivalent (semantic) so G doesn't
            # trivially reduce to the BM25 baseline on step 0.
            "last_tool": ToolAction.SEMANTIC.value,
            "last_filters": None,
            "retrieved_chunks": [],
            "all_retrieved_ids": [],
            "tool_log": [],
            "n_steps": 0,
            "max_agent_steps": self.max_agent_steps,
            "final_answer": None,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }
        final = self._graph.invoke(initial)

        logger.info("G tools=%s", " ".join(final.get("tool_log", [])))

        answer = final.get("final_answer") or "No answer."
        final_chunk_ids = [h["chunk_id"] for h in final.get("retrieved_chunks", [])]

        return RunResult(
            answer=answer,
            retrieved_chunk_ids=final_chunk_ids,
            hhem_score=None,
            flagged=None,
            n_steps=final["n_steps"],
            tokens_in=final["tokens_in"],
            tokens_out=final["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=final["cost_usd"],
        )
