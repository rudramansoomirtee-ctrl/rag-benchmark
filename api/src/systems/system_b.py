"""System B: agentic RAG with LangGraph.

State machine: RETRIEVE -> ROUTE -> (REFORMULATE -> RETRIEVE) or ANSWER

Each non-final iteration is ONE free-text LLM call that both decides and acts:
it replies either `ANSWER` (synthesise now) or `SEARCH: <query>` (the next
search query). When it answers — or on the forced final step — a dedicated
`generate()` with System A's ANSWER_SYSTEM_PROMPT produces the final response so
answer quality matches A/F.

This replaces an earlier (RouteDecision via `instructor`) + (execute) pair. The
typed one-field schema was assumed robust across providers but had never been
run on Nova until exp25, where it failed ~78% of the time: Nova Lite emits the
decision as prose, not the JSON `instructor` parses, so `action` came back
missing and the whole run errored. Free-text keyword routing is robust on
Nova/Qwen3/Anthropic alike and roughly halves B's LLM calls (the route step no
longer needs a separate execute call to write the query).

Evidence ACCUMULATES across iterations (IRCoT-style union): both the route and
answer calls operate on the RRF-fused merge of every iteration's ranked
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
import re
import time

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

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
from src.trace import trace_event

logger = logging.getLogger("rag.system_b")


ROUTE_SYSTEM_PROMPT = (
    "You are an iterative research assistant answering a multi-hop question over "
    "a news corpus. You are given the ORIGINAL question and the EVIDENCE gathered "
    "so far — the best-ranked chunks fused across every search run so far.\n"
    "\n"
    "Reply in EXACTLY one of these two forms, on a single line:\n"
    "  ANSWER\n"
    "      - the gathered evidence already contains every fact needed.\n"
    "  SEARCH: <query>\n"
    "      - a fact is still missing. Write ONE new search query naming the "
    "specific missing entity, date, or relationship, with distinct keywords from "
    "earlier searches (no trivial paraphrase).\n"
    "\n"
    "Reply with ONLY 'ANSWER' or 'SEARCH: <query>' — no other text."
)


class AgentState(TypedDict):
    original_query: str
    current_query: str
    iteration_hits: list[list[dict]]
    answer_chunks: list[dict]
    n_steps: int
    max_agent_steps: int
    semantic_only: bool
    final_answer: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float


def _retrieve_node(state: AgentState) -> AgentState:
    hits = retrieve(state["current_query"], top_k=settings.top_k, semantic_only=state["semantic_only"])
    state["iteration_hits"].append(hits)
    state["n_steps"] += 1
    return state


def _parse_route(text: str) -> tuple[str, str | None]:
    """Parse the free-text route reply into (action, query).

    A 'SEARCH: <query>' line means reformulate with that query; anything else
    (including a bare 'ANSWER') falls through to answering from current evidence
    — a safe default that cannot stall the step-bounded loop. Robust where the
    old instructor JSON schema failed: Nova/Qwen emit the keyword as plain text."""
    m = re.search(r"search\s*:\s*(.+)", text, re.IGNORECASE)
    if m:
        q = m.group(1).strip().splitlines()[0].strip().strip("\"'")
        if q:
            return "reformulate", q
    return "answer", None


def _route_node(state: AgentState) -> AgentState:
    """One free-text call decides the next move (and supplies the new query when
    searching); a dedicated ANSWER_SYSTEM_PROMPT call runs only when answering.

    Both calls see the same RRF-fused working memory — judging sufficiency on
    only the latest batch would re-request facts already held. On the forced
    final step the route call is skipped (the outcome is known) and we answer.
    """
    working = rrf_fuse(state["iteration_hits"])[:FUSED_ANSWER_TOP_K]
    context = format_context(working)
    state["answer_chunks"] = working
    forced = state["n_steps"] >= state["max_agent_steps"]

    if forced:
        action, query = "answer", None
    else:
        route = generate(messages=[
            {"role": "system", "content": ROUTE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Original question: {state['original_query']}\n\n"
                f"Evidence gathered so far:\n{context}\n\n"
                f"Steps taken: {state['n_steps']}/{state['max_agent_steps']}"
            )},
        ])
        _accum(state, route)
        action, query = _parse_route(route["content"] or "")

    if action == "reformulate":
        state["current_query"] = query or state["current_query"]
        trace_event("route", step=state["n_steps"], action="search", query=state["current_query"])
        logger.debug("B step %s/%s -> SEARCH %r", state["n_steps"], state["max_agent_steps"], state["current_query"])
    else:
        synth = generate(messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['original_query']}"},
        ])
        _accum(state, synth)
        state["final_answer"] = synth["content"] or "No answer produced."
        trace_event("route", step=state["n_steps"], action="answer", forced=forced)
        logger.debug("B step %s/%s -> ANSWER%s", state["n_steps"], state["max_agent_steps"], " (forced)" if forced else "")

    return state


def _accum(state: AgentState, gen: dict) -> None:
    """Accumulate a `generate()` call's tokens + cost into the agent state."""
    state["tokens_in"] += gen["tokens_in"]
    state["tokens_out"] += gen["tokens_out"]
    state["cost_usd"] += gen["cost_usd"]


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

    def __init__(self, max_agent_steps: int | None = None, semantic_only: bool = False):
        self.max_agent_steps = (
            max_agent_steps if max_agent_steps is not None else settings.max_agent_steps
        )
        self.semantic_only = semantic_only
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
            "semantic_only": self.semantic_only,
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
            n_steps=final["n_steps"],
            tokens_in=final["tokens_in"],
            tokens_out=final["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=final["cost_usd"],
        )
