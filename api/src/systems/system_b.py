"""System B: agentic RAG with LangGraph.

State machine: RETRIEVE -> DECIDE -> (REFORMULATE -> RETRIEVE) or ANSWER
Uses instructor for typed decisions so the agent's action choice cannot
parse-fail. Retrieval shares the hybrid+rerank pipeline used by A/C/D so
the comparison isolates agent behaviour from retrieval quality.

The persisted `retrieved_chunk_ids` represents the chunks present at the
agent's final iteration — i.e. the evidence it actually used to answer.
The full per-step trace is on Phoenix; P@k/R@k against the final set is
what the runner persists.
"""
import time

import instructor
from langgraph.graph import StateGraph, END
from litellm import completion
from typing_extensions import TypedDict

from src.config import settings
from src.retrieval.retrieve import retrieve
from src.systems.base import RunResult
from src.systems.schemas import AgentAction, AgentDecision


DECIDE_SYSTEM_PROMPT = (
    "You are an iterative research assistant answering multi-hop questions over "
    "a news corpus.\n"
    "\n"
    "Each step you receive:\n"
    "- The ORIGINAL multi-hop question (may require connecting facts from MULTIPLE articles).\n"
    "- The CURRENT retrieved context (top-k chunks for the latest query).\n"
    "- How many steps you have used out of the budget.\n"
    "\n"
    "Choose exactly one action:\n"
    "\n"
    "1. ANSWER — the current context contains every fact needed. Produce the final\n"
    "   answer now, synthesising across MULTIPLE chunks if the question is multi-hop.\n"
    "   Cite chunk IDs in brackets for each claim.\n"
    "\n"
    "2. REFORMULATE — at least one fact is still missing or ambiguous. Write a NEW\n"
    "   query that targets the missing piece. Good reformulations:\n"
    "   - Name a specific missing entity, date, or relationship rather than rephrasing.\n"
    "   - Decompose the question into one sub-question if it has multiple parts.\n"
    "   - Use distinct keywords from the original query — avoid trivial paraphrase.\n"
    "\n"
    "Hard rules:\n"
    "- If you have used >= 4 of your steps, prefer ANSWER with current evidence.\n"
    "- Do not invent facts. If context cannot answer, ANSWER with\n"
    "  'The provided context does not contain the answer.'"
)


class AgentState(TypedDict):
    original_query: str
    current_query: str
    retrieved_chunks: list[dict]
    all_retrieved_ids: list[str]
    n_steps: int
    final_answer: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float


def _retrieve_node(state: AgentState) -> AgentState:
    hits = retrieve(state["current_query"], top_k=settings.top_k)
    state["retrieved_chunks"] = hits
    state["all_retrieved_ids"].extend(h["chunk_id"] for h in hits)
    state["n_steps"] += 1
    return state


def _decide_node(state: AgentState) -> AgentState:
    client = instructor.from_litellm(completion)
    context = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in state["retrieved_chunks"])

    decision, raw = client.chat.completions.create_with_completion(
        model=settings.litellm_model,
        response_model=AgentDecision,
        aws_region_name=settings.aws_region,
        temperature=0,
        messages=[
            {"role": "system", "content": DECIDE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original question: {state['original_query']}\n\n"
                    f"Current context:\n{context}\n\n"
                    f"Steps taken: {state['n_steps']}/{settings.max_agent_steps}"
                ),
            },
        ],
    )

    usage = getattr(raw, "usage", None)
    if usage is not None:
        state["tokens_in"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        state["tokens_out"] += int(getattr(usage, "completion_tokens", 0) or 0)
    hidden = getattr(raw, "_hidden_params", None) or {}
    state["cost_usd"] += float(hidden.get("response_cost") or 0.0)

    if decision.action == AgentAction.ANSWER or state["n_steps"] >= settings.max_agent_steps:
        state["final_answer"] = decision.final_answer or "No answer produced."
    else:
        state["current_query"] = decision.reformulated_query or state["current_query"]

    return state


def _route(state: AgentState) -> str:
    if state["final_answer"] is not None:
        return "end"
    return "retrieve"


def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("decide", _decide_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "decide")
    g.add_conditional_edges("decide", _route, {"retrieve": "retrieve", "end": END})
    return g.compile()


class SystemB:
    name = "B"

    def __init__(self):
        self._graph = _build_graph()

    def answer(self, query: str) -> RunResult:
        t0 = time.time()
        initial: AgentState = {
            "original_query": query,
            "current_query": query,
            "retrieved_chunks": [],
            "all_retrieved_ids": [],
            "n_steps": 0,
            "final_answer": None,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }
        final = self._graph.invoke(initial)

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
