"""System B: agentic RAG with LangGraph.

State machine: RETRIEVE -> DECIDE -> (REFORMULATE -> RETRIEVE) or ANSWER
Uses Instructor for typed decisions, so we can never parse-fail on the agent's action.

This is a skeleton — adapt from your production LangGraph code. The structure here
keeps the same interface as System A so the runner stays uniform.
"""
import time

import instructor
from langgraph.graph import StateGraph, END
from litellm import completion
from typing_extensions import TypedDict

from src.config import settings
from src.llm.client import generate
from src.retrieval.embeddings import embed_one
from src.retrieval.opensearch_client import knn_search
from src.systems.base import RunResult
from src.systems.schemas import AgentAction, AgentDecision


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
    qvec = embed_one(state["current_query"])
    hits = knn_search(qvec, top_k=settings.top_k)
    state["retrieved_chunks"] = hits
    state["all_retrieved_ids"].extend(h["chunk_id"] for h in hits)
    state["n_steps"] += 1
    return state


def _decide_node(state: AgentState) -> AgentState:
    client = instructor.from_litellm(completion)
    context = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in state["retrieved_chunks"])

    decision: AgentDecision = client.chat.completions.create(
        model=settings.litellm_model,
        response_model=AgentDecision,
        aws_region_name=settings.aws_region,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Given the retrieved context and the original question, decide: "
                    "either reformulate the query for better retrieval, or answer if the "
                    "context is sufficient."
                ),
            },
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

        # Token + cost accounting from the agent loop is captured by Phoenix spans;
        # for SQL aggregation, do a final synthesis call only if the loop ended without one.
        answer = final.get("final_answer") or "No answer."

        return RunResult(
            answer=answer,
            retrieved_chunk_ids=list(dict.fromkeys(final["all_retrieved_ids"])),
            hhem_score=None,
            flagged=None,
            n_steps=final["n_steps"],
            # Token/cost tallies come from Phoenix in v2; populate from the instructor
            # response_cost in your iteration. Stubbed for now.
            tokens_in=0,
            tokens_out=0,
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=0.0,
        )
