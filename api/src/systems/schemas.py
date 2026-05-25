"""Typed schemas for the agent's structured outputs (used by Systems B and F)."""
from enum import Enum

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    REFORMULATE = "reformulate"
    ANSWER = "answer"


class AgentDecision(BaseModel):
    """The agent's choice for the next step in the RETRIEVE/REFORMULATE/ANSWER loop."""

    action: AgentAction = Field(description="What to do next given retrieved context.")
    reformulated_query: str | None = Field(
        default=None,
        description="Required if action=reformulate. The improved query.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Required if action=answer. The final response.",
    )
    reasoning: str = Field(default="", description="Brief justification, <=30 words.")


class Decomposition(BaseModel):
    """Single-hop sub-questions that together answer a multi-hop question (System F)."""

    subquestions: list[str] = Field(
        default_factory=list,
        description="2-4 standalone single-hop sub-questions; empty if already single-hop.",
    )
    reasoning: str = Field(default="", description="Brief justification, <=30 words.")


class RouteDecision(BaseModel):
    """Tiny one-field schema used by Option-1 re-architecture of System B.

    Splitting the agent loop into (route, execute) calls — instead of a single
    multi-field instructor call — makes B robust across providers (Qwen3, Nova,
    DeepSeek). Each call now has a trivial schema that any function-calling LLM
    can populate reliably.
    """

    action: AgentAction = Field(description="reformulate (gather more evidence) | answer (synthesise now).")
    reasoning: str = Field(default="", description="Brief justification, <=20 words.")


class JudgeLabel(str, Enum):
    PERFECT = "perfect"
    ACCEPTABLE = "acceptable"
    MISSING = "missing"
    INCORRECT = "incorrect"


class JudgeVerdict(BaseModel):
    """CRAG-rubric answer-quality label for the secondary LLM-as-judge metric."""

    label: JudgeLabel = Field(
        description="perfect | acceptable | missing | incorrect, per the CRAG rubric."
    )
    reasoning: str = Field(default="", description="Brief justification, <=30 words.")
