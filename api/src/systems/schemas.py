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


class ToolAction(str, Enum):
    """Retrieval tools available to System G's agent. ANSWER ends the loop."""

    SEMANTIC = "retrieve_semantic"
    BM25 = "retrieve_bm25"
    FILTERED = "retrieve_filtered"
    ANSWER = "answer"


class RouteDecision(BaseModel):
    """Tiny one-field schema used by Option-1 re-architecture of System B.

    Splitting the agent loop into (route, execute) calls — instead of a single
    multi-field instructor call — makes B robust across providers (Qwen3, Nova,
    DeepSeek). Each call now has a trivial schema that any function-calling LLM
    can populate reliably.
    """

    action: AgentAction = Field(description="reformulate (gather more evidence) | answer (synthesise now).")
    reasoning: str = Field(default="", description="Brief justification, <=20 words.")


class ToolRouteDecision(BaseModel):
    """One-field route schema for System G — picks the retrieval tool or ANSWER.
    Tool *arguments* (query, filters) are gathered in a second call so this
    schema stays trivial even when the model is below tool-use grade."""

    action: ToolAction = Field(description="Pick the tool best suited to the missing fact, or ANSWER.")
    reasoning: str = Field(default="", description="Brief justification, <=20 words.")


class FilteredToolArgs(BaseModel):
    """Tool arguments populated only when action=retrieve_filtered. Two- or
    three-field schema — much simpler than the original AgentDecisionG."""

    query: str = Field(description="Targeted search string for the metadata-scoped retrieval.")
    filter_source: str | None = Field(
        default=None,
        description="Exact publisher name (e.g. 'Hacker News', 'TechCrunch', 'The Verge') or None.",
    )
    filter_category: str | None = Field(
        default=None,
        description="One of: sports, technology, entertainment, business, science, health, or None.",
    )


class AgentDecisionG(BaseModel):
    """System G's per-step decision: which retrieval tool to call next, or ANSWER.

    Filter fields are deliberately FLAT (filter_source, filter_category) rather
    than a nested FilterSpec object — Amazon Nova-class models emit multiple
    tool calls when forced to populate nested schemas, which instructor cannot
    merge. Flat fields work uniformly across Anthropic and Nova on Bedrock.
    """

    action: ToolAction = Field(
        description="Pick the tool best suited to the missing information, or ANSWER."
    )
    query: str | None = Field(
        default=None,
        description="Search string for retrieve_* actions. Targeted, not a paraphrase of the original.",
    )
    filter_source: str | None = Field(
        default=None,
        description="When action=retrieve_filtered, the exact publisher name, e.g. 'Hacker News', 'TechCrunch', 'The Verge'. Leave empty otherwise.",
    )
    filter_category: str | None = Field(
        default=None,
        description="When action=retrieve_filtered, one of: sports, technology, entertainment, business, science, health. Leave empty otherwise.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Required if action=answer. The final response, citing chunk IDs.",
    )
    reasoning: str = Field(default="", description="Brief justification, <=30 words.")


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
