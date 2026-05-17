"""Typed schemas for the agent's structured outputs (used by Systems B and D)."""
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
    reasoning: str = Field(description="Brief justification, <=30 words.")
