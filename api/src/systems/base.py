"""Shared interface for the four systems. Each implements .answer(query) -> RunResult."""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class RunResult:
    answer: str
    retrieved_chunk_ids: list[str]
    hhem_score: float | None
    flagged: bool | None
    n_steps: int
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    phoenix_trace_id: str | None = None


class System(Protocol):
    name: str
    def answer(self, query: str) -> RunResult: ...
