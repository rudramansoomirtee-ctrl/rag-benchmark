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
    # Union of every chunk retrieved across the run (all agent iterations / all
    # sub-question fan-out), deduped. `retrieved_chunk_ids` is only the FINAL
    # answering context; this is "evidence ever seen", for the retrieval-ceiling
    # analysis. Systems that retrieve once leave it None → runner falls back to
    # `retrieved_chunk_ids`.
    all_retrieved_chunk_ids: list[str] | None = None


class System(Protocol):
    name: str
    def answer(self, query: str) -> RunResult: ...
