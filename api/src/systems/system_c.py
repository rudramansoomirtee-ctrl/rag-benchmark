"""System C: System A wrapped with an HHEM faithfulness gate.

If the HHEM score for the (context, answer) pair falls below the calibrated
threshold, the answer is `flagged`. Composition over inheritance — System A
is the engine; this just adds the gate.
"""
from src.config import settings
from src.faithfulness.hhem import score as hhem_score
from src.systems.base import RunResult
from src.systems.system_a import SystemA


class SystemC:
    name = "C"

    def __init__(self):
        self._inner = SystemA()

    def answer(self, query: str) -> RunResult:
        result = self._inner.answer(query)

        # Build the premise from the retrieved chunks (the model sees the same context).
        # In production you'd cache these alongside the run — for now re-fetching is fine.
        from src.retrieval.opensearch_client import get_client
        client = get_client()
        chunks_resp = client.mget(
            index=settings.opensearch_index,
            body={"ids": result.retrieved_chunk_ids},
        )
        premise = "\n\n".join(
            d["_source"]["text"] for d in chunks_resp["docs"] if d.get("found")
        )

        s = hhem_score([(premise, result.answer)])[0]
        result.hhem_score = s
        result.flagged = s < settings.hhem_threshold
        return result
