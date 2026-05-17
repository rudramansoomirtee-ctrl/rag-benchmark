"""System D: System B wrapped with an HHEM faithfulness gate."""
from src.config import settings
from src.faithfulness.hhem import score as hhem_score
from src.systems.base import RunResult
from src.systems.system_b import SystemB


class SystemD:
    name = "D"

    def __init__(self):
        self._inner = SystemB()

    def answer(self, query: str) -> RunResult:
        result = self._inner.answer(query)

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
