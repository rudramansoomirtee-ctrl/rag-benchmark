"""POST /api/ask — run one query through one system. Returns the full RunResult
plus the retrieved chunk text (re-fetched from OpenSearch so the UI can show it).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.evaluation.runner import SYSTEM_REGISTRY
from src.retrieval.opensearch_client import get_client

router = APIRouter(prefix="/api", tags=["ask"])

# Lazy singletons: each system pays its init cost on first request only.
_instances: dict = {}


class AskRequest(BaseModel):
    system: str
    query: str
    trace: bool = False


def _get_system(name: str):
    if name not in SYSTEM_REGISTRY:
        raise HTTPException(400, f"unknown system: {name}")
    if name not in _instances:
        _instances[name] = SYSTEM_REGISTRY[name]()
    return _instances[name]


def _fetch_chunk_text(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    resp = get_client().mget(index=settings.opensearch_index, body={"ids": ids})
    return [
        {"id": d["_id"], "text": d["_source"]["text"]}
        for d in resp["docs"] if d.get("found")
    ]


@router.post("/ask")
def ask(req: AskRequest):
    from src.trace import capture

    system = _get_system(req.system)
    events: list = []
    try:
        if req.trace:
            with capture() as events:
                result = system.answer(req.query)
        else:
            result = system.answer(req.query)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")

    return {
        "answer": result.answer,
        "retrieved_chunks": _fetch_chunk_text(result.retrieved_chunk_ids),
        "n_steps": result.n_steps,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "latency_ms": result.latency_ms,
        "cost_usd": result.cost_usd,
        "trace": events,
    }
