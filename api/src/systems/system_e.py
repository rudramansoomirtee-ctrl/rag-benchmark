"""System E: OpenRag (ultimate_rag) plugged in as a benchmarked system.

Retrieval is delegated to a running OpenRag server — its real multi-strategy
pipeline (HyDE + BM25 + query-decomposition + RAPTOR) with neural reranking.
We call it over HTTP exactly as OpenRag's own benchmark does (POST /query), so
System E exercises OpenRag's actual implementation, not a reimplementation.

Scoring alignment: this harness scores retrieval by URL-keyed chunk IDs
(chunks.external_id = article url), but OpenRag returns chunk *text* with no
source URL — both of its ingest paths drop it, which is why OpenRag's own eval
substring-matches evidence text instead of comparing IDs. We therefore recover
each retrieved chunk's article URL by matching its text back to the MultiHop
corpus already in Postgres, then dedupe to a URL list (preserving OpenRag's
reranked order). System E is then measured by the same recall@k / precision@k
as Systems A-D.

Answer generation reuses the shared Bedrock LLM and System A's prompt, so the
A-E comparison isolates retrieval strategy, not the generator.

Caveats:
  - RAPTOR summary nodes (and any snippet not contained in a single article)
    recover no URL and are skipped — they cannot map to one article anyway.
  - cost_usd reflects only the answer-generation call; OpenRag's retrieval-side
    spend (OpenAI embeddings, Cohere rerank, HyDE/decomp LLM calls) happens on
    the OpenRag server and is not visible to LiteLLM here.

Prerequisites: an OpenRag server reachable at settings.openrag_url with the same
MultiHop corpus ingested, and `ingest-dataset multihop` already run here (so the
corpus bodies exist in Postgres for URL recovery).
"""
import re
import time
from functools import lru_cache

import httpx
from sqlalchemy import select

from src.config import settings
from src.db.models import Chunk
from src.db.session import get_session
from src.llm.client import generate
from src.systems.base import RunResult
from src.systems.system_a import ANSWER_SYSTEM_PROMPT


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


@lru_cache(maxsize=1)
def _corpus_index() -> tuple[tuple[str, str], ...]:
    """(url, whitespace-stripped lowercased body) for every MultiHop article."""
    session = get_session()
    try:
        rows = session.execute(
            select(Chunk.external_id, Chunk.text).where(Chunk.dataset == "multihop")
        ).all()
    finally:
        session.close()
    return tuple((url, _norm(body)) for url, body in rows)


def _recover_url(chunk_text: str) -> str | None:
    """Map an OpenRag chunk back to its source article URL by text containment."""
    needle = _norm(chunk_text)
    if not needle:
        return None
    for url, body in _corpus_index():
        if needle in body:
            return url
    return None


class SystemE:
    name = "E"

    def answer(self, query: str) -> RunResult:
        t0 = time.time()

        resp = httpx.post(
            f"{settings.openrag_url}/query",
            json={
                "query": query,
                "top_k": settings.retrieval_pool,
                "mode": settings.openrag_mode,
                "include_graph": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        hits = resp.json().get("results", [])

        retrieved_urls: list[str] = []
        context_parts: list[str] = []
        for h in hits:
            text = h.get("text", "")
            url = _recover_url(text)
            context_parts.append(f"[{url or 'unmapped'}] {text}")
            if url and url not in retrieved_urls:
                retrieved_urls.append(url)

        context = "\n\n".join(context_parts)
        result = generate(
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
        )

        return RunResult(
            answer=result["content"],
            retrieved_chunk_ids=retrieved_urls,
            hhem_score=None,
            flagged=None,
            n_steps=1,
            tokens_in=result["tokens_in"],
            tokens_out=result["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=result["cost_usd"],
        )
