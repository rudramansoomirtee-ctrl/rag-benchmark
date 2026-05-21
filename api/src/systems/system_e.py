"""System E: vendored OpenRag (ultimate_rag) as a benchmarked system, in-process.

The OpenRag engine lives in the repo as the top-level `ultimate_rag` /
`knowledge_base` packages. System E calls its `UltimateRetriever` directly —
no HTTP, no separate service — exercising the real multi-strategy pipeline
(HyDE + BM25 + query-decomposition + RAPTOR) with Cohere neural reranking.

Index: OpenRag retrieves over an in-memory RAPTOR forest, not OpenSearch. The
forest is built once from the MultiHop corpus and persisted to disk
(`build-openrag-index`); System E loads it lazily on first query.

Scoring alignment: this harness scores retrieval by URL-keyed chunk IDs, but
OpenRag returns chunk *text* with no source URL. We recover each chunk's article
URL by matching its text back to the MultiHop corpus already in Postgres, then
dedupe (preserving OpenRag's reranked order) so System E is measured by the same
recall@k / precision@k as Systems A-D. Answer generation reuses the shared
Bedrock LLM and System A's prompt, so E differs from A only in retrieval.

Heavy imports (ultimate_rag / knowledge_base, which pull openai/cohere/umap/…)
are deferred into the lazy singleton below, so importing this module — which
`runner.py` does alongside A-D — never requires OpenRag's deps or a built tree.

Caveats:
  - RAPTOR summary nodes / snippets not contained in a single article recover no
    URL and are skipped — they cannot map to one article anyway.
  - cost_usd covers only the answer call; OpenRag's retrieval-side OpenAI/Cohere
    spend is not visible to LiteLLM.

Prerequisites: OPENAI_API_KEY + COHERE_API_KEY in the api container,
`ingest-dataset multihop`, then `build-openrag-index multihop`.
"""
import asyncio
import re
import sys
import time
from functools import lru_cache

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


def _install_pickle_shim() -> None:
    """Legacy RAPTOR pickles reference `raptor.*`; the module is `knowledge_base.raptor`.

    Aliasing keeps pickle.load happy for trees saved/loaded across that path.
    """
    if "raptor" in sys.modules:
        return
    try:
        from knowledge_base import raptor as kb_raptor

        sys.modules["raptor"] = kb_raptor
        if hasattr(kb_raptor, "tree_structures"):
            sys.modules["raptor.tree_structures"] = kb_raptor.tree_structures
    except Exception:
        pass


def _run_async(coro):
    """Run a coroutine from sync code, whether or not a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


@lru_cache(maxsize=1)
def _get_retriever():
    """Load the persisted RAPTOR forest once and build the OpenRag retriever."""
    _install_pickle_shim()
    from ultimate_rag.agents.observations import ObservationCollector
    from ultimate_rag.core.node import TreeForest
    from ultimate_rag.core.persistence import TreePersistence
    from ultimate_rag.graph.graph import KnowledgeGraph
    from ultimate_rag.retrieval.retriever import RetrievalConfig, UltimateRetriever

    tree = TreePersistence(local_dir=settings.openrag_tree_dir).load_tree(
        settings.openrag_tree_name
    )
    if tree is None:
        raise RuntimeError(
            f"OpenRag tree '{settings.openrag_tree_name}' not found in "
            f"{settings.openrag_tree_dir}. Build it first: "
            "python -m src.cli build-openrag-index multihop"
        )

    forest = TreeForest(forest_id="default", name="System E forest")
    forest.add_tree(tree)
    return UltimateRetriever(
        forest=forest,
        graph=KnowledgeGraph(),
        observation_collector=ObservationCollector(),
        config=RetrievalConfig(),
    )


def build_index(
    dataset: str = "multihop",
    num_layers: int | None = None,
    target_top: int | None = None,
) -> int:
    """Build OpenRag's RAPTOR forest from a dataset's corpus and persist it.

    Reads article bodies from Postgres (already loaded by `ingest-dataset`),
    runs RAPTOR clustering + summarisation (OpenAI), and saves a pickle under
    settings.openrag_tree_dir. Returns the number of source articles. One-time
    and resumable-by-rerun (re-running rebuilds the tree).
    """
    _install_pickle_shim()
    from ultimate_rag.core.persistence import TreePersistence
    from ultimate_rag.raptor.tree_building import RaptorTreeBuilder, TreeBuildConfig

    session = get_session()
    try:
        rows = session.scalars(
            select(Chunk).where(Chunk.dataset == dataset).order_by(Chunk.id)
        ).all()
        texts = [c.text for c in rows]
    finally:
        session.close()

    if not texts:
        raise RuntimeError(
            f"no corpus chunks for dataset '{dataset}' — run `ingest-dataset {dataset}` first"
        )

    config = TreeBuildConfig(
        num_layers=num_layers or settings.openrag_num_layers,
        target_top_nodes=target_top or settings.openrag_target_top_nodes,
    )
    tree = RaptorTreeBuilder(config).build_from_texts(
        texts, tree_name=settings.openrag_tree_name
    )
    TreePersistence(local_dir=settings.openrag_tree_dir).save_tree(tree, to_local=True)
    return len(texts)


class SystemE:
    name = "E"

    def answer(self, query: str) -> RunResult:
        from ultimate_rag.retrieval.retriever import RetrievalMode

        t0 = time.time()
        retriever = _get_retriever()
        result = _run_async(
            retriever.retrieve(
                query=query,
                top_k=settings.retrieval_pool,
                mode=RetrievalMode(settings.openrag_mode),
            )
        )

        retrieved_urls: list[str] = []
        context_parts: list[str] = []
        for chunk in result.chunks:
            text = getattr(chunk, "text", "") or ""
            url = _recover_url(text)
            context_parts.append(f"[{url or 'unmapped'}] {text}")
            if url and url not in retrieved_urls:
                retrieved_urls.append(url)

        context = "\n\n".join(context_parts)
        gen = generate(
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
        )

        return RunResult(
            answer=gen["content"],
            retrieved_chunk_ids=retrieved_urls,
            hhem_score=None,
            flagged=None,
            n_steps=1,
            tokens_in=gen["tokens_in"],
            tokens_out=gen["tokens_out"],
            latency_ms=int((time.time() - t0) * 1000),
            cost_usd=gen["cost_usd"],
        )
