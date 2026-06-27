"""The experiment loop.

Iterates over (system, dataset, query). For each combination:
  1. Calls system.answer(query)
  2. Scores correctness vs ground truth
  3. Computes precision@k / recall@k
  4. Persists a `runs` row to Postgres
  5. Phoenix captures the full trace automatically

The runner is resumable: the UNIQUE(experiment_id, system, query_id) constraint
on `runs` means re-running this picks up where it left off — useful when Bedrock
throttles mid-eval.
"""
import logging
from typing import Callable

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.config import settings
from src.db.models import Chunk, Experiment, Query, Run
from src.db.session import get_session
from src.evaluation.metrics import exact_match, contains_match, answer_match
from src.systems.base import System, RunResult
from src.systems.system_a import SystemA
from src.systems.system_b import SystemB
from src.systems.system_f import SystemF
from src.systems.system_f_tuned import SystemFTuned

logger = logging.getLogger("rag.runner")


def _git_sha() -> str | None:
    """Code version for provenance. Prefers an explicit GIT_SHA env (the only
    reliable source inside the api container, which mounts src/ but not .git),
    falling back to `git rev-parse` when the runner is invoked on a host checkout."""
    import os
    import subprocess

    env = os.environ.get("GIT_SHA")
    if env:
        return env.strip()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__), timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _litellm_version() -> str | None:
    """Pin-of-record for cost reproducibility: litellm's pricing map (which produces
    every run's cost_usd) changes between versions, and requirements allow a range.
    Recording the installed version per experiment makes costs attributable without
    needing a hard requirements pin."""
    try:
        from importlib.metadata import version
        return version("litellm")
    except Exception:
        return None


def _corpus_fingerprint(session, datasets: list[str]) -> dict:
    """Per-dataset chunk census so the experiment record proves which corpus
    build it ran against. `granularity` is 'passage' (all `<url>#p<i>`),
    'article' (none), or 'mixed' — the last flags the additive-ingest hazard
    (C8) where an article index was never wiped before a passage re-ingest."""
    fp: dict = {}
    for ds in datasets:
        total = session.scalar(
            select(func.count(Chunk.id)).where(Chunk.dataset == ds)
        ) or 0
        passages = session.scalar(
            select(func.count(Chunk.id)).where(
                Chunk.dataset == ds, Chunk.external_id.like("%#p%")
            )
        ) or 0
        if total == 0:
            granularity = "empty"
        elif passages == total:
            granularity = "passage"
        elif passages == 0:
            granularity = "article"
        else:
            granularity = "mixed"
        fp[ds] = {"n_chunks": total, "n_passage_chunks": passages, "granularity": granularity}
    return fp


# Final lineup: A (naive), B (iterative agent, default 5-step budget), F
# (decomposition + RRF), F-tuned (decomposition + CoT + source-aware retrieval
# + single-final-rerank). System G (multi-tool agentic) and the B1/B3/B5
# iteration sweep were both removed after the n=50 runs showed neither added
# value over the baseline systems. Historical runs remain in the DB.
SYSTEM_REGISTRY: dict[str, Callable[[], System]] = {
    "A": SystemA,
    "B": SystemB,
    "F": SystemF,
    "F-tuned": SystemFTuned,
}


def run_experiment(
    name: str,
    systems: list[str],
    datasets: list[str],
    split: str = "eval",
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 42,
    stratify: bool = True,
    min_gold_articles: int | None = None,
    query_ids: list[int] | None = None,
) -> int:
    """Run all (system, dataset, query) combinations and persist to `runs`.

    Returns the experiment_id.
    """
    console = Console()
    session = get_session()

    try:
        # Load queries, then select. --sample = seeded (optionally stratified by
        # question_type) random draw → a defensible subset. --limit = first-N, a
        # quick smoke test only (NOT random). --sample wins if both are given.
        queries: list[Query] = session.scalars(
            select(Query).where(Query.dataset.in_(datasets), Query.split == split).order_by(Query.id)
        ).all()

        # Hard-subset filter: keep only queries needing >= N gold articles (a hop-count
        # proxy). Null queries (0 gold) are excluded — they don't test retrieval coverage.
        # Applied before sampling so --sample draws from the hard subset.
        if min_gold_articles:
            queries = [q for q in queries if len(q.relevant_chunk_ids or []) >= min_gold_articles]

        selection: dict = {"method": "all", "n": len(queries)}
        if query_ids:
            idset = set(query_ids)
            queries = [q for q in queries if q.id in idset]
            queries.sort(key=lambda q: q.id)
            selection = {"method": "explicit_ids", "n": len(queries), "query_ids": [q.id for q in queries]}
        elif sample:
            import random
            from collections import defaultdict
            rng = random.Random(seed)
            if stratify:
                buckets: dict = defaultdict(list)
                for q in queries:
                    buckets[(q.query_metadata or {}).get("question_type") or "_none"].append(q)
                total = len(queries) or 1
                picked: list[Query] = []
                for qs in buckets.values():
                    k = min(len(qs), max(1, round(sample * len(qs) / total)))
                    picked.extend(rng.sample(qs, k))
                rng.shuffle(picked)
                queries = picked[:sample]
            else:
                queries = rng.sample(queries, min(sample, len(queries)))
            queries.sort(key=lambda q: q.id)
            if limit:
                console.print("[yellow]--sample given; ignoring --limit[/yellow]")
            selection = {
                "method": "stratified_sample" if stratify else "random_sample",
                "n": len(queries),
                "seed": seed,
                "query_ids": [q.id for q in queries],
            }
        elif limit:
            queries = queries[:limit]
            selection = {"method": "first_n", "n": len(queries)}

        exp = Experiment(
            name=name,
            config_json={
                "systems": systems,
                "datasets": datasets,
                "split": split,
                "min_gold_articles": min_gold_articles,
                "selection": selection,
                "top_k": settings.top_k,
                "fused_answer_top_k": settings.fused_answer_top_k,
                "max_agent_steps": settings.max_agent_steps,
                "agent_steps_by_system": {
                    s: settings.max_agent_steps for s in systems if s == "B"
                },
                "model": settings.litellm_model,
                "embedding_model": settings.embedding_model,
                # Provenance — lets the DB prove environment constancy across the
                # 4×3 matrix (C2). Without these, two experiments are not known
                # to share a pipeline even when they share a model string.
                "git_sha": _git_sha(),
                "litellm_version": _litellm_version(),
                "reranker_model": settings.reranker_model,
                "rerank_provider": settings.rerank_provider,
                "retrieval_pool": settings.retrieval_pool,
                "retrieval_stratify_sources": settings.retrieval_stratify_sources,
                "retrieval_semantic_only": settings.retrieval_semantic_only,
                "corpus": _corpus_fingerprint(session, datasets),
            },
        )
        session.add(exp)
        session.commit()
        session.refresh(exp)
        exp_id = exp.id

        console.print(
            f"[bold]Experiment[/bold] [cyan]{name}[/cyan] (id={exp_id}) — "
            f"{len(queries)} queries × {len(systems)} systems = {len(queries) * len(systems)} runs"
        )
        logger.info(
            "experiment '%s' id=%s: %d queries × %d systems (%s); selection=%s",
            name, exp_id, len(queries), len(systems), ",".join(systems), selection.get("method"),
        )

        # Instantiate systems once
        system_instances: dict[str, System] = {s: SYSTEM_REGISTRY[s]() for s in systems}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for sys_name, system in system_instances.items():
                logger.info("system %s: %d queries", sys_name, len(queries))
                task = progress.add_task(f"System {sys_name}", total=len(queries))
                for q in queries:
                    _run_one(session, exp_id, sys_name, system, q)
                    progress.advance(task)

        session.query(Experiment).filter_by(id=exp_id).update(
            {"finished_at": __import__("sqlalchemy").func.now()}
        )
        session.commit()
        return exp_id
    finally:
        session.close()


def _run_one(session, exp_id: int, sys_name: str, system: System, q: Query) -> None:
    """Execute one (system, query) run. Idempotent via UNIQUE constraint upsert."""
    try:
        result: RunResult = system.answer(q.query_text)
    except Exception as e:
        logger.warning("system %s failed on query %s: %s", sys_name, q.id, e)
        # Stub row so the UNIQUE upsert skips this query on resume.
        # Clear with `DELETE FROM runs WHERE answer IS NULL` to retry failures.
        stmt = insert(Run).values(
            experiment_id=exp_id,
            system=sys_name,
            query_id=q.id,
            retrieved_chunk_ids=[],
        ).on_conflict_do_nothing(index_elements=["experiment_id", "system", "query_id"])
        session.execute(stmt)
        session.commit()
        return

    if not q.ground_truth:
        is_correct = None
    elif q.dataset == "musique":
        # MuSiQue ships multiple acceptable surface forms; score against gold + aliases
        # with bidirectional containment (committed terse answers like '140' ⊆ '140 mi').
        golds = [q.ground_truth] + ((q.query_metadata or {}).get("answer_aliases") or [])
        is_correct = answer_match(result.answer, golds)
    else:
        is_correct = contains_match(result.answer, q.ground_truth)
    # Systems that retrieve once leave all_retrieved_chunk_ids None → fall back to
    # the final context (for them the two are identical).
    all_retrieved = result.all_retrieved_chunk_ids or result.retrieved_chunk_ids

    stmt = insert(Run).values(
        experiment_id=exp_id,
        system=sys_name,
        query_id=q.id,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        all_retrieved_chunk_ids=all_retrieved,
        answer=result.answer,
        n_steps=result.n_steps,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        is_correct=is_correct,
        phoenix_trace_id=result.phoenix_trace_id,
    ).on_conflict_do_nothing(index_elements=["experiment_id", "system", "query_id"])
    session.execute(stmt)
    session.commit()
    logger.debug(
        "%s q%s: correct=%s steps=%s cost=$%.4f %sms",
        sys_name, q.id, is_correct, result.n_steps, float(result.cost_usd or 0), result.latency_ms,
    )
