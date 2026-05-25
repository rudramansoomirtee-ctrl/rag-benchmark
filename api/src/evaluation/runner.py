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

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from src.config import settings
from src.db.models import Experiment, Query, Run
from src.db.session import get_session
from src.evaluation.metrics import exact_match, contains_match
from src.systems.base import System, RunResult
from src.systems.system_a import SystemA
from src.systems.system_b import SystemB
from src.systems.system_f import SystemF
from src.systems.system_f_tuned import SystemFTuned

logger = logging.getLogger("rag.runner")


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

        selection: dict = {"method": "all", "n": len(queries)}
        if sample:
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
                "selection": selection,
                "top_k": settings.top_k,
                "max_agent_steps": settings.max_agent_steps,
                "agent_steps_by_system": {
                    s: settings.max_agent_steps for s in systems if s == "B"
                },
                "model": settings.litellm_model,
                "embedding_model": settings.embedding_model,
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


def _faithfulness(chunk_ids: list[str], answer: str | None) -> tuple[float | None, bool | None]:
    """HHEM faithfulness of `answer` against its retrieved chunks, for every system.

    Premise = the retrieved chunk texts (re-fetched from OpenSearch). Returns
    (score, flagged) or (None, None) when there's nothing to score or HHEM errs —
    a faithfulness miss must never lose the run itself.
    """
    if not chunk_ids or not answer:
        return None, None
    try:
        from src.faithfulness.hhem import score as hhem_score
        from src.retrieval.opensearch_client import get_client

        resp = get_client().mget(index=settings.opensearch_index, body={"ids": chunk_ids})
        premise = "\n\n".join(
            d["_source"]["text"] for d in resp["docs"] if d.get("found")
        )
        if not premise:
            return None, None
        s = hhem_score([(premise, answer)])[0]
        return s, s < settings.hhem_threshold
    except Exception:
        return None, None


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

    is_correct = (
        contains_match(result.answer, q.ground_truth) if q.ground_truth else None
    )
    hhem, flagged = _faithfulness(result.retrieved_chunk_ids, result.answer)

    stmt = insert(Run).values(
        experiment_id=exp_id,
        system=sys_name,
        query_id=q.id,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        answer=result.answer,
        hhem_score=hhem,
        flagged=flagged,
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
