"""Read-only browsing of experiments, runs, and aggregated metrics.

Experiment creation stays in the CLI — a full run takes hours and shouldn't
be tied to an HTTP request lifecycle.
"""
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.db.models import Experiment, Metric, Query, Run
from src.db.session import get_session

router = APIRouter(prefix="/api", tags=["experiments"])


@router.get("/experiments")
def list_experiments():
    session = get_session()
    try:
        rows = session.scalars(
            select(Experiment).order_by(Experiment.id.desc())
        ).all()
        return [
            {
                "id": e.id,
                "name": e.name,
                "config": e.config_json,
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "finished_at": e.finished_at.isoformat() if e.finished_at else None,
            }
            for e in rows
        ]
    finally:
        session.close()


@router.get("/experiments/{exp_id}/runs")
def runs(exp_id: int, limit: int = 100):
    session = get_session()
    try:
        rows = session.execute(
            select(Run, Query)
            .join(Query, Run.query_id == Query.id)
            .where(Run.experiment_id == exp_id)
            .order_by(Run.id)
            .limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "query_id": r.query_id,
                "system": r.system,
                "query": q.query_text,
                "ground_truth": q.ground_truth,
                "answer": r.answer,
                "is_correct": r.is_correct,
                "llm_judge_label": r.llm_judge_label,
                "n_steps": r.n_steps,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "latency_ms": r.latency_ms,
                "cost_usd": float(r.cost_usd) if r.cost_usd is not None else None,
            }
            for r, q in rows
        ]
    finally:
        session.close()


_DATASET_INDEX = {"multihop": "rag-chunks", "musique": "rag-chunks-musique"}


@router.post("/experiments/{exp_id}/runs/{run_id}/replay")
def replay_run(exp_id: int, run_id: int):
    """Re-execute one run's (system, query) with glass-box trace capture, against
    that run's dataset index, and return the full trace. Lets the Experiments UI
    replay any historical run and show its pipeline (decomposition / iteration /
    retrieval / rerank reordering). Deterministic (temperature=0), so it faithfully
    reproduces the original run."""
    from src.config import settings
    from src.evaluation.runner import SYSTEM_REGISTRY
    from src.trace import capture

    session = get_session()
    try:
        row = session.execute(
            select(Run, Query).join(Query, Run.query_id == Query.id)
            .where(Run.id == run_id, Run.experiment_id == exp_id)
        ).first()
    finally:
        session.close()
    if not row:
        raise HTTPException(404, "run not found")
    run, q = row
    if run.system not in SYSTEM_REGISTRY:
        raise HTTPException(400, f"cannot replay system {run.system}")

    # Point retrieval at the run's dataset index for the duration of the replay.
    # Single-user tool: a brief global swap (restored in finally) is acceptable.
    prev = settings.opensearch_index
    settings.opensearch_index = _DATASET_INDEX.get(q.dataset, prev)
    try:
        with capture() as events:
            result = SYSTEM_REGISTRY[run.system]().answer(q.query_text)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        settings.opensearch_index = prev

    return {
        "system": run.system,
        "query": q.query_text,
        "ground_truth": q.ground_truth,
        "answer": result.answer,
        "trace": events,
    }


@router.get("/experiments/{exp_id}/by-type")
def by_type(exp_id: int):
    """Per-question-type accuracy breakdown (inference/comparison/temporal/null).

    Computed on the fly from runs + queries.metadata['question_type'] — same logic
    as the `metrics-by-type` CLI, surfaced for the SPA.
    """
    from collections import defaultdict
    from src.evaluation.metrics import exact_match, token_f1, _post_marker
    from src.evaluation.judge import CRAG_SCORE

    def _cands(qq):
        return [c for c in [qq.ground_truth] + ((qq.query_metadata or {}).get("answer_aliases") or []) if c]

    session = get_session()
    try:
        rows = session.execute(
            select(Run, Query).join(Query, Run.query_id == Query.id).where(Run.experiment_id == exp_id)
        ).all()
        grouped = defaultdict(list)
        for r, q in rows:
            qt = ((q.query_metadata or {}).get("question_type") or "unknown").replace("_query", "").lower()
            grouped[(r.system, qt)].append((r, q))
        out = []
        for (system, qt), pairs in sorted(grouped.items()):
            n = len(pairs)
            correct = sum(1 for r, _ in pairs if r.is_correct)
            exact = sum(max((float(exact_match(_post_marker(r.answer or ""), c)) for c in _cands(q)), default=0.0)
                        for r, q in pairs) / n if n else 0.0
            tf1 = sum(max((token_f1(r.answer or "", c) for c in _cands(q)), default=0.0)
                      for r, q in pairs) / n if n else 0.0
            judged = [r.llm_judge_label for r, _ in pairs if r.llm_judge_label]
            crag = (sum(CRAG_SCORE.get(lbl, 0.0) for lbl in judged) / len(judged)) if judged else None
            out.append({
                "system": system, "question_type": qt, "n": n,
                "accuracy": (correct / n if n else 0.0), "accuracy_exact": exact,
                "token_f1": tf1, "crag_score": crag,
            })
        return out
    finally:
        session.close()


@router.get("/experiments/{exp_id}/metrics")
def metrics(exp_id: int):
    session = get_session()
    try:
        rows = session.scalars(
            select(Metric).where(Metric.experiment_id == exp_id)
        ).all()
        return [
            {
                "system": m.system,
                "dataset": m.dataset,
                "n_queries": m.n_queries,
                "precision_at_5": m.precision_at_5,
                "recall_at_5": m.recall_at_5,
                "accuracy": m.accuracy,
                "accuracy_exact": m.accuracy_exact,
                "avg_token_f1": m.avg_token_f1,
                "crag_score": m.crag_score,
                "pct_failed": m.pct_failed,
                "avg_trajectory_length": m.avg_trajectory_length,
                "total_cost_usd": float(m.total_cost_usd) if m.total_cost_usd is not None else None,
                "cost_per_query": (float(m.total_cost_usd) / m.n_queries) if (m.total_cost_usd is not None and m.n_queries) else None,
                "cost_per_correct": float(m.cost_per_correct) if m.cost_per_correct is not None else None,
            }
            for m in rows
        ]
    finally:
        session.close()
