"""Read-only browsing of experiments, runs, and aggregated metrics.

Experiment creation stays in the CLI — a full run takes hours and shouldn't
be tied to an HTTP request lifecycle.
"""
from fastapi import APIRouter
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
                "system": r.system,
                "query": q.query_text,
                "ground_truth": q.ground_truth,
                "answer": r.answer,
                "is_correct": r.is_correct,
                "hhem_score": r.hhem_score,
                "flagged": r.flagged,
                "n_steps": r.n_steps,
                "latency_ms": r.latency_ms,
                "cost_usd": float(r.cost_usd) if r.cost_usd is not None else None,
            }
            for r, q in rows
        ]
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
                "crag_score": m.crag_score,
                "avg_faithfulness": m.avg_faithfulness,
                "pct_flagged": m.pct_flagged,
                "avg_trajectory_length": m.avg_trajectory_length,
                "total_cost_usd": float(m.total_cost_usd) if m.total_cost_usd is not None else None,
                "cost_per_correct": float(m.cost_per_correct) if m.cost_per_correct is not None else None,
            }
            for m in rows
        ]
    finally:
        session.close()
