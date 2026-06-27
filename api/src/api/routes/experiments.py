"""Read-only browsing of experiments, runs, and aggregated metrics.

Experiment creation stays in the CLI — a full run takes hours and shouldn't
be tied to an HTTP request lifecycle.
"""
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from src.db.models import Experiment, Metric, Query, Run
from src.db.session import get_session

router = APIRouter(prefix="/api", tags=["experiments"])


def _question_type(q) -> str:
    """Normalised question type: MultiHop `inference_query` -> `inference`,
    MuSiQue `2hop` unchanged. `unknown` when absent."""
    return ((q.query_metadata or {}).get("question_type") or "unknown").replace("_query", "").lower()


def _answer_cands(q) -> list[str]:
    """Gold answer plus any dataset-provided aliases (MuSiQue `answer_aliases`)."""
    return [c for c in [q.ground_truth] + ((q.query_metadata or {}).get("answer_aliases") or []) if c]


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
def runs(exp_id: int, limit: int = 2000):
    from src.evaluation.metrics import token_f1
    from src.evaluation.stats import covered

    session = get_session()
    try:
        rows = session.execute(
            select(Run, Query)
            .join(Query, Run.query_id == Query.id)
            .where(Run.experiment_id == exp_id)
            .order_by(Run.id)
            .limit(limit)
        ).all()
        out = []
        for r, q in rows:
            cov = covered(r.all_retrieved_chunk_ids or r.retrieved_chunk_ids, q.relevant_chunk_ids)
            if r.is_correct or cov is None:
                failure_type = None
            else:
                failure_type = "generation" if cov else "retrieval"
            out.append({
                "id": r.id,
                "query_id": r.query_id,
                "system": r.system,
                "query": q.query_text,
                "ground_truth": q.ground_truth,
                "answer": r.answer,
                "is_correct": r.is_correct,
                "llm_judge_label": r.llm_judge_label,
                "question_type": _question_type(q),
                "covered": cov,
                "failure_type": failure_type,
                "token_f1": max((token_f1(r.answer or "", c) for c in _answer_cands(q)), default=0.0),
                "hhem_score": r.hhem_score,
                "flagged": r.flagged,
                "has_trace": bool(r.trace_json),
                "n_steps": r.n_steps,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "latency_ms": r.latency_ms,
                "cost_usd": float(r.cost_usd) if r.cost_usd is not None else None,
            })
        return out
    finally:
        session.close()


@router.get("/experiments/{exp_id}/runs/{run_id}/trace")
def run_trace(exp_id: int, run_id: int):
    """Return a run's stored glass-box trace (`trace_json`) if one was captured at
    run time, so the UI can render it instantly without a live replay."""
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
    return {
        "system": run.system,
        "query": q.query_text,
        "ground_truth": q.ground_truth,
        "answer": run.answer,
        "trace": run.trace_json or [],
    }


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

    session = get_session()
    try:
        rows = session.execute(
            select(Run, Query).join(Query, Run.query_id == Query.id).where(Run.experiment_id == exp_id)
        ).all()
        grouped = defaultdict(list)
        for r, q in rows:
            grouped[(r.system, _question_type(q))].append((r, q))
        out = []
        for (system, qt), pairs in sorted(grouped.items()):
            n = len(pairs)
            correct = sum(1 for r, _ in pairs if r.is_correct)
            exact = sum(max((float(exact_match(_post_marker(r.answer or ""), c)) for c in _answer_cands(q)), default=0.0)
                        for r, q in pairs) / n if n else 0.0
            tf1 = sum(max((token_f1(r.answer or "", c) for c in _answer_cands(q)), default=0.0)
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


def _nan_to_none(v):
    return None if (isinstance(v, float) and v != v) else v


@router.get("/experiments/{exp_id}/analysis")
def analysis(exp_id: int):
    """Per-system analytical depth for one experiment, computed server-side with the
    canonical scoring code (mirrors notebook cells N2/N3/N4/N5): accuracy + bootstrap CI,
    Token-F1/EM, CRAG, P@5/R@5, retrieval-coverage ceiling + retrieval-vs-generation
    failure attribution, cost & latency dispersion, the cost-accuracy Pareto frontier,
    and the metric-agreement matrix."""
    import statistics
    from collections import defaultdict
    from src.evaluation.metrics import (
        contains_match, exact_match, token_f1, precision_at_k, recall_at_k,
    )
    from src.evaluation.judge import CRAG_SCORE
    from src.evaluation import stats as st

    session = get_session()
    try:
        rows = session.execute(
            select(Run, Query).join(Query, Run.query_id == Query.id).where(Run.experiment_id == exp_id)
        ).all()
    finally:
        session.close()

    grouped = defaultdict(list)
    for r, q in rows:
        grouped[r.system].append((r, q))

    per_system = []
    pareto_pts = []
    sig = {"contains": [], "exact": [], "tf1": [], "crag": []}  # pooled for the agreement matrix
    _CRAG_BIN = {"perfect": 1, "acceptable": 1, "missing": 0, "incorrect": 0}

    for system in sorted(grouped):
        pairs = grouped[system]
        n = len(pairs)
        correct_flags = [1 if r.is_correct else 0 for r, _ in pairs]
        n_correct = sum(correct_flags)
        accuracy = n_correct / n if n else 0.0
        ci_lo, ci_hi = st.bootstrap_ci(correct_flags)

        exact_vals, tf1_vals, costs, lats, steps, judged = [], [], [], [], [], []
        cov_flags, cov_correct = [], []
        err_retrieval = err_generation = 0
        for r, q in pairs:
            cands = _answer_cands(q)
            cm = 1 if any(contains_match(r.answer or "", c) for c in cands) else 0
            em = 1 if any(exact_match(r.answer or "", c) for c in cands) else 0
            tf1 = max((token_f1(r.answer or "", c) for c in cands), default=0.0)
            exact_vals.append(em)
            tf1_vals.append(tf1)
            sig["contains"].append(cm)
            sig["exact"].append(em)
            sig["tf1"].append(1 if tf1 >= 0.5 else 0)
            sig["crag"].append(_CRAG_BIN.get(r.llm_judge_label) if r.llm_judge_label else None)
            if r.llm_judge_label:
                judged.append(CRAG_SCORE.get(r.llm_judge_label, 0.0))
            cov = st.covered(r.all_retrieved_chunk_ids or r.retrieved_chunk_ids, q.relevant_chunk_ids)
            if cov is not None:
                ok = bool(r.is_correct)
                cov_flags.append(1 if cov else 0)
                cov_correct.append((cov, ok))
                if not ok and not cov:
                    err_retrieval += 1
                elif not ok and cov:
                    err_generation += 1
            if r.cost_usd is not None:
                costs.append(float(r.cost_usd))
            if r.latency_ms is not None:
                lats.append(float(r.latency_ms))
            if r.n_steps is not None:
                steps.append(float(r.n_steps))

        n_cov = len(cov_flags)
        n_cov_yes = sum(cov_flags)
        cost_total = sum(costs)
        c_lo, c_hi = st.bootstrap_ci(costs) if costs else (None, None)
        ps = {
            "system": system, "n": n,
            "accuracy": accuracy, "acc_ci_lo": ci_lo, "acc_ci_hi": ci_hi,
            "accuracy_exact": (sum(exact_vals) / n if n else 0.0),
            "token_f1": (sum(tf1_vals) / n if n else 0.0),
            "crag_score": (sum(judged) / len(judged)) if judged else None,
            "precision_at_5": statistics.mean([precision_at_k(r.retrieved_chunk_ids, q.relevant_chunk_ids, 5) for r, q in pairs]) if n else None,
            "recall_at_5": statistics.mean([recall_at_k(r.retrieved_chunk_ids, q.relevant_chunk_ids, 5) for r, q in pairs]) if n else None,
            "n_gold_bearing": n_cov,
            "coverage": (n_cov_yes / n_cov) if n_cov else None,
            "acc_if_covered": (sum(1 for c, ok in cov_correct if c and ok) / n_cov_yes) if n_cov_yes else None,
            "err_retrieval": (err_retrieval / n_cov) if n_cov else None,
            "err_generation": (err_generation / n_cov) if n_cov else None,
            "pct_failed": (sum(1 for r, _ in pairs if r.answer is None) / n) if n else 0.0,
            "cost_mean": (cost_total / len(costs)) if costs else None,
            "cost_median": statistics.median(costs) if costs else None,
            "cost_p95": st.percentile(costs, 95) if costs else None,
            "cost_ci_lo": c_lo, "cost_ci_hi": c_hi,
            "cost_total": cost_total if costs else None,
            "cost_per_correct": (cost_total / n_correct) if (costs and n_correct) else None,
            "lat_p50": statistics.median(lats) if lats else None,
            "lat_p95": st.percentile(lats, 95) if lats else None,
            "lat_mean": (sum(lats) / len(lats)) if lats else None,
            "lat_std": st.stdev(lats) if lats else None,
            "mean_steps": (sum(steps) / len(steps)) if steps else None,
        }
        per_system.append({k: _nan_to_none(v) for k, v in ps.items()})
        if ps["cost_per_correct"]:
            pareto_pts.append((system, accuracy, ps["cost_per_correct"]))

    front = {lbl for lbl, _, _ in st.pareto_frontier(pareto_pts)}
    pareto = [
        {"system": lbl, "accuracy": a, "cost_per_correct": c, "on_frontier": lbl in front}
        for lbl, a, c in pareto_pts
    ]

    names = ["contains", "exact", "tf1", "crag"]
    labels = {"contains": "contains", "exact": "exact", "tf1": "tokenF1≥.5", "crag": "CRAG good"}
    agreement = {
        "labels": [labels[k] for k in names],
        "matrix": [[_nan_to_none(st.agreement_rate(sig[i], sig[j])) for j in names] for i in names],
    }
    return {"per_system": per_system, "pareto": pareto, "agreement": agreement}


@router.get("/experiments/compare")
def compare(ids: str, dataset: str = "multihop"):
    """Cross-experiment rank stability (notebook N1) over the selected experiments:
    a system×experiment accuracy pivot, the pairwise Kendall tau-b between experiments
    (rank stability across model tiers), and per-(system,experiment) accuracy/cost for
    grouped bars and a combined Pareto.

    Computed directly from `runs` (not the `metrics` table) so it works whether or not
    `compute-metrics` has been run for the selected experiments."""
    from collections import defaultdict
    from src.evaluation import stats as st

    try:
        exp_ids = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "ids must be comma-separated integers")
    if not exp_ids:
        raise HTTPException(400, "no experiment ids given")

    session = get_session()
    try:
        exps = session.scalars(select(Experiment).where(Experiment.id.in_(exp_ids))).all()
        rows = session.execute(
            select(Run, Query).join(Query, Run.query_id == Query.id)
            .where(Run.experiment_id.in_(exp_ids), Query.dataset == dataset)
        ).all()
    finally:
        session.close()

    label = {e.id: f"{(e.config_json or {}).get('model') or '?'} (#{e.id})" for e in exps}
    ordered = [eid for eid in exp_ids if eid in label]  # preserve selection order

    agg = defaultdict(lambda: {"n": 0, "correct": 0, "cost": 0.0})
    systems_set = set()
    for r, q in rows:
        a = agg[(r.experiment_id, r.system)]
        a["n"] += 1
        if r.is_correct:
            a["correct"] += 1
        if r.cost_usd is not None:
            a["cost"] += float(r.cost_usd)
        systems_set.add(r.system)
    systems = sorted(systems_set)
    acc = {k: (v["correct"] / v["n"] if v["n"] else None) for k, v in agg.items()}
    cost_pc = {k: (v["cost"] / v["correct"] if v["correct"] else None) for k, v in agg.items()}

    pivot = {
        "experiments": [label[e] for e in ordered],
        "systems": systems,
        "rows": [[acc.get((e, s)) for e in ordered] for s in systems],
    }

    tau = {"labels": [label[e] for e in ordered], "matrix": [], "note": None}
    if len(ordered) >= 2:
        for a in ordered:
            row = []
            for b in ordered:
                paired = [
                    (acc.get((a, s)), acc.get((b, s))) for s in systems
                    if acc.get((a, s)) is not None and acc.get((b, s)) is not None
                ]
                row.append(_nan_to_none(st.kendall_tau_b([u for u, _ in paired], [v for _, v in paired]))
                           if len(paired) >= 2 else None)
            tau["matrix"].append(row)
    else:
        tau["note"] = "Need ≥2 experiments for rank stability."

    per_system_per_exp = [
        {"experiment": label[e], "system": s, "accuracy": acc.get((e, s)), "cost_per_correct": cost_pc.get((e, s))}
        for e in ordered for s in systems
    ]
    return {"pivot": pivot, "kendall_tau": tau, "per_system_per_exp": per_system_per_exp}
