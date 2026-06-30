"""Marimo notebook for Chapter 4 analysis.

Open with: `marimo edit notebooks/analysis.py`

Reads directly from Postgres so figures regenerate any time you re-run the eval.
"""
import marimo as mo


__generated_with__ = "marimo"
app = mo.App()


@app.cell
def __():
    import pandas as pd
    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine

    engine = create_engine("postgresql+psycopg://rag:ragbench@localhost:5432/ragbench")

    # Presentation labels for figures/tables. Internal codes (runs.system) are kept
    # stable for DB + historical comparability; these map code -> descriptive name
    # (with the code retained in parentheses). Two axes: orchestration strategy, and
    # the dense-only retriever ablation (the "-minus" variants).
    SYSTEM_DISPLAY_NAMES = {
        "A": "Single-pass RAG (A)",
        "A-minus": "Single-pass RAG · dense-only (A-minus)",
        "B": "Iterative RAG (B)",
        "B-minus": "Iterative RAG · dense-only (B-minus)",
        "F": "Parallel decomposition (F)",
        "F-seq": "Sequential decomposition / Self-Ask (F-seq)",
        "F-tuned": "F-tuned · stacked, retired (F-tuned)",
        "G": "Multi-tool agentic · retired (G)",
    }

    def disp(code):
        return SYSTEM_DISPLAY_NAMES.get(code, code)

    return SYSTEM_DISPLAY_NAMES, disp, engine, pd, plt


@app.cell
def __(engine, pd):
    metrics = pd.read_sql("SELECT * FROM metrics ORDER BY system, dataset", engine)
    metrics
    return (metrics,)


@app.cell
def __(disp, metrics, plt):
    # Pareto: cost_per_correct vs accuracy
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in metrics.iterrows():
        ax.scatter(row["accuracy"], float(row["cost_per_correct"] or 0), s=80)
        ax.annotate(f"{disp(row['system'])} · {row['dataset']}",
                    (row["accuracy"], float(row["cost_per_correct"] or 0)),
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Cost per correct answer (USD)")
    ax.set_title("Cost-accuracy Pareto")
    return (fig,)


@app.cell
def __(engine, pd):
    # Per-question-type accuracy (MultiHop). Type lives in metadata->>'question_type',
    # not task_type. Denominator matches the CLI: NULL is_correct counts as wrong.
    by_type = pd.read_sql(
        """
        SELECT r.system,
               lower(split_part(q.metadata->>'question_type', '_query', 1)) AS question_type,
               AVG((COALESCE(r.is_correct, false))::int)::float AS accuracy,
               COUNT(*) AS n
        FROM runs r
        JOIN queries q ON q.id = r.query_id
        WHERE q.dataset = 'multihop'
        GROUP BY r.system, question_type
        ORDER BY r.system, question_type
        """,
        engine,
    )
    by_type
    return (by_type,)


@app.cell
def __(SYSTEM_DISPLAY_NAMES, by_type):
    # Grouped bars: accuracy per question type, one cluster per system.
    pivot = by_type.pivot(index="question_type", columns="system", values="accuracy")
    pivot = pivot.rename(columns=SYSTEM_DISPLAY_NAMES)
    ax2 = pivot.plot(kind="bar", figsize=(8, 5), ylim=(0, 1))
    ax2.set_ylabel("Accuracy (containment)")
    ax2.set_xlabel("MultiHop question type")
    ax2.set_title("Per-question-type accuracy by system")
    fig2 = ax2.get_figure()
    return (fig2, pivot)


# =====================================================================
#  Analysis helpers (audit N-series). Imports the canonical metric
#  functions so notebook numbers match `compute-metrics` exactly, rather
#  than reimplementing scoring here. Run marimo from the repo root (or
#  notebooks/) so the api/src package resolves.
# =====================================================================
@app.cell
def __():
    import json
    import pathlib
    import sys

    import numpy as np

    _candidates = [
        pathlib.Path.cwd() / "api",
        pathlib.Path.cwd().parent / "api",
        (pathlib.Path(__file__).resolve().parent.parent / "api") if "__file__" in globals() else None,
    ]
    _api = next((p for p in _candidates if p and (p / "src").is_dir()), None)
    if _api and str(_api) not in sys.path:
        sys.path.insert(0, str(_api))
    # Single source of truth: the canonical scoring + statistical helpers live in
    # api/src/evaluation/{metrics,stats}.py, so the notebook, `compute-metrics`, and
    # the web dashboard can never drift.
    from src.evaluation.metrics import contains_match, exact_match, token_f1
    from src.evaluation.stats import (
        bootstrap_ci, covered, kendall_tau_b, pareto_frontier, _aslist,
    )

    def agreement(a, b):
        """Fraction of rows where two binary metric series agree (both non-null).
        Pandas wrapper over the same equality-rate logic as `stats.agreement_rate`."""
        m = a.notna() & b.notna()
        return float((a[m] == b[m]).mean()) if m.any() else float("nan")

    return (
        agreement, bootstrap_ci, contains_match, covered, exact_match,
        kendall_tau_b, np, pareto_frontier, token_f1, _aslist,
    )


@app.cell
def __(engine, pd):
    # Raw per-run frame for the failure-attribution and metric-agreement analyses.
    runs_raw = pd.read_sql(
        """
        SELECT r.experiment_id, r.system, r.answer, r.llm_judge_label,
               r.cost_usd, r.latency_ms, r.n_steps,
               r.retrieved_chunk_ids,
               COALESCE(r.all_retrieved_chunk_ids, r.retrieved_chunk_ids) AS all_retrieved_chunk_ids,
               q.ground_truth, q.relevant_chunk_ids,
               lower(split_part(q.metadata->>'question_type', '_query', 1)) AS question_type
        FROM runs r
        JOIN queries q ON q.id = r.query_id
        WHERE q.dataset = 'multihop'
        ORDER BY r.experiment_id, r.system
        """,
        engine,
    )
    runs_raw
    return (runs_raw,)


@app.cell
def __(contains_match, covered, exact_match, runs_raw, token_f1):
    # Recompute every metric under the CURRENT scoring code (stored is_correct may
    # have been written by an older contains_match — see audit C5), so the
    # agreement analysis is internally consistent.
    runs_m = runs_raw.copy()
    runs_m["m_contains"] = [
        int(contains_match(a or "", g or "")) for a, g in zip(runs_m.answer, runs_m.ground_truth)
    ]
    runs_m["m_exact"] = [
        int(exact_match(a or "", g or "")) for a, g in zip(runs_m.answer, runs_m.ground_truth)
    ]
    runs_m["m_tf1"] = [
        token_f1(a or "", g or "") for a, g in zip(runs_m.answer, runs_m.ground_truth)
    ]
    runs_m["m_tf1_bin"] = (runs_m["m_tf1"] >= 0.5).astype(int)
    runs_m["m_crag_bin"] = runs_m["llm_judge_label"].map(
        {"perfect": 1, "acceptable": 1, "missing": 0, "incorrect": 0}
    )
    # Coverage is scored against evidence EVER seen (all_retrieved_chunk_ids), so
    # System B's iterative retrievals count, not just its final answering context.
    runs_m["covered"] = [
        covered(ar, rel) for ar, rel in zip(runs_m.all_retrieved_chunk_ids, runs_m.relevant_chunk_ids)
    ]
    runs_m
    return (runs_m,)


# ---- N3: retrieval ceiling + failure attribution (A4/O6) ----
@app.cell
def __(disp, pd, plt, runs_m):
    # Every wrong answer is either a RETRIEVAL failure (gold never retrieved) or a
    # GENERATION failure (gold was present, model still wrong). Coverage is the
    # retrieval ceiling: accuracy cannot meaningfully exceed it. Null-type queries
    # (no gold evidence) are excluded.
    _d = runs_m.dropna(subset=["covered"])
    _recs = []
    for _s, _g in _d.groupby("system"):
        _cov = _g["covered"].astype(bool)
        _ok = _g["m_contains"].astype(bool)
        _recs.append(dict(
            system=_s, n=len(_g),
            coverage=_cov.mean(),
            accuracy=_ok.mean(),
            acc_if_covered=(_ok[_cov].mean() if _cov.any() else float("nan")),
            err_retrieval=((~_cov) & (~_ok)).mean(),
            err_generation=(_cov & (~_ok)).mean(),
        ))
    ceiling = pd.DataFrame(_recs).sort_values("system").reset_index(drop=True)

    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _sys = ceiling["system"].map(disp)
    _ax.barh(_sys, ceiling["accuracy"], color="#2a9d8f", label="correct")
    _ax.barh(_sys, ceiling["err_generation"], left=ceiling["accuracy"], color="#e9c46a", label="generation failure (had evidence)")
    _ax.barh(_sys, ceiling["err_retrieval"], left=ceiling["accuracy"] + ceiling["err_generation"], color="#e76f51", label="retrieval failure (no evidence)")
    _ax.scatter(ceiling["coverage"], _sys, color="black", zorder=5, label="coverage (ceiling)")
    _ax.set_xlim(0, 1)
    _ax.set_xlabel("Fraction of queries")
    _ax.set_title("Failure attribution & retrieval ceiling (MultiHop, gold-bearing queries)")
    _ax.legend(loc="lower right", fontsize=8)
    fig_ceiling = _fig
    ceiling
    return ceiling, fig_ceiling


# ---- N4: metric agreement / divergence (A4/O6) ----
@app.cell
def __(agreement, np, pd, plt, runs_m):
    # How often do the four answer-correctness signals agree? Divergence is the
    # point: containment is lenient, exact is strict, token-F1 is graded, CRAG is
    # an LLM. CRAG cells use only judged runs.
    _cols = {
        "contains": "m_contains",
        "exact": "m_exact",
        "tokenF1>=.5": "m_tf1_bin",
        "CRAG good": "m_crag_bin",
    }
    _names = list(_cols)
    agree_mat = pd.DataFrame(index=_names, columns=_names, dtype=float)
    for _i in _names:
        for _j in _names:
            agree_mat.loc[_i, _j] = agreement(runs_m[_cols[_i]], runs_m[_cols[_j]])

    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    _im = _ax.imshow(agree_mat.values.astype(float), vmin=0.5, vmax=1.0, cmap="YlGn")
    _ax.set_xticks(range(len(_names)), _names, rotation=45, ha="right")
    _ax.set_yticks(range(len(_names)), _names)
    for _i in range(len(_names)):
        for _j in range(len(_names)):
            _v = agree_mat.values[_i, _j]
            _ax.text(_j, _i, "—" if np.isnan(_v) else f"{_v:.2f}", ha="center", va="center", fontsize=9)
    _ax.set_title("Pairwise metric agreement rate")
    _fig.colorbar(_im, fraction=0.046, pad=0.04)
    fig_agreement = _fig
    agree_mat
    return agree_mat, fig_agreement


# ---- N2: cost & latency dispersion + accuracy bootstrap CI (RQ2/RQ4) ----
@app.cell
def __(bootstrap_ci, pd, runs_m):
    _recs = []
    for _s, _g in runs_m.groupby("system"):
        _ok = _g["m_contains"].astype(bool)
        _lat = _g["latency_ms"].dropna().astype(float)
        _cost = _g["cost_usd"].dropna().astype(float)
        _lo, _hi = bootstrap_ci(_ok.astype(int).tolist())
        _ncorrect = int(_ok.sum())
        _recs.append(dict(
            system=_s, n=len(_g),
            accuracy=_ok.mean(), acc_ci_lo=_lo, acc_ci_hi=_hi,
            lat_p50=_lat.median(), lat_p95=_lat.quantile(0.95),
            lat_mean=_lat.mean(), lat_std=_lat.std(),
            cost_mean=_cost.mean(), cost_std=_cost.std(), cost_total=_cost.sum(),
            cost_per_correct=(_cost.sum() / _ncorrect if _ncorrect else float("nan")),
        ))
    variance_tbl = pd.DataFrame(_recs).sort_values("system").reset_index(drop=True)
    variance_tbl
    return (variance_tbl,)


# ---- Cost per query per system (RQ2) ----
@app.cell
def __(bootstrap_ci, pd, plt, runs_m):
    # The per-query economic unit. Retrieval is local (embeddings + OpenSearch +
    # cross-encoder all run in-process, $0), so 100% of per-query cost is Bedrock
    # LLM generation — differences are purely #LLM-calls × tokens. mean_steps is
    # the driver: A≈1 call, F/F-tuned≈2 (decompose+answer), B≈2×iterations.
    _recs = []
    for _s, _g in runs_m.groupby("system"):
        _c = _g["cost_usd"].dropna().astype(float)
        if _c.empty:
            continue
        _lo, _hi = bootstrap_ci(_c.tolist())
        _recs.append(dict(
            system=_s, n=len(_c),
            cost_per_query=_c.mean(),
            ci_lo=_lo, ci_hi=_hi,
            median=_c.median(), p95=_c.quantile(0.95), std=_c.std(),
            mean_steps=_g["n_steps"].dropna().mean(),
            total_cost=_c.sum(),
        ))
    cost_per_query = pd.DataFrame(_recs).sort_values("cost_per_query").reset_index(drop=True)

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _x = list(range(len(cost_per_query)))
    _err = [
        (cost_per_query.cost_per_query - cost_per_query.ci_lo).clip(lower=0),
        (cost_per_query.ci_hi - cost_per_query.cost_per_query).clip(lower=0),
    ]
    _ax.bar(_x, cost_per_query.cost_per_query, yerr=_err, capsize=5, color="#4878a8")
    _ax.set_xticks(_x, cost_per_query.system)
    _ax.set_ylabel("Cost per query (USD)")
    _ax.set_title("Mean cost per query by system (95% bootstrap CI)")
    fig_cost_per_query = _fig
    cost_per_query
    return cost_per_query, fig_cost_per_query


# ---- N1: rank stability across models/experiments (RQ3/RQ4) ----
@app.cell
def __(engine, kendall_tau_b, pd):
    # One column per experiment (labelled with its model). Kendall tau-b between
    # columns measures whether the system ranking is stable across model tiers.
    # Needs >=2 experiments (the 4x3 matrix) to be meaningful.
    _mx = pd.read_sql(
        """
        SELECT m.system, m.accuracy,
               e.id AS experiment_id, e.config_json->>'model' AS model
        FROM metrics m
        JOIN experiments e ON e.id = m.experiment_id
        WHERE m.dataset = 'multihop'
        ORDER BY e.id
        """,
        engine,
    )
    _piv = _mx.pivot_table(index="system", columns="experiment_id", values="accuracy")
    _labels = _mx.drop_duplicates("experiment_id").set_index("experiment_id")["model"]
    _cols = list(_piv.columns)
    if len(_cols) < 2:
        rank_pivot = _piv
        rank_tau = "Need >=2 experiments (run the 4x3 matrix) for rank stability."
    else:
        rank_tau = pd.DataFrame(index=_cols, columns=_cols, dtype=float)
        for _a in _cols:
            for _b in _cols:
                # Select columns individually (not via _piv[[_a, _b]], which would
                # duplicate-select on the diagonal where _a == _b and yield a frame,
                # not a series). Pairwise-drop rows missing either system.
                _xa, _xb = _piv[_a], _piv[_b]
                _mask = _xa.notna() & _xb.notna()
                rank_tau.loc[_a, _b] = kendall_tau_b(_xa[_mask].tolist(), _xb[_mask].tolist())
        _names = [f"{_labels.get(c, '?')} (#{c})" for c in _cols]
        rank_tau.index = _names
        rank_tau.columns = _names
        rank_pivot = _piv.rename(columns={c: f"{_labels.get(c, '?')} (#{c})" for c in _cols})
    rank_pivot
    return rank_pivot, rank_tau


# ---- N5: cost-accuracy Pareto frontier (A3/O5) ----
@app.cell
def __(disp, metrics, pareto_frontier, plt):
    _pts = [
        (str(r["system"]), float(r["accuracy"] or 0), float(r["cost_per_correct"] or 0))
        for _, r in metrics.iterrows()
        if r["dataset"] == "multihop" and r["cost_per_correct"]
    ]
    _front = pareto_frontier(_pts)
    _fig, _ax = plt.subplots(figsize=(7, 5))
    for _lbl, _a, _c in _pts:
        _ax.scatter(_a, _c, s=80, color="#bbbbbb", zorder=2)
        _ax.annotate(disp(_lbl), (_a, _c), xytext=(5, 5), textcoords="offset points", fontsize=8)
    if _front:
        _fx = [a for _, a, _c in _front]
        _fy = [c for _, _a, c in _front]
        _ax.plot(_fx, _fy, "-o", color="#2a9d8f", zorder=3, label="Pareto frontier")
        for _lbl, _a, _c in _front:
            _ax.scatter(_a, _c, s=120, edgecolor="#2a9d8f", facecolor="none", linewidth=2, zorder=4)
    _ax.set_xlabel("Accuracy (containment)")
    _ax.set_ylabel("Cost per correct answer (USD)")
    _ax.set_title("Cost–accuracy Pareto frontier")
    _ax.legend()
    fig_pareto = _fig
    fig_pareto
    return (fig_pareto,)


if __name__ == "__main__":
    app.run()
