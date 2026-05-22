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
    return engine, pd, plt


@app.cell
def __(engine, pd):
    metrics = pd.read_sql("SELECT * FROM metrics ORDER BY system, dataset", engine)
    metrics
    return (metrics,)


@app.cell
def __(metrics, plt):
    # Pareto: cost_per_correct vs accuracy
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in metrics.iterrows():
        ax.scatter(row["accuracy"], float(row["cost_per_correct"] or 0), s=80)
        ax.annotate(f"{row['system']} ({row['dataset']})",
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
def __(by_type):
    # Grouped bars: accuracy per question type, one cluster per system.
    pivot = by_type.pivot(index="question_type", columns="system", values="accuracy")
    ax2 = pivot.plot(kind="bar", figsize=(8, 5), ylim=(0, 1))
    ax2.set_ylabel("Accuracy (containment)")
    ax2.set_xlabel("MultiHop question type")
    ax2.set_title("Per-question-type accuracy by system")
    fig2 = ax2.get_figure()
    return (fig2, pivot)


if __name__ == "__main__":
    app.run()
