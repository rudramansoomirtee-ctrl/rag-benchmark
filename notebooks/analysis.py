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


if __name__ == "__main__":
    app.run()
