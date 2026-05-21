"""Project CLI.

Usage:
    docker compose run --rm api python -m src.cli ingest-dataset multihop
    docker compose run --rm api python -m src.cli index-corpus multihop
    docker compose run --rm api python -m src.cli calibrate
    docker compose run --rm api python -m src.cli run-experiment --name v1 --systems A,B,C,D --datasets multihop
    docker compose run --rm api python -m src.cli compute-metrics --experiment 1
    docker compose run --rm api python -m src.cli export --experiment 1
"""
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from src.config import settings
from src.db.models import Run, Query
from src.db.session import get_session
from src.tracing import init_tracing

app = typer.Typer(add_completion=False)
console = Console()


@app.callback()
def _setup():
    """Runs before every command — initialise tracing once."""
    init_tracing()


@app.command()
def ingest_dataset(name: str):
    """Load a dataset (multihop | ragtruth) into Postgres."""
    if name == "multihop":
        from src.datasets.multihop import ingest
        n_q, n_c = ingest()
        console.print(f"[green]multihop:[/green] {n_q} queries, {n_c} chunks")
    elif name == "ragtruth":
        from src.datasets.ragtruth import ingest
        n = ingest()
        console.print(f"[green]ragtruth:[/green] {n} rows")
    else:
        raise typer.BadParameter(f"unknown dataset: {name}")


@app.command()
def index_corpus(name: str):
    """Embed chunks for a dataset and load them into OpenSearch."""
    from src.retrieval.indexer import index_corpus as _idx
    n = _idx(name)
    console.print(f"[green]indexed[/green] {n} chunks into {settings.opensearch_index}")


@app.command()
def build_openrag_index(name: str = "multihop"):
    """Build & persist the vendored OpenRag RAPTOR forest for System E (one-time).

    Reads the dataset corpus from Postgres, runs RAPTOR (OpenAI embeddings +
    gpt-4o-mini summaries), and saves a pickle under settings.openrag_tree_dir.
    Requires OPENAI_API_KEY; re-running rebuilds the tree.
    """
    from src.systems.system_e import build_index
    n = build_index(name)
    console.print(
        f"[green]built OpenRag tree[/green] '{settings.openrag_tree_name}' "
        f"from {n} articles -> {settings.openrag_tree_dir}"
    )


@app.command()
def calibrate(
    calibration_split: str = "ragtruth.calibration",
    output: str = "/data/results/threshold.json",
):
    """Fit HHEM threshold on RAGTruth labels. Writes threshold.json + ROC curve."""
    from src.evaluation.calibration import fit_threshold
    from src.faithfulness.hhem import score as hhem_score

    session = get_session()
    try:
        dataset, split = calibration_split.split(".", 1)
        rows = session.scalars(
            select(Query).where(Query.dataset == dataset, Query.split == split)
        ).all()
        console.print(f"[bold]calibrating on {len(rows)} RAGTruth rows[/bold]")

        # Each row's prompt is the premise; ground_truth holds the response.
        # `hallucination` label (1=halluc, 0=faithful) lives in query_metadata.
        pairs = [(r.query_text, r.ground_truth or "") for r in rows]
        labels = [int((r.query_metadata or {}).get("hallucination", 0)) for r in rows]
        scores = hhem_score(pairs)

        summary = fit_threshold(scores, labels, output_dir=str(Path(output).parent))
        console.print(f"[green]threshold = {summary['threshold']:.3f}[/green] "
                      f"(F1 = {summary['f1']:.3f}, AUC = {summary['auc']:.3f})")
    finally:
        session.close()


@app.command()
def run_experiment(
    name: str = typer.Option(..., "--name"),
    systems: str = typer.Option("A,B,C,D", "--systems"),
    datasets: str = typer.Option("multihop", "--datasets"),
    split: str = typer.Option("eval", "--split"),
    limit: int = typer.Option(None, "--limit"),
):
    """Run all four systems over all queries. Resumable on failure."""
    from src.evaluation.runner import run_experiment as _run
    exp_id = _run(
        name=name,
        systems=systems.split(","),
        datasets=datasets.split(","),
        split=split,
        limit=limit,
    )
    console.print(f"[green]experiment finished[/green] id={exp_id}")


@app.command()
def compute_metrics(experiment: int = typer.Option(..., "--experiment")):
    """Compute aggregate metrics for an experiment and write to the `metrics` table."""
    from src.evaluation.metrics import precision_at_k, recall_at_k
    from src.db.models import Metric

    session = get_session()
    try:
        # Pull all runs joined with queries
        rows = session.execute(
            select(Run, Query).join(Query, Run.query_id == Query.id).where(Run.experiment_id == experiment)
        ).all()

        # Group by (system, dataset)
        from collections import defaultdict
        grouped = defaultdict(list)
        for run, q in rows:
            grouped[(run.system, q.dataset)].append((run, q))

        table = Table(title=f"Metrics for experiment {experiment}")
        for col in ["System", "Dataset", "N", "P@5", "R@5", "Accuracy", "AvgHHEM", "Cost", "$/correct"]:
            table.add_column(col)

        for (system, dataset), pairs in grouped.items():
            n = len(pairs)
            p = sum(precision_at_k(r.retrieved_chunk_ids, q.relevant_chunk_ids, settings.top_k) for r, q in pairs) / n
            rec = sum(recall_at_k(r.retrieved_chunk_ids, q.relevant_chunk_ids, settings.top_k) for r, q in pairs) / n
            correct = sum(1 for r, _ in pairs if r.is_correct)
            acc = correct / n
            hhems = [r.hhem_score for r, _ in pairs if r.hhem_score is not None]
            avg_hhem = sum(hhems) / len(hhems) if hhems else None
            flagged = [r.flagged for r, _ in pairs if r.flagged is not None]
            pct_flagged = sum(1 for f in flagged if f) / len(flagged) if flagged else None
            n_steps = [r.n_steps for r, _ in pairs if r.n_steps is not None]
            avg_steps = sum(n_steps) / len(n_steps) if n_steps else None
            total_cost = sum(float(r.cost_usd or 0) for r, _ in pairs)
            cost_per_correct = total_cost / correct if correct else None

            # Upsert
            existing = session.scalar(
                select(Metric).where(
                    Metric.experiment_id == experiment,
                    Metric.system == system,
                    Metric.dataset == dataset,
                )
            )
            data = dict(
                experiment_id=experiment,
                system=system,
                dataset=dataset,
                n_queries=n,
                precision_at_5=p,
                recall_at_5=rec,
                avg_faithfulness=avg_hhem,
                pct_flagged=pct_flagged,
                avg_trajectory_length=avg_steps,
                accuracy=acc,
                total_cost_usd=total_cost,
                cost_per_correct=cost_per_correct,
            )
            if existing:
                for k, v in data.items():
                    setattr(existing, k, v)
            else:
                session.add(Metric(**data))

            table.add_row(
                system, dataset, str(n),
                f"{p:.3f}", f"{rec:.3f}", f"{acc:.3f}",
                f"{avg_hhem:.3f}" if avg_hhem is not None else "-",
                f"${total_cost:.3f}",
                f"${cost_per_correct:.4f}" if cost_per_correct else "-",
            )

        session.commit()
        console.print(table)
    finally:
        session.close()


@app.command()
def export(
    experiment: int = typer.Option(..., "--experiment"),
    fmt: str = typer.Option("json", "--format"),
    output: str = typer.Option("/data/results/", "--output"),
):
    """Export runs + metrics for a given experiment to disk."""
    from src.db.models import Metric

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    session = get_session()
    try:
        runs = session.scalars(select(Run).where(Run.experiment_id == experiment)).all()
        metrics = session.scalars(select(Metric).where(Metric.experiment_id == experiment)).all()

        runs_data = [
            {c.name: getattr(r, c.name) for c in r.__table__.columns} for r in runs
        ]
        metrics_data = [
            {c.name: getattr(m, c.name) for c in m.__table__.columns} for m in metrics
        ]
    finally:
        session.close()

    # JSON-friendly conversion
    def _default(o):
        from datetime import datetime
        from decimal import Decimal
        if isinstance(o, (datetime,)):
            return o.isoformat()
        if isinstance(o, Decimal):
            return float(o)
        return str(o)

    (out / f"exp{experiment}_runs.json").write_text(json.dumps(runs_data, default=_default, indent=2))
    (out / f"exp{experiment}_metrics.json").write_text(json.dumps(metrics_data, default=_default, indent=2))
    console.print(f"[green]wrote[/green] {out}/exp{experiment}_runs.json and exp{experiment}_metrics.json")


@app.command()
def healthcheck():
    """One-line health check across all services."""
    import httpx
    from sqlalchemy import text
    from src.db.session import engine
    from src.retrieval.opensearch_client import get_client

    try:
        with engine.connect() as conn:
            console.print(f"[green]postgres:[/green] {conn.execute(text('SELECT 1')).scalar()}")
    except Exception as e:
        console.print(f"[red]postgres FAILED:[/red] {e}")

    try:
        console.print(f"[green]opensearch:[/green] {get_client().info()['version']['number']}")
    except Exception as e:
        console.print(f"[red]opensearch FAILED:[/red] {e}")

    try:
        r = httpx.get(f"{settings.phoenix_collector_endpoint}/healthz", timeout=5.0)
        console.print(f"[green]phoenix:[/green] HTTP {r.status_code}")
    except Exception as e:
        console.print(f"[red]phoenix FAILED:[/red] {e}")

    try:
        from src.faithfulness.hhem import score
        s = score([("a cat sat on the mat", "a feline rested on the rug")])
        console.print(f"[green]hhem:[/green] {s[0]:.3f}")
    except Exception as e:
        console.print(f"[red]hhem FAILED:[/red] {e}")

    try:
        from src.llm.client import generate
        out = generate([{"role": "user", "content": "say only the word: ok"}])
        console.print(f"[green]bedrock:[/green] '{out['content'][:30]}' (cost ${out['cost_usd']:.6f})")
    except Exception as e:
        console.print(f"[red]bedrock FAILED:[/red] {e}")


if __name__ == "__main__":
    app()
