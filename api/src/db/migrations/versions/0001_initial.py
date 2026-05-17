"""initial schema: experiments, queries, chunks, runs, metrics

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-17

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("config_json", postgresql.JSONB, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("notes", sa.Text),
    )

    op.create_table(
        "queries",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset", sa.Text, nullable=False),
        sa.Column("external_id", sa.Text, nullable=False),
        sa.Column("split", sa.Text, nullable=False),
        sa.Column("task_type", sa.Text),
        sa.Column("query_text", sa.Text, nullable=False),
        sa.Column("ground_truth", sa.Text),
        sa.Column("relevant_chunk_ids", postgresql.JSONB, nullable=False),
        sa.Column("metadata", postgresql.JSONB),
        sa.UniqueConstraint("dataset", "external_id"),
    )
    op.create_index("ix_queries_dataset_split", "queries", ["dataset", "split"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("dataset", sa.Text, nullable=False),
        sa.Column("external_id", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSONB),
        sa.UniqueConstraint("dataset", "external_id"),
    )

    op.create_table(
        "runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("experiment_id", sa.Integer, sa.ForeignKey("experiments.id", ondelete="CASCADE")),
        sa.Column("system", sa.Text, nullable=False),
        sa.Column("query_id", sa.Integer, sa.ForeignKey("queries.id")),
        sa.Column("retrieved_chunk_ids", postgresql.JSONB, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("hhem_score", sa.Float),
        sa.Column("flagged", sa.Boolean),
        sa.Column("n_steps", sa.Integer),
        sa.Column("tokens_in", sa.Integer),
        sa.Column("tokens_out", sa.Integer),
        sa.Column("latency_ms", sa.Integer),
        sa.Column("cost_usd", sa.Numeric(10, 6)),
        sa.Column("is_correct", sa.Boolean),
        sa.Column("phoenix_trace_id", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("experiment_id", "system", "query_id"),
    )
    op.create_index("ix_runs_experiment_system", "runs", ["experiment_id", "system"])

    op.create_table(
        "metrics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("experiment_id", sa.Integer, sa.ForeignKey("experiments.id", ondelete="CASCADE")),
        sa.Column("system", sa.Text, nullable=False),
        sa.Column("dataset", sa.Text, nullable=False),
        sa.Column("n_queries", sa.Integer, nullable=False),
        sa.Column("precision_at_5", sa.Float),
        sa.Column("recall_at_5", sa.Float),
        sa.Column("avg_faithfulness", sa.Float),
        sa.Column("pct_flagged", sa.Float),
        sa.Column("avg_trajectory_length", sa.Float),
        sa.Column("accuracy", sa.Float),
        sa.Column("total_cost_usd", sa.Numeric(10, 4)),
        sa.Column("cost_per_correct", sa.Numeric(10, 6)),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("experiment_id", "system", "dataset"),
    )


def downgrade() -> None:
    op.drop_table("metrics")
    op.drop_table("runs")
    op.drop_table("chunks")
    op.drop_table("queries")
    op.drop_table("experiments")
