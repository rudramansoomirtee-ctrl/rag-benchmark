"""secondary answer metrics: exact-match + CRAG LLM-as-judge

Adds the secondary correctness columns. Primary correctness (contains_match)
is unchanged. `runs.llm_judge_label` holds the per-answer CRAG label; the
`metrics` aggregates hold normalized exact-match accuracy and the mean CRAG
truthfulness score.

Revision ID: 0002_secondary_metrics
Revises: 0001_initial
Create Date: 2026-05-22

"""
from alembic import op
import sqlalchemy as sa


revision = "0002_secondary_metrics"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("llm_judge_label", sa.Text))
    op.add_column("metrics", sa.Column("accuracy_exact", sa.Float))
    op.add_column("metrics", sa.Column("crag_score", sa.Float))


def downgrade() -> None:
    op.drop_column("metrics", "crag_score")
    op.drop_column("metrics", "accuracy_exact")
    op.drop_column("runs", "llm_judge_label")
