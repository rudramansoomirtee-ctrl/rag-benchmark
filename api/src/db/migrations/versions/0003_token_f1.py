"""secondary answer metric: SQuAD-style token F1

Adds `metrics.avg_token_f1` — the mean token-overlap F1 per (system, dataset),
complementary to the primary `contains_match` accuracy and the existing
`accuracy_exact`. Computed on the fly from stored answers in `compute-metrics`,
so it can be backfilled over historical experiments without re-running systems.

Revision ID: 0003_token_f1
Revises: 0002_secondary_metrics
Create Date: 2026-06-11

"""
from alembic import op
import sqlalchemy as sa


revision = "0003_token_f1"
down_revision = "0002_secondary_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("metrics", sa.Column("avg_token_f1", sa.Float))


def downgrade() -> None:
    op.drop_column("metrics", "avg_token_f1")
