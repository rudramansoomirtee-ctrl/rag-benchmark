"""retrieval-ceiling evidence + failure visibility

Adds:
- `runs.all_retrieved_chunk_ids` — union of every chunk a run ever retrieved
  (all agent iterations / all sub-question fan-out), as opposed to
  `retrieved_chunk_ids` which is only the final answering context. Powers the
  retrieval-ceiling / failure-attribution analysis for iterative System B.
- `metrics.pct_failed` — fraction of a (system, dataset) group whose runs errored
  (answer IS NULL). Those rows already count as wrong in `accuracy` (deliberate
  "a crash is a wrong answer" policy); this column surfaces them instead of
  hiding them in the denominator.

Both nullable so historical rows are unaffected.

Revision ID: 0004_ceiling_and_failure
Revises: 0003_token_f1
Create Date: 2026-06-11

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "0004_ceiling_and_failure"
down_revision = "0003_token_f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("all_retrieved_chunk_ids", JSONB))
    op.add_column("metrics", sa.Column("pct_failed", sa.Float))


def downgrade() -> None:
    op.drop_column("metrics", "pct_failed")
    op.drop_column("runs", "all_retrieved_chunk_ids")
