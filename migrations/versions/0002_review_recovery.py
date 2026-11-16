"""Retain uncertainty and explicit restart confirmation for model dispatch recovery."""

import sqlalchemy as sa
from alembic import op

revision = "0002_review_recovery"
down_revision = "0001_core"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "vlm_reviews",
        sa.Column(
            "runtime_uncertain", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
    )
    op.add_column(
        "vlm_reviews",
        sa.Column("runtime_restart_confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        "UPDATE vlm_reviews SET runtime_uncertain = true WHERE status IN ('running','timeout') OR error_code = 'interrupted'"
    )


def downgrade():
    raise RuntimeError("Destructive downgrade is disabled. Restore a verified backup instead.")
