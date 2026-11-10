"""Track ingestion start time and terminal failure reasons."""
from alembic import op
import sqlalchemy as sa

revision = "d4e8f1a9c3b7"
down_revision = "b53d720c9645"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("failure_reason", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "failure_reason")
    op.drop_column("documents", "processing_started_at")
