"""Store complete chunk text; do not pretend old previews contain lost content."""
from alembic import op
import sqlalchemy as sa

revision = "a42c619b8534"
down_revision = "f31b508a7423"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("text", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "text")
