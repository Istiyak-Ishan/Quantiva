"""Retain groundedness, charts, guardrail flags and actual provider usage."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "b53d720c9645"
down_revision = "a42c619b8534"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns: list[tuple[str, sa.types.TypeEngine]] = [("confidence", sa.Float()), ("chart_url", sa.Text()), ("guardrail_flags", JSONB()), ("input_tokens", sa.Integer()), ("output_tokens", sa.Integer()), ("model_used", sa.String())]
    for name, column_type in columns:
        op.add_column("messages", sa.Column(name, column_type, nullable=True))
    op.add_column("messages", sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    for name in ("cache_hit", "model_used", "output_tokens", "input_tokens", "guardrail_flags", "chart_url", "confidence"):
        op.drop_column("messages", name)
