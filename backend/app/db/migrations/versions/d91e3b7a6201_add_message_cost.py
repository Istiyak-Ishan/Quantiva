"""Add the message cost column missing from the original chat migration.

Revision ID: d91e3b7a6201
Revises: c58ae3901b52

Some existing development databases already have ``messages.cost_usd`` (for
example, added manually or by an earlier intermediate revision) while their
Alembic version is still behind. Adding the column blindly aborts the whole
upgrade. Check the live schema first so the migration is additive and safe on
both clean and partially-upgraded databases.
"""
from alembic import op, context
import sqlalchemy as sa

revision = "d91e3b7a6201"
down_revision = "c58ae3901b52"
branch_labels = None
depends_on = None


def _has_message_cost_column() -> bool:
    if context.is_offline_mode():
        return False
    inspector = sa.inspect(op.get_bind())
    return any(column["name"] == "cost_usd" for column in inspector.get_columns("messages"))


def upgrade() -> None:
    if not _has_message_cost_column():
        op.add_column("messages", sa.Column("cost_usd", sa.Float(), nullable=True, server_default="0.0"))


def downgrade() -> None:
    if _has_message_cost_column():
        op.drop_column("messages", "cost_usd")
