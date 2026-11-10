"""Scope existing evaluation runs to their document owner, preserving rows."""
from alembic import op
import sqlalchemy as sa

revision = "f31b508a7423"
down_revision = "e20a497f6312"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("eval_runs", sa.Column("user_id", sa.UUID(), nullable=True))
    op.execute("UPDATE eval_runs SET user_id = documents.user_id FROM documents WHERE eval_runs.doc_id = documents.id")
    op.alter_column("eval_runs", "user_id", nullable=False)
    op.create_foreign_key("fk_eval_runs_user_id", "eval_runs", "users", ["user_id"], ["id"], ondelete="CASCADE")
    op.create_index("ix_eval_runs_user_id", "eval_runs", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_eval_runs_user_id", table_name="eval_runs")
    op.drop_constraint("fk_eval_runs_user_id", "eval_runs", type_="foreignkey")
    op.drop_column("eval_runs", "user_id")
