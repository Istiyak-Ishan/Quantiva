"""create eval runs table

Revision ID: c58ae3901b52
Revises: b47dc6e638e8
Create Date: 2026-11-08 19:58:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c58ae3901b52'
down_revision: Union[str, Sequence[str], None] = 'b47dc6e638e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table(
        'eval_runs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('doc_id', sa.UUID(), nullable=False),
        sa.Column('avg_faithfulness', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('avg_context_relevance', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('avg_answer_relevance', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('per_question_scores', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('status', sa.Text(), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['doc_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    op.drop_table('eval_runs')
