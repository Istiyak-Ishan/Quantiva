import uuid
from datetime import datetime, timezone
from sqlalchemy import Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.database import Base

class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    avg_faithfulness: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_context_relevance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_answer_relevance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    
    # Store the individual question evaluations as JSON
    per_question_scores: Mapped[list] = mapped_column(JSONB, nullable=False, default=[])
    
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending") # pending, completed, failed
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
