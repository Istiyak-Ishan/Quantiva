import uuid
from sqlalchemy import String, Integer, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.db.database import Base

class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text_preview: Mapped[str] = mapped_column(Text, nullable=False)
    full_text: Mapped[str | None] = mapped_column("text", Text, nullable=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=True)
    chunk_type: Mapped[str] = mapped_column(String, default="text")
    
    # Links to the vector database entry
    qdrant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, unique=True, default=uuid.uuid4)

    @property
    def text(self) -> str:
        # Legacy rows need explicit reingestion; previews cannot be reconstructed.
        return self.full_text if self.full_text is not None else self.text_preview

    # Optional: Relationship mapping
    # document = relationship("Document", back_populates="chunks")
