"""SQLite-compatible ORM schema for the local-first Quantiva desktop app."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def uuid_string() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class SchemaVersion(Base):
    __tablename__ = "schema_versions"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    app_version: Mapped[str] = mapped_column(String(32), nullable=False)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    description: Mapped[str | None] = mapped_column(Text)


class ApplicationProfile(TimestampMixin, Base):
    __tablename__ = "application_profiles"

    profile_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=uuid_string
    )
    display_name: Mapped[str] = mapped_column(
        String(255), nullable=False, default="Local User"
    )
    workspace_name: Mapped[str] = mapped_column(
        String(255), nullable=False, default="My Workspace"
    )
    workspace_preferences_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )


class ModelConfiguration(TimestampMixin, Base):
    __tablename__ = "model_configurations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    provider_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(2048))
    credential_key_id: Mapped[str | None] = mapped_column(String(255))
    has_api_key: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.2)
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=4096)
    context_window: Mapped[int | None] = mapped_column(Integer)
    capabilities_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )

    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="model_configuration"
    )


class EmbeddingConfiguration(TimestampMixin, Base):
    __tablename__ = "embedding_configurations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    provider_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_url: Mapped[str | None] = mapped_column(String(2048))
    credential_key_id: Mapped[str | None] = mapped_column(String(255))
    has_api_key: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=32)
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )

    knowledge_bases: Mapped[list[KnowledgeBase]] = relationship(
        back_populates="embedding_configuration"
    )


class RetrievalConfiguration(Base):
    __tablename__ = "retrieval_configurations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    search_mode: Mapped[str] = mapped_column(
        String(32), nullable=False, default="hybrid"
    )
    dense_top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    sparse_top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    rrf_k: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    dense_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    sparse_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    enable_reranking: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    reranker_provider: Mapped[str | None] = mapped_column(String(64))
    reranker_model: Mapped[str | None] = mapped_column(String(255))
    final_top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    enable_agentic_mode: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    max_agent_iterations: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )

    knowledge_bases: Mapped[list[KnowledgeBase]] = relationship(
        back_populates="retrieval_configuration"
    )


class KnowledgeBase(TimestampMixin, Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    embedding_config_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("embedding_configurations.id", ondelete="SET NULL"),
        index=True,
    )
    retrieval_config_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("retrieval_configurations.id", ondelete="SET NULL"),
        index=True,
    )
    document_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="ready", index=True
    )

    embedding_configuration: Mapped[EmbeddingConfiguration | None] = relationship(
        back_populates="knowledge_bases"
    )
    retrieval_configuration: Mapped[RetrievalConfiguration | None] = relationship(
        back_populates="knowledge_bases"
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    jobs: Mapped[list[ProcessingJob]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    index_metadata: Mapped[IndexMetadata | None] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id", "sha256_hash", name="uq_documents_kb_sha256"
        ),
        Index("ix_documents_kb_status", "knowledge_base_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    knowledge_base_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_rel_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    processed_rel_path: Mapped[str | None] = mapped_column(String(2048))
    file_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    ocr_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    jobs: Mapped[list[ProcessingJob]] = relationship(
        back_populates="document",
        passive_deletes=True,
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_index", name="uq_document_chunks_document_index"
        ),
        Index("ix_document_chunks_kb_document", "knowledge_base_id", "document_id"),
        Index("ix_document_chunks_content_hash", "content_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(String(1024))
    sheet_name: Mapped[str | None] = mapped_column(String(255))
    is_table: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")
    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="chunks")


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_kb_updated", "knowledge_base_id", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    knowledge_base_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    model_config_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("model_configurations.id", ondelete="SET NULL"),
        index=True,
    )
    is_pinned: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(
        back_populates="conversations"
    )
    model_configuration: Mapped[ModelConfiguration | None] = relationship(
        back_populates="conversations"
    )
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning_trace_json: Mapped[list[Any] | dict[str, Any] | None] = mapped_column(JSON)
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    retrieved_chunks_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    token_usage_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    provider_used: Mapped[str | None] = mapped_column(String(64))
    model_used: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        Index("ix_processing_jobs_status_created", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    knowledge_base_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        index=True,
    )
    document_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="queued", index=True
    )
    stage: Mapped[str | None] = mapped_column(String(64))
    progress_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    message: Mapped[str | None] = mapped_column(Text)
    error_details: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    knowledge_base: Mapped[KnowledgeBase | None] = relationship(back_populates="jobs")
    document: Mapped[Document | None] = relationship(back_populates="jobs")


class IndexMetadata(Base):
    __tablename__ = "index_metadata"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    knowledge_base_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    vector_collection_name: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    embedding_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sparse_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    needs_reindex: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    knowledge_base: Mapped[KnowledgeBase] = relationship(
        back_populates="index_metadata"
    )


__all__ = [
    "ApplicationProfile",
    "Base",
    "Conversation",
    "Document",
    "DocumentChunk",
    "EmbeddingConfiguration",
    "IndexMetadata",
    "KnowledgeBase",
    "Message",
    "ModelConfiguration",
    "ProcessingJob",
    "RetrievalConfiguration",
    "SchemaVersion",
]
