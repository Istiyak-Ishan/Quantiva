"""
backend/app/core/knowledge_bases.py
────────────────────────────────────────────────────────────────────────────────
Knowledge Base Service & Independent Index Isolation for Quantiva Desktop.

Responsibilities:
1. Knowledge Base Lifecycle Management:
   - Creation with strict embedding dimension locking in IndexMetadata.
   - Dedicated local vector collection allocation (kb_{kb_id}).
   - Listing with aggregated statistics and health checks.
   - Deep inspection and status reporting (documents by status, chunk totals).
   - Updates with model space divergence detection (IndexMetadata.needs_reindex).
   - Clean deletion purging vectors, FTS5 sparse rows, original/processed files,
     and cascading SQLite records.
2. Document Lifecycle Management within Knowledge Bases:
   - Upload with SHA-256 duplicate detection scoped per Knowledge Base.
   - Force-overwrite support (safely replacing existing documents).
   - Streaming file write via LocalFileStore.
   - Asynchronous ingestion job submission via LocalJobManager.
   - Deletion purging vectors, FTS5 entries, files, and DB records while
     decrementing counters atomically.
3. Strict Index Isolation & Dimension Safety:
   - Early dimension validation before hitting the vector store.
   - Multi-collection isolation ensuring zero cross-KB leakage.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import io
import logging
from pathlib import Path
import threading
from typing import Any, BinaryIO
import uuid

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, settings
from app.ingestion.types import DocumentChunkPayload
from app.jobs.indexing import submit_knowledge_base_reindex_job
from app.jobs.ingestion import submit_document_ingestion_job
from app.jobs.manager import JobStatus, LocalJobManager, TERMINAL_STATUSES, get_local_job_manager
from app.providers.interfaces.exceptions import EmbeddingDimensionMismatchError
from app.storage.database import SessionLocal, session_scope
from app.storage.file_store import LocalFileStore, StoredFile, sanitize_filename
from app.storage.interfaces import SearchHit
from app.storage.models import (
    Document,
    DocumentChunk,
    EmbeddingConfiguration,
    IndexMetadata,
    KnowledgeBase,
    ProcessingJob,
    RetrievalConfiguration,
)
from app.storage.vector_store import (
    LocalVectorStore,
    VectorStoreManager,
    collection_name_for_knowledge_base,
)

logger = logging.getLogger(__name__)


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class DuplicateDocumentError(FileExistsError):
    """Raised when an identical document hash already exists in the target knowledge base."""

    def __init__(
        self,
        message: str,
        *,
        existing_document_id: str | None = None,
        sha256_hash: str | None = None,
        knowledge_base_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.existing_document_id = existing_document_id
        self.sha256_hash = sha256_hash
        self.knowledge_base_id = knowledge_base_id


# ── Data Transfer Objects ─────────────────────────────────────────────────────

@dataclass
class KnowledgeBaseSummary:
    """Summary of a knowledge base for list views."""

    id: str
    name: str
    description: str | None
    status: str
    document_count: int
    chunk_count: int
    embedding_config_id: str | None
    embedding_provider: str | None
    embedding_model_name: str | None
    embedding_dimensions: int | None
    retrieval_config_id: str | None
    needs_reindex: bool
    created_at: datetime | None
    updated_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeBaseDetails:
    """Detailed information for a single knowledge base."""

    id: str
    name: str
    description: str | None
    status: str
    document_count: int
    chunk_count: int
    documents_by_status: dict[str, int]
    embedding_config_id: str | None
    embedding_provider: str | None
    embedding_model_name: str | None
    embedding_dimensions: int | None
    retrieval_config_id: str | None
    vector_collection_name: str
    vector_count: int
    sparse_count: int
    needs_reindex: bool
    is_healthy: bool
    last_indexed_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentSummary:
    """Summary of a document record."""

    id: str
    knowledge_base_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    sha256_hash: str
    status: str
    chunk_count: int
    error_message: str | None
    ocr_applied: bool
    metadata: dict[str, Any]
    created_at: datetime | None
    updated_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Knowledge Base Service Implementation ─────────────────────────────────────

class KnowledgeBaseService:
    """Central service managing Knowledge Base and Document lifecycles.

    Guarantees strict independent index isolation:
    - Every KnowledgeBase locks its embedding model and dimensions in IndexMetadata.
    - Vector collections are named 'kb_{kb_id}' and isolated in embedded Qdrant.
    - Sparse FTS5 entries are scoped by knowledge_base_id.
    - Document files are organized in dedicated directory trees per knowledge_base_id.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
        *,
        vector_store: LocalVectorStore | None = None,
        file_store: LocalFileStore | None = None,
        job_manager: LocalJobManager | None = None,
    ) -> None:
        self.session_factory = session_factory or SessionLocal
        self.vector_store = vector_store or LocalVectorStore()
        self.file_store = file_store or LocalFileStore()
        self.job_manager = job_manager

    # ── Knowledge Base Lifecycle ──────────────────────────────────────────────

    async def create_knowledge_base(
        self,
        name: str,
        description: str | None = None,
        *,
        embedding_config_id: str | None = None,
        retrieval_config_id: str | None = None,
    ) -> KnowledgeBase:
        """Create a new Knowledge Base and initialize its locked vector collection."""
        name_clean = str(name).strip()
        if not name_clean:
            raise ValueError("Knowledge base name cannot be empty")

        with self.session_factory() as session:
            # Check name uniqueness
            existing = session.scalar(
                select(KnowledgeBase).where(KnowledgeBase.name == name_clean)
            )
            if existing is not None:
                raise ValueError(f"KnowledgeBase with name '{name_clean}' already exists.")

            # Resolve Embedding Configuration
            if embedding_config_id:
                emb_config = session.get(EmbeddingConfiguration, embedding_config_id)
                if emb_config is None:
                    raise KeyError(f"EmbeddingConfiguration not found: {embedding_config_id}")
            else:
                emb_config = session.scalar(
                    select(EmbeddingConfiguration).where(EmbeddingConfiguration.is_default == True)
                )
                if emb_config is None:
                    emb_config = session.scalar(select(EmbeddingConfiguration))
                if emb_config is None:
                    raise ValueError(
                        "No EmbeddingConfiguration found in system. Please configure an embedding model first."
                    )

            # Resolve Retrieval Configuration
            if retrieval_config_id:
                ret_config = session.get(RetrievalConfiguration, retrieval_config_id)
                if ret_config is None:
                    raise KeyError(f"RetrievalConfiguration not found: {retrieval_config_id}")
            else:
                ret_config = session.scalar(
                    select(RetrievalConfiguration).where(RetrievalConfiguration.is_default == True)
                )
                if ret_config is None:
                    ret_config = session.scalar(select(RetrievalConfiguration))

            kb_id = str(uuid.uuid4())
            collection_name = collection_name_for_knowledge_base(kb_id)

            kb = KnowledgeBase(
                id=kb_id,
                name=name_clean,
                description=description.strip() if description else None,
                embedding_config_id=emb_config.id,
                retrieval_config_id=ret_config.id if ret_config else None,
                document_count=0,
                chunk_count=0,
                status="ready",
            )
            session.add(kb)

            # Lock embedding dimensions and model space in IndexMetadata
            index_metadata = IndexMetadata(
                knowledge_base_id=kb_id,
                vector_collection_name=collection_name,
                embedding_provider=emb_config.provider_type,
                embedding_model_name=emb_config.model_name,
                embedding_dimensions=int(emb_config.dimensions),
                vector_count=0,
                sparse_count=0,
                needs_reindex=False,
            )
            session.add(index_metadata)
            session.commit()

            # Ensure the dedicated vector collection is allocated in embedded Qdrant
            await self.vector_store.ensure_collection(
                collection_name,
                embedding_dimensions=int(emb_config.dimensions),
                index_metadata=index_metadata,
            )

            # Expunge so caller can access attributes safely outside session
            session.refresh(kb)
            session.expunge(kb)
            return kb

    def list_knowledge_bases(self) -> list[KnowledgeBaseSummary]:
        """List all knowledge bases with aggregated statistics."""
        with self.session_factory() as session:
            stmt = (
                select(KnowledgeBase, IndexMetadata)
                .outerjoin(IndexMetadata, KnowledgeBase.id == IndexMetadata.knowledge_base_id)
                .order_by(KnowledgeBase.created_at.desc())
            )
            rows = session.execute(stmt).all()
            summaries: list[KnowledgeBaseSummary] = []
            for kb, index_meta in rows:
                summaries.append(
                    KnowledgeBaseSummary(
                        id=kb.id,
                        name=kb.name,
                        description=kb.description,
                        status=kb.status,
                        document_count=kb.document_count,
                        chunk_count=kb.chunk_count,
                        embedding_config_id=kb.embedding_config_id,
                        embedding_provider=index_meta.embedding_provider if index_meta else None,
                        embedding_model_name=index_meta.embedding_model_name if index_meta else None,
                        embedding_dimensions=index_meta.embedding_dimensions if index_meta else None,
                        retrieval_config_id=kb.retrieval_config_id,
                        needs_reindex=index_meta.needs_reindex if index_meta else False,
                        created_at=kb.created_at,
                        updated_at=kb.updated_at,
                    )
                )
            return summaries

    async def get_knowledge_base_details(self, knowledge_base_id: str) -> KnowledgeBaseDetails:
        """Return deep status and index health diagnostics for a knowledge base."""
        with self.session_factory() as session:
            kb = session.get(KnowledgeBase, knowledge_base_id)
            if kb is None:
                raise KeyError(f"KnowledgeBase not found: {knowledge_base_id}")

            index_meta = session.scalar(
                select(IndexMetadata).where(IndexMetadata.knowledge_base_id == knowledge_base_id)
            )

            # Calculate document status breakdown
            doc_status_rows = session.execute(
                select(Document.status, func.count(Document.id))
                .where(Document.knowledge_base_id == knowledge_base_id)
                .group_by(Document.status)
            ).all()

            docs_by_status: dict[str, int] = {
                status: count for status, count in doc_status_rows
            }
            docs_by_status["total"] = sum(docs_by_status.values())

            # Real chunk count
            total_chunks = session.scalar(
                select(func.count(DocumentChunk.id)).where(
                    DocumentChunk.knowledge_base_id == knowledge_base_id
                )
            ) or 0

            collection_name = (
                index_meta.vector_collection_name
                if index_meta
                else collection_name_for_knowledge_base(knowledge_base_id)
            )

            # Check collection health from vector store
            coll_info = await self.vector_store.get_collection_info(collection_name)
            is_healthy = (coll_info is not None) and not (
                index_meta.needs_reindex if index_meta else False
            )

            return KnowledgeBaseDetails(
                id=kb.id,
                name=kb.name,
                description=kb.description,
                status=kb.status,
                document_count=kb.document_count,
                chunk_count=total_chunks,
                documents_by_status=docs_by_status,
                embedding_config_id=kb.embedding_config_id,
                embedding_provider=index_meta.embedding_provider if index_meta else None,
                embedding_model_name=index_meta.embedding_model_name if index_meta else None,
                embedding_dimensions=index_meta.embedding_dimensions if index_meta else None,
                retrieval_config_id=kb.retrieval_config_id,
                vector_collection_name=collection_name,
                vector_count=coll_info.point_count if coll_info else (index_meta.vector_count if index_meta else 0),
                sparse_count=index_meta.sparse_count if index_meta else 0,
                needs_reindex=index_meta.needs_reindex if index_meta else False,
                is_healthy=is_healthy,
                last_indexed_at=index_meta.last_indexed_at if index_meta else None,
                created_at=kb.created_at,
                updated_at=kb.updated_at,
            )

    async def update_knowledge_base(
        self,
        knowledge_base_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        embedding_config_id: str | None = None,
        retrieval_config_id: str | None = None,
        trigger_reindex: bool = False,
    ) -> KnowledgeBase:
        """Update Knowledge Base metadata and detect embedding configuration divergence."""
        reindex_job_id: str | None = None

        with self.session_factory() as session:
            kb = session.get(KnowledgeBase, knowledge_base_id)
            if kb is None:
                raise KeyError(f"KnowledgeBase not found: {knowledge_base_id}")

            if name is not None:
                name_clean = name.strip()
                if not name_clean:
                    raise ValueError("Knowledge base name cannot be empty")
                existing = session.scalar(
                    select(KnowledgeBase).where(
                        KnowledgeBase.name == name_clean,
                        KnowledgeBase.id != knowledge_base_id,
                    )
                )
                if existing is not None:
                    raise ValueError(f"KnowledgeBase with name '{name_clean}' already exists.")
                kb.name = name_clean

            if description is not None:
                kb.description = description.strip() if description else None

            if retrieval_config_id is not None:
                ret_config = session.get(RetrievalConfiguration, retrieval_config_id)
                if ret_config is None:
                    raise KeyError(f"RetrievalConfiguration not found: {retrieval_config_id}")
                kb.retrieval_config_id = retrieval_config_id

            if embedding_config_id is not None and embedding_config_id != kb.embedding_config_id:
                new_emb = session.get(EmbeddingConfiguration, embedding_config_id)
                if new_emb is None:
                    raise KeyError(f"EmbeddingConfiguration not found: {embedding_config_id}")

                index_meta = session.scalar(
                    select(IndexMetadata).where(IndexMetadata.knowledge_base_id == knowledge_base_id)
                )
                if index_meta is not None:
                    # Model space divergence check
                    if (
                        new_emb.dimensions != index_meta.embedding_dimensions
                        or new_emb.model_name != index_meta.embedding_model_name
                        or new_emb.provider_type != index_meta.embedding_provider
                    ):
                        index_meta.needs_reindex = True
                        logger.info(
                            "Knowledge base %s embedding model changed from %s (%dd) to %s (%dd). Marked needs_reindex=True.",
                            knowledge_base_id,
                            index_meta.embedding_model_name,
                            index_meta.embedding_dimensions,
                            new_emb.model_name,
                            new_emb.dimensions,
                        )

                kb.embedding_config_id = embedding_config_id

            session.commit()
            session.refresh(kb)
            session.expunge(kb)

        if trigger_reindex and embedding_config_id is not None:
            reindex_job_id = await self.trigger_reindex(
                knowledge_base_id, new_embedding_config_id=embedding_config_id
            )

        return kb

    async def delete_knowledge_base(self, knowledge_base_id: str) -> bool:
        """Completely purge a knowledge base across vector, FTS5, files, and SQLite stores."""
        with self.session_factory() as session:
            kb = session.get(KnowledgeBase, knowledge_base_id)
            if kb is None:
                return False

            # 1. Cancel active jobs
            active_jobs = session.scalars(
                select(ProcessingJob.id).where(
                    ProcessingJob.knowledge_base_id == knowledge_base_id,
                    ProcessingJob.status.notin_(TERMINAL_STATUSES),
                )
            ).all()

        if self.job_manager is not None:
            for job_id in active_jobs:
                try:
                    await self.job_manager.cancel(job_id)
                except Exception:
                    logger.debug("Failed cancelling job %s during KB deletion", job_id)

        # 2. Delete vector collection in embedded Qdrant
        collection_name = collection_name_for_knowledge_base(knowledge_base_id)
        try:
            await self.vector_store.delete_collection(collection_name)
        except Exception as exc:
            logger.warning("Error deleting vector collection %s: %s", collection_name, exc)

        # 3. Purge sparse FTS5 entries
        try:
            with self.session_factory() as session:
                session.execute(
                    text("DELETE FROM document_chunks_fts WHERE knowledge_base_id = :kb_id"),
                    {"kb_id": knowledge_base_id},
                )
                session.commit()
        except Exception as exc:
            logger.debug("FTS5 sparse cleanup notice for KB %s: %s", knowledge_base_id, exc)

        # 4. Delete all original and processed files
        try:
            self.file_store.delete_knowledge_base(knowledge_base_id)
        except Exception as exc:
            logger.warning("Error deleting document files for KB %s: %s", knowledge_base_id, exc)

        # 5. Delete SQLite records (cascades to documents, chunks, index_metadata, jobs)
        with self.session_factory() as session:
            target = session.get(KnowledgeBase, knowledge_base_id)
            if target is not None:
                session.delete(target)
                session.commit()

        logger.info("Successfully deleted knowledge base %s and all associated assets.", knowledge_base_id)
        return True

    # ── Document Lifecycle ────────────────────────────────────────────────────

    async def upload_document(
        self,
        knowledge_base_id: str,
        file_content: bytes | BinaryIO | AsyncIterator[bytes],
        filename: str,
        *,
        file_type: str | None = None,
        allow_overwrite: bool = False,
        metadata_json: dict[str, Any] | None = None,
    ) -> Document:
        """Stream save an uploaded document, compute SHA-256, and enqueue ingestion."""
        filename_clean = sanitize_filename(filename)
        if not filename_clean or filename_clean == "document":
            filename_clean = Path(filename).name or "document.bin"

        with self.session_factory() as session:
            kb = session.get(KnowledgeBase, knowledge_base_id)
            if kb is None:
                raise KeyError(f"KnowledgeBase not found: {knowledge_base_id}")

        doc_id = str(uuid.uuid4())

        # Stream save file and compute SHA-256 in a single pass
        if hasattr(file_content, "__aiter__"):
            stored_file: StoredFile = await self.file_store.save_original_stream(
                knowledge_base_id=knowledge_base_id,
                document_id=doc_id,
                filename=filename_clean,
                stream=file_content,
            )
        else:
            stored_file = await asyncio.to_thread(
                self.file_store.save_original,
                knowledge_base_id=knowledge_base_id,
                document_id=doc_id,
                filename=filename_clean,
                content=file_content,
            )

        # Check for duplicate SHA-256 hash inside this knowledge base
        with self.session_factory() as session:
            existing_doc = session.scalar(
                select(Document).where(
                    Document.knowledge_base_id == knowledge_base_id,
                    Document.sha256_hash == stored_file.sha256_hash,
                )
            )

        if existing_doc is not None:
            if not allow_overwrite:
                # Clean up the newly written file to avoid orphaned storage
                self.file_store.delete_document(
                    knowledge_base_id=knowledge_base_id,
                    document_id=doc_id,
                    original_rel_path=stored_file.relative_path,
                )
                raise DuplicateDocumentError(
                    f"Document '{filename}' with hash {stored_file.sha256_hash} "
                    f"already exists in knowledge base '{knowledge_base_id}' (doc ID: {existing_doc.id}).",
                    existing_document_id=existing_doc.id,
                    sha256_hash=stored_file.sha256_hash,
                    knowledge_base_id=knowledge_base_id,
                )
            else:
                # Safely delete existing document first
                logger.info(
                    "Force-overwriting duplicate document %s in knowledge base %s.",
                    existing_doc.id,
                    knowledge_base_id,
                )
                await self.delete_document(knowledge_base_id, existing_doc.id)

        # Determine file type
        resolved_type = (
            file_type
            or Path(filename).suffix.lstrip(".").lower()
            or "bin"
        )

        # Insert Document record
        with self.session_factory() as session:
            kb = session.get(KnowledgeBase, knowledge_base_id)
            doc = Document(
                id=doc_id,
                knowledge_base_id=knowledge_base_id,
                filename=filename_clean,
                original_rel_path=stored_file.relative_path,
                file_type=resolved_type,
                file_size_bytes=stored_file.file_size_bytes,
                sha256_hash=stored_file.sha256_hash,
                status="QUEUED",
                chunk_count=0,
                metadata_json=metadata_json or {},
            )
            session.add(doc)
            if kb is not None:
                kb.document_count += 1
            session.commit()
            session.refresh(doc)
            session.expunge(doc)

        # Enqueue document ingestion job
        if self.job_manager is not None:
            try:
                await submit_document_ingestion_job(
                    document_id=doc_id,
                    knowledge_base_id=knowledge_base_id,
                    manager=self.job_manager,
                )
            except Exception as exc:
                logger.warning(
                    "Ingestion job submission failed for document %s: %s",
                    doc_id,
                    exc,
                )

        return doc

    async def delete_document(self, knowledge_base_id: str, document_id: str) -> bool:
        """Purge a single document across vectors, sparse FTS5, files, and SQLite records."""
        with self.session_factory() as session:
            doc = session.get(Document, document_id)
            if doc is None:
                return False
            if doc.knowledge_base_id != knowledge_base_id:
                raise ValueError(
                    f"Document {document_id} belongs to KB {doc.knowledge_base_id}, not {knowledge_base_id}"
                )

            original_path = doc.original_rel_path
            processed_path = doc.processed_rel_path
            chunk_count = doc.chunk_count or 0

            # Find active jobs to cancel
            active_jobs = session.scalars(
                select(ProcessingJob.id).where(
                    ProcessingJob.document_id == document_id,
                    ProcessingJob.status.notin_(TERMINAL_STATUSES),
                )
            ).all()

        # 1. Cancel active jobs
        if self.job_manager is not None:
            for job_id in active_jobs:
                try:
                    await self.job_manager.cancel(job_id)
                except Exception:
                    logger.debug("Failed cancelling job %s during doc deletion", job_id)

        # 2. Delete vectors from embedded Qdrant collection
        collection_name = collection_name_for_knowledge_base(knowledge_base_id)
        try:
            await self.vector_store.delete_by_document_id(
                collection_name=collection_name,
                document_id=document_id,
            )
        except Exception as exc:
            logger.warning("Error deleting vectors for document %s: %s", document_id, exc)

        # 3. Purge sparse FTS5 entries (defense-in-depth in addition to SQLite CASCADE trigger)
        try:
            with self.session_factory() as session:
                session.execute(
                    text("DELETE FROM document_chunks_fts WHERE document_id = :doc_id"),
                    {"doc_id": document_id},
                )
                session.commit()
        except Exception as exc:
            logger.debug("FTS5 cleanup notice for document %s: %s", document_id, exc)

        # 4. Delete original and processed files from disk
        try:
            self.file_store.delete_document(
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                original_rel_path=original_path,
                processed_rel_path=processed_path,
            )
        except Exception as exc:
            logger.warning("Error deleting files for document %s: %s", document_id, exc)

        # 5. Delete SQLite Document record and update KB counters atomically
        with self.session_factory() as session:
            target_doc = session.get(Document, document_id)
            if target_doc is not None:
                session.delete(target_doc)

            target_kb = session.get(KnowledgeBase, knowledge_base_id)
            if target_kb is not None:
                target_kb.document_count = session.scalar(
                    select(func.count(Document.id)).where(
                        Document.knowledge_base_id == knowledge_base_id,
                        Document.id != document_id,
                    )
                ) or 0
                target_kb.chunk_count = session.scalar(
                    select(func.count(DocumentChunk.id)).where(
                        DocumentChunk.knowledge_base_id == knowledge_base_id,
                        DocumentChunk.document_id != document_id,
                    )
                ) or 0

            session.commit()

        logger.info(
            "Successfully deleted document %s from knowledge base %s.",
            document_id,
            knowledge_base_id,
        )
        return True

    def list_documents(
        self,
        knowledge_base_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentSummary]:
        """List documents in a knowledge base with pagination and optional status filter."""
        with self.session_factory() as session:
            stmt = select(Document).where(Document.knowledge_base_id == knowledge_base_id)
            if status is not None:
                stmt = stmt.where(Document.status == status.strip().lower())
            stmt = stmt.order_by(Document.created_at.desc()).offset(offset).limit(limit)

            docs = session.scalars(stmt).all()
            summaries: list[DocumentSummary] = []
            for d in docs:
                summaries.append(
                    DocumentSummary(
                        id=d.id,
                        knowledge_base_id=d.knowledge_base_id,
                        filename=d.filename,
                        file_type=d.file_type,
                        file_size_bytes=d.file_size_bytes,
                        sha256_hash=d.sha256_hash,
                        status=d.status,
                        chunk_count=d.chunk_count,
                        error_message=d.error_message,
                        ocr_applied=d.ocr_applied,
                        metadata=dict(d.metadata_json or {}),
                        created_at=d.created_at,
                        updated_at=d.updated_at,
                    )
                )
            return summaries

    def get_document(
        self,
        knowledge_base_id: str,
        document_id: str,
    ) -> DocumentSummary | None:
        """Fetch a single document summary."""
        with self.session_factory() as session:
            doc = session.get(Document, document_id)
            if doc is None or doc.knowledge_base_id != knowledge_base_id:
                return None
            return DocumentSummary(
                id=doc.id,
                knowledge_base_id=doc.knowledge_base_id,
                filename=doc.filename,
                file_type=doc.file_type,
                file_size_bytes=doc.file_size_bytes,
                sha256_hash=doc.sha256_hash,
                status=doc.status,
                chunk_count=doc.chunk_count,
                error_message=doc.error_message,
                ocr_applied=doc.ocr_applied,
                metadata=dict(doc.metadata_json or {}),
                created_at=doc.created_at,
                updated_at=doc.updated_at,
            )

    # ── Reindexing & Search ───────────────────────────────────────────────────

    async def trigger_reindex(
        self,
        knowledge_base_id: str,
        new_embedding_config_id: str | None = None,
    ) -> str:
        """Submit a background reindex job using shadow collections."""
        with self.session_factory() as session:
            kb = session.get(KnowledgeBase, knowledge_base_id)
            if kb is None:
                raise KeyError(f"KnowledgeBase not found: {knowledge_base_id}")

            target_config_id = new_embedding_config_id or kb.embedding_config_id
            if not target_config_id:
                raise ValueError("Knowledge base has no embedding configuration configured.")

            target_config = session.get(EmbeddingConfiguration, target_config_id)
            if target_config is None:
                raise KeyError(f"EmbeddingConfiguration not found: {target_config_id}")

        job_id = await submit_knowledge_base_reindex_job(
            knowledge_base_id=knowledge_base_id,
            new_embedding_config_id=target_config_id,
            manager=self.job_manager,
        )
        logger.info(
            "Submitted reindex job %s for knowledge base %s with embedding config %s",
            job_id,
            knowledge_base_id,
            target_config_id,
        )
        return job_id

    def validate_query_dimensions(self, knowledge_base_id: str, query_dimensions: int) -> None:
        """Validate query vector dimensionality against the locked IndexMetadata.

        Raises EmbeddingDimensionMismatchError immediately before hitting the vector store.
        """
        with self.session_factory() as session:
            index_meta = session.scalar(
                select(IndexMetadata).where(IndexMetadata.knowledge_base_id == knowledge_base_id)
            )
            if index_meta is None:
                raise KeyError(f"IndexMetadata not found for knowledge base: {knowledge_base_id}")

            if query_dimensions != index_meta.embedding_dimensions:
                raise EmbeddingDimensionMismatchError(
                    expected=index_meta.embedding_dimensions,
                    actual=query_dimensions,
                    provider=index_meta.embedding_provider,
                    model=index_meta.embedding_model_name,
                )

    async def search(
        self,
        knowledge_base_id: str,
        query_vector: Sequence[float],
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search nearest vectors in the knowledge base collection after dimension verification."""
        with self.session_factory() as session:
            index_meta = session.scalar(
                select(IndexMetadata).where(IndexMetadata.knowledge_base_id == knowledge_base_id)
            )
            if index_meta is None:
                raise KeyError(f"IndexMetadata not found for knowledge base: {knowledge_base_id}")

            # Eagerly load values before session closes
            expected_dims = index_meta.embedding_dimensions
            provider = index_meta.embedding_provider
            model_name = index_meta.embedding_model_name
            coll_name = index_meta.vector_collection_name

        query_dims = len(query_vector)
        if query_dims != expected_dims:
            raise EmbeddingDimensionMismatchError(
                expected=expected_dims,
                actual=query_dims,
                provider=provider,
                model=model_name,
            )

        return await self.vector_store.search_dense(
            collection_name=coll_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

    async def upsert_chunks(
        self,
        knowledge_base_id: str,
        chunks: Sequence[DocumentChunkPayload],
        embeddings: Sequence[Sequence[float]],
    ) -> list[str]:
        """Convenience method to index chunks into the dedicated collection for this KB."""
        with self.session_factory() as session:
            index_meta = session.scalar(
                select(IndexMetadata).where(IndexMetadata.knowledge_base_id == knowledge_base_id)
            )
            if index_meta is None:
                raise KeyError(f"IndexMetadata not found for knowledge base: {knowledge_base_id}")
            expected_dims = index_meta.embedding_dimensions
            coll_name = index_meta.vector_collection_name

        if embeddings:
            vector_dims = len(embeddings[0])
            if vector_dims != expected_dims:
                raise EmbeddingDimensionMismatchError(
                    expected=expected_dims,
                    actual=vector_dims,
                    provider=index_meta.embedding_provider,
                    model=index_meta.embedding_model_name,
                )

        return await self.vector_store.upsert_chunks(
            collection_name=coll_name,
            chunks=chunks,
            embeddings=embeddings,
            metadata={"embedding_dimensions": expected_dims},
        )


# ── Global Singleton Accessors ────────────────────────────────────────────────

_global_kb_service_lock = threading.RLock()
_global_kb_service: KnowledgeBaseService | None = None


def get_knowledge_base_service(
    session_factory: sessionmaker[Session] | None = None,
    vector_store: LocalVectorStore | None = None,
    file_store: LocalFileStore | None = None,
    job_manager: LocalJobManager | None = None,
) -> KnowledgeBaseService:
    """Return the thread-safe global KnowledgeBaseService singleton."""
    global _global_kb_service
    if _global_kb_service is None:
        with _global_kb_service_lock:
            if _global_kb_service is None:
                _global_kb_service = KnowledgeBaseService(
                    session_factory=session_factory,
                    vector_store=vector_store,
                    file_store=file_store,
                    job_manager=job_manager,
                )
    return _global_kb_service


def reset_knowledge_base_service() -> None:
    """Reset the global singleton (used in test teardown)."""
    global _global_kb_service
    with _global_kb_service_lock:
        _global_kb_service = None


__all__ = [
    "DocumentSummary",
    "DuplicateDocumentError",
    "KnowledgeBaseDetails",
    "KnowledgeBaseService",
    "KnowledgeBaseSummary",
    "get_knowledge_base_service",
    "reset_knowledge_base_service",
]
