"""Persistent local document-ingestion job definition."""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.types import DocumentChunkPayload
from app.jobs.manager import JobContext, JobStage, LocalJobManager, get_local_job_manager
from app.providers.interfaces.embedding import EmbeddingProvider
from app.storage.models import Document, DocumentChunk, KnowledgeBase

EmbeddingResolver = Callable[[str], EmbeddingProvider | Awaitable[EmbeddingProvider]]
FileLoader = Callable[[Document], bytes | Awaitable[bytes]]


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class _DenseIngestionAdapter:
    def __init__(
        self,
        vector_store: Any,
        collection_name: str,
        dimensions: int,
    ) -> None:
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.dimensions = dimensions

    async def index_dense(
        self,
        chunks: Sequence[DocumentChunkPayload],
        embeddings: Sequence[list[float]],
    ) -> list[str]:
        return await self.vector_store.upsert_chunks(
            self.collection_name,
            chunks,
            embeddings,
            metadata={"embedding_dimensions": self.dimensions},
        )


class _SparseIngestionAdapter:
    def __init__(self, sparse_store: Any, knowledge_base_id: str) -> None:
        self.sparse_store = sparse_store
        self.knowledge_base_id = knowledge_base_id

    async def index_sparse(self, chunks: Sequence[DocumentChunkPayload]) -> None:
        await self.sparse_store.index_chunks(self.knowledge_base_id, chunks)


class DocumentIngestionJobHandler:
    JOB_TYPE = "DOCUMENT_INGESTION"

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        embedding_provider_resolver: EmbeddingResolver,
        vector_store: Any,
        sparse_store: Any,
        data_dir: Path | str | None = None,
        pipeline_factory: Callable[..., IngestionPipeline] | None = None,
        file_loader: FileLoader | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.embedding_provider_resolver = embedding_provider_resolver
        self.vector_store = vector_store
        self.sparse_store = sparse_store
        self.data_dir = Path(data_dir or settings.QUANTIVA_DATA_DIR).resolve()
        self.pipeline_factory = pipeline_factory or IngestionPipeline
        self.file_loader = file_loader or self._load_file

    def _load_file(self, document: Document) -> bytes:
        path = (self.data_dir / document.original_rel_path).resolve()
        try:
            path.relative_to(self.data_dir)
        except ValueError as exc:
            raise ValueError("Document path escapes QUANTIVA_DATA_DIR") from exc
        return path.read_bytes()

    async def _load_document(self, document_id: str) -> dict[str, Any]:
        def load() -> dict[str, Any]:
            with self.session_factory() as session:
                document = session.get(Document, document_id)
                if document is None:
                    raise KeyError(f"Document not found: {document_id}")
                values: dict[str, Any] = {
                    "id": document.id,
                    "knowledge_base_id": document.knowledge_base_id,
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "original_rel_path": document.original_rel_path,
                }
                session.expunge(document)
                values["document"] = document
                return values

        return await asyncio.to_thread(load)

    async def _set_processing(self, document_id: str) -> None:
        def persist() -> None:
            with self.session_factory() as session:
                document = session.get(Document, document_id)
                if document is None:
                    raise KeyError(document_id)
                document.status = "PROCESSING"
                document.error_message = None
                session.commit()

        await asyncio.to_thread(persist)

    async def _persist_result(self, result) -> None:
        def persist() -> None:
            with self.session_factory() as session:
                document = session.get(Document, result.document_id)
                if document is None:
                    raise KeyError(result.document_id)
                session.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.document_id == result.document_id
                    )
                )
                for payload in result.chunks:
                    session.add(
                        DocumentChunk(
                            id=str(payload["chunk_id"]),
                            document_id=str(payload["document_id"]),
                            knowledge_base_id=str(payload["knowledge_base_id"]),
                            chunk_index=int(payload["chunk_index"]),
                            page_number=payload.get("page_number"),
                            section_title=payload.get("section_title"),
                            sheet_name=payload.get("sheet_name"),
                            is_table=bool(payload["is_table"]),
                            content=str(payload["content"]),
                            content_hash=str(payload["content_hash"]),
                            char_count=int(payload["char_count"]),
                            metadata_json={},
                        )
                    )
                document.page_count = result.metadata.get("page_count")
                document.chunk_count = len(result.chunks)
                document.ocr_applied = bool(result.metadata.get("ocr"))
                document.metadata_json = dict(result.metadata)
                document.status = "READY"
                document.error_message = None
                knowledge_base = session.get(KnowledgeBase, result.knowledge_base_id)
                if knowledge_base is not None:
                    knowledge_base.document_count = int(
                        session.scalar(
                            select(func.count())
                            .select_from(Document)
                            .where(
                                Document.knowledge_base_id == result.knowledge_base_id,
                                Document.status == "READY",
                            )
                        )
                        or 0
                    )
                    knowledge_base.chunk_count = int(
                        session.scalar(
                            select(func.count())
                            .select_from(DocumentChunk)
                            .where(
                                DocumentChunk.knowledge_base_id
                                == result.knowledge_base_id
                            )
                        )
                        or 0
                    )
                session.commit()

        await asyncio.to_thread(persist)

    async def _cleanup_partial(
        self,
        document_id: str,
        knowledge_base_id: str,
        collection_name: str,
    ) -> None:
        await self.vector_store.delete_by_document_id(
            collection_name,
            document_id,
            filters={"knowledge_base_id": knowledge_base_id},
        )
        await self.sparse_store.delete_by_document_id(
            knowledge_base_id,
            document_id,
        )

        def purge() -> None:
            with self.session_factory() as session:
                session.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.document_id == document_id
                    )
                )
                session.commit()

        await asyncio.to_thread(purge)

    async def __call__(
        self,
        context: JobContext,
        payload: Mapping[str, Any],
    ) -> None:
        document_id = str(payload["document_id"])
        knowledge_base_id = str(payload["knowledge_base_id"])
        await context.report(JobStage.VALIDATING, 1.0, "Loading document")
        values = await self._load_document(document_id)
        if values["knowledge_base_id"] != knowledge_base_id:
            raise ValueError("Document does not belong to the requested knowledge base")
        await self._set_processing(document_id)
        provider = await _maybe_await(
            self.embedding_provider_resolver(knowledge_base_id)
        )
        dimensions = int(provider.get_dimensions())
        collection_name = self.vector_store.collection_name(knowledge_base_id)

        async def cleanup() -> None:
            await self._cleanup_partial(
                document_id,
                knowledge_base_id,
                collection_name,
            )

        context.add_cleanup(cleanup)
        detached_document = values["document"]
        if inspect.iscoroutinefunction(self.file_loader):
            file_bytes = await self.file_loader(detached_document)
        else:
            file_bytes = await context.run_cpu(self.file_loader, detached_document)
        dense_adapter = _DenseIngestionAdapter(
            self.vector_store,
            collection_name,
            dimensions,
        )
        sparse_adapter = _SparseIngestionAdapter(
            self.sparse_store,
            knowledge_base_id,
        )
        pipeline = self.pipeline_factory(
            provider,
            dense_indexer=dense_adapter,
            sparse_indexer=sparse_adapter,
        )
        stage_map = {
            "validate": JobStage.VALIDATING,
            "extract": JobStage.EXTRACTING,
            "ocr_if_needed": JobStage.OCR,
            "clean": JobStage.EXTRACTING,
            "chunk": JobStage.CHUNKING,
            "embed": JobStage.EMBEDDING,
            "index_dense": JobStage.INDEXING_VECTOR,
            "index_sparse": JobStage.INDEXING_SPARSE,
        }

        def progress_callback(stage: str, percent: float, message: str) -> None:
            context.raise_if_cancelled()
            mapped = stage_map.get(stage, JobStage.EXTRACTING)
            context.report_nowait(mapped, percent, message)

        result = await pipeline.run(
            file_bytes,
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            file_name=str(values["filename"]),
            mime_type=str(values["file_type"]),
            progress_callback=progress_callback,
            allow_duplicates=True,
        )
        await context.flush_reports()
        await context.report(JobStage.FINALIZING, 99.0, "Saving document chunks")
        await self._persist_result(result)
        context.remove_cleanup(cleanup)


async def submit_document_ingestion_job(
    document_id: str,
    knowledge_base_id: str,
    *,
    manager: LocalJobManager | None = None,
) -> str:
    manager = manager or get_local_job_manager()
    return await manager.submit(
        DocumentIngestionJobHandler.JOB_TYPE,
        payload={
            "document_id": str(document_id),
            "knowledge_base_id": str(knowledge_base_id),
        },
        knowledge_base_id=str(knowledge_base_id),
        document_id=str(document_id),
        stage=JobStage.VALIDATING,
    )


__all__ = ["DocumentIngestionJobHandler", "submit_document_ingestion_job"]
