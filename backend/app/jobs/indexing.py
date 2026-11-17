"""Safe shadow-collection knowledge-base reindex job."""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.ingestion.types import DocumentChunkPayload
from app.jobs.manager import JobContext, JobStage, LocalJobManager, get_local_job_manager
from app.providers.interfaces.embedding import EmbeddingProvider
from app.storage.models import (
    Document,
    DocumentChunk,
    EmbeddingConfiguration,
    IndexMetadata,
    KnowledgeBase,
    ProcessingJob,
)

logger = logging.getLogger(__name__)
EmbeddingConfigResolver = Callable[
    [EmbeddingConfiguration],
    EmbeddingProvider | Awaitable[EmbeddingProvider],
]


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class KnowledgeBaseReindexJobHandler:
    JOB_TYPE = "KNOWLEDGE_BASE_REINDEX"

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        embedding_provider_resolver: EmbeddingConfigResolver,
        vector_store: Any,
    ) -> None:
        self.session_factory = session_factory
        self.embedding_provider_resolver = embedding_provider_resolver
        self.vector_store = vector_store

    async def _load_inputs(
        self,
        knowledge_base_id: str,
        embedding_config_id: str,
    ) -> tuple[
        EmbeddingConfiguration,
        str | None,
        list[DocumentChunkPayload],
    ]:
        def load() -> tuple[
            EmbeddingConfiguration,
            str | None,
            list[DocumentChunkPayload],
        ]:
            with self.session_factory() as session:
                knowledge_base = session.get(KnowledgeBase, knowledge_base_id)
                if knowledge_base is None:
                    raise KeyError(f"Knowledge base not found: {knowledge_base_id}")
                config = session.get(EmbeddingConfiguration, embedding_config_id)
                if config is None:
                    raise KeyError(
                        f"Embedding configuration not found: {embedding_config_id}"
                    )
                current_index = session.scalar(
                    select(IndexMetadata).where(
                        IndexMetadata.knowledge_base_id == knowledge_base_id
                    )
                )
                old_collection = (
                    current_index.vector_collection_name
                    if current_index is not None
                    else None
                )
                rows = session.execute(
                    select(DocumentChunk, Document.filename)
                    .join(Document, Document.id == DocumentChunk.document_id)
                    .where(DocumentChunk.knowledge_base_id == knowledge_base_id)
                    .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
                ).all()
                chunks: list[DocumentChunkPayload] = []
                for chunk, filename in rows:
                    chunks.append(
                        {
                            "chunk_id": chunk.id,
                            "document_id": chunk.document_id,
                            "knowledge_base_id": chunk.knowledge_base_id,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "source_file": filename,
                            "section_title": chunk.section_title,
                            "sheet_name": chunk.sheet_name,
                            "is_table": chunk.is_table,
                            "content_hash": chunk.content_hash,
                            "char_count": chunk.char_count,
                        }
                    )
                session.expunge(config)
                return config, old_collection, chunks

        return await asyncio.to_thread(load)

    async def _promote_shadow(
        self,
        context: JobContext,
        knowledge_base_id: str,
        embedding_config_id: str,
        config: EmbeddingConfiguration,
        provider: EmbeddingProvider,
        shadow_collection: str,
        vector_count: int,
    ) -> None:
        def persist() -> None:
            with self.session_factory() as session:
                knowledge_base = session.get(KnowledgeBase, knowledge_base_id)
                if knowledge_base is None:
                    raise KeyError(knowledge_base_id)
                metadata = session.scalar(
                    select(IndexMetadata).where(
                        IndexMetadata.knowledge_base_id == knowledge_base_id
                    )
                )
                if metadata is None:
                    metadata = IndexMetadata(
                        knowledge_base_id=knowledge_base_id,
                        vector_collection_name=shadow_collection,
                        embedding_provider=provider.provider_name(),
                        embedding_model_name=provider.model_name(),
                        embedding_dimensions=provider.get_dimensions(),
                    )
                    session.add(metadata)
                metadata.vector_collection_name = shadow_collection
                metadata.embedding_provider = provider.provider_name()
                metadata.embedding_model_name = provider.model_name()
                metadata.embedding_dimensions = provider.get_dimensions()
                metadata.vector_count = vector_count
                metadata.needs_reindex = False
                metadata.last_indexed_at = datetime.now(timezone.utc)
                knowledge_base.embedding_config_id = embedding_config_id
                knowledge_base.chunk_count = vector_count

                # Promotion and its crash-recovery marker share one SQLite
                # transaction. Recovery will never delete an active shadow.
                job = session.get(ProcessingJob, context.job_id)
                if job is None:
                    raise KeyError(context.job_id)
                job.payload_json = {
                    **dict(job.payload_json or {}),
                    "shadow_collection": shadow_collection,
                    "shadow_promoted": True,
                    "embedding_dimensions": config.dimensions,
                }
                session.commit()

        await asyncio.to_thread(persist)

    async def __call__(
        self,
        context: JobContext,
        payload: Mapping[str, Any],
    ) -> None:
        knowledge_base_id = str(payload["knowledge_base_id"])
        embedding_config_id = str(payload["new_embedding_config_id"])
        await context.report(JobStage.VALIDATING, 1.0, "Loading chunks for reindex")
        config, old_collection, chunks = await self._load_inputs(
            knowledge_base_id,
            embedding_config_id,
        )
        provider = await _maybe_await(self.embedding_provider_resolver(config))
        dimensions = int(provider.get_dimensions())
        if dimensions != int(config.dimensions):
            raise ValueError(
                "Embedding provider dimensions do not match the selected configuration"
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        base_collection = self.vector_store.collection_name(knowledge_base_id)
        shadow_collection = f"{base_collection}_reindex_{timestamp}"
        await context.update_payload(
            shadow_collection=shadow_collection,
            shadow_promoted=False,
        )
        await self.vector_store.ensure_collection(shadow_collection, dimensions)
        promoted = False

        async def remove_shadow() -> None:
            if not promoted:
                await self.vector_store.delete_collection(shadow_collection)

        context.add_cleanup(remove_shadow)
        try:
            total = len(chunks)
            batch_size = max(1, int(config.batch_size or 32))
            for start in range(0, total, batch_size):
                await context.checkpoint()
                batch = chunks[start : start + batch_size]
                await context.report(
                    JobStage.EMBEDDING,
                    5.0 + 70.0 * (start / max(1, total)),
                    f"Embedding chunks {start + 1}-{start + len(batch)} of {total}",
                )
                embeddings = await provider.embed_documents(
                    [chunk["content"] for chunk in batch],
                    batch_size=len(batch),
                )
                if len(embeddings) != len(batch):
                    raise ValueError("Embedding provider returned an incomplete batch")
                for embedding in embeddings:
                    provider.validate_dimensions(embedding)
                await context.checkpoint()
                await context.report(
                    JobStage.INDEXING_VECTOR,
                    75.0 + 20.0 * ((start + len(batch)) / max(1, total)),
                    f"Indexing {start + len(batch)}/{total} vectors",
                )
                await self.vector_store.upsert_chunks(
                    shadow_collection,
                    batch,
                    embeddings,
                    metadata={"embedding_dimensions": dimensions},
                )

            await context.checkpoint()
            await context.report(
                JobStage.FINALIZING,
                98.0,
                "Promoting completed shadow index",
            )
            await self._promote_shadow(
                context,
                knowledge_base_id,
                embedding_config_id,
                config,
                provider,
                shadow_collection,
                total,
            )
            promoted = True
            context.remove_cleanup(remove_shadow)
        except BaseException:
            if not promoted:
                await self.vector_store.delete_collection(shadow_collection)
                context.remove_cleanup(remove_shadow)
            raise

        # The active pointer now targets a complete collection. Failure to
        # remove the obsolete collection wastes disk only; it must not roll back
        # or mark the successfully promoted index as failed.
        if old_collection and old_collection != shadow_collection:
            try:
                await self.vector_store.delete_collection(old_collection)
            except Exception:
                logger.warning(
                    "Unable to remove old vector collection %s after promotion",
                    old_collection,
                    exc_info=True,
                )


async def submit_knowledge_base_reindex_job(
    knowledge_base_id: str,
    new_embedding_config_id: str,
    *,
    manager: LocalJobManager | None = None,
) -> str:
    manager = manager or get_local_job_manager()
    return await manager.submit(
        KnowledgeBaseReindexJobHandler.JOB_TYPE,
        payload={
            "knowledge_base_id": str(knowledge_base_id),
            "new_embedding_config_id": str(new_embedding_config_id),
        },
        knowledge_base_id=str(knowledge_base_id),
        stage=JobStage.VALIDATING,
    )


__all__ = [
    "KnowledgeBaseReindexJobHandler",
    "submit_knowledge_base_reindex_job",
]
