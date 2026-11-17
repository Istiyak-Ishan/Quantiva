"""Task-owned async resources: no pooled connections survive asyncio.run()."""
import asyncio
import logging
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.config import settings
from app.core.metrics import DOCS_INGESTED
from app.workers.celery_app import celery_app
import app.models  # noqa: F401 - ensure all ORM models (e.g. User) are registered in Base.metadata
from app.models.document import Document
from app.models.chunk import Chunk
from app.core.storage import get_file
from app.services.ingestion.ingestion_service import process_document as process_file
from app.services.retrieval.embedding_service import embedding_service
from app.services.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

_METRIC_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/csv": "csv",
    "application/csv": "csv",
    "text/plain": "txt",
    "text/markdown": "md",
}


def _metric_file_type(mime: str) -> str:
    return _METRIC_TYPES.get(mime, "other")


async def _process_document_async(doc_id: str):
    document_id = uuid.UUID(doc_id)
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = VectorStore()
    try:
        async with sessions() as session:
            # Held through ingestion: competing tasks and deletes cannot interleave.
            doc = await session.scalar(select(Document).where(Document.id == document_id).with_for_update())
            if doc is None or doc.status in {"ready", "deleting"}:
                return
            try:
                file_bytes = await asyncio.to_thread(get_file, doc.storage_path)
                chunks = await asyncio.to_thread(process_file, str(doc.id), file_bytes, doc.file_type)
                if not chunks:
                    raise ValueError("No readable content was extracted")
                embeddings = []
                for start in range(0, len(chunks), 100):
                    embeddings.extend(await asyncio.to_thread(embedding_service.embed_batch, [c["text"] for c in chunks[start:start + 100]]))
                await store.create_collection()
                await store.delete_document(str(doc.id), user_id=str(doc.user_id))
                qdrant_ids = await store.upsert_chunks(chunks, embeddings, str(doc.id), str(doc.user_id))
                # Transactionally replace derived chunk rows on an explicit retry.
                # Raw documents and existing collections are never deleted here.
                await session.execute(delete(Chunk).where(Chunk.doc_id == document_id))
                session.add_all([Chunk(doc_id=doc.id, chunk_index=i, text_preview=chunk["text"][:200], full_text=chunk["text"], page_number=chunk["metadata"].get("page"), chunk_type=chunk["metadata"].get("chunk_type", "text"), qdrant_id=uuid.UUID(qid)) for i, (chunk, qid) in enumerate(zip(chunks, qdrant_ids))])
                doc.status = "ready"
                doc.chunk_count = len(chunks)
                doc.failure_reason = None
                doc.processing_started_at = None
                doc.metadata_json = {**(doc.metadata_json or {}), "embedding_model": settings.EMBEDDING_MODEL, "embedding_dimensions": settings.EMBEDDING_DIMENSIONS, "embedding_collection": store.collection_name}
                await session.commit()
                # Worker-process counter; exposed only with Prometheus multiprocess
                # aggregation, never inferred by the API process.
                DOCS_INGESTED.labels(file_type=_metric_file_type(doc.file_type)).inc()
            except Exception:
                logger.exception("Document ingestion failed for %s", document_id)
                await session.rollback()
                await session.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(
                        status="failed",
                        failure_reason="Processing failed. Retry indexing.",
                        processing_started_at=None,
                    )
                )
                await session.commit()
                # Celery records failure too; never report a successful task.
                raise
    finally:
        await store.client.close()
        await engine.dispose()


@celery_app.task(name="process_document", acks_late=True, reject_on_worker_lost=True)
def process_document(doc_id: str):
    asyncio.run(_process_document_async(doc_id))
