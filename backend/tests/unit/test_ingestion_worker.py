"""Real PDF parsing + SQL + local Qdrant; Google and S3 are explicit seams.

The sync test runs Celery tasks through separate asyncio.run invocations to
exercise task resource ownership without touching the configured application DB.
"""
import asyncio
import uuid
from unittest.mock import patch

import pymupdf
import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.config import settings
from app.db.database import Base
from app.models.user import User
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.llm.errors import EmbeddingError
from app.services.retrieval.vector_store import VectorStore
from app.workers.ingestion_worker import process_document


def pdf_bytes():
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Quantiva regression document. Quarterly revenue is 42 million dollars.")
    data = pdf.tobytes()
    pdf.close()
    return data


def local_store():
    store = VectorStore.__new__(VectorStore)
    store.client = AsyncQdrantClient(location=":memory:")
    store.collection_name = "worker_test"
    store.vector_size = settings.EMBEDDING_DIMENSIONS
    return store


async def seed(url, ids):
    engine = create_async_engine(url)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            user = User(email="worker@example.com", hashed_password="not-used")
            db.add(user)
            await db.flush()
            db.add_all([Document(id=doc_id, user_id=user.id, filename="real.pdf", file_type="application/pdf", file_size=100, storage_path="pdf", status="processing") for doc_id in ids])
            await db.commit()
    finally:
        await engine.dispose()


async def read_results(url):
    engine = create_async_engine(url)
    try:
        async with async_sessionmaker(engine)() as db:
            docs = list((await db.scalars(select(Document))).all())
            chunks = list((await db.scalars(select(Chunk))).all())
            return [d.status for d in docs], [c.text for c in chunks]
    finally:
        await engine.dispose()


def test_two_sequential_celery_tasks_and_retry_use_fresh_resources(tmp_path, monkeypatch):
    url = f"sqlite+aiosqlite:///{(tmp_path / 'worker.db').as_posix()}"
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    ids = [uuid.uuid4(), uuid.uuid4()]
    asyncio.run(seed(url, ids))
    with patch("app.workers.ingestion_worker.VectorStore", side_effect=local_store), patch("app.workers.ingestion_worker.get_file", return_value=pdf_bytes()), patch("app.workers.ingestion_worker.embedding_service.embed_batch", side_effect=lambda texts: [[1.] + [0.] * (settings.EMBEDDING_DIMENSIONS - 1) for _ in texts]):
        process_document.run(str(ids[0]))
        process_document.run(str(ids[1]))
        process_document.run(str(ids[0]))  # ready document is idempotently skipped
    statuses, texts = asyncio.run(read_results(url))
    assert statuses == ["ready", "ready"]
    assert len(texts) == 2
    assert all("42 million" in text for text in texts)


def test_embedding_failure_marks_document_failed_not_ready(tmp_path, monkeypatch):
    url = f"sqlite+aiosqlite:///{(tmp_path / 'failure.db').as_posix()}"
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    doc_id = uuid.uuid4()
    asyncio.run(seed(url, [doc_id]))
    with patch("app.workers.ingestion_worker.VectorStore", side_effect=local_store), patch("app.workers.ingestion_worker.get_file", return_value=pdf_bytes()), patch("app.workers.ingestion_worker.embedding_service.embed_batch", side_effect=EmbeddingError()):
        with pytest.raises(EmbeddingError):
            process_document.run(str(doc_id))
    statuses, texts = asyncio.run(read_results(url))
    assert statuses == ["failed"]
    assert texts == []
