"""Lifecycle, recovery, cancellation and safe-reindex tests."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.jobs.indexing import (
    KnowledgeBaseReindexJobHandler,
    submit_knowledge_base_reindex_job,
)
from app.jobs.ingestion import (
    DocumentIngestionJobHandler,
    submit_document_ingestion_job,
)
from app.jobs.manager import JobStage, JobStatus, LocalJobManager
from app.ingestion.types import IngestionResult
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.types import ConnectionTestResult
from app.storage.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from app.storage.models import (
    Document,
    DocumentChunk,
    EmbeddingConfiguration,
    IndexMetadata,
    KnowledgeBase,
    ProcessingJob,
)


@pytest.fixture
def job_database(tmp_path: Path):
    engine = create_sqlite_engine(tmp_path / "job-data")
    initialize_database(engine)
    factory = create_session_factory(engine)
    try:
        yield engine, factory
    finally:
        engine.dispose()


def create_document(factory, *, status: str = "PENDING") -> tuple[str, str]:
    with factory() as session:
        knowledge_base = KnowledgeBase(name=f"KB-{id(factory)}-{status}")
        session.add(knowledge_base)
        session.flush()
        document = Document(
            knowledge_base_id=knowledge_base.id,
            filename="job.txt",
            original_rel_path="documents/original/job.txt",
            file_type="txt",
            file_size_bytes=10,
            sha256_hash=(knowledge_base.id.replace("-", "") * 2)[:64],
            status=status,
        )
        session.add(document)
        session.commit()
        return knowledge_base.id, document.id


async def test_full_async_job_progression_pending_running_completed(job_database):
    _engine, factory = job_database
    manager = LocalJobManager(factory, worker_count=2)

    async def handler(context, payload):
        await context.report(JobStage.VALIDATING, 5, "Validated")
        cpu_result = await context.run_cpu(lambda value: value * 2, payload["value"])
        assert cpu_result == 42
        await context.report(JobStage.EXTRACTING, 25, "Extracted")
        await context.report(JobStage.CHUNKING, 50, "Chunked")
        await context.report(JobStage.EMBEDDING, 75, "Embedded")
        await context.report(JobStage.INDEXING_VECTOR, 90, "Indexed")

    manager.register_handler("TEST_INGESTION", handler)
    await manager.start()
    try:
        job_id = await manager.submit("TEST_INGESTION", payload={"value": 21})
        job = await manager.wait_for_job(job_id, timeout=5)

        assert job.status == JobStatus.COMPLETED.value
        assert job.stage == JobStage.FINALIZING.value
        assert job.progress_percent == 100
        history = manager.history(job_id)
        assert history[0].status == JobStatus.PENDING.value
        assert JobStatus.RUNNING.value in {event.status for event in history}
        assert history[-1].status == JobStatus.COMPLETED.value
        assert {
            JobStage.VALIDATING.value,
            JobStage.EXTRACTING.value,
            JobStage.CHUNKING.value,
            JobStage.EMBEDDING.value,
            JobStage.INDEXING_VECTOR.value,
            JobStage.FINALIZING.value,
        }.issubset({event.stage for event in history})
    finally:
        await manager.stop()


class SuccessfulEmbeddingProvider(EmbeddingProvider):
    def provider_name(self) -> str:
        return "successful-mock"

    def model_name(self) -> str:
        return "embedding-4d"

    def get_dimensions(self) -> int:
        return 4

    async def embed_documents(self, texts, batch_size=32):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, query):
        return [1.0, 0.0, 0.0, 0.0]

    async def test_embedding(self):
        return ConnectionTestResult(
            success=True,
            latency_ms=0,
            provider=self.provider_name(),
            model=self.model_name(),
            detected_dimensions=4,
        )


class FakeIngestionVectorStore:
    def __init__(self) -> None:
        self.indexed: list[str] = []

    @staticmethod
    def collection_name(knowledge_base_id: str) -> str:
        return f"kb_{knowledge_base_id}"

    async def upsert_chunks(self, collection, chunks, embeddings, **kwargs):
        self.indexed.extend(chunk["chunk_id"] for chunk in chunks)
        return list(self.indexed)

    async def delete_by_document_id(self, *args, **kwargs):
        self.indexed.clear()


class FakeIngestionSparseStore:
    def __init__(self) -> None:
        self.indexed: list[str] = []

    async def index_chunks(self, knowledge_base_id, chunks, **kwargs):
        self.indexed.extend(chunk["chunk_id"] for chunk in chunks)

    async def delete_by_document_id(self, *args, **kwargs):
        self.indexed.clear()


class FakeSuccessfulPipeline:
    def __init__(self, provider, *, dense_indexer, sparse_indexer):
        self.provider = provider
        self.dense_indexer = dense_indexer
        self.sparse_indexer = sparse_indexer

    async def run(
        self,
        file_bytes,
        *,
        document_id,
        knowledge_base_id,
        file_name,
        progress_callback,
        **kwargs,
    ):
        payload = {
            "chunk_id": "ingested-chunk",
            "document_id": document_id,
            "knowledge_base_id": knowledge_base_id,
            "page_number": 1,
            "chunk_index": 0,
            "content": file_bytes.decode(),
            "source_file": file_name,
            "section_title": None,
            "sheet_name": None,
            "is_table": False,
            "content_hash": "9" * 64,
            "char_count": len(file_bytes),
        }
        stages = [
            ("validate", 5),
            ("extract", 20),
            ("ocr_if_needed", 35),
            ("chunk", 60),
            ("embed", 85),
        ]
        for stage, percent in stages:
            progress_callback(stage, percent, stage)
        embeddings = await self.provider.embed_documents([payload["content"]])
        progress_callback("index_dense", 90, "vectors")
        point_ids = await self.dense_indexer.index_dense([payload], embeddings)
        progress_callback("index_sparse", 97, "keywords")
        await self.sparse_indexer.index_sparse([payload])
        return IngestionResult(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            file_name=file_name,
            file_type="txt",
            file_hash="8" * 64,
            chunks=[payload],
            embeddings=embeddings,
            dense_point_ids=point_ids,
            sparse_indexed=True,
            metadata={"page_count": 1, "ocr": False},
        )


async def test_document_ingestion_handler_persists_completed_chunks(job_database):
    _engine, factory = job_database
    kb_id, document_id = create_document(factory, status="PENDING")
    vectors = FakeIngestionVectorStore()
    sparse = FakeIngestionSparseStore()
    handler = DocumentIngestionJobHandler(
        factory,
        embedding_provider_resolver=lambda _kb_id: SuccessfulEmbeddingProvider(),
        vector_store=vectors,
        sparse_store=sparse,
        pipeline_factory=FakeSuccessfulPipeline,
        file_loader=lambda _document: b"local ingestion content",
    )
    manager = LocalJobManager(
        factory,
        worker_count=1,
        vector_store=vectors,
        sparse_store=sparse,
    )
    manager.register_handler(handler.JOB_TYPE, handler)
    await manager.start()
    try:
        job_id = await submit_document_ingestion_job(
            document_id,
            kb_id,
            manager=manager,
        )
        job = await manager.wait_for_job(job_id, timeout=5)
        with factory() as session:
            document = session.get(Document, document_id)
            chunks = list(
                session.scalars(
                    select(DocumentChunk).where(
                        DocumentChunk.document_id == document_id
                    )
                )
            )

        assert job.status == JobStatus.COMPLETED.value
        assert document.status == "READY"
        assert document.chunk_count == 1
        assert [chunk.id for chunk in chunks] == ["ingested-chunk"]
        assert vectors.indexed == ["ingested-chunk"]
        assert sparse.indexed == ["ingested-chunk"]
    finally:
        await manager.stop()


class FakeCleanupStore:
    def __init__(self) -> None:
        self.document_deletions: list[tuple[str, str]] = []
        self.collection_deletions: list[str] = []

    async def delete_by_document_id(self, collection, document_id, **kwargs):
        self.document_deletions.append((collection, document_id))

    async def delete_collection(self, collection):
        self.collection_deletions.append(collection)


class FakeSparseCleanupStore:
    def __init__(self) -> None:
        self.deletions: list[tuple[str, str]] = []

    async def delete_by_document_id(self, knowledge_base_id, document_id, **kwargs):
        self.deletions.append((knowledge_base_id, document_id))


async def test_startup_recovery_marks_interrupted_and_purges_partial_data(job_database):
    _engine, factory = job_database
    kb_id, document_id = create_document(factory, status="PROCESSING")
    with factory() as session:
        session.add(
            DocumentChunk(
                document_id=document_id,
                knowledge_base_id=kb_id,
                chunk_index=0,
                content="partial chunk",
                content_hash="a" * 64,
                char_count=13,
            )
        )
        job = ProcessingJob(
            job_type="DOCUMENT_INGESTION",
            knowledge_base_id=kb_id,
            document_id=document_id,
            status="RUNNING",
            stage="EMBEDDING",
            progress_percent=70,
            payload_json={"document_id": document_id},
        )
        session.add(job)
        session.commit()
        job_id = job.id

    vectors = FakeCleanupStore()
    sparse = FakeSparseCleanupStore()
    manager = LocalJobManager(
        factory,
        vector_store=vectors,
        sparse_store=sparse,
    )
    await manager.start()
    try:
        recovered = await manager.get_job(job_id)
        with factory() as session:
            document = session.get(Document, document_id)
            chunk_count = session.scalar(
                select(func.count()).select_from(DocumentChunk)
            )

        assert recovered.status == JobStatus.INTERRUPTED.value
        assert document.status == "INTERRUPTED_NEEDS_RETRY"
        assert chunk_count == 0
        assert vectors.document_deletions == [(f"kb_{kb_id}", document_id)]
        assert sparse.deletions == [(kb_id, document_id)]
    finally:
        await manager.stop()


async def test_mid_job_cancellation_runs_cleanup_and_marks_document(job_database):
    _engine, factory = job_database
    kb_id, document_id = create_document(factory, status="PENDING")
    manager = LocalJobManager(factory, worker_count=1)
    running = asyncio.Event()

    async def handler(context, payload):
        def insert_partial():
            with factory() as session:
                document = session.get(Document, document_id)
                document.status = "PROCESSING"
                session.add(
                    DocumentChunk(
                        document_id=document_id,
                        knowledge_base_id=kb_id,
                        chunk_index=0,
                        content="temporary",
                        content_hash="b" * 64,
                        char_count=9,
                    )
                )
                session.commit()

        def purge_partial():
            with factory() as session:
                chunks = list(
                    session.scalars(
                        select(DocumentChunk).where(
                            DocumentChunk.document_id == document_id
                        )
                    )
                )
                for chunk in chunks:
                    session.delete(chunk)
                session.commit()

        await context.run_cpu(insert_partial)
        context.add_cleanup(lambda: context.run_cpu(purge_partial))
        await context.report(JobStage.CHUNKING, 50, "Partial chunks written")
        running.set()
        while True:
            await asyncio.sleep(0.01)
            await context.checkpoint()

    manager.register_handler("CANCELLABLE_INGESTION", handler)
    await manager.start()
    try:
        job_id = await manager.submit(
            "CANCELLABLE_INGESTION",
            knowledge_base_id=kb_id,
            document_id=document_id,
        )
        await asyncio.wait_for(running.wait(), timeout=5)
        assert await manager.cancel(job_id) is True
        job = await manager.wait_for_job(job_id, timeout=5)

        with factory() as session:
            document = session.get(Document, document_id)
            chunk_count = session.scalar(
                select(func.count()).select_from(DocumentChunk)
            )
        assert job.status == JobStatus.CANCELLED.value
        assert document.status == "CANCELLED"
        assert chunk_count == 0
    finally:
        await manager.stop()


class FailingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 4) -> None:
        self.dimensions = dimensions
        self.calls = 0

    def provider_name(self) -> str:
        return "failing-mock"

    def model_name(self) -> str:
        return "new-embedding"

    def get_dimensions(self) -> int:
        return self.dimensions

    async def embed_documents(self, texts, batch_size=32):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("embedding batch failed")
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, query):
        return [1.0, 0.0, 0.0, 0.0]

    async def test_embedding(self):
        return ConnectionTestResult(
            success=True,
            latency_ms=0,
            provider=self.provider_name(),
            model=self.model_name(),
            detected_dimensions=self.dimensions,
        )


class FakeReindexVectorStore:
    def __init__(self) -> None:
        self.collections: dict[str, list[str]] = {"kb-old": ["old-vector"]}
        self.deleted: list[str] = []

    @staticmethod
    def collection_name(knowledge_base_id: str) -> str:
        return f"kb_{knowledge_base_id}"

    async def ensure_collection(self, collection_name, dimensions):
        self.collections.setdefault(collection_name, [])

    async def upsert_chunks(
        self, collection_name, chunks, embeddings, *, metadata=None
    ):
        self.collections.setdefault(collection_name, []).extend(
            chunk["chunk_id"] for chunk in chunks
        )
        return [chunk["chunk_id"] for chunk in chunks]

    async def delete_collection(self, collection_name):
        self.deleted.append(collection_name)
        self.collections.pop(collection_name, None)


async def test_safe_reindex_rolls_back_shadow_when_embedding_fails(job_database):
    _engine, factory = job_database
    with factory() as session:
        old_config = EmbeddingConfiguration(
            name="Old embedding",
            provider_type="mock",
            model_name="old",
            dimensions=4,
        )
        new_config = EmbeddingConfiguration(
            name="New embedding",
            provider_type="mock",
            model_name="new",
            dimensions=4,
            batch_size=1,
        )
        session.add_all([old_config, new_config])
        session.flush()
        kb = KnowledgeBase(name="Reindex KB", embedding_config_id=old_config.id)
        session.add(kb)
        session.flush()
        document = Document(
            knowledge_base_id=kb.id,
            filename="reindex.txt",
            original_rel_path="documents/original/reindex.txt",
            file_type="txt",
            file_size_bytes=20,
            sha256_hash="c" * 64,
            status="READY",
        )
        session.add(document)
        session.flush()
        for index in range(3):
            session.add(
                DocumentChunk(
                    document_id=document.id,
                    knowledge_base_id=kb.id,
                    chunk_index=index,
                    content=f"chunk {index}",
                    content_hash=str(index) * 64,
                    char_count=7,
                )
            )
        session.add(
            IndexMetadata(
                knowledge_base_id=kb.id,
                vector_collection_name="kb-old",
                embedding_provider="mock",
                embedding_model_name="old",
                embedding_dimensions=4,
                vector_count=3,
            )
        )
        session.commit()
        kb_id, old_config_id, new_config_id = kb.id, old_config.id, new_config.id

    vectors = FakeReindexVectorStore()
    provider = FailingEmbeddingProvider()
    handler = KnowledgeBaseReindexJobHandler(
        factory,
        embedding_provider_resolver=lambda _config: provider,
        vector_store=vectors,
    )
    manager = LocalJobManager(factory, worker_count=1, vector_store=vectors)
    manager.register_handler(handler.JOB_TYPE, handler)
    await manager.start()
    try:
        job_id = await submit_knowledge_base_reindex_job(
            kb_id,
            new_config_id,
            manager=manager,
        )
        job = await manager.wait_for_job(job_id, timeout=5)

        with factory() as session:
            kb = session.get(KnowledgeBase, kb_id)
            metadata = session.scalar(
                select(IndexMetadata).where(
                    IndexMetadata.knowledge_base_id == kb_id
                )
            )
        assert job.status == JobStatus.FAILED.value
        assert "embedding batch failed" in job.error_details
        assert kb.embedding_config_id == old_config_id
        assert metadata.vector_collection_name == "kb-old"
        assert "kb-old" in vectors.collections
        shadow_names = [name for name in vectors.deleted if "_reindex_" in name]
        assert len(shadow_names) == 1
        assert shadow_names[0] not in vectors.collections
        assert "kb-old" not in vectors.deleted
    finally:
        await manager.stop()
