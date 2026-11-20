"""
backend/tests/core/test_knowledge_bases.py
────────────────────────────────────────────────────────────────────────────────
Tests for Knowledge Base Service & Independent Index Isolation (Prompt P5.1).
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select

from app.core.knowledge_bases import (
    DuplicateDocumentError,
    KnowledgeBaseDetails,
    KnowledgeBaseService,
    KnowledgeBaseSummary,
)
from app.ingestion.types import DocumentChunkPayload
from app.jobs.manager import JobStatus, LocalJobManager
from app.providers.interfaces.exceptions import EmbeddingDimensionMismatchError
from app.storage.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from app.storage.file_store import LocalFileStore
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


def make_vector(dimensions: int, value: float = 1.0) -> list[float]:
    vec = [0.0] * dimensions
    vec[0] = value
    return vec


def make_chunk_payload(
    chunk_id: str,
    kb_id: str,
    doc_id: str,
    chunk_index: int = 0,
    content: str = "Test chunk text",
) -> DocumentChunkPayload:
    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "knowledge_base_id": kb_id,
        "page_number": 1,
        "chunk_index": chunk_index,
        "content": content,
        "source_file": f"{doc_id}.txt",
        "section_title": "Overview",
        "sheet_name": None,
        "is_table": False,
        "content_hash": f"hash-{chunk_id}",
        "char_count": len(content),
    }


@pytest.fixture
def kb_test_env(tmp_path: Path):
    """Provides an isolated environment for KnowledgeBaseService testing."""
    data_dir = tmp_path / "quantiva-kb-test"
    data_dir.mkdir(parents=True, exist_ok=True)

    # SQLite
    engine = create_sqlite_engine(data_dir)
    initialize_database(engine)
    factory = create_session_factory(engine)

    # Embedded Vector Store
    v_manager = VectorStoreManager()
    v_manager.close()
    vec_dir = data_dir / "vector_store"
    vec_manager = VectorStoreManager(data_dir=data_dir, vector_store_dir=vec_dir)
    vec_manager.start()
    vector_store = LocalVectorStore(vec_manager)

    # File Store
    file_store = LocalFileStore(data_dir=data_dir)

    # Seed Embedding & Retrieval Configurations
    with factory() as session:
        emb_small = EmbeddingConfiguration(
            id="emb-config-384",
            name="Test Small Embedding (384d)",
            provider_type="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
            dimensions=384,
            is_default=True,
        )
        emb_large = EmbeddingConfiguration(
            id="emb-config-768",
            name="Test Large Embedding (768d)",
            provider_type="sentence_transformers",
            model_name="intfloat/multilingual-e5-base",
            dimensions=768,
            is_default=False,
        )
        ret_default = RetrievalConfiguration(
            id="ret-config-default",
            name="Default Hybrid Retrieval",
            search_mode="hybrid",
            dense_top_k=20,
            sparse_top_k=20,
            is_default=True,
        )
        session.add_all([emb_small, emb_large, ret_default])
        session.commit()

    # Job manager mock or instance
    job_manager = MagicMock(spec=LocalJobManager)
    job_manager.submit = AsyncMock(return_value="test-job-id-123")
    job_manager.cancel = AsyncMock(return_value=True)

    service = KnowledgeBaseService(
        session_factory=factory,
        vector_store=vector_store,
        file_store=file_store,
        job_manager=job_manager,
    )

    try:
        yield service, factory, vector_store, file_store, job_manager, data_dir
    finally:
        vec_manager.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_create_knowledge_base_locks_dimensions_and_collection(kb_test_env):
    service, factory, vector_store, file_store, job_manager, data_dir = kb_test_env

    kb = await service.create_knowledge_base(
        name="Finance Research",
        description="Q4 earnings and financial models",
        embedding_config_id="emb-config-384",
    )

    assert kb.id is not None
    assert kb.name == "Finance Research"
    assert kb.document_count == 0

    expected_collection = f"kb_{kb.id}"

    # Verify IndexMetadata is locked in database
    with factory() as session:
        index_meta = session.scalar(
            select(IndexMetadata).where(IndexMetadata.knowledge_base_id == kb.id)
        )
        assert index_meta is not None
        assert index_meta.embedding_dimensions == 384
        assert index_meta.embedding_provider == "sentence_transformers"
        assert index_meta.embedding_model_name == "all-MiniLM-L6-v2"
        assert index_meta.vector_collection_name == expected_collection
        assert index_meta.needs_reindex is False

    # Verify collection exists in vector store with 384 dimensions
    coll_info = await vector_store.get_collection_info(expected_collection)
    assert coll_info is not None
    assert coll_info.name == expected_collection
    assert coll_info.dimensions == 384


@pytest.mark.asyncio
async def test_create_knowledge_base_duplicate_name_rejected(kb_test_env):
    service, factory, vector_store, file_store, job_manager, data_dir = kb_test_env

    await service.create_knowledge_base(name="Unique KB")

    with pytest.raises(ValueError, match="already exists"):
        await service.create_knowledge_base(name="Unique KB")


@pytest.mark.asyncio
async def test_multi_kb_strict_isolation_and_dimension_safety(kb_test_env):
    """Verify two knowledge bases with different dimensions remain strictly isolated."""
    service, factory, vector_store, file_store, job_manager, data_dir = kb_test_env

    # 1. Create kb_alpha with 384-dim embeddings
    kb_alpha = await service.create_knowledge_base(
        name="KB Alpha (384d)",
        embedding_config_id="emb-config-384",
    )

    # 2. Create kb_beta with 768-dim embeddings
    kb_beta = await service.create_knowledge_base(
        name="KB Beta (768d)",
        embedding_config_id="emb-config-768",
    )

    assert kb_alpha.id != kb_beta.id

    # 3. Upsert chunks into kb_alpha with 384d vector
    chunk_a = make_chunk_payload("chunk-a", kb_alpha.id, "doc-a", 0, "Alpha contents")
    vec_384 = make_vector(384, 1.0)
    await service.upsert_chunks(kb_alpha.id, [chunk_a], [vec_384])

    # 4. Upsert chunks into kb_beta with 768d vector
    chunk_b = make_chunk_payload("chunk-b", kb_beta.id, "doc-b", 0, "Beta contents")
    vec_768 = make_vector(768, 1.0)
    await service.upsert_chunks(kb_beta.id, [chunk_b], [vec_768])

    # 5. Search kb_alpha with 384d vector -> succeeds, only hits chunk_a
    hits_alpha = await service.search(kb_alpha.id, vec_384, top_k=5)
    assert len(hits_alpha) == 1
    assert hits_alpha[0].chunk_id == "chunk-a"
    assert hits_alpha[0].content == "Alpha contents"

    # 6. Search kb_beta with 768d vector -> succeeds, only hits chunk_b
    hits_beta = await service.search(kb_beta.id, vec_768, top_k=5)
    assert len(hits_beta) == 1
    assert hits_beta[0].chunk_id == "chunk-b"
    assert hits_beta[0].content == "Beta contents"

    # 7. Query kb_alpha with 768d vector -> raises EmbeddingDimensionMismatchError before search
    with pytest.raises(EmbeddingDimensionMismatchError) as exc_info_alpha:
        await service.search(kb_alpha.id, vec_768)
    assert exc_info_alpha.value.expected == 384
    assert exc_info_alpha.value.actual == 768

    # 8. Query kb_beta with 384d vector -> raises EmbeddingDimensionMismatchError before search
    with pytest.raises(EmbeddingDimensionMismatchError) as exc_info_beta:
        await service.search(kb_beta.id, vec_384)
    assert exc_info_beta.value.expected == 768
    assert exc_info_beta.value.actual == 384

    # 9. Direct validate_query_dimensions verification
    service.validate_query_dimensions(kb_alpha.id, 384)  # No error
    with pytest.raises(EmbeddingDimensionMismatchError):
        service.validate_query_dimensions(kb_alpha.id, 768)


@pytest.mark.asyncio
async def test_upload_document_lifecycle_and_hashing(kb_test_env):
    service, factory, vector_store, file_store, job_manager, data_dir = kb_test_env

    kb = await service.create_knowledge_base(name="Doc Upload KB")

    file_bytes = b"Hello Quantiva. This is a sample document for testing ingestion."
    doc = await service.upload_document(
        knowledge_base_id=kb.id,
        file_content=file_bytes,
        filename="sample_report.pdf",
    )

    assert doc.id is not None
    assert doc.filename == "sample_report.pdf"
    assert doc.file_type == "pdf"
    assert doc.file_size_bytes == len(file_bytes)
    assert doc.status == "QUEUED"
    assert doc.sha256_hash is not None

    # Check file exists in file store under original directory
    read_bytes = file_store.read_original(kb.id, doc.id, doc.filename)
    assert read_bytes == file_bytes

    # Check database record and counter update
    with factory() as session:
        kb_db = session.get(KnowledgeBase, kb.id)
        assert kb_db.document_count == 1

        doc_db = session.get(Document, doc.id)
        assert doc_db is not None
        assert doc_db.sha256_hash == doc.sha256_hash
        assert doc_db.status == "QUEUED"

    # Check ingestion job was submitted
    assert job_manager.submit.called


@pytest.mark.asyncio
async def test_duplicate_document_detection_and_force_overwrite(kb_test_env):
    service, factory, vector_store, file_store, job_manager, data_dir = kb_test_env

    kb1 = await service.create_knowledge_base(name="Dup Test KB 1")
    kb2 = await service.create_knowledge_base(name="Dup Test KB 2")

    content = b"Exact identical document content for duplicate testing."

    # 1. First upload in KB 1 succeeds
    doc1 = await service.upload_document(
        knowledge_base_id=kb1.id,
        file_content=content,
        filename="test_file.txt",
    )
    assert doc1 is not None

    # 2. Uploading same content in KB 1 without allow_overwrite raises DuplicateDocumentError
    with pytest.raises(DuplicateDocumentError) as exc_info:
        await service.upload_document(
            knowledge_base_id=kb1.id,
            file_content=content,
            filename="another_name.txt",
            allow_overwrite=False,
        )
    assert exc_info.value.existing_document_id == doc1.id
    assert exc_info.value.sha256_hash == doc1.sha256_hash

    # Verify no second document was added to DB
    docs_in_kb1 = service.list_documents(kb1.id)
    assert len(docs_in_kb1) == 1

    # 3. Uploading same content in KB 2 succeeds because duplicates are scoped per KB
    doc_in_kb2 = await service.upload_document(
        knowledge_base_id=kb2.id,
        file_content=content,
        filename="test_file.txt",
    )
    assert doc_in_kb2 is not None
    assert doc_in_kb2.id != doc1.id

    # 4. Uploading same content in KB 1 with allow_overwrite=True replaces doc1
    doc1_replaced = await service.upload_document(
        knowledge_base_id=kb1.id,
        file_content=content,
        filename="test_file_v2.txt",
        allow_overwrite=True,
    )
    assert doc1_replaced.id != doc1.id

    # Old doc1 must be gone, exactly 1 doc in KB 1
    docs_after = service.list_documents(kb1.id)
    assert len(docs_after) == 1
    assert docs_after[0].id == doc1_replaced.id


@pytest.mark.asyncio
async def test_delete_document_cleans_all_stores_isolated(kb_test_env):
    service, factory, vector_store, file_store, job_manager, data_dir = kb_test_env

    kb_alpha = await service.create_knowledge_base(name="Delete Test KB Alpha", embedding_config_id="emb-config-384")
    kb_beta = await service.create_knowledge_base(name="Delete Test KB Beta", embedding_config_id="emb-config-384")

    # Upload doc to alpha
    doc_a = await service.upload_document(kb_alpha.id, b"Doc A content", "doc_a.txt")
    # Upload doc to beta
    doc_b = await service.upload_document(kb_beta.id, b"Doc B content", "doc_b.txt")

    # Add chunk to alpha vector store
    chunk_a = make_chunk_payload("chunk-da", kb_alpha.id, doc_a.id, 0, "Doc A chunk")
    await service.upsert_chunks(kb_alpha.id, [chunk_a], [make_vector(384)])

    # Add chunk to beta vector store
    chunk_b = make_chunk_payload("chunk-db", kb_beta.id, doc_b.id, 0, "Doc B chunk")
    await service.upsert_chunks(kb_beta.id, [chunk_b], [make_vector(384)])

    # Add chunks to SQLite chunks table
    with factory() as session:
        session.add(
            DocumentChunk(
                id="chunk-da",
                document_id=doc_a.id,
                knowledge_base_id=kb_alpha.id,
                chunk_index=0,
                content="Doc A chunk",
                content_hash="hash-a",
                char_count=11,
            )
        )
        session.add(
            DocumentChunk(
                id="chunk-db",
                document_id=doc_b.id,
                knowledge_base_id=kb_beta.id,
                chunk_index=0,
                content="Doc B chunk",
                content_hash="hash-b",
                char_count=11,
            )
        )
        # Update chunk counts
        kb_a_row = session.get(KnowledgeBase, kb_alpha.id)
        kb_a_row.chunk_count = 1
        doc_a_row = session.get(Document, doc_a.id)
        doc_a_row.chunk_count = 1

        kb_b_row = session.get(KnowledgeBase, kb_beta.id)
        kb_b_row.chunk_count = 1
        doc_b_row = session.get(Document, doc_b.id)
        doc_b_row.chunk_count = 1
        session.commit()

    # Now delete doc_a from kb_alpha
    success = await service.delete_document(kb_alpha.id, doc_a.id)
    assert success is True

    # 1. File store check: doc_a original is gone
    with pytest.raises(FileNotFoundError):
        file_store.read_original(kb_alpha.id, doc_a.id)

    # 2. Vector store check: chunk-da is gone from kb_alpha
    hits_a = await service.search(kb_alpha.id, make_vector(384), top_k=5)
    assert len(hits_a) == 0

    # 3. Vector store check: chunk-db in kb_beta is completely intact!
    hits_b = await service.search(kb_beta.id, make_vector(384), top_k=5)
    assert len(hits_b) == 1
    assert hits_b[0].chunk_id == "chunk-db"

    # 4. Database checks: doc_a and chunk-da are deleted
    with factory() as session:
        assert session.get(Document, doc_a.id) is None
        assert session.get(DocumentChunk, "chunk-da") is None
        assert session.get(Document, doc_b.id) is not None
        assert session.get(DocumentChunk, "chunk-db") is not None

        kb_a_ref = session.get(KnowledgeBase, kb_alpha.id)
        assert kb_a_ref.document_count == 0
        assert kb_a_ref.chunk_count == 0


@pytest.mark.asyncio
async def test_delete_knowledge_base_purges_everything(kb_test_env):
    service, factory, vector_store, file_store, job_manager, data_dir = kb_test_env

    kb1 = await service.create_knowledge_base(name="To Delete KB", embedding_config_id="emb-config-384")
    kb2 = await service.create_knowledge_base(name="Keep KB", embedding_config_id="emb-config-384")

    # Upload doc to kb1
    doc1 = await service.upload_document(kb1.id, b"Data to purge", "purge.txt")
    chunk1 = make_chunk_payload("chunk-p1", kb1.id, doc1.id, 0, "Purge text")
    await service.upsert_chunks(kb1.id, [chunk1], [make_vector(384)])

    # Upload doc to kb2
    doc2 = await service.upload_document(kb2.id, b"Data to keep", "keep.txt")
    chunk2 = make_chunk_payload("chunk-k2", kb2.id, doc2.id, 0, "Keep text")
    await service.upsert_chunks(kb2.id, [chunk2], [make_vector(384)])

    coll_kb1 = collection_name_for_knowledge_base(kb1.id)
    coll_kb2 = collection_name_for_knowledge_base(kb2.id)

    # Delete kb1
    success = await service.delete_knowledge_base(kb1.id)
    assert success is True

    # 1. Vector store: collection for kb1 is gone
    info1 = await vector_store.get_collection_info(coll_kb1)
    assert info1 is None

    # Collection for kb2 is intact
    info2 = await vector_store.get_collection_info(coll_kb2)
    assert info2 is not None

    # 2. File store: directory for kb1 is removed
    kb1_dir = file_store.original_dir / kb1.id
    assert not kb1_dir.exists()

    # 3. Database: KB 1 and its IndexMetadata are gone
    with factory() as session:
        assert session.get(KnowledgeBase, kb1.id) is None
        assert session.scalar(select(IndexMetadata).where(IndexMetadata.knowledge_base_id == kb1.id)) is None
        assert session.get(Document, doc1.id) is None

        # KB 2 is intact
        assert session.get(KnowledgeBase, kb2.id) is not None


@pytest.mark.asyncio
async def test_update_knowledge_base_reindex_detection(kb_test_env):
    service, factory, vector_store, file_store, job_manager, data_dir = kb_test_env

    kb = await service.create_knowledge_base(
        name="Original Name",
        description="Original Desc",
        embedding_config_id="emb-config-384",
    )

    # 1. Simple update of name & description
    updated_kb = await service.update_knowledge_base(
        kb.id,
        name="New Name",
        description="New Desc",
    )
    assert updated_kb.name == "New Name"
    assert updated_kb.description == "New Desc"

    with factory() as session:
        idx_meta = session.scalar(select(IndexMetadata).where(IndexMetadata.knowledge_base_id == kb.id))
        assert idx_meta.needs_reindex is False

    # 2. Switch embedding configuration to 768d model
    updated_kb2 = await service.update_knowledge_base(
        kb.id,
        embedding_config_id="emb-config-768",
        trigger_reindex=False,
    )
    assert updated_kb2.embedding_config_id == "emb-config-768"

    # IndexMetadata must now indicate needs_reindex=True
    with factory() as session:
        idx_meta = session.scalar(select(IndexMetadata).where(IndexMetadata.knowledge_base_id == kb.id))
        assert idx_meta.needs_reindex is True


@pytest.mark.asyncio
async def test_get_knowledge_base_details_and_listing(kb_test_env):
    service, factory, vector_store, file_store, job_manager, data_dir = kb_test_env

    kb = await service.create_knowledge_base(
        name="Diagnostics KB",
        embedding_config_id="emb-config-384",
    )

    await service.upload_document(kb.id, b"Doc 1", "file1.txt")
    await service.upload_document(kb.id, b"Doc 2", "file2.txt")

    # List KBs
    summaries = service.list_knowledge_bases()
    assert len(summaries) >= 1
    match = next(s for s in summaries if s.id == kb.id)
    assert match.name == "Diagnostics KB"
    assert match.document_count == 2
    assert match.embedding_dimensions == 384
    assert match.needs_reindex is False

    # Deep details
    details = await service.get_knowledge_base_details(kb.id)
    assert isinstance(details, KnowledgeBaseDetails)
    assert details.id == kb.id
    assert details.document_count == 2
    assert details.documents_by_status.get("QUEUED") == 2
    assert details.documents_by_status.get("total") == 2
    assert details.is_healthy is True
    assert details.vector_collection_name == f"kb_{kb.id}"
