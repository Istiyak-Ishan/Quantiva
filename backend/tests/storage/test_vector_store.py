"""Persistence and isolation tests for the embedded local Qdrant repository."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.types import DocumentChunkPayload
from app.providers.interfaces.exceptions import EmbeddingDimensionMismatchError
from app.storage.vector_store import LocalVectorStore, VectorStoreManager


def vector(dimensions: int, position: int = 0) -> list[float]:
    values = [0.0] * dimensions
    values[position] = 1.0
    return values


def chunk(
    chunk_id: str,
    *,
    knowledge_base_id: str,
    document_id: str,
    chunk_index: int,
    content: str,
) -> DocumentChunkPayload:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "knowledge_base_id": knowledge_base_id,
        "page_number": chunk_index + 1,
        "chunk_index": chunk_index,
        "content": content,
        "source_file": f"{document_id}.pdf",
        "section_title": "Results",
        "sheet_name": None,
        "is_table": chunk_index % 2 == 1,
        "content_hash": f"hash-{chunk_id}",
        "char_count": len(content),
    }


@pytest.fixture
def local_store(tmp_path: Path):
    # The manager is process-wide by design.  Closing first lets each test select
    # an isolated directory without ever opening two clients concurrently.
    existing = VectorStoreManager()
    existing.close()
    manager = VectorStoreManager(data_dir=tmp_path)
    manager.start()
    store = LocalVectorStore(manager)
    try:
        yield store, manager
    finally:
        manager.close()


async def test_collections_are_isolated_by_knowledge_base_and_dimensions(local_store):
    store, _manager = local_store

    info_384 = await store.ensure_knowledge_base_collection("small", 384)
    info_768 = await store.ensure_knowledge_base_collection("large", 768)

    assert info_384.name == "kb_small"
    assert info_384.dimensions == 384
    assert info_384.distance_metric == "cosine"
    assert info_768.name == "kb_large"
    assert info_768.dimensions == 768

    with pytest.raises(EmbeddingDimensionMismatchError):
        await store.ensure_collection("kb_small", 768)


async def test_chunk_upsert_and_filtered_similarity_search(local_store):
    store, _manager = local_store
    collection = store.collection_name("finance")
    chunks = [
        chunk(
            "chunk-a",
            knowledge_base_id="finance",
            document_id="doc-a",
            chunk_index=0,
            content="Revenue increased substantially.",
        ),
        chunk(
            "chunk-b",
            knowledge_base_id="finance",
            document_id="doc-b",
            chunk_index=1,
            content="Operating costs declined.",
        ),
        chunk(
            "chunk-c",
            knowledge_base_id="finance",
            document_id="doc-a",
            chunk_index=2,
            content="Revenue guidance was unchanged.",
        ),
    ]
    embeddings = [vector(384, 0), vector(384, 1), vector(384, 2)]

    point_ids = await store.upsert_chunks(
        collection,
        chunks,
        embeddings,
        metadata={"embedding_dimensions": 384},
    )
    all_results = await store.search_dense(collection, vector(384, 0), 3)
    filtered = await store.search_dense(
        collection,
        vector(384, 0),
        3,
        filters={"document_id": "doc-a", "knowledge_base_id": "finance"},
    )

    assert len(point_ids) == 3
    assert len(set(point_ids)) == 3
    assert [result.chunk_id for result in all_results][0] == "chunk-a"
    assert {result.chunk_id for result in filtered} == {"chunk-a", "chunk-c"}
    first = filtered[0]
    assert first.content == "Revenue increased substantially."
    assert first.metadata["document_id"] == "doc-a"
    assert first.metadata["knowledge_base_id"] == "finance"
    assert first.metadata["page_number"] == 1
    assert first.metadata["chunk_index"] == 0
    assert first.metadata["source_file"] == "doc-a.pdf"
    assert first.metadata["is_table"] is False


async def test_dimension_validation_happens_before_upsert_or_query(local_store):
    store, _manager = local_store
    collection = store.collection_name("dimensions")
    payload = chunk(
        "chunk-dim",
        knowledge_base_id="dimensions",
        document_id="doc-dim",
        chunk_index=0,
        content="Dimension test",
    )

    await store.ensure_collection(collection, 384)
    with pytest.raises(EmbeddingDimensionMismatchError):
        await store.upsert_chunks(
            collection,
            [payload],
            [vector(768)],
            metadata={"embedding_dimensions": 768},
        )
    with pytest.raises(EmbeddingDimensionMismatchError):
        await store.search_dense(collection, vector(768), 5)
    with pytest.raises(EmbeddingDimensionMismatchError):
        await store.search_dense(
            collection,
            vector(384),
            5,
            index_metadata={
                "vector_collection_name": collection,
                "embedding_dimensions": 768,
            },
        )

    assert (await store.get_collection_info(collection)).point_count == 0


async def test_per_document_deletion_and_atomic_collection_deletion(local_store):
    store, _manager = local_store
    collection = store.collection_name("deletion")
    chunks = [
        chunk(
            "delete-a",
            knowledge_base_id="deletion",
            document_id="doc-delete",
            chunk_index=0,
            content="Delete this vector.",
        ),
        chunk(
            "keep-b",
            knowledge_base_id="deletion",
            document_id="doc-keep",
            chunk_index=1,
            content="Keep this vector.",
        ),
    ]
    await store.upsert_chunks(
        collection,
        chunks,
        [vector(384, 0), vector(384, 1)],
        metadata={"embedding_dimensions": 384},
    )

    await store.delete_by_document_id(collection, "doc-delete")
    remaining = await store.search_dense(collection, vector(384, 1), 10)
    assert [result.chunk_id for result in remaining] == ["keep-b"]

    await store.delete_collection(collection)
    assert await store.get_collection_info(collection) is None
    # Missing collections are explicitly idempotent.
    await store.delete_collection(collection)


async def test_clean_shutdown_and_reopen_persists_vectors(tmp_path: Path):
    manager = VectorStoreManager()
    manager.close()
    manager = VectorStoreManager(data_dir=tmp_path)
    manager.start()
    store = LocalVectorStore(manager)
    collection = store.collection_name("persistent")
    payload = chunk(
        "persisted-chunk",
        knowledge_base_id="persistent",
        document_id="persisted-doc",
        chunk_index=0,
        content="This vector must survive process shutdown.",
    )
    await store.upsert_chunks(
        collection,
        [payload],
        [vector(384, 7)],
        metadata={"embedding_dimensions": 384},
    )
    manager.close()
    assert manager.is_started is False

    reopened = VectorStoreManager(data_dir=tmp_path)
    assert reopened is manager
    reopened.start()
    try:
        results = await LocalVectorStore(reopened).search_dense(
            collection,
            vector(384, 7),
            5,
        )
        assert [result.chunk_id for result in results] == ["persisted-chunk"]
        assert results[0].content == "This vector must survive process shutdown."
    finally:
        reopened.close()


def test_singleton_rejects_second_active_storage_directory(tmp_path: Path):
    manager = VectorStoreManager()
    manager.close()
    manager = VectorStoreManager(data_dir=tmp_path / "first")
    manager.start()
    try:
        assert VectorStoreManager() is manager
        with pytest.raises(RuntimeError, match="already active"):
            VectorStoreManager(data_dir=tmp_path / "second")
    finally:
        manager.close()
