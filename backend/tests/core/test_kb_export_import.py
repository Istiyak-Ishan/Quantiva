"""
backend/tests/core/test_kb_export_import.py
────────────────────────────────────────────────────────────────────────────────
Tests for portable Knowledge Base export/import (.qkb.zip) — Prompt P5.2.

Verifies that an exported Knowledge Base, imported under a fresh identity,
preserves 100% of documents, chunk text, FTS5/BM25 sparse results, and dense
vector search results without re-extraction or re-embedding.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import func, select

from app.core.kb_archive import (
    KBArchiveError,
    KBArchiveValidationError,
    export_knowledge_base,
    export_knowledge_base_async,
    import_knowledge_base,
    import_knowledge_base_async,
)
from app.core.knowledge_bases import KnowledgeBaseService
from app.ingestion.types import DocumentChunkPayload
from app.jobs.manager import LocalJobManager
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
    RetrievalConfiguration,
)
from app.storage.sparse_store import SQLiteSparseIndex
from app.storage.vector_store import LocalVectorStore, VectorStoreManager

DIMENSIONS = 384


def make_vector(index: int, dimensions: int = DIMENSIONS) -> list[float]:
    """Deterministic, non-zero, one-hot-ish unit vectors for stable ranking."""
    vector = [0.0] * dimensions
    vector[index % dimensions] = 1.0
    return vector


def ranking_query(dimensions: int = DIMENSIONS) -> list[float]:
    """Query vector producing a strict, tie-free ordering over make_vector(0..3)."""
    vector = [0.0] * dimensions
    vector[0:4] = [1.0, 0.9, 0.5, 0.2]
    return vector


def _chunk_payload(
    chunk_id: str,
    document_id: str,
    knowledge_base_id: str,
    chunk_index: int,
    content: str,
    filename: str,
) -> DocumentChunkPayload:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "knowledge_base_id": knowledge_base_id,
        "page_number": 1,
        "chunk_index": chunk_index,
        "content": content,
        "source_file": filename,
        "section_title": "Body",
        "sheet_name": None,
        "is_table": False,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "char_count": len(content),
    }


@pytest.fixture
def kb_archive_env(tmp_path: Path):
    """Isolated SQLite + embedded Qdrant + file store environment."""
    data_dir = tmp_path / "quantiva-qkb-test"
    data_dir.mkdir(parents=True, exist_ok=True)

    engine = create_sqlite_engine(data_dir)
    initialize_database(engine)
    factory = create_session_factory(engine)

    VectorStoreManager().close()
    vec_manager = VectorStoreManager(
        data_dir=data_dir,
        vector_store_dir=data_dir / "vector_store",
    )
    vec_manager.start()
    vector_store = LocalVectorStore(vec_manager)
    file_store = LocalFileStore(data_dir=data_dir)

    with factory() as session:
        session.add_all(
            [
                EmbeddingConfiguration(
                    id="emb-config-384",
                    name="Test Small Embedding (384d)",
                    provider_type="sentence_transformers",
                    model_name="all-MiniLM-L6-v2",
                    dimensions=DIMENSIONS,
                    is_default=True,
                ),
                EmbeddingConfiguration(
                    id="emb-config-768",
                    name="Test Large Embedding (768d)",
                    provider_type="sentence_transformers",
                    model_name="intfloat/multilingual-e5-base",
                    dimensions=768,
                    is_default=False,
                ),
                RetrievalConfiguration(
                    id="ret-config-default",
                    name="Default Hybrid Retrieval",
                    search_mode="hybrid",
                    is_default=True,
                ),
            ]
        )
        session.commit()

    job_manager = MagicMock(spec=LocalJobManager)
    job_manager.submit = AsyncMock(return_value="test-job-id")
    job_manager.cancel = AsyncMock(return_value=True)

    service = KnowledgeBaseService(
        session_factory=factory,
        vector_store=vector_store,
        file_store=file_store,
        job_manager=job_manager,
    )

    try:
        yield service, factory, vector_store, file_store, engine, data_dir
    finally:
        vec_manager.close()
        engine.dispose()


async def _upload_and_seed(
    service: KnowledgeBaseService,
    factory: Any,
    knowledge_base_id: str,
    documents: list[tuple[bytes, str, list[tuple[str, str, list[float]]]]],
    *,
    status: str = "READY",
) -> list[Document]:
    """Upload source files, persist chunk rows, and index dense + sparse data."""
    uploaded: list[Document] = []
    payloads: list[DocumentChunkPayload] = []
    vectors: list[list[float]] = []

    for content, filename, chunk_defs in documents:
        document = await service.upload_document(
            knowledge_base_id=knowledge_base_id,
            file_content=content,
            filename=filename,
        )
        uploaded.append(document)

        with factory() as session:
            row = session.get(Document, document.id)
            row.status = status
            row.chunk_count = len(chunk_defs)
            row.page_count = 1
            for index, (chunk_id, text, _vector) in enumerate(chunk_defs):
                session.add(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document.id,
                        knowledge_base_id=knowledge_base_id,
                        chunk_index=index,
                        page_number=1,
                        section_title="Body",
                        is_table=False,
                        content=text,
                        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        char_count=len(text),
                        metadata_json={"seeded": True},
                    )
                )
            session.commit()

        for index, (chunk_id, text, vector) in enumerate(chunk_defs):
            payloads.append(
                _chunk_payload(
                    chunk_id=chunk_id,
                    document_id=document.id,
                    knowledge_base_id=knowledge_base_id,
                    chunk_index=index,
                    content=text,
                    filename=filename,
                )
            )
            vectors.append(vector)

    with factory() as session:
        kb_row = session.get(KnowledgeBase, knowledge_base_id)
        kb_row.chunk_count = len(payloads)
        session.commit()

    await service.upsert_chunks(knowledge_base_id, payloads, vectors)
    return uploaded


def _rewrite_zip_member(
    archive: Path,
    member: str,
    mutate: Callable[[bytes], bytes],
) -> None:
    """Rewrite one zip member in place while leaving the rest untouched."""
    with zipfile.ZipFile(archive, "r") as zf:
        members = [(info, zf.read(info.filename)) for info in zf.infolist()]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for info, data in members:
            if info.filename == member:
                data = mutate(data)
            zf.writestr(info, data)


def _rewrite_kb_metadata(archive: Path, mutate: Callable[[dict], None]) -> None:
    def _mutate(raw: bytes) -> bytes:
        metadata = json.loads(raw.decode("utf-8"))
        mutate(metadata)
        return json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")

    _rewrite_zip_member(archive, "kb_metadata.json", _mutate)


# ── 1. Full round-trip fidelity ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_import_round_trip_preserves_everything(kb_archive_env):
    service, factory, vector_store, file_store, engine, data_dir = kb_archive_env

    source = await service.create_knowledge_base(
        name="Portable Source",
        description="Round-trip fidelity fixture",
        embedding_config_id="emb-config-384",
    )

    pdf_bytes = b"%PDF-1.4\nQuarterly earnings report payload."
    xlsx_bytes = b"PK\x03\x04\nBudget workbook payload."
    document_defs = [
        (
            pdf_bytes,
            "q4_report.pdf",
            [
                ("src-c0", "Quarterly revenue increased by eighteen percent.", make_vector(0)),
                ("src-c1", "Operating margin expanded across all regions.", make_vector(1)),
            ],
        ),
        (
            xlsx_bytes,
            "budget.xlsx",
            [
                ("src-c2", "Employee headcount reached four thousand people.", make_vector(2)),
                ("src-c3", "Research and development spending doubled.", make_vector(3)),
            ],
        ),
    ]
    uploaded = await _upload_and_seed(service, factory, source.id, document_defs)

    # Record the reference results before export.
    source_dense = await service.search(source.id, ranking_query(), top_k=4)
    sparse_index = SQLiteSparseIndex(engine)
    source_sparse = await sparse_index.search_sparse(
        source.id, "quarterly revenue", top_k=10
    )
    assert [hit.content for hit in source_dense] == [
        "Quarterly revenue increased by eighteen percent.",
        "Operating margin expanded across all regions.",
        "Employee headcount reached four thousand people.",
        "Research and development spending doubled.",
    ]
    assert [hit.content for hit in source_sparse] == [
        "Quarterly revenue increased by eighteen percent."
    ]

    # Export through the public synchronous API (exercised while a loop runs).
    archive = export_knowledge_base(source.id, service=service)
    assert archive.is_file()
    assert archive.parent == (data_dir / "exports").resolve()
    assert archive.name.endswith(".qkb.zip")

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        assert {
            "kb_metadata.json",
            "documents.json",
            "chunks.jsonl",
            "vectors.jsonl",
        } <= names
        assert sum(1 for name in names if name.startswith("original_files/")) == 2
        metadata = json.loads(zf.read("kb_metadata.json"))
        assert metadata["format"] == "quantiva-knowledge-base"
        assert metadata["format_version"] == 1
        assert metadata["counts"] == {
            "documents": 2,
            "chunks": 4,
            "vectors": 4,
            "skipped_vectors": 0,
        }
        assert metadata["embedding_configuration"]["model_name"] == "all-MiniLM-L6-v2"
        assert metadata["index_metadata"]["embedding_dimensions"] == DIMENSIONS
        assert len(zf.read("chunks.jsonl").splitlines()) == 4
        # Original payloads travel with the archive.
        assert pdf_bytes in b"".join(
            zf.read(name) for name in names if name.startswith("original_files/")
        )

    # Import through the sync API on a worker thread.
    imported = await asyncio.to_thread(
        import_knowledge_base, archive, None, service=service
    )

    # ── Fresh identity ────────────────────────────────────────────────────────
    assert imported.id != source.id
    assert imported.name == "Portable Source (Imported)"
    assert imported.description == "Round-trip fidelity fixture"
    assert imported.document_count == 2
    assert imported.chunk_count == 4
    assert imported.embedding_config_id == "emb-config-384"

    with factory() as session:
        imported_docs = session.scalars(
            select(Document).where(Document.knowledge_base_id == imported.id)
        ).all()
        imported_chunks = session.scalars(
            select(DocumentChunk).where(
                DocumentChunk.knowledge_base_id == imported.id
            )
        ).all()
        assert len(imported_docs) == 2
        assert len(imported_chunks) == 4

        old_doc_ids = {doc.id for doc in uploaded}
        old_chunk_ids = {
            chunk_id
            for _, _, chunk_defs in document_defs
            for chunk_id, _, _ in chunk_defs
        }
        assert old_doc_ids.isdisjoint({doc.id for doc in imported_docs})
        assert old_chunk_ids.isdisjoint({chunk.id for chunk in imported_chunks})
        assert all(
            doc.original_rel_path.startswith(
                f"documents/original/{imported.id}/"
            )
            for doc in imported_docs
        )
        assert all(chunk.metadata_json == {"seeded": True} for chunk in imported_chunks)

        # Documents keep their text and per-document chunk counters.
        assert sorted(doc.filename for doc in imported_docs) == [
            "budget.xlsx",
            "q4_report.pdf",
        ]
        assert {doc.filename: doc.chunk_count for doc in imported_docs} == {
            "q4_report.pdf": 2,
            "budget.xlsx": 2,
        }
        assert all(doc.status == "READY" for doc in imported_docs)

        imported_index = session.scalar(
            select(IndexMetadata).where(
                IndexMetadata.knowledge_base_id == imported.id
            )
        )
        assert imported_index is not None
        assert imported_index.vector_collection_name == f"kb_{imported.id}"
        assert imported_index.embedding_dimensions == DIMENSIONS
        assert imported_index.embedding_provider == "sentence_transformers"
        assert imported_index.embedding_model_name == "all-MiniLM-L6-v2"
        assert imported_index.needs_reindex is False
        assert imported_index.vector_count == 4
        assert imported_index.sparse_count == 4
        assert imported_index.last_indexed_at is not None

    # ── Original files restored byte-for-byte ─────────────────────────────────
    with factory() as session:
        imported_by_name = {
            doc.filename: doc
            for doc in session.scalars(
                select(Document).where(Document.knowledge_base_id == imported.id)
            ).all()
        }
    assert file_store.read_original(
        imported.id,
        imported_by_name["q4_report.pdf"].id,
        "q4_report.pdf",
    ) == pdf_bytes
    assert file_store.read_original(
        imported.id,
        imported_by_name["budget.xlsx"].id,
        "budget.xlsx",
    ) == xlsx_bytes

    # ── Dense vectors restored without re-embedding ───────────────────────────
    imported_dense = await service.search(imported.id, ranking_query(), top_k=4)
    assert [hit.content for hit in imported_dense] == [
        hit.content for hit in source_dense
    ]
    for expected, actual in zip(source_dense, imported_dense):
        assert actual.score == pytest.approx(expected.score, abs=1e-6)
        assert actual.chunk_id not in old_chunk_ids

    # ── Sparse FTS5 results preserved (ids remapped by rank order) ────────────
    imported_sparse = await sparse_index.search_sparse(
        imported.id, "quarterly revenue", top_k=10
    )
    assert [hit.content for hit in imported_sparse] == [
        hit.content for hit in source_sparse
    ]
    assert imported_sparse[0].metadata["source_file"] == "q4_report.pdf"
    assert imported_sparse[0].chunk_id not in old_chunk_ids

    # ── Health diagnostics line up ────────────────────────────────────────────
    details = await service.get_knowledge_base_details(imported.id)
    assert details.is_healthy is True
    assert details.vector_count == 4
    assert details.sparse_count == 4
    assert details.documents_by_status.get("READY") == 2


# ── 2. Repeated import must never collide ─────────────────────────────────────

@pytest.mark.asyncio
async def test_import_same_archive_twice_uses_fresh_uuids(kb_archive_env):
    service, factory, vector_store, file_store, engine, data_dir = kb_archive_env

    source = await service.create_knowledge_base(
        name="Repeatable", embedding_config_id="emb-config-384"
    )
    await _upload_and_seed(
        service,
        factory,
        source.id,
        [
            (
                b"repeatable bytes",
                "repeatable.txt",
                [("rep-c0", "Repeatable chunk contents for import twice.", make_vector(5))],
            )
        ],
    )
    archive = await export_knowledge_base_async(source.id, service=service)

    first = await asyncio.to_thread(
        import_knowledge_base, archive, None, service=service
    )
    second = await asyncio.to_thread(
        import_knowledge_base, archive, None, service=service
    )
    async_third = await import_knowledge_base_async(
        archive, "Explicit Async Name", service=service
    )

    assert len({source.id, first.id, second.id, async_third.id}) == 4
    assert first.name == "Repeatable (Imported)"
    assert second.name == "Repeatable (Imported) 2"
    assert async_third.name == "Explicit Async Name"

    with factory() as session:
        assert session.scalar(
            select(func.count(KnowledgeBase.id)).where(
                KnowledgeBase.id.in_([first.id, second.id, async_third.id])
            )
        ) == 3
        all_docs = session.scalars(
            select(Document).where(
                Document.knowledge_base_id.in_([first.id, second.id, async_third.id])
            )
        ).all()
        all_chunks = session.scalars(
            select(DocumentChunk).where(
                DocumentChunk.knowledge_base_id.in_(
                    [first.id, second.id, async_third.id]
                )
            )
        ).all()
        assert len(all_docs) == 3
        assert len(all_chunks) == 3
        assert len({doc.id for doc in all_docs}) == 3
        assert len({chunk.id for chunk in all_chunks}) == 3
        # Each import owns an isolated dense collection.
        for kb_id in (first.id, second.id, async_third.id):
            assert (
                session.scalar(
                    select(IndexMetadata).where(IndexMetadata.knowledge_base_id == kb_id)
                ).vector_collection_name
                == f"kb_{kb_id}"
            )

    # Explicit duplicate names are rejected instead of silently overwriting.
    with pytest.raises(ValueError, match="already exists"):
        await asyncio.to_thread(
            import_knowledge_base, archive, "Explicit Async Name", service=service
        )

    # Each copy remains independently searchable.
    for kb in (first, second, async_third):
        hits = await service.search(kb.id, make_vector(5), top_k=5)
        assert len(hits) == 1
        assert hits[0].content == "Repeatable chunk contents for import twice."


# ── 3. Embedding space divergence flags a reindex ─────────────────────────────

@pytest.mark.asyncio
async def test_import_flags_reindex_when_embedding_space_differs(kb_archive_env):
    service, factory, vector_store, file_store, engine, data_dir = kb_archive_env

    source = await service.create_knowledge_base(
        name="Divergent", embedding_config_id="emb-config-384"
    )
    await _upload_and_seed(
        service,
        factory,
        source.id,
        [
            (
                b"divergent bytes",
                "divergent.txt",
                [
                    ("div-c0", "Alpha divergent chunk.", make_vector(7)),
                    ("div-c1", "Beta divergent chunk.", make_vector(8)),
                ],
            )
        ],
    )
    archive = export_knowledge_base(source.id, service=service)

    def switch_embedding_space(metadata: dict[str, Any]) -> None:
        metadata["embedding_configuration"]["id"] = "remote-768-config"
        metadata["embedding_configuration"]["name"] = "Remote 768 Embedding"
        metadata["embedding_configuration"]["model_name"] = "all-mpnet-base-v2"
        metadata["embedding_configuration"]["dimensions"] = 768

    _rewrite_kb_metadata(archive, switch_embedding_space)

    imported = await asyncio.to_thread(
        import_knowledge_base, archive, "Divergent Imported", service=service
    )

    # Text survives fully even though the vectors cannot be reused.
    assert imported.document_count == 1
    assert imported.chunk_count == 2

    with factory() as session:
        embedding = session.get(EmbeddingConfiguration, imported.embedding_config_id)
        assert embedding is not None
        assert embedding.model_name == "all-mpnet-base-v2"
        assert embedding.dimensions == 768
        assert embedding.id != "remote-768-config"

        index_metadata = session.scalar(
            select(IndexMetadata).where(
                IndexMetadata.knowledge_base_id == imported.id
            )
        )
        assert index_metadata.embedding_dimensions == 768
        assert index_metadata.embedding_model_name == "all-mpnet-base-v2"
        assert index_metadata.needs_reindex is True
        assert index_metadata.vector_count == 0
        assert index_metadata.sparse_count == 2

        chunks = session.scalars(
            select(DocumentChunk).where(
                DocumentChunk.knowledge_base_id == imported.id
            )
        ).all()
        assert {chunk.content for chunk in chunks} == {
            "Alpha divergent chunk.",
            "Beta divergent chunk.",
        }

    # Sparse search still works immediately.
    sparse_index = SQLiteSparseIndex(engine)
    sparse_hits = await sparse_index.search_sparse(imported.id, "alpha", top_k=5)
    assert [hit.content for hit in sparse_hits] == ["Alpha divergent chunk."]

    # The locked dimension is enforced before touching the vector store.
    with pytest.raises(EmbeddingDimensionMismatchError):
        await service.search(imported.id, make_vector(0, dimensions=384))
    assert await service.search(imported.id, make_vector(0, dimensions=768)) == []


# ── 4. Archive integrity / validation ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_import_rejects_missing_or_tampered_archives(kb_archive_env, tmp_path):
    service, factory, vector_store, file_store, engine, data_dir = kb_archive_env

    source = await service.create_knowledge_base(
        name="Tamper Target", embedding_config_id="emb-config-384"
    )
    await _upload_and_seed(
        service,
        factory,
        source.id,
        [
            (
                b"tamper bytes",
                "tamper.txt",
                [("tamper-c0", "Tamper detection chunk.", make_vector(9))],
            )
        ],
    )
    archive = export_knowledge_base(source.id, service=service)

    # 4a. Missing manifest
    missing_manifest = tmp_path / "missing_manifest.qkb.zip"
    with zipfile.ZipFile(archive) as zf:
        with zipfile.ZipFile(missing_manifest, "w") as target:
            for name in zf.namelist():
                if name != "kb_metadata.json":
                    target.writestr(name, zf.read(name))
    with pytest.raises(KBArchiveValidationError, match="kb_metadata.json"):
        await asyncio.to_thread(
            import_knowledge_base, missing_manifest, None, service=service
        )

    # 4b. Tampered payload with stale checksum
    tampered = tmp_path / "tampered.qkb.zip"
    tampered.write_bytes(archive.read_bytes())
    _rewrite_zip_member(
        tampered,
        "chunks.jsonl",
        lambda raw: raw.replace(b"Tamper detection chunk.", b"Evil injected chunk."),
    )
    with pytest.raises(KBArchiveValidationError, match="integrity verification failed"):
        await asyncio.to_thread(
            import_knowledge_base, tampered, None, service=service
        )

    # 4c. Path traversal member
    traversal = tmp_path / "traversal.qkb.zip"
    with zipfile.ZipFile(archive) as zf:
        metadata_bytes = zf.read("kb_metadata.json")
    with zipfile.ZipFile(traversal, "w") as zf:
        zf.writestr("kb_metadata.json", metadata_bytes)
        zf.writestr("../escape.txt", b"nope")
    with pytest.raises(KBArchiveValidationError, match="Unsafe archive member"):
        await asyncio.to_thread(
            import_knowledge_base, traversal, None, service=service
        )

    # 4d. Not a zip at all
    not_a_zip = tmp_path / "not_a_zip.qkb.zip"
    not_a_zip.write_bytes(b"definitely not a zip")
    with pytest.raises(KBArchiveValidationError, match="not a valid zip"):
        await asyncio.to_thread(
            import_knowledge_base, not_a_zip, None, service=service
        )

    # 4e. Missing archive path
    with pytest.raises(FileNotFoundError):
        await asyncio.to_thread(
            import_knowledge_base,
            tmp_path / "does_not_exist.qkb.zip",
            None,
            service=service,
        )

    # No partially imported Knowledge Base leaked from any failed attempt.
    assert len(service.list_knowledge_bases()) == 1

    # 4f. Exporting an unknown Knowledge Base fails loudly.
    with pytest.raises(KBArchiveError, match="not found"):
        export_knowledge_base("00000000-0000-0000-0000-000000000000", service=service)
