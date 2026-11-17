"""Tests for SQLite FTS5 synchronization, sanitization and BM25 retrieval."""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import delete, text

from app.ingestion.types import DocumentChunkPayload
from app.storage.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from app.storage.models import Document, DocumentChunk, KnowledgeBase
from app.storage.sparse_store import SQLiteSparseIndex, sanitize_fts5_query


@pytest.fixture
def sparse_database(tmp_path: Path):
    engine = create_sqlite_engine(tmp_path / "data")
    initialize_database(engine)
    factory = create_session_factory(engine)
    store = SQLiteSparseIndex(engine)
    try:
        yield engine, factory, store
    finally:
        engine.dispose()


def add_document(
    factory,
    *,
    knowledge_base_id: str,
    name: str,
    contents: list[str],
) -> tuple[str, list[str]]:
    with factory() as session:
        document = Document(
            knowledge_base_id=knowledge_base_id,
            filename=f"{name}.pdf",
            original_rel_path=f"documents/original/{name}.pdf",
            file_type="pdf",
            file_size_bytes=100,
            sha256_hash=(name.encode().hex() + "0" * 64)[:64],
            status="ready",
        )
        session.add(document)
        session.flush()
        chunks = [
            DocumentChunk(
                document_id=document.id,
                knowledge_base_id=knowledge_base_id,
                chunk_index=index,
                page_number=index + 1,
                section_title="Quarterly Revenue" if index == 0 else "Notes",
                is_table=index == 0,
                content=content,
                content_hash=f"{name}-{index}".ljust(64, "0")[:64],
                char_count=len(content),
                metadata_json={"ordinal": index},
            )
            for index, content in enumerate(contents)
        ]
        session.add_all(chunks)
        session.commit()
        return document.id, [chunk.id for chunk in chunks]


async def test_bm25_ranks_most_relevant_chunk_first(sparse_database):
    _engine, factory, store = sparse_database
    with factory() as session:
        knowledge_base = KnowledgeBase(name="Ranking KB")
        session.add(knowledge_base)
        session.commit()
        kb_id = knowledge_base.id
    _document_id, chunk_ids = add_document(
        factory,
        knowledge_base_id=kb_id,
        name="ranking",
        contents=[
            "Q4 revenue revenue revenue reached 42 million USD after strong growth.",
            "The appendix briefly mentions revenue among many unrelated disclosures.",
            "Employee headcount and office locations are listed here.",
        ],
    )

    results = await store.search_sparse(kb_id, "Q4 revenue growth", 10)

    assert results
    assert results[0].chunk_id == chunk_ids[0]
    assert results[0].score > 0
    assert all(result.score >= 0 for result in results)
    assert results == sorted(results, key=lambda item: item.score, reverse=True)
    assert results[0].metadata["source_file"] == "ranking.pdf"
    assert results[0].metadata["page_number"] == 1
    assert results[0].metadata["is_table"] is True


@pytest.mark.parametrize(
    "hostile_query",
    [
        '"What is Q4\'s revenue (in $USD) vs. 2024-2025? [ table: 1 ] *AND* NOT null"',
        '((("""***))) AND OR NOT NEAR / : - + ?',
        "' ; DROP TABLE document_chunks; -- revenue*:* (Q4)",
        "\x00\x01\x02 revenue / 2025",
        "AND OR NOT",
    ],
)
async def test_adversarial_punctuation_never_causes_fts_syntax_error(
    sparse_database,
    hostile_query: str,
):
    _engine, factory, store = sparse_database
    with factory() as session:
        knowledge_base = KnowledgeBase(name=f"Hostile {abs(hash(hostile_query))}")
        session.add(knowledge_base)
        session.commit()
        kb_id = knowledge_base.id
    add_document(
        factory,
        knowledge_base_id=kb_id,
        name=f"hostile-{abs(hash(hostile_query))}",
        contents=["Q4 revenue was 42 USD in 2025; the table value is not null."],
    )

    sanitized = sanitize_fts5_query(hostile_query)
    results = await store.search_sparse(kb_id, hostile_query, 5)

    assert "*" not in sanitized
    assert "(" not in sanitized
    assert ")" not in sanitized
    assert ":" not in sanitized
    assert not any(
        token in {"AND", "NOT", "NEAR"}
        for token in sanitized.replace('"', "").split()
    )
    assert isinstance(results, list)


async def test_knowledge_base_and_document_filters_are_strict(sparse_database):
    _engine, factory, store = sparse_database
    with factory() as session:
        first_kb = KnowledgeBase(name="First isolated KB")
        second_kb = KnowledgeBase(name="Second isolated KB")
        session.add_all([first_kb, second_kb])
        session.commit()
        first_kb_id, second_kb_id = first_kb.id, second_kb.id

    first_doc, first_chunks = add_document(
        factory,
        knowledge_base_id=first_kb_id,
        name="first-document",
        contents=["confidential alpha revenue evidence"],
    )
    second_doc_same_kb, second_chunks_same_kb = add_document(
        factory,
        knowledge_base_id=first_kb_id,
        name="second-document",
        contents=["public alpha revenue evidence"],
    )
    _other_doc, second_kb_chunks = add_document(
        factory,
        knowledge_base_id=second_kb_id,
        name="other-kb-document",
        contents=["alpha revenue evidence from another knowledge base"],
    )

    first_results = await store.search_sparse(first_kb_id, "alpha revenue", 10)
    filtered_results = await store.search_sparse(
        first_kb_id,
        "alpha revenue",
        10,
        filters={"document_ids": [first_doc]},
    )
    second_results = await store.search_sparse(second_kb_id, "alpha revenue", 10)

    assert {result.chunk_id for result in first_results} == {
        first_chunks[0],
        second_chunks_same_kb[0],
    }
    assert [result.chunk_id for result in filtered_results] == [first_chunks[0]]
    assert [result.chunk_id for result in second_results] == [second_kb_chunks[0]]
    assert all(
        result.metadata["knowledge_base_id"] == first_kb_id
        for result in first_results
    )
    assert second_doc_same_kb != first_doc


async def test_document_delete_cascade_automatically_purges_fts_rows(sparse_database):
    engine, factory, store = sparse_database
    with factory() as session:
        knowledge_base = KnowledgeBase(name="Cleanup KB")
        session.add(knowledge_base)
        session.commit()
        kb_id = knowledge_base.id
    document_id, _chunk_ids = add_document(
        factory,
        knowledge_base_id=kb_id,
        name="cleanup",
        contents=["orphan prevention evidence", "additional prevention evidence"],
    )
    assert len(await store.search_sparse(kb_id, "prevention evidence", 10)) == 2

    with factory() as session:
        session.execute(delete(Document).where(Document.id == document_id))
        session.commit()

    assert await store.search_sparse(kb_id, "prevention evidence", 10) == []
    with engine.connect() as connection:
        count = connection.execute(
            text(
                "SELECT count(*) FROM document_chunks_fts "
                "WHERE document_id = :document_id"
            ),
            {"document_id": document_id},
        ).scalar_one()
    assert count == 0


async def test_explicit_batch_index_and_delete_are_transactional(sparse_database):
    _engine, _factory, store = sparse_database
    payload: DocumentChunkPayload = {
        "chunk_id": "direct-chunk",
        "document_id": "direct-document",
        "knowledge_base_id": "direct-kb",
        "page_number": 1,
        "chunk_index": 0,
        "content": "directly indexed sparse evidence",
        "source_file": "direct.txt",
        "section_title": "Direct",
        "sheet_name": None,
        "is_table": False,
        "content_hash": "f" * 64,
        "char_count": 32,
    }

    await store.index_chunks("direct-kb", [payload])
    results = await store.search_sparse("direct-kb", "sparse evidence", 5)
    assert [result.chunk_id for result in results] == ["direct-chunk"]

    await store.delete_by_document_id("direct-kb", "direct-document")
    assert await store.search_sparse("direct-kb", "sparse evidence", 5) == []
