"""SQLite FTS5 / BM25 sparse retrieval repository."""
from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import Engine, bindparam, text

from app.ingestion.types import DocumentChunkPayload
from app.storage.interfaces import SearchHit, SparseIndexRepository

FTS_TABLE = "document_chunks_fts"

CREATE_FTS_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    knowledge_base_id UNINDEXED,
    source_file,
    section_title,
    content,
    tokenize='porter unicode61 remove_diacritics 2'
);
"""

CREATE_TRIGGER_SQL: tuple[str, ...] = (
    """
    CREATE TRIGGER IF NOT EXISTS document_chunks_fts_after_insert
    AFTER INSERT ON document_chunks
    BEGIN
        DELETE FROM document_chunks_fts WHERE chunk_id = NEW.id;
        INSERT INTO document_chunks_fts(
            chunk_id, document_id, knowledge_base_id,
            source_file, section_title, content
        ) VALUES (
            NEW.id,
            NEW.document_id,
            NEW.knowledge_base_id,
            COALESCE(
                (SELECT filename FROM documents WHERE id = NEW.document_id),
                ''
            ),
            COALESCE(NEW.section_title, ''),
            NEW.content
        );
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS document_chunks_fts_after_delete
    AFTER DELETE ON document_chunks
    BEGIN
        DELETE FROM document_chunks_fts WHERE chunk_id = OLD.id;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS document_chunks_fts_after_update
    AFTER UPDATE ON document_chunks
    BEGIN
        DELETE FROM document_chunks_fts WHERE chunk_id = OLD.id;
        INSERT INTO document_chunks_fts(
            chunk_id, document_id, knowledge_base_id,
            source_file, section_title, content
        ) VALUES (
            NEW.id,
            NEW.document_id,
            NEW.knowledge_base_id,
            COALESCE(
                (SELECT filename FROM documents WHERE id = NEW.document_id),
                ''
            ),
            COALESCE(NEW.section_title, ''),
            NEW.content
        );
    END;
    """,
    # Defense in depth for rows inserted through SparseIndexRepository before
    # their relational chunk transaction is committed.
    """
    CREATE TRIGGER IF NOT EXISTS documents_fts_after_delete
    AFTER DELETE ON documents
    BEGIN
        DELETE FROM document_chunks_fts WHERE document_id = OLD.id;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS knowledge_bases_fts_after_delete
    AFTER DELETE ON knowledge_bases
    BEGIN
        DELETE FROM document_chunks_fts WHERE knowledge_base_id = OLD.id;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS documents_fts_after_filename_update
    AFTER UPDATE OF filename ON documents
    BEGIN
        UPDATE document_chunks_fts
        SET source_file = NEW.filename
        WHERE document_id = NEW.id;
    END;
    """,
)

BACKFILL_FTS_SQL = """
INSERT INTO document_chunks_fts(
    chunk_id, document_id, knowledge_base_id,
    source_file, section_title, content
)
SELECT
    chunk.id,
    chunk.document_id,
    chunk.knowledge_base_id,
    document.filename,
    COALESCE(chunk.section_title, ''),
    chunk.content
FROM document_chunks AS chunk
JOIN documents AS document ON document.id = chunk.document_id
WHERE NOT EXISTS (
    SELECT 1
    FROM document_chunks_fts AS fts
    WHERE fts.chunk_id = chunk.id
);
"""

BM25_EXPRESSION = (
    "bm25(document_chunks_fts, 0.0, 0.0, 0.0, 1.5, 2.0, 1.0)"
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_FTS_OPERATORS = frozenset({"and", "or", "not", "near"})
_QUESTION_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "please",
        "show",
        "tell",
        "than",
        "that",
        "the",
        "this",
        "to",
        "vs",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
    }
)


def sanitize_fts5_query(raw_query: str) -> str:
    """Convert arbitrary natural language into a syntax-safe FTS5 OR query.

    Only Unicode alphanumeric tokens survive, so quotes, column selectors,
    wildcards, parentheses, slashes, hyphens and all FTS operators lose their
    special meaning.  Question stop words are removed when useful content terms
    remain; all-stop-word queries retain their non-operator words rather than
    unexpectedly becoming empty.
    """
    if not raw_query or not raw_query.strip():
        return ""

    raw_tokens = [
        token.casefold()[:64]
        for token in _TOKEN_RE.findall(raw_query)
        if token and len(token) <= 256
    ]
    non_operators = [token for token in raw_tokens if token not in _FTS_OPERATORS]
    content_tokens = [
        token for token in non_operators if token not in _QUESTION_STOP_WORDS
    ]
    selected = content_tokens or non_operators

    # Deduplicate without losing the user's term order and cap pathological
    # prompts before they can create an oversized MATCH expression.
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in selected:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
        if len(unique_tokens) >= 64:
            break

    # Tokens are already operator-free, but doubling quotes makes this safe if
    # tokenization is ever broadened in the future.
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique_tokens
    )


def initialize_fts5_schema(bind: Engine) -> None:
    """Create FTS5 storage/triggers and backfill existing relational chunks."""
    with bind.begin() as connection:
        connection.exec_driver_sql(CREATE_FTS_TABLE_SQL)
        for statement in CREATE_TRIGGER_SQL:
            connection.exec_driver_sql(statement)
        # Remove any legacy orphan before idempotent backfill.
        connection.exec_driver_sql(
            """
            DELETE FROM document_chunks_fts
            WHERE chunk_id NOT IN (SELECT id FROM document_chunks);
            """
        )
        connection.exec_driver_sql(BACKFILL_FTS_SQL)


class SQLiteSparseIndex(SparseIndexRepository):
    """Thread-safe async facade over SQLite's built-in FTS5 BM25 engine."""

    def __init__(self, bind: Engine | None = None) -> None:
        if bind is None:
            from app.storage.database import engine

            bind = engine
        self.bind = bind
        self._lock = threading.RLock()
        initialize_fts5_schema(self.bind)

    async def _run(self, operation):
        def locked():
            with self._lock:
                return operation()

        return await asyncio.to_thread(locked)

    async def index_chunks(
        self,
        knowledge_base_id: str,
        chunks: Sequence[DocumentChunkPayload],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not knowledge_base_id:
            raise ValueError("knowledge_base_id is required")
        for chunk in chunks:
            if str(chunk["knowledge_base_id"]) != str(knowledge_base_id):
                raise ValueError("Chunk knowledge_base_id does not match the index")

        def operation() -> None:
            with self.bind.begin() as connection:
                for chunk in chunks:
                    connection.execute(
                        text(
                            "DELETE FROM document_chunks_fts "
                            "WHERE chunk_id = :chunk_id"
                        ),
                        {"chunk_id": str(chunk["chunk_id"])},
                    )
                    connection.execute(
                        text(
                            """
                            INSERT INTO document_chunks_fts(
                                chunk_id, document_id, knowledge_base_id,
                                source_file, section_title, content
                            ) VALUES (
                                :chunk_id, :document_id, :knowledge_base_id,
                                :source_file, :section_title, :content
                            )
                            """
                        ),
                        {
                            "chunk_id": str(chunk["chunk_id"]),
                            "document_id": str(chunk["document_id"]),
                            "knowledge_base_id": str(knowledge_base_id),
                            "source_file": str(chunk["source_file"]),
                            "section_title": str(chunk.get("section_title") or ""),
                            "content": str(chunk["content"]),
                        },
                    )

        await self._run(operation)

    async def search_sparse(
        self,
        knowledge_base_id: str,
        query_text: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        if not knowledge_base_id:
            raise ValueError("knowledge_base_id is required")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        safe_query = sanitize_fts5_query(query_text)
        if not safe_query:
            return []

        document_ids: list[str] = []
        if filters:
            if filters.get("document_id") is not None:
                document_ids = [str(filters["document_id"])]
            elif filters.get("document_ids"):
                document_ids = [str(value) for value in filters["document_ids"]]

        sql = f"""
            SELECT
                document_chunks_fts.chunk_id AS chunk_id,
                document_chunks_fts.document_id AS document_id,
                document_chunks_fts.knowledge_base_id AS knowledge_base_id,
                document_chunks_fts.source_file AS source_file,
                document_chunks_fts.section_title AS section_title,
                document_chunks_fts.content AS content,
                chunk.page_number AS page_number,
                chunk.chunk_index AS chunk_index,
                chunk.is_table AS is_table,
                chunk.metadata_json AS metadata_json,
                {BM25_EXPRESSION} AS bm25_rank
            FROM document_chunks_fts
            LEFT JOIN document_chunks AS chunk
                ON chunk.id = document_chunks_fts.chunk_id
            WHERE document_chunks_fts MATCH :query
              AND document_chunks_fts.knowledge_base_id = :knowledge_base_id
        """
        if document_ids:
            sql += " AND document_chunks_fts.document_id IN :document_ids"
        sql += " ORDER BY bm25_rank ASC LIMIT :top_k"
        statement = text(sql)
        if document_ids:
            statement = statement.bindparams(bindparam("document_ids", expanding=True))

        parameters: dict[str, Any] = {
            "query": safe_query,
            "knowledge_base_id": str(knowledge_base_id),
            "top_k": int(top_k),
        }
        if document_ids:
            parameters["document_ids"] = document_ids

        def operation() -> list[SearchHit]:
            with self.bind.connect() as connection:
                rows = connection.execute(statement, parameters).mappings().all()
            results: list[SearchHit] = []
            for row in rows:
                raw_rank = float(row["bm25_rank"] or 0.0)
                raw_metadata = row["metadata_json"]
                if isinstance(raw_metadata, str):
                    try:
                        parsed_metadata = json.loads(raw_metadata)
                    except json.JSONDecodeError:
                        parsed_metadata = {}
                else:
                    parsed_metadata = dict(raw_metadata or {})
                result_metadata = {
                    **parsed_metadata,
                    "document_id": str(row["document_id"]),
                    "knowledge_base_id": str(row["knowledge_base_id"]),
                    "source_file": str(row["source_file"] or ""),
                    "section_title": row["section_title"] or None,
                    "page_number": row["page_number"],
                    "chunk_index": row["chunk_index"],
                    "is_table": bool(row["is_table"] or False),
                }
                results.append(
                    SearchHit(
                        chunk_id=str(row["chunk_id"]),
                        score=max(0.0, -raw_rank),
                        content=str(row["content"] or ""),
                        metadata=result_metadata,
                    )
                )
            return results

        return await self._run(operation)

    async def delete_by_document_id(
        self,
        knowledge_base_id: str,
        document_id: str,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        if not knowledge_base_id or not document_id:
            raise ValueError("knowledge_base_id and document_id are required")

        def operation() -> None:
            with self.bind.begin() as connection:
                connection.execute(
                    text(
                        """
                        DELETE FROM document_chunks_fts
                        WHERE knowledge_base_id = :knowledge_base_id
                          AND document_id = :document_id
                        """
                    ),
                    {
                        "knowledge_base_id": str(knowledge_base_id),
                        "document_id": str(document_id),
                    },
                )

        await self._run(operation)


# Descriptive alias for callers that prefer storage-engine terminology.
LocalSparseStore = SQLiteSparseIndex
SQLiteSparseStore = SQLiteSparseIndex
FTS5SparseStore = SQLiteSparseIndex
initialize_fts5 = initialize_fts5_schema


__all__ = [
    "BM25_EXPRESSION",
    "CREATE_FTS_TABLE_SQL",
    "FTS_TABLE",
    "FTS5SparseStore",
    "LocalSparseStore",
    "SQLiteSparseIndex",
    "SQLiteSparseStore",
    "initialize_fts5",
    "initialize_fts5_schema",
    "sanitize_fts5_query",
]
