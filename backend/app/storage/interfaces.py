"""Abstract repository interfaces for dense and sparse chunk storage.

This module is intentionally free of any vector-database, FTS5, or provider
SDK import.  Retrieval code depends only on these contracts; the application
context supplies adapters backed by the singleton embedded Qdrant/LanceDB
client and the local SQLite FTS5/BM25 index.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.types import DocumentChunkPayload


# ── Result / metadata types ───────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SearchHit:
    """One scored chunk returned by a dense or sparse repository.

    ``metadata`` must contain at least the citation fields available to the
    adapter (``source_file``, ``page_number`` and ``chunk_index`` are preferred
    names).  ``content`` is the full chunk text used for reranking and context
    construction; adapters should never return a truncated preview here.
    """

    chunk_id: str
    score: float
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


#: Semantic aliases used by each side of the hybrid retriever.
VectorSearchResult = SearchHit
SparseSearchResult = SearchHit


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """Minimal description of a dense vector collection."""

    name: str
    dimensions: int
    distance_metric: str = "cosine"
    point_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Dense vector repository ───────────────────────────────────────────────────

class VectorStoreRepository(ABC):
    """Persistence contract for dense chunk vectors.

    Implementations must be safe for concurrent coroutine access and must use
    the application-wide singleton client rather than opening a new embedded
    database connection per call.
    """

    @abstractmethod
    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[DocumentChunkPayload],
        embeddings: Sequence[Sequence[float]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """Insert or update *chunks* with their corresponding *embeddings*.

        Returns the dense point identifiers in the same order as *chunks*.
        Implementations must reject length mismatches and non-finite vectors.
        """

    @abstractmethod
    async def search_dense(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return the *top_k* nearest chunks ordered by descending similarity.

        ``filters`` is an adapter-defined mapping (for example
        ``{"knowledge_base_id": ..., "document_ids": [...]}``).  It must be
        applied inside the storage layer, never after the search.
        """

    @abstractmethod
    async def delete_by_document_id(
        self,
        collection_name: str,
        document_id: str,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        """Delete every dense point belonging to *document_id*."""

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """Delete an entire collection.  Missing collections are a no-op."""

    @abstractmethod
    async def get_collection_info(
        self,
        collection_name: str,
    ) -> CollectionInfo | None:
        """Return collection metadata, or ``None`` when it does not exist."""


# ── Sparse / BM25 repository ──────────────────────────────────────────────────

class SparseIndexRepository(ABC):
    """Persistence contract for local BM25/FTS5 keyword indexes."""

    @abstractmethod
    async def index_chunks(
        self,
        knowledge_base_id: str,
        chunks: Sequence[DocumentChunkPayload],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Add or replace *chunks* in the knowledge base's sparse index."""

    @abstractmethod
    async def search_sparse(
        self,
        knowledge_base_id: str,
        query_text: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return the *top_k* keyword matches ordered by descending BM25 score.

        ``query_text`` has already been normalised and tokenised by
        :class:`app.retrieval.sparse.SparseRetriever`.
        """

    @abstractmethod
    async def delete_by_document_id(
        self,
        knowledge_base_id: str,
        document_id: str,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        """Delete every sparse entry belonging to *document_id*."""


__all__ = [
    "CollectionInfo",
    "SearchHit",
    "SparseIndexRepository",
    "SparseSearchResult",
    "VectorSearchResult",
    "VectorStoreRepository",
]
