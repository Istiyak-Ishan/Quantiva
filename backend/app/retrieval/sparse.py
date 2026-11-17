"""Sparse (BM25 / FTS5) retrieval stage.

Query normalisation and tokenisation live here so every sparse adapter receives
the same deterministic term stream regardless of whether it is backed by
SQLite FTS5, an in-memory BM25 index, or another local engine.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from app.retrieval.types import RetrievedChunk, chunk_from_hit
from app.storage.interfaces import SparseIndexRepository

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_query(text: str) -> str:
    """Unicode-normalise, case-fold and collapse whitespace in *text*."""
    if not text:
        return ""
    normalised = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalised.split())


def tokenize_query(text: str) -> list[str]:
    """Return deterministic lowercase alphanumeric query terms."""
    return _TOKEN_RE.findall(normalize_query(text))


def build_sparse_query(text: str) -> str:
    """Return the normalised term string passed to a sparse repository."""
    return " ".join(tokenize_query(text))


class SparseRetriever:
    """Normalise a query and retrieve BM25 keyword matches."""

    def __init__(self, sparse_index: SparseIndexRepository) -> None:
        if sparse_index is None:
            raise ValueError("A sparse_index repository is required")
        self.sparse_index = sparse_index

    async def retrieve(
        self,
        query: str,
        *,
        knowledge_base_id: str | None = None,
        top_k: int = 20,
        filters: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """Return sparse candidates ordered by descending BM25 score."""
        if not knowledge_base_id:
            raise ValueError("knowledge_base_id is required for sparse retrieval")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        sparse_query = build_sparse_query(query)
        if not sparse_query:
            return []

        hits = await self.sparse_index.search_sparse(
            str(knowledge_base_id),
            sparse_query,
            top_k,
            filters,
        )
        return [
            chunk_from_hit(hit, sparse_rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]

    async def search(
        self,
        query: str,
        *,
        knowledge_base_id: str | None = None,
        top_k: int = 20,
        filters: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """Alias for :meth:`retrieve` for callers using search terminology."""
        return await self.retrieve(
            query,
            knowledge_base_id=knowledge_base_id,
            top_k=top_k,
            filters=filters,
        )


__all__ = [
    "SparseRetriever",
    "build_sparse_query",
    "normalize_query",
    "tokenize_query",
]
