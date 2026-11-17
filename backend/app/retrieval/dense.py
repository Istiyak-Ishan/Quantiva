"""Dense (embedding) retrieval stage.

The retriever owns no provider SDK and no vector-database client.  It receives
an :class:`~app.providers.interfaces.embedding.EmbeddingProvider` and a
:class:`~app.storage.interfaces.VectorStoreRepository`, validates the query
vector against the knowledge base's locked :class:`IndexMetadata`, and only
then issues the search.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.exceptions import EmbeddingDimensionMismatchError
from app.retrieval.types import RetrievedChunk, chunk_from_hit
from app.storage.interfaces import VectorStoreRepository

if TYPE_CHECKING:
    from app.providers.registry import IndexMetadata


class DenseRetriever:
    """Embed a query and retrieve nearest chunks from a vector repository."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreRepository,
        index_metadata: IndexMetadata,
    ) -> None:
        if embedding_provider is None:
            raise ValueError("An embedding_provider is required")
        if vector_store is None:
            raise ValueError("A vector_store is required")
        if index_metadata is None:
            raise ValueError("index_metadata is required")
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.index_metadata = index_metadata

    @property
    def expected_dimensions(self) -> int:
        dimensions = int(self.index_metadata.embedding_dimensions)
        if dimensions <= 0:
            raise ValueError("IndexMetadata.embedding_dimensions must be positive")
        return dimensions

    async def retrieve(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        top_k: int = 20,
        filters: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """Return dense candidates ordered by descending similarity."""
        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        expected = self.expected_dimensions
        declared = int(self.embedding_provider.get_dimensions())
        if declared != expected:
            raise EmbeddingDimensionMismatchError(
                expected=expected,
                actual=declared,
                provider=self.embedding_provider.provider_name(),
                model=self.embedding_provider.model_name(),
            )

        query_vector = await self.embedding_provider.embed_query(query)
        actual = len(query_vector)
        if actual != expected:
            raise EmbeddingDimensionMismatchError(
                expected=expected,
                actual=actual,
                provider=self.embedding_provider.provider_name(),
                model=self.embedding_provider.model_name(),
            )

        resolved_collection = collection_name or getattr(
            self.index_metadata, "collection_name", None
        )
        if not resolved_collection:
            raise ValueError(
                "collection_name is required (pass it explicitly or set it on "
                "IndexMetadata)"
            )

        hits = await self.vector_store.search_dense(
            str(resolved_collection),
            list(query_vector),
            top_k,
            filters,
        )
        return [
            chunk_from_hit(hit, dense_rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        top_k: int = 20,
        filters: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        """Alias for :meth:`retrieve` for callers using search terminology."""
        return await self.retrieve(
            query,
            collection_name=collection_name,
            top_k=top_k,
            filters=filters,
        )


__all__ = ["DenseRetriever"]
