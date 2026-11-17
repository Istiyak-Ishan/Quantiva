"""End-to-end retrieval orchestration.

Supports three modes:

``hybrid``
    Dense + sparse retrieval, weighted RRF fusion, optional reranking.
``dense_only``
    Embedding similarity search, optional reranking.
``sparse_only``
    BM25/FTS5 keyword search, optional reranking.

No concrete provider, vector database, or FTS implementation is imported here.
The embedding provider, dense repository, sparse repository, and reranker are
all injected through their abstract interfaces.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.reranker import RerankerProvider
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranking import RerankingStage
from app.retrieval.sparse import SparseRetriever
from app.retrieval.types import (
    CitationAnchor,
    ContextBlock,
    RETRIEVAL_MODES,
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
)
from app.storage.interfaces import SparseIndexRepository, VectorStoreRepository

if TYPE_CHECKING:
    from app.providers.registry import IndexMetadata


def build_context_blocks(chunks: Sequence[RetrievedChunk]) -> list[ContextBlock]:
    """Render reranked chunks as citation-anchored context blocks."""
    return [
        ContextBlock(
            citation=CitationAnchor.from_chunk(chunk).format(),
            content=chunk.content,
            chunk=chunk,
        )
        for chunk in chunks
    ]


class RetrievalPipeline:
    """Provider-agnostic hybrid retrieval pipeline."""

    def __init__(
        self,
        mode: RetrievalMode = "hybrid",
        *,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStoreRepository | None = None,
        sparse_index: SparseIndexRepository | None = None,
        index_metadata: IndexMetadata | None = None,
        reranker: RerankerProvider | None = None,
        rerank_enabled: bool = True,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        initial_top_k: int = 25,
        final_top_k: int = 8,
    ) -> None:
        self.mode = self._validate_mode(mode)
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("RRF weights must be non-negative")
        if dense_weight == 0 and sparse_weight == 0:
            raise ValueError("At least one RRF weight must be positive")
        for name, value in (
            ("dense_top_k", dense_top_k),
            ("sparse_top_k", sparse_top_k),
            ("initial_top_k", initial_top_k),
            ("final_top_k", final_top_k),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.index_metadata = index_metadata
        self.rrf_k = int(rrf_k)
        self.dense_weight = float(dense_weight)
        self.sparse_weight = float(sparse_weight)
        self.dense_top_k = int(dense_top_k)
        self.sparse_top_k = int(sparse_top_k)
        self.initial_top_k = int(initial_top_k)
        self.final_top_k = int(final_top_k)

        self._dense: DenseRetriever | None = None
        if (
            embedding_provider is not None
            and vector_store is not None
            and index_metadata is not None
        ):
            self._dense = DenseRetriever(
                embedding_provider,
                vector_store,
                index_metadata,
            )
        self._sparse = (
            SparseRetriever(sparse_index) if sparse_index is not None else None
        )
        self._reranking = RerankingStage(reranker, enabled=rerank_enabled)

        self._require_dependencies(self.mode)

    # ── Validation / resolution ─────────────────────────────────────────────

    @staticmethod
    def _validate_mode(mode: str) -> RetrievalMode:
        normalised = str(mode).strip().lower()
        if normalised not in RETRIEVAL_MODES:
            supported = ", ".join(sorted(RETRIEVAL_MODES))
            raise ValueError(
                f"Unsupported retrieval mode '{mode}'. Expected one of: {supported}"
            )
        return cast(RetrievalMode, normalised)

    def _require_dependencies(self, mode: RetrievalMode) -> None:
        if mode in ("hybrid", "dense_only"):
            if self._dense is None:
                raise ValueError(
                    f"Retrieval mode '{mode}' requires embedding_provider, "
                    "vector_store and index_metadata"
                )
        if mode in ("hybrid", "sparse_only") and self._sparse is None:
            raise ValueError(
                f"Retrieval mode '{mode}' requires a sparse_index repository"
            )

    def _resolve_knowledge_base_id(
        self, knowledge_base_id: str | None
    ) -> str | None:
        if knowledge_base_id:
            return str(knowledge_base_id)
        metadata = self.index_metadata
        if metadata is not None:
            value = getattr(metadata, "knowledge_base_id", None)
            if value:
                return str(value)
        return None

    def _resolve_collection_name(
        self,
        collection_name: str | None,
        knowledge_base_id: str | None,
    ) -> str:
        if collection_name:
            return str(collection_name)
        metadata = self.index_metadata
        if metadata is not None:
            value = getattr(metadata, "collection_name", None)
            if value:
                return str(value)
        if knowledge_base_id:
            return knowledge_base_id
        raise ValueError(
            "collection_name is required for dense retrieval (pass it "
            "explicitly or set it on IndexMetadata)"
        )

    # ── Main entry point ────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        *,
        knowledge_base_id: str | None = None,
        collection_name: str | None = None,
        filters: Mapping[str, object] | None = None,
        mode: RetrievalMode | None = None,
    ) -> RetrievalResult:
        """Run the configured retrieval mode and return citation-ready context."""
        effective_mode = self._validate_mode(mode) if mode is not None else self.mode
        self._require_dependencies(effective_mode)

        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")

        kb_id = self._resolve_knowledge_base_id(knowledge_base_id)
        if effective_mode in ("hybrid", "sparse_only") and not kb_id:
            raise ValueError(
                "knowledge_base_id is required for sparse retrieval (pass it "
                "explicitly or set it on IndexMetadata)"
            )

        candidates: list[RetrievedChunk]
        if effective_mode == "hybrid":
            assert self._dense is not None and self._sparse is not None
            resolved_collection = self._resolve_collection_name(
                collection_name, kb_id
            )
            dense_results, sparse_results = await asyncio.gather(
                self._dense.retrieve(
                    query,
                    collection_name=resolved_collection,
                    top_k=self.dense_top_k,
                    filters=filters,
                ),
                self._sparse.retrieve(
                    query,
                    knowledge_base_id=kb_id,
                    top_k=self.sparse_top_k,
                    filters=filters,
                ),
            )
            candidates = reciprocal_rank_fusion(
                dense_results,
                sparse_results,
                k=self.rrf_k,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
                top_k=self.initial_top_k,
            )
        elif effective_mode == "dense_only":
            assert self._dense is not None
            resolved_collection = self._resolve_collection_name(
                collection_name, kb_id
            )
            dense_results = await self._dense.retrieve(
                query,
                collection_name=resolved_collection,
                top_k=self.dense_top_k,
                filters=filters,
            )
            candidates = dense_results[: self.initial_top_k]
        else:
            assert self._sparse is not None
            sparse_results = await self._sparse.retrieve(
                query,
                knowledge_base_id=kb_id,
                top_k=self.sparse_top_k,
                filters=filters,
            )
            candidates = sparse_results[: self.initial_top_k]

        final_chunks = await self._reranking.apply(
            query,
            candidates,
            top_k=self.final_top_k,
        )
        context_blocks = build_context_blocks(final_chunks)

        return RetrievalResult(
            query=query,
            mode=effective_mode,
            chunks=final_chunks,
            context_blocks=context_blocks,
            candidates_considered=len(candidates),
            reranked=self._reranking.active and bool(candidates),
        )

    async def search(
        self,
        query: str,
        *,
        knowledge_base_id: str | None = None,
        collection_name: str | None = None,
        filters: Mapping[str, object] | None = None,
        mode: RetrievalMode | None = None,
    ) -> RetrievalResult:
        """Alias for :meth:`retrieve` for callers using search terminology."""
        return await self.retrieve(
            query,
            knowledge_base_id=knowledge_base_id,
            collection_name=collection_name,
            filters=filters,
            mode=mode,
        )


__all__ = ["RetrievalPipeline", "build_context_blocks"]
