"""Configurable reranking stage.

The stage is a thin adapter between :class:`RetrievedChunk` candidates and the
provider-agnostic :class:`~app.providers.interfaces.reranker.RerankerProvider`.
When reranking is disabled (or no provider is configured) the upstream order is
preserved and only ``final_top_k`` is applied.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from app.providers.interfaces.reranker import (
    RerankDocument,
    RerankerProvider,
)
from app.retrieval.types import RetrievedChunk


class RerankingStage:
    """Apply an optional cross-encoder reranker to fused candidates."""

    def __init__(
        self,
        reranker: RerankerProvider | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.reranker = reranker
        self.enabled = bool(enabled)

    @property
    def active(self) -> bool:
        """True only when a provider is configured and reranking is enabled."""
        return self.enabled and self.reranker is not None

    async def apply(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        """Return the final ``top_k`` chunks, reranked when active."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidates = list(candidates)
        if not candidates:
            return []

        if not self.active:
            return [
                replace(chunk, final_rank=rank, rerank_score=None)
                for rank, chunk in enumerate(candidates[:top_k], start=1)
            ]

        assert self.reranker is not None  # narrowed by ``self.active``
        documents = [
            RerankDocument(
                id=chunk.chunk_id,
                content=chunk.content,
                metadata=dict(chunk.metadata),
                initial_score=chunk.score,
            )
            for chunk in candidates
        ]
        results = await self.reranker.rerank(query, documents, top_k=top_k)

        by_id = {chunk.chunk_id: chunk for chunk in candidates}
        ordered_results = sorted(
            results,
            key=lambda result: (-float(result.rerank_score), result.id),
        )

        reranked: list[RetrievedChunk] = []
        seen: set[str] = set()
        for result in ordered_results:
            if result.id in seen:
                continue
            base = by_id.get(result.id)
            if base is None:
                continue
            seen.add(result.id)

            metadata = dict(base.metadata)
            for key, value in result.metadata.items():
                if key not in metadata or metadata[key] is None or metadata[key] == "":
                    metadata[key] = value

            reranked.append(
                replace(
                    base,
                    content=base.content or result.content,
                    metadata=metadata,
                    rerank_score=float(result.rerank_score),
                    final_rank=len(reranked) + 1,
                )
            )

        if results and not reranked:
            raise ValueError(
                "Reranker returned no chunk_ids matching the retrieval candidates"
            )
        return reranked


__all__ = ["RerankingStage"]
