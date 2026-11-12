"""
app/providers/interfaces/reranker.py
──────────────────────────────────────────────────────────────────────────────
Abstract base class for all reranker provider implementations.

Rerankers are invoked after the dense + sparse hybrid retrieval merge step to
reorder candidates by cross-attention relevance.  The existing
``CrossEncoderReranker`` (``cross-encoder/ms-marco-MiniLM-L-6-v2``) implements
this interface as a local sentence-transformers provider.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.providers.interfaces.types import ConnectionTestResult


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RerankDocument:
    """An input candidate for reranking.

    ``id`` is used to correlate :class:`RerankResult` entries back to the
    original retrieval results without relying on list position.
    """

    id: str
    """Unique identifier for the candidate chunk (e.g. Qdrant point UUID)."""

    content: str
    """Full chunk text that will be compared to the query by the cross-encoder."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value metadata carried through unchanged."""

    initial_score: float = 0.0
    """Score assigned by the upstream retrieval stage (e.g. RRF score)."""


@dataclass
class RerankResult:
    """A single reranked output document.

    Results are returned in descending order of :attr:`rerank_score` by all
    implementations.
    """

    id: str
    """Matches the ``id`` of the input :class:`RerankDocument`."""

    content: str
    """Chunk text (unchanged from the input document)."""

    metadata: dict[str, Any]
    """Metadata carried through from the input document (unchanged)."""

    rerank_score: float
    """Cross-encoder relevance score (higher = more relevant)."""

    initial_score: float
    """Original upstream retrieval score preserved for comparison / audit."""

    rank: int
    """Zero-based position in the reranked output list."""


# ── Abstract provider ─────────────────────────────────────────────────────────

class RerankerProvider(ABC):
    """Provider-agnostic interface for cross-encoder reranking.

    Thread-safety
    -------------
    Model loading must be deferred and serialised with a threading lock.
    The model must not be loaded at import time, as the first request may
    arrive before the model weights are on disk.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. ``"cross_encoder"``."""

    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier, e.g. ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``."""

    # ── Reranking ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[RerankDocument],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """Rerank *documents* by cross-attention relevance to *query*.

        Parameters
        ----------
        query:
            The user's search query string.
        documents:
            Candidate documents from the upstream hybrid retrieval step.
        top_k:
            Maximum number of results to return.  The returned list is
            truncated to the top *top_k* documents by :attr:`RerankResult.rerank_score`.

        Returns
        -------
        list[RerankResult]
            Results sorted by ``rerank_score`` descending, length <= *top_k*.

        Raises
        ------
        ProviderConnectionError
            If a remote reranker service is unavailable.
        """

    # ── Connectivity probe ────────────────────────────────────────────────────

    @abstractmethod
    async def test_reranker(self) -> ConnectionTestResult:
        """Probe reranker availability with a minimal synthetic request.

        Sends a fixed query and two synthetic documents through the full
        reranking path.  Reports latency and any errors via
        :class:`~types.ConnectionTestResult`.

        Must never raise — all failures are reported via ``success=False``.
        """
