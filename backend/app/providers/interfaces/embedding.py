"""
app/providers/interfaces/embedding.py
──────────────────────────────────────────────────────────────────────────────
Abstract base class for all embedding provider implementations.

Embedding providers are strictly decoupled from LLM providers so that a
knowledge base can be indexed with a different model than the one used for
chat.  Every ``KnowledgeBase`` (future) locks its ``embedding_provider``,
``embedding_model_name``, and ``embedding_dimensions`` into ``IndexMetadata``.

Querying a knowledge base with a mismatched embedding dimension must raise
:class:`~exceptions.EmbeddingDimensionMismatchError` before any vector-store
operation occurs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.providers.interfaces.types import ConnectionTestResult


class EmbeddingProvider(ABC):
    """Provider-agnostic interface for text embedding.

    Implementations live in ``app/providers/<provider>/`` and must import ONLY
    from this interfaces package plus their own SDK.

    Thread-safety
    -------------
    Implementations must be safe for concurrent coroutine access.  Expensive
    model loading (e.g. loading a local SentenceTransformer model from disk)
    must be deferred and serialised with a threading lock — not performed at
    import time.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. ``"gemini"`` or ``"sentence_transformers"``."""

    @abstractmethod
    def model_name(self) -> str:
        """Active embedding model identifier, e.g. ``"gemini-embedding-001"``."""

    # ── Dimensions ────────────────────────────────────────────────────────────

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the fixed output vector length for the active model.

        This value is used to validate compatibility with an existing vector
        collection before any embedding call is made.  Implementations must
        return a known, non-zero value.
        """

    # ── Embedding operations ──────────────────────────────────────────────────

    @abstractmethod
    async def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed a list of document texts for indexing.

        Parameters
        ----------
        texts:
            Non-empty list of strings to embed.  Empty strings should be
            rejected with ``ValueError``.
        batch_size:
            Maximum number of texts to send in a single API call.
            Implementations must handle batching internally.

        Returns
        -------
        list[list[float]]
            L2-normalised embedding vectors in the same order as ``texts``.
            Each inner list has length equal to :meth:`get_dimensions`.

        Raises
        ------
        EmbeddingDimensionMismatchError
            If the returned vectors do not match the declared dimension.
        ProviderConnectionError
            Network failure or local provider not running.
        ProviderAuthenticationError
            API key invalid.
        ProviderRateLimitError
            Quota exceeded.
        """

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for similarity search.

        Many providers/models use a different task type for query embedding vs.
        document embedding (e.g. ``"RETRIEVAL_QUERY"`` vs.
        ``"RETRIEVAL_DOCUMENT"`` in Gemini).  Implementations must apply the
        correct task type automatically.

        Parameters
        ----------
        query:
            Non-empty query string.

        Returns
        -------
        list[float]
            L2-normalised embedding vector of length :meth:`get_dimensions`.
        """

    # ── Connectivity probe ────────────────────────────────────────────────────

    @abstractmethod
    async def test_embedding(self) -> ConnectionTestResult:
        """Probe provider connectivity and return the actual vector length.

        Sends the fixed probe text ``"Quantiva connection test"`` through the
        full embedding path and reports:

        * ``success`` — whether the call completed without error.
        * ``latency_ms`` — round-trip time in milliseconds.
        * ``detected_dimensions`` — length of the returned vector (must match
          :meth:`get_dimensions`).
        * ``error_code`` / ``error_message`` — populated only on failure.

        Must never raise — all failures are reported via ``success=False``.

        Used by the UI settings panel to validate embedding configuration and
        to alert the user to dimension mismatches before ingestion begins.
        """

    # ── Utility ───────────────────────────────────────────────────────────────

    def validate_dimensions(self, vector: list[float]) -> None:
        """Assert that *vector* matches the declared dimension.

        Raises
        ------
        EmbeddingDimensionMismatchError
            If ``len(vector) != self.get_dimensions()``.
        """
        from app.providers.interfaces.exceptions import EmbeddingDimensionMismatchError

        expected = self.get_dimensions()
        actual = len(vector)
        if actual != expected:
            raise EmbeddingDimensionMismatchError(
                expected=expected,
                actual=actual,
                provider=self.provider_name(),
                model=self.model_name(),
            )
