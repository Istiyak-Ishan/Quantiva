"""
app/providers/embeddings/ollama.py
──────────────────────────────────────────────────────────────────────────────
Local Ollama text-embedding provider adapter.

Uses Ollama's batch ``/api/embed`` endpoint when available and automatically
falls back to the legacy single-input ``/api/embeddings`` endpoint on older
Ollama builds.  Output dimensions are detected from the first successful
response unless explicitly configured.

Conforms to :class:`~app.providers.interfaces.embedding.EmbeddingProvider`.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import httpx

from app.config import settings
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.exceptions import (
    ProviderError,
)
from app.providers.interfaces.types import ConnectionTestResult
from app.providers.ollama_common import (
    is_model_not_found_response,
    normalize_base_url,
    raise_for_ollama_status,
    translate_transport_error,
)


class _BatchEndpointUnavailable(Exception):
    """Internal signal: this Ollama build lacks the batch /api/embed route."""


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding adapter implementing the unified EmbeddingProvider interface."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float | None = None,
        connect_timeout_seconds: float | None = None,
    ) -> None:
        self._base_url = normalize_base_url(
            base_url or getattr(settings, "OLLAMA_BASE_URL", None)
        )
        self._model_name = (
            model
            or getattr(settings, "OLLAMA_EMBEDDING_MODEL", None)
            or "nomic-embed-text"
        )
        # 0 means "not yet detected"; the first successful vector locks it.
        self._dimensions = int(dimensions) if dimensions else 0
        self._client = client
        self._transport = transport
        self._owns_client = client is None
        self._timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 120.0)
        )
        self._connect_timeout_seconds = float(
            connect_timeout_seconds
            if connect_timeout_seconds is not None
            else getattr(settings, "OLLAMA_CONNECT_TIMEOUT_SECONDS", 5.0)
        )
        # None = unknown, True = batch endpoint works, False = legacy only.
        self._batch_endpoint_supported: bool | None = None

    # ── Identity & dimensions ─────────────────────────────────────────────────

    def provider_name(self) -> str:
        return "ollama"

    def model_name(self) -> str:
        return self._model_name

    def base_url(self) -> str:
        return self._base_url

    def get_dimensions(self) -> int:
        """Return detected dimensions, or ``0`` until the first embedding call."""
        return self._dimensions

    # ── HTTP client lifecycle ─────────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                self._timeout_seconds,
                connect=self._connect_timeout_seconds,
            )
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=timeout,
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ── Low-level HTTP helpers ────────────────────────────────────────────────

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        client = self._get_client()
        try:
            return await client.post(
                f"{self._base_url}{path}",
                json=payload,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise translate_transport_error(
                exc, base_url=self._base_url, model=self._model_name
            ) from exc

    @staticmethod
    def _decode_json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "Ollama returned invalid JSON from the embedding endpoint.",
                provider="ollama",
                retriable=True,
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(
                "Ollama returned an unexpected embedding payload.",
                provider="ollama",
                retriable=True,
            )
        return data

    # ── Endpoint strategies ───────────────────────────────────────────────────

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the modern batch endpoint or signal an unsupported route."""
        response = await self._post_json(
            "/api/embed",
            {"model": self._model_name, "input": texts},
        )

        if response.status_code in (404, 405):
            # A missing *model* is a 404 too; distinguish it from an older
            # Ollama build that simply does not expose /api/embed.
            if is_model_not_found_response(
                response.status_code, response.text
            ):
                raise_for_ollama_status(
                    response.status_code,
                    response.text,
                    base_url=self._base_url,
                    model=self._model_name,
                )
            raise _BatchEndpointUnavailable()

        if not response.is_success:
            raise_for_ollama_status(
                response.status_code,
                response.text,
                base_url=self._base_url,
                model=self._model_name,
            )

        data = self._decode_json(response)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ProviderError(
                (
                    "Ollama /api/embed returned "
                    f"{len(embeddings) if isinstance(embeddings, list) else 0} "
                    f"embeddings for a batch of {len(texts)}."
                ),
                provider="ollama",
                model=self._model_name,
                retriable=True,
            )

        vectors: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list):
                raise ProviderError(
                    "Ollama returned a malformed embedding vector.",
                    provider="ollama",
                    model=self._model_name,
                    retriable=True,
                )
            vectors.append(self._normalize_vector(embedding))
        self._batch_endpoint_supported = True
        return vectors

    async def _embed_legacy_single(self, text: str) -> list[float]:
        """Call the legacy single-input /api/embeddings endpoint."""
        response = await self._post_json(
            "/api/embeddings",
            {"model": self._model_name, "prompt": text},
        )

        if not response.is_success:
            raise_for_ollama_status(
                response.status_code,
                response.text,
                base_url=self._base_url,
                model=self._model_name,
            )

        data = self._decode_json(response)
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise ProviderError(
                "Ollama /api/embeddings returned no embedding vector.",
                provider="ollama",
                model=self._model_name,
                retriable=True,
            )
        self._batch_endpoint_supported = False
        return self._normalize_vector(embedding)

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """Validate, dimension-check, and L2-normalise one embedding vector."""
        if not vector:
            raise ProviderError(
                "Ollama returned an empty embedding vector.",
                provider="ollama",
                model=self._model_name,
            )

        values: list[float] = []
        for value in vector:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ProviderError(
                    "Ollama returned a non-numeric embedding value.",
                    provider="ollama",
                    model=self._model_name,
                ) from exc
            if not math.isfinite(numeric):
                raise ProviderError(
                    "Ollama returned non-finite embedding values.",
                    provider="ollama",
                    model=self._model_name,
                )
            values.append(numeric)

        if self._dimensions == 0:
            self._dimensions = len(values)
        else:
            # Raises EmbeddingDimensionMismatchError before retrieval proceeds.
            self.validate_dimensions(values)

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise ProviderError(
                "Ollama returned a zero-magnitude embedding vector.",
                provider="ollama",
                model=self._model_name,
            )
        return [value / norm for value in values]

    # ── Embedding API ─────────────────────────────────────────────────────────

    async def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("texts list must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"texts[{index}] must be a non-empty string"
                )

        if self._batch_endpoint_supported is not False:
            vectors: list[list[float]] = []
            try:
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    vectors.extend(await self._embed_batch(batch))
                return vectors
            except _BatchEndpointUnavailable:
                # Older Ollama: discard partial results and redo every text
                # through the legacy endpoint so ordering is preserved.
                self._batch_endpoint_supported = False

        return [await self._embed_legacy_single(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        vectors = await self.embed_documents([query])
        return vectors[0]

    # ── Connectivity probe ────────────────────────────────────────────────────

    async def test_embedding(self) -> ConnectionTestResult:
        start_time = time.perf_counter()
        try:
            probe_vector = await self.embed_query("Quantiva connection test")
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=True,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                detected_dimensions=len(probe_vector),
            )
        except ProviderError as err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                detected_dimensions=None,
                error_code=type(err).__name__,
                error_message=err.user_message,
            )
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                detected_dimensions=None,
                error_code="UnknownError",
                error_message=str(exc),
            )
