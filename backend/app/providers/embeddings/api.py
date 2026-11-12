"""
app/providers/embeddings/api.py
──────────────────────────────────────────────────────────────────────────────
OpenAI-compatible embedding adapter for OpenAI, LM Studio, llama.cpp server,
and custom ``/v1/embeddings`` endpoints.

Handles batching, dynamic output-dimension detection, and L2 normalisation.

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
    ProviderAuthenticationError,
    ProviderError,
)
from app.providers.interfaces.types import ConnectionTestResult
from app.providers.openai_common import (
    DEFAULT_LLAMA_CPP_BASE_URL,
    DEFAULT_LM_STUDIO_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    build_openai_headers,
    normalize_openai_base_url,
    raise_for_openai_status,
    translate_openai_transport_error,
)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Embedding adapter for any OpenAI-compatible ``/v1/embeddings`` API."""

    def __init__(
        self,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 10.0,
        dimensions: int | None = None,
        provider_label: str = "openai_compatible",
        is_local: bool = False,
        requires_api_key: bool = False,
        send_dimensions: bool = False,
    ) -> None:
        self._base_url = normalize_openai_base_url(base_url)
        self._model_name = model
        if api_key is not None:
            self._api_key = api_key
        elif requires_api_key or provider_label == "openai":
            self._api_key = getattr(settings, "OPENAI_API_KEY", None)
        else:
            self._api_key = None

        self._custom_headers = dict(headers or {})
        self._client = client
        self._transport = transport
        self._owns_client = client is None
        self._timeout_seconds = float(timeout_seconds)
        self._connect_timeout_seconds = float(connect_timeout_seconds)
        # 0 means "not yet detected"; the first vector locks the dimension.
        self._dimensions = int(dimensions) if dimensions else 0
        self._provider_label = provider_label
        self._is_local = is_local
        self._requires_api_key = requires_api_key
        self._send_dimensions = send_dimensions

    # ── Identity & dimensions ─────────────────────────────────────────────────

    def provider_name(self) -> str:
        return self._provider_label

    def model_name(self) -> str:
        return self._model_name

    def base_url(self) -> str:
        return self._base_url

    def get_dimensions(self) -> int:
        """Return detected dimensions, or ``0`` until the first embedding call."""
        return self._dimensions

    # ── HTTP lifecycle ────────────────────────────────────────────────────────

    def _embeddings_url(self) -> str:
        return f"{self._base_url}/embeddings"

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

    def _request_headers(self) -> dict[str, str]:
        if self._requires_api_key and not self._api_key:
            raise ProviderAuthenticationError(
                (
                    f"An API key is required for '{self._provider_label}'. "
                    "Add it in Settings or set OPENAI_API_KEY."
                ),
                provider=self._provider_label,
                model=self._model_name,
            )
        return build_openai_headers(self._api_key, self._custom_headers)

    # ── Embedding request ─────────────────────────────────────────────────────

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self._send_dimensions and self._dimensions:
            payload["dimensions"] = self._dimensions

        headers = self._request_headers()
        client = self._get_client()
        try:
            response = await client.post(
                self._embeddings_url(),
                json=payload,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            raise translate_openai_transport_error(
                exc,
                provider=self._provider_label,
                base_url=self._base_url,
                model=self._model_name,
                is_local=self._is_local,
            ) from exc

        if not response.is_success:
            raise_for_openai_status(
                response.status_code,
                response.text,
                provider=self._provider_label,
                model=self._model_name,
                base_url=self._base_url,
            )

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"{self._provider_label} returned invalid JSON from /embeddings.",
                provider=self._provider_label,
                model=self._model_name,
                retriable=True,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError(
                f"{self._provider_label} returned an unexpected embeddings payload.",
                provider=self._provider_label,
                model=self._model_name,
                retriable=True,
            )

        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise ProviderError(
                (
                    f"{self._provider_label} returned "
                    f"{len(items) if isinstance(items, list) else 0} embeddings "
                    f"for a batch of {len(texts)}."
                ),
                provider=self._provider_label,
                model=self._model_name,
                retriable=True,
            )

        # OpenAI guarantees order via ``index``; sort defensively for proxies.
        if all(
            isinstance(item, dict) and isinstance(item.get("index"), int)
            for item in items
        ):
            items = sorted(items, key=lambda item: item["index"])

        vectors: list[list[float]] = []
        for item in items:
            if not isinstance(item, dict):
                raise ProviderError(
                    f"{self._provider_label} returned a malformed embedding item.",
                    provider=self._provider_label,
                    model=self._model_name,
                    retriable=True,
                )
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise ProviderError(
                    f"{self._provider_label} returned no embedding vector.",
                    provider=self._provider_label,
                    model=self._model_name,
                    retriable=True,
                )
            vectors.append(self._normalize_vector(embedding))

        return vectors

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """Validate, dimension-check, and L2-normalise one embedding vector."""
        if not vector:
            raise ProviderError(
                f"{self._provider_label} returned an empty embedding vector.",
                provider=self._provider_label,
                model=self._model_name,
            )

        values: list[float] = []
        for value in vector:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ProviderError(
                    f"{self._provider_label} returned a non-numeric embedding value.",
                    provider=self._provider_label,
                    model=self._model_name,
                ) from exc
            if not math.isfinite(numeric):
                raise ProviderError(
                    f"{self._provider_label} returned non-finite embedding values.",
                    provider=self._provider_label,
                    model=self._model_name,
                )
            values.append(numeric)

        if self._dimensions == 0:
            self._dimensions = len(values)
        else:
            self.validate_dimensions(values)

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise ProviderError(
                f"{self._provider_label} returned a zero-magnitude embedding vector.",
                provider=self._provider_label,
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
                raise ValueError(f"texts[{index}] must be a non-empty string")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

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

    # ── Preset factory constructors ───────────────────────────────────────────

    @classmethod
    def for_openai(
        cls,
        model: str = "text-embedding-3-small",
        **kwargs: Any,
    ) -> OpenAIEmbeddingProvider:
        return OpenAIEmbeddingProvider(model=model, **kwargs)

    @classmethod
    def for_lm_studio(
        cls,
        model: str = "local-model",
        **kwargs: Any,
    ) -> LMStudioEmbeddingProvider:
        return LMStudioEmbeddingProvider(model=model, **kwargs)

    @classmethod
    def for_llama_cpp(
        cls,
        model: str = "local-model",
        **kwargs: Any,
    ) -> LlamaCppEmbeddingProvider:
        return LlamaCppEmbeddingProvider(model=model, **kwargs)


class OpenAIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Preset for OpenAI's official embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("is_local", False)
        kwargs.setdefault("provider_label", "openai")
        kwargs.setdefault("requires_api_key", True)
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            **kwargs,
        )


class LMStudioEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Preset for LM Studio's local embeddings endpoint."""

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("is_local", True)
        kwargs.setdefault("provider_label", "lm_studio")
        kwargs.setdefault("requires_api_key", False)
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            **kwargs,
        )


class LlamaCppEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Preset for llama.cpp's OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = DEFAULT_LLAMA_CPP_BASE_URL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("is_local", True)
        kwargs.setdefault("provider_label", "llama_cpp")
        kwargs.setdefault("requires_api_key", False)
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            **kwargs,
        )
