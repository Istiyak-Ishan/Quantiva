"""
app/providers/embeddings/gemini_embedding.py
──────────────────────────────────────────────────────────────────────────────
Google Gemini text embedding provider adapter.

Conforms to :class:`~app.providers.interfaces.embedding.EmbeddingProvider`.
Supports both gemini-embedding-001 and text-embedding-004.
"""
from __future__ import annotations

import math
import time
from typing import Any

from google import genai
from google.genai import types

from app.config import settings
from app.providers.gemini_common import build_gemini_client, translate_gemini_error
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.exceptions import (
    EmbeddingDimensionMismatchError,
    ProviderError,
)
from app.providers.interfaces.types import ConnectionTestResult


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding adapter implementing the unified EmbeddingProvider interface."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        client: genai.Client | None = None,
    ) -> None:
        self._model_name = model or settings.EMBEDDING_MODEL
        self._dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
        self._client = client or build_gemini_client(api_key=api_key)

    def provider_name(self) -> str:
        return "gemini"

    def model_name(self) -> str:
        return self._model_name

    def get_dimensions(self) -> int:
        return self._dimensions

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """Validate dimensions and L2-normalize an embedding vector."""
        self.validate_dimensions(vector)
        if not all(math.isfinite(v) for v in vector):
            raise ProviderError(
                "Gemini returned non-finite embedding values.",
                provider=self.provider_name(),
                model=self._model_name,
            )
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            raise ProviderError(
                "Gemini returned zero-magnitude embedding vector.",
                provider=self.provider_name(),
                model=self._model_name,
            )
        return [v / norm for v in vector]

    async def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("texts list must not be empty")
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"texts[{i}] must be a non-empty string")

        all_vectors: list[list[float]] = []

        # Process in batches respecting batch_size
        for idx in range(0, len(texts), batch_size):
            batch = texts[idx : idx + batch_size]
            contents = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in batch
            ]
            config = types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._dimensions,
            )

            try:
                response = await self._client.aio.models.embed_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                raise translate_gemini_error(exc, model=self._model_name) from exc

            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise ProviderError(
                    f"Gemini returned {len(embeddings)} embeddings for batch of {len(batch)}.",
                    provider=self.provider_name(),
                    model=self._model_name,
                )

            for emb in embeddings:
                raw_values = getattr(emb, "values", None) or []
                all_vectors.append(self._normalize_vector(raw_values))

        return all_vectors

    async def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        contents = [types.Content(parts=[types.Part.from_text(text=query)])]
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=self._dimensions,
        )

        try:
            response = await self._client.aio.models.embed_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise translate_gemini_error(exc, model=self._model_name) from exc

        embeddings = getattr(response, "embeddings", None) or []
        if not embeddings:
            raise ProviderError(
                "Gemini returned no embeddings for query.",
                provider=self.provider_name(),
                model=self._model_name,
            )

        raw_values = getattr(embeddings[0], "values", None) or []
        return self._normalize_vector(raw_values)

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
        except Exception as exc:
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
