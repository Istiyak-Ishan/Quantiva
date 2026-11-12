"""
app/providers/rerankers/api.py
──────────────────────────────────────────────────────────────────────────────
Remote reranker adapter for JSON ``/rerank`` endpoints.

Supports three request/response dialects with one class:

* ``cohere``            — ``POST https://api.cohere.com/v1/rerank``
* ``jina``              — ``POST https://api.jina.ai/v1/rerank``
* ``openai_compatible`` — any ``/v1/rerank`` endpoint (vLLM, Infinity,
  text-embeddings-inference with ``document_field="texts"``)

Conforms to :class:`~app.providers.interfaces.reranker.RerankerProvider`.
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.config import settings
from app.providers.interfaces.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
)
from app.providers.interfaces.reranker import (
    RerankDocument,
    RerankerProvider,
    RerankResult,
)
from app.providers.interfaces.types import ConnectionTestResult
from app.providers.openai_common import (
    build_openai_headers,
    raise_for_openai_status,
    translate_openai_transport_error,
)

DEFAULT_COHERE_BASE_URL = "https://api.cohere.com/v1"
DEFAULT_JINA_BASE_URL = "https://api.jina.ai/v1"
DEFAULT_COHERE_MODEL = "rerank-v3.5"
DEFAULT_JINA_MODEL = "jina-reranker-v2-base-multilingual"

STYLE_COHERE = "cohere"
STYLE_JINA = "jina"
STYLE_OPENAI_COMPATIBLE = "openai_compatible"

_STYLES = (STYLE_COHERE, STYLE_JINA, STYLE_OPENAI_COMPATIBLE)
_SCORE_KEYS = ("relevance_score", "score", "rerank_score")


def _extract_error_detail(body: str) -> str:
    """Pull a readable message out of Cohere/Jina/vLLM error envelopes."""
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body.strip()

    if isinstance(payload, dict):
        for key in ("message", "detail", "error_description"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        error = payload.get("error")
        if isinstance(error, str) and error:
            return error
        if isinstance(error, dict):
            message = error.get("message") or error.get("type")
            if message:
                return str(message)
    return body.strip()


class APIRerankerProvider(RerankerProvider):
    """Reranker backed by a remote JSON ``/rerank`` HTTP endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_COHERE_BASE_URL,
        model: str = DEFAULT_COHERE_MODEL,
        api_key: str | None = None,
        *,
        style: str = STYLE_OPENAI_COMPATIBLE,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
        connect_timeout_seconds: float = 5.0,
        provider_label: str | None = None,
        is_local: bool = False,
        requires_api_key: bool = False,
        document_field: str = "documents",
        send_top_n: bool = True,
        return_documents: bool = False,
    ) -> None:
        normalized_style = (style or STYLE_OPENAI_COMPATIBLE).strip().lower()
        if normalized_style not in _STYLES:
            raise ValueError(
                f"style must be one of {_STYLES}, got {style!r}"
            )

        self._base_url = (base_url or DEFAULT_COHERE_BASE_URL).strip().rstrip("/")
        self._model_name = model
        self._api_key = api_key or ""
        self._style = normalized_style
        self._client = client
        self._transport = transport
        self._owns_client = client is None
        self._custom_headers = dict(headers or {})
        self._timeout_seconds = float(timeout_seconds)
        self._connect_timeout_seconds = float(connect_timeout_seconds)
        self._provider_label = provider_label or normalized_style
        self._is_local = is_local
        self._requires_api_key = requires_api_key
        self._document_field = document_field
        self._send_top_n = send_top_n
        self._return_documents = return_documents

    # ── Identity ──────────────────────────────────────────────────────────────

    def provider_name(self) -> str:
        return self._provider_label

    def model_name(self) -> str:
        return self._model_name

    def base_url(self) -> str:
        return self._base_url

    # ── HTTP lifecycle ────────────────────────────────────────────────────────

    def _rerank_url(self) -> str:
        return f"{self._base_url}/rerank"

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
                    f"Add it in Settings or set "
                    f"{self._provider_label.upper()}_API_KEY."
                ),
                provider=self._provider_label,
                model=self._model_name,
            )
        return build_openai_headers(self._api_key, self._custom_headers)

    # ── Request construction ──────────────────────────────────────────────────

    def _build_payload(
        self,
        query: str,
        documents: list[RerankDocument],
        top_k: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "query": query,
            self._document_field: [document.content for document in documents],
        }
        if self._send_top_n:
            payload["top_n"] = min(top_k, len(documents))
        if self._style in (STYLE_COHERE, STYLE_JINA) or self._return_documents:
            payload["return_documents"] = False
        return payload

    # ── Response parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _item_index(
        item: dict[str, Any],
        documents: list[RerankDocument],
    ) -> int | None:
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(documents):
            return index
        document = item.get("document")
        if isinstance(document, dict):
            text = document.get("text")
            if isinstance(text, str):
                for position, candidate in enumerate(documents):
                    if candidate.content == text:
                        return position
        return None

    @staticmethod
    def _item_score(item: dict[str, Any]) -> float | None:
        for key in _SCORE_KEYS:
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _parse_results(
        self,
        payload: Any,
        documents: list[RerankDocument],
        top_k: int,
    ) -> list[RerankResult]:
        if isinstance(payload, list):
            raw_results: Any = payload
        elif isinstance(payload, dict):
            raw_results = (
                payload.get("results")
                or payload.get("data")
                or payload.get("rerank_results")
            )
        else:
            raw_results = None

        if not isinstance(raw_results, list):
            raise ProviderError(
                f"{self._provider_label} returned no rerank results.",
                provider=self._provider_label,
                model=self._model_name,
                retriable=True,
            )

        best: dict[int, float] = {}
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            index = self._item_index(item, documents)
            score = self._item_score(item)
            if index is None or score is None:
                continue
            if index not in best or score > best[index]:
                best[index] = score

        if not best:
            raise ProviderError(
                f"{self._provider_label} returned rerank results without usable "
                "indices or scores.",
                provider=self._provider_label,
                model=self._model_name,
                retriable=True,
            )

        ranked = sorted(best.items(), key=lambda pair: pair[1], reverse=True)
        return [
            RerankResult(
                id=documents[index].id,
                content=documents[index].content,
                metadata=dict(documents[index].metadata),
                rerank_score=score,
                initial_score=documents[index].initial_score,
                rank=rank,
            )
            for rank, (index, score) in enumerate(ranked[:top_k])
        ]

    # ── Reranking API ─────────────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        documents: list[RerankDocument],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not documents:
            return []

        payload = self._build_payload(query, documents, top_k)
        headers = self._request_headers()
        client = self._get_client()

        try:
            response = await client.post(
                self._rerank_url(),
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
                _extract_error_detail(response.text),
                provider=self._provider_label,
                model=self._model_name,
                base_url=self._base_url,
            )

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"{self._provider_label} returned invalid JSON from /rerank.",
                provider=self._provider_label,
                model=self._model_name,
                retriable=True,
            ) from exc

        return self._parse_results(data, documents, top_k)

    # ── Connectivity probe ────────────────────────────────────────────────────

    async def test_reranker(self) -> ConnectionTestResult:
        start_time = time.perf_counter()
        try:
            results = await self.rerank(
                "Quantiva connection test",
                [
                    RerankDocument(
                        id="probe-1", content="Quantiva connection test"
                    ),
                    RerankDocument(
                        id="probe-2", content="Unrelated candidate content"
                    ),
                ],
                top_k=2,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            if not results:
                raise ProviderError(
                    f"{self._provider_label} probe returned no results.",
                    provider=self._provider_label,
                    model=self._model_name,
                    retriable=True,
                )
            return ConnectionTestResult(
                success=True,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
            )
        except ProviderError as err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
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
                error_code="UnknownError",
                error_message=str(exc),
            )

    # ── Preset constructors ───────────────────────────────────────────────────

    @classmethod
    def for_cohere(
        cls,
        model: str = DEFAULT_COHERE_MODEL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> APIRerankerProvider:
        return cls(
            base_url=DEFAULT_COHERE_BASE_URL,
            model=model,
            api_key=(
                api_key
                if api_key is not None
                else getattr(settings, "COHERE_API_KEY", "")
            ),
            style=STYLE_COHERE,
            provider_label="cohere",
            requires_api_key=True,
            is_local=False,
            **kwargs,
        )

    @classmethod
    def for_jina(
        cls,
        model: str = DEFAULT_JINA_MODEL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> APIRerankerProvider:
        return cls(
            base_url=DEFAULT_JINA_BASE_URL,
            model=model,
            api_key=(
                api_key
                if api_key is not None
                else getattr(settings, "JINA_API_KEY", "")
            ),
            style=STYLE_JINA,
            provider_label="jina",
            requires_api_key=True,
            is_local=False,
            **kwargs,
        )

    @classmethod
    def for_openai_compatible(
        cls,
        base_url: str,
        model: str = "rerank-model",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> APIRerankerProvider:
        kwargs.setdefault("style", STYLE_OPENAI_COMPATIBLE)
        kwargs.setdefault("provider_label", "openai_compatible")
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            **kwargs,
        )
