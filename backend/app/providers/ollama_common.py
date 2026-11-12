"""
app/providers/ollama_common.py
──────────────────────────────────────────────────────────────────────────────
Shared helpers for the Ollama LLM, embedding, and runtime-management adapters.

The adapters talk to Ollama through its plain HTTP JSON API using
:mod:`httpx` (no Ollama Python SDK).  Keeping the transport layer SDK-free
means the packaged desktop application stays small and PyInstaller-friendly.
"""
from __future__ import annotations

from typing import Any

import httpx

from app.providers.interfaces.exceptions import (
    ContextWindowExceededError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from app.providers.interfaces.types import GenerationConfig

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# Keep error bodies short enough to safely surface in logs/UI.
_MAX_ERROR_BODY_CHARS = 600


def normalize_base_url(base_url: str | None) -> str:
    """Return a canonical Ollama base URL without a trailing slash."""
    return (base_url or DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")


def daemon_unreachable_message(base_url: str) -> str:
    """Exact, actionable message used whenever the daemon cannot be reached."""
    return (
        f"Cannot connect to Ollama at {base_url}. "
        "Ensure Ollama is installed and running."
    )


def model_not_installed_message(model: str) -> str:
    """Exact, actionable message used when Ollama does not have the model."""
    return (
        f"Model '{model}' is not installed in Ollama. "
        f"Pull it via Model Studio or run 'ollama pull {model}'."
    )


def _trim(body: str) -> str:
    clean = " ".join((body or "").split())
    if len(clean) > _MAX_ERROR_BODY_CHARS:
        return clean[:_MAX_ERROR_BODY_CHARS] + "…"
    return clean


def translate_transport_error(
    exc: Exception,
    *,
    base_url: str,
    model: str | None = None,
) -> ProviderError:
    """Translate an :mod:`httpx` transport exception into a provider error.

    A refused connection (Ollama not installed / not started) gets the exact
    actionable message required by the desktop UI.
    """
    if isinstance(exc, ProviderError):
        return exc

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        message = daemon_unreachable_message(base_url)
        error = ProviderConnectionError(
            message,
            provider="ollama",
            model=model,
        )
        # The UI displays ``user_message``; keep the actionable instruction
        # identical to the exception message for this specific failure.
        error.user_message = message
        return error

    if isinstance(exc, httpx.TimeoutException):
        return ProviderConnectionError(
            (
                f"Ollama at {base_url} did not respond in time. "
                "The model may still be loading; please retry."
            ),
            provider="ollama",
            model=model,
        )

    if isinstance(exc, httpx.RequestError):
        return ProviderConnectionError(
            (
                f"Ollama request to {base_url} failed: {_trim(str(exc))}. "
                "Ensure Ollama is installed and running."
            ),
            provider="ollama",
            model=model,
        )

    return ProviderError(
        f"Unexpected Ollama error: {_trim(str(exc))}",
        provider="ollama",
        model=model,
        retriable=False,
    )


def _looks_like_model_not_found(status_code: int, body: str) -> bool:
    if status_code != 404:
        return False
    lowered = body.lower()
    return (
        "model" in lowered
        and (
            "not found" in lowered
            or "not installed" in lowered
            or "no such" in lowered
        )
    )


def raise_for_ollama_status(
    status_code: int,
    body: str,
    *,
    base_url: str,
    model: str,
) -> None:
    """Raise the appropriate typed exception for a non-success HTTP status."""
    if 200 <= status_code < 300:
        return

    lowered = (body or "").lower()

    if status_code == 404 or _looks_like_model_not_found(status_code, body):
        message = model_not_installed_message(model)
        error = ModelNotFoundError(
            message,
            provider="ollama",
            model=model,
        )
        error.user_message = message
        raise error

    if status_code in (401, 403):
        raise ProviderAuthenticationError(
            f"Ollama rejected the request (HTTP {status_code}): {_trim(body)}",
            provider="ollama",
            model=model,
        )

    if status_code == 429:
        raise ProviderRateLimitError(
            f"Ollama rate limit exceeded: {_trim(body)}",
            provider="ollama",
            model=model,
        )

    if status_code == 400 and any(
        token in lowered
        for token in (
            "context",
            "too long",
            "too large",
            "exceeds",
            "token limit",
        )
    ):
        raise ContextWindowExceededError(
            f"Ollama context window exceeded: {_trim(body)}",
            provider="ollama",
            model=model,
        )

    if status_code >= 500:
        raise ProviderError(
            f"Ollama server error (HTTP {status_code}): {_trim(body)}",
            provider="ollama",
            model=model,
            retriable=True,
        )

    raise ProviderError(
        f"Ollama request failed (HTTP {status_code}): {_trim(body)}",
        provider="ollama",
        model=model,
        retriable=False,
    )


def is_model_not_found_response(status_code: int, body: str) -> bool:
    """True when a 404 represents a missing model rather than a missing route."""
    return _looks_like_model_not_found(status_code, body)


def build_chat_options(config: GenerationConfig | None) -> dict[str, Any]:
    """Translate normalised generation config into Ollama ``options``."""
    if config is None:
        return {}

    options: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "num_predict": config.max_tokens,
    }
    if config.stop_sequences:
        options["stop"] = list(config.stop_sequences)
    return options


def request_timeout(config: GenerationConfig | None, default: float = 120.0) -> float:
    """Resolve the per-request timeout from a :class:`GenerationConfig`."""
    if config is not None and config.timeout_seconds:
        return float(config.timeout_seconds)
    return default
