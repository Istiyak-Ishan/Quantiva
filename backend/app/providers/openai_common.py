"""
app/providers/openai_common.py
──────────────────────────────────────────────────────────────────────────────
Shared helpers for OpenAI-compatible chat, embedding, LM Studio, llama.cpp,
OpenAI official, and Anthropic-style API adapters.

All adapters use :mod:`httpx` directly so the packaged desktop application
does not depend on vendor SDK wrappers.
"""
from __future__ import annotations

import json
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
from app.providers.interfaces.types import ConnectionTestResult, TokenUsage

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080/v1"

_MAX_ERROR_BODY_CHARS = 700


def normalize_openai_base_url(base_url: str | None) -> str:
    """Return a canonical base URL without a trailing slash."""
    return (base_url or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")


def _trim(value: str) -> str:
    clean = " ".join((value or "").split())
    if len(clean) > _MAX_ERROR_BODY_CHARS:
        return clean[:_MAX_ERROR_BODY_CHARS] + "…"
    return clean


def extract_api_error_message(body: str) -> str:
    """Extract a human-readable message from an OpenAI-style error body."""
    if not body:
        return ""

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return _trim(body)

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type")
            if message:
                return _trim(str(message))
        elif isinstance(error, str):
            return _trim(error)
        message = payload.get("message")
        if message:
            return _trim(str(message))

    return _trim(body)


def translate_openai_transport_error(
    exc: Exception,
    *,
    provider: str,
    base_url: str,
    model: str | None = None,
    is_local: bool = False,
) -> ProviderError:
    """Translate an httpx transport failure into a typed provider error."""
    if isinstance(exc, ProviderError):
        return exc

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        if is_local:
            message = (
                f"Cannot connect to {provider} at {base_url}. "
                "Ensure the local model server is running."
            )
        else:
            message = (
                f"Could not connect to {provider} at {base_url}. "
                "Check your internet connection and API endpoint URL."
            )
        error = ProviderConnectionError(message, provider=provider, model=model)
        error.user_message = message
        return error

    if isinstance(exc, httpx.TimeoutException):
        return ProviderConnectionError(
            (
                f"{provider} at {base_url} did not respond in time. "
                "The model may still be loading; please retry."
            ),
            provider=provider,
            model=model,
        )

    if isinstance(exc, httpx.RequestError):
        return ProviderConnectionError(
            f"{provider} request failed: {_trim(str(exc))}",
            provider=provider,
            model=model,
        )

    return ProviderError(
        f"Unexpected {provider} error: {_trim(str(exc))}",
        provider=provider,
        model=model,
        retriable=False,
    )


def raise_for_openai_status(
    status_code: int,
    body: str,
    *,
    provider: str,
    model: str,
    base_url: str,
) -> None:
    """Raise a normalised exception for a non-success OpenAI-style response."""
    if 200 <= status_code < 300:
        return

    detail = extract_api_error_message(body)
    lowered = detail.lower()

    if status_code in (401, 403):
        raise ProviderAuthenticationError(
            f"{provider} authentication failed: {detail or 'invalid API key'}",
            provider=provider,
            model=model,
        )

    if status_code == 404:
        message = (
            f"Model '{model}' was not found on '{provider}'. "
            f"Verify the model identifier or load the model at {base_url}."
        )
        error = ModelNotFoundError(message, provider=provider, model=model)
        error.user_message = message
        raise error

    if status_code == 429:
        raise ProviderRateLimitError(
            f"{provider} rate limit exceeded: {detail or 'too many requests'}",
            provider=provider,
            model=model,
        )

    if status_code == 400 and any(
        token in lowered
        for token in (
            "context",
            "too long",
            "maximum token",
            "token limit",
            "exceeds",
        )
    ):
        raise ContextWindowExceededError(
            f"{provider} context window exceeded: {detail}",
            provider=provider,
            model=model,
        )

    if status_code >= 500:
        raise ProviderError(
            f"{provider} server error (HTTP {status_code}): {detail}",
            provider=provider,
            model=model,
            retriable=True,
        )

    raise ProviderError(
        f"{provider} request failed (HTTP {status_code}): {detail}",
        provider=provider,
        model=model,
        retriable=False,
    )


def build_openai_headers(
    api_key: str | None,
    custom_headers: dict[str, str] | None = None,
    *,
    accept: str = "application/json",
) -> dict[str, str]:
    """Build request headers, allowing custom headers to override defaults."""
    headers: dict[str, str] = {
        "Accept": accept,
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if custom_headers:
        headers.update(custom_headers)
    return headers


def decode_tool_arguments(value: Any) -> dict[str, Any]:
    """Decode OpenAI-style tool arguments (dict or JSON text) into a dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return {"_raw": value}
        return decoded if isinstance(decoded, dict) else {"value": decoded}
    return {"value": value}


def extract_openai_text(content: Any) -> str:
    """Flatten string or content-part arrays into plain assistant text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type in (None, "text", "output_text", "input_text"):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)
    return str(content)


def usage_from_openai(payload: dict[str, Any]) -> TokenUsage | None:
    """Convert an OpenAI ``usage`` object into normalised TokenUsage."""
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(
        usage.get("total_tokens") or prompt_tokens + completion_tokens
    )
    if not prompt_tokens and not completion_tokens and not total_tokens:
        return None
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def failure_result(
    *,
    start_time: float,
    provider: str,
    model: str,
    error: ProviderError,
) -> ConnectionTestResult:
    """Build a ConnectionTestResult for a failed connectivity probe."""
    import time

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return ConnectionTestResult(
        success=False,
        latency_ms=round(latency_ms, 2),
        provider=provider,
        model=model,
        error_code=type(error).__name__,
        error_message=error.user_message,
    )
