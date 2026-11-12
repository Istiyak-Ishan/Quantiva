"""
app/providers/gemini_common.py
──────────────────────────────────────────────────────────────────────────────
Shared utilities for Google Gemini providers (LLM and Embeddings).
"""
from __future__ import annotations

import asyncio

from google import genai
from google.genai import types

from app.config import settings
from app.providers.interfaces.exceptions import (
    ContextWindowExceededError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)


def build_gemini_client(
    api_key: str | None = None,
    timeout_ms: int | None = None,
) -> genai.Client:
    """Create a configured google-genai Client instance."""
    resolved_key = api_key or settings.GOOGLE_API_KEY
    resolved_timeout = timeout_ms or settings.GOOGLE_TIMEOUT_MS
    return genai.Client(
        api_key=resolved_key,
        http_options=types.HttpOptions(timeout=resolved_timeout),
    )


def translate_gemini_error(
    exc: Exception,
    *,
    model: str | None = None,
) -> ProviderError:
    """Translate Google API and network exceptions into the Quantiva ProviderError hierarchy."""
    if isinstance(exc, ProviderError):
        return exc

    msg = str(exc)
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    status_str = str(status) if status else ""
    full_text = f"{code} {status_str} {msg}".upper()

    # 1. Authentication / Permissions
    if (
        code in (401, 403)
        or "PERMISSION_DENIED" in full_text
        or "UNAUTHENTICATED" in full_text
        or "API_KEY_INVALID" in full_text
    ):
        return ProviderAuthenticationError(
            f"Gemini authentication failed: {msg}",
            provider="gemini",
            model=model,
        )

    # 2. Rate limit / Quota exceeded
    if (
        code == 429
        or "RESOURCE_EXHAUSTED" in full_text
        or "QUOTA" in full_text
        or "RATE LIMIT" in full_text
    ):
        return ProviderRateLimitError(
            f"Gemini rate limit exceeded: {msg}",
            provider="gemini",
            model=model,
        )

    # 3. Model Not Found
    if code == 404 or "NOT_FOUND" in full_text:
        return ModelNotFoundError(
            f"Gemini model not found: {msg}",
            provider="gemini",
            model=model,
        )

    # 4. Context Window Exceeded
    if code == 400 and any(
        kw in full_text for kw in ("CONTEXT", "TOKEN", "TOO LONG", "MAXIMUM CONTEXT")
    ):
        return ContextWindowExceededError(
            f"Gemini context window exceeded: {msg}",
            provider="gemini",
            model=model,
        )

    # 5. Connection / Network / Timeout / 5xx
    if (
        code in (500, 502, 503, 504)
        or "DEADLINE_EXCEEDED" in full_text
        or "UNAVAILABLE" in full_text
        or isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError))
    ):
        return ProviderConnectionError(
            f"Gemini connection error: {msg}",
            provider="gemini",
            model=model,
        )

    # Generic provider error fallback
    return ProviderError(
        f"Gemini error: {msg}",
        provider="gemini",
        model=model,
        retriable=False,
    )
