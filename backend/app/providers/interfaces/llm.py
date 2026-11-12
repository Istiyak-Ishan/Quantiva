"""
app/providers/interfaces/llm.py
──────────────────────────────────────────────────────────────────────────────
Abstract base class for all LLM provider implementations.

Concrete implementations live in ``app/providers/<provider>/`` and must
import ONLY from this interfaces package plus the provider's own SDK — never
from ``app/ingestion/``, ``app/retrieval/``, ``app/agents/``, or ``app/api/``.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator

from app.providers.interfaces.types import (
    ChatMessage,
    ConnectionTestResult,
    GenerationConfig,
    LLMResponse,
    LLMStreamChunk,
    ProviderCapabilities,
    ToolDefinition,
)

if TYPE_CHECKING:
    pass


class LLMProvider(ABC):
    """Provider-agnostic interface for text generation and chat completion.

    All public methods are either abstract (must be overridden) or provide
    sensible default implementations that concrete providers may override.

    Thread-safety
    -------------
    Implementations must be safe for concurrent use across coroutines.
    Stateful SDK clients should be managed as singletons via
    ``app/core/application_context.py``, not instantiated per-request.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name, e.g. ``"gemini"`` or ``"ollama"``.

        Used in :class:`~app.providers.interfaces.types.LLMResponse` metadata
        and observability labels.
        """

    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier, e.g. ``"gemini-2.0-flash"`` or ``"llama3.2"``.

        Must reflect the model that will actually be called — not a default or
        fallback that was silently substituted.
        """

    # ── Capabilities ──────────────────────────────────────────────────────────

    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return the static capability descriptor for this provider/model pair.

        Results may be cached by the provider; capabilities do not change
        without an application restart.
        """

    # ── Generation ────────────────────────────────────────────────────────────

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Run a single (non-streaming) generation request.

        Parameters
        ----------
        messages:
            Ordered conversation history in normalised :class:`ChatMessage`
            format.  The provider adapter translates to its native format.
        config:
            Sampling configuration.  ``None`` uses the provider's defaults.
        tools:
            Optional list of tool definitions the model may invoke.  Passing
            tools to a provider that sets
            :attr:`ProviderCapabilities.supports_function_calling` to ``False``
            raises :class:`~exceptions.CapabilityNotSupportedError`.

        Returns
        -------
        LLMResponse
            Normalised response including content, any tool calls, token usage,
            and provider metadata.

        Raises
        ------
        ProviderAuthenticationError
            API key invalid or missing.
        ProviderConnectionError
            Network failure, timeout, or local provider not running.
        ModelNotFoundError
            Requested model unavailable on this provider.
        ContextWindowExceededError
            Prompt exceeds the model's context window.
        ProviderRateLimitError
            Quota or rate limit exceeded.
        CapabilityNotSupportedError
            Tools requested but provider does not support function calling.
        """

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream generation results incrementally.

        Yields :class:`~types.LLMStreamChunk` objects until the final chunk
        (``is_final == True``) is emitted.  The final chunk carries
        ``usage`` and ``finish_reason`` when available.

        Implementations that set
        :attr:`ProviderCapabilities.supports_streaming` to ``False`` must
        raise :class:`~exceptions.CapabilityNotSupportedError`.
        """
        # Satisfy mypy's AsyncIterator protocol; implementations must override.
        raise NotImplementedError  # pragma: no cover
        yield  # type: ignore[misc]

    # ── Connectivity probe ────────────────────────────────────────────────────

    @abstractmethod
    async def test_connection(self) -> ConnectionTestResult:
        """Probe provider connectivity and measure latency.

        Sends a minimal low-cost request (e.g. 1–2 token prompt) and returns a
        :class:`~types.ConnectionTestResult`.  Must never raise — all failures
        are reported through ``success=False`` and ``error_message``.

        Used by the UI settings panel to validate provider configuration before
        any user-facing query is submitted.
        """

    # ── Token estimation ──────────────────────────────────────────────────────

    def count_tokens_estimate(self, text: str) -> int:
        """Return a fast, local token-count estimate without an API call.

        The default implementation uses a character-based heuristic
        (~4 chars / token) that is accurate to within ±20% for typical English
        text.  Concrete providers with access to a local tokenizer should
        override this method for higher accuracy.

        Parameters
        ----------
        text:
            Raw text to estimate tokens for.

        Returns
        -------
        int
            Estimated token count (always >= 1 for non-empty input).
        """
        if not text:
            return 0
        # Strip markdown/code noise for a tighter estimate.
        clean = re.sub(r"\s+", " ", text).strip()
        # ~4 characters per token is a reasonable default across common models.
        return max(1, len(clean) // 4)
