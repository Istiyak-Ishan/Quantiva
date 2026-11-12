"""
app/providers/interfaces/types.py
──────────────────────────────────────────────────────────────────────────────
Normalized, provider-agnostic type definitions for the Quantiva provider
abstraction layer.

No external SDK dependency is imported here — this module is used by both
concrete providers AND by the rest of the app (ingestion, retrieval, agents)
so it must remain SDK-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Provider taxonomy ─────────────────────────────────────────────────────────

class ProviderType(str, Enum):
    """Supported AI provider back-ends."""

    OLLAMA = "ollama"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LM_STUDIO = "lm_studio"
    LLAMA_CPP = "llama_cpp"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CROSS_ENCODER = "cross_encoder"


# ── Provider capabilities ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProviderCapabilities:
    """Immutable capability descriptor returned by each provider implementation.

    Consumers must check these flags before invoking optional features (e.g.
    function calling) and fall back gracefully when unsupported.
    """

    supports_streaming: bool
    """The provider can stream tokens incrementally."""

    supports_function_calling: bool
    """The provider accepts native tool/function-call schemas."""

    supports_structured_output: bool
    """The provider can be constrained to emit valid JSON matching a schema."""

    supports_vision: bool
    """The provider can accept image content in messages."""

    supports_system_prompt: bool
    """The provider honours an explicit system message role."""

    max_context_window: int
    """Maximum combined input + output tokens the provider accepts."""

    is_local: bool
    """True when inference runs on the local machine (Ollama, LM Studio, llama.cpp)."""


# ── Chat roles ────────────────────────────────────────────────────────────────

class ChatRole(str, Enum):
    """Normalised message roles shared across all providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ── Tool call types ───────────────────────────────────────────────────────────

@dataclass
class ToolCallRequest:
    """A tool/function invocation requested by the model.

    ``arguments`` is always a decoded dict — providers that return JSON strings
    must decode before constructing this object.
    """

    id: str
    """Provider-assigned call identifier (used to correlate results)."""

    name: str
    """Name of the registered tool to invoke."""

    arguments: dict[str, Any]
    """Decoded keyword arguments for the tool function."""


@dataclass
class ToolCallResult:
    """The result of executing a :class:`ToolCallRequest`.

    Pass back to the LLM in a :attr:`ChatRole.TOOL` message.
    """

    tool_call_id: str
    """Must match the :attr:`ToolCallRequest.id` that triggered this result."""

    name: str
    """Name of the tool that produced this result."""

    content: str
    """Serialised (usually JSON) result string to include in the next prompt."""

    is_error: bool = False
    """Set to ``True`` when tool execution raised an exception."""


# ── Chat message ──────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    """A single turn in a normalised conversation history.

    Exactly one of ``content``, ``tool_calls``, or ``tool_result`` should carry
    meaningful data for a given role, following these conventions:

    * ``SYSTEM`` / ``USER`` / ``ASSISTANT`` text turns → ``content``
    * ``ASSISTANT`` tool-call turn                     → ``tool_calls``
    * ``TOOL`` result turn                             → ``tool_result``
    """

    role: ChatRole
    content: str = ""
    tool_calls: list[ToolCallRequest] | None = None
    tool_result: ToolCallResult | None = None


# ── Tool definition ───────────────────────────────────────────────────────────

@dataclass
class ToolDefinition:
    """Schema for a callable tool registered with the LLM.

    ``parameters_json_schema`` must be a valid JSON Schema object (OpenAI-style
    ``{"type": "object", "properties": {...}, "required": [...]}``) —
    provider adapters translate this to their native format.
    """

    name: str
    description: str
    parameters_json_schema: dict[str, Any]


# ── Generation config ─────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """Provider-agnostic sampling parameters.

    All fields carry sensible defaults.  Provider implementations silently
    ignore parameters they do not support and must not raise on them.
    """

    temperature: float = 0.2
    """Sampling temperature (0 = deterministic, 1 = very random)."""

    max_tokens: int = 4096
    """Maximum number of completion tokens to generate."""

    top_p: float = 0.95
    """Nucleus sampling probability mass."""

    stop_sequences: list[str] | None = None
    """Optional token sequences that stop generation early."""

    response_json_schema: dict[str, Any] | None = None
    """When set, instruct the provider to emit JSON conforming to this schema."""

    timeout_seconds: float = 120.0
    """Wall-clock timeout for a single generation request."""


# ── Token usage ───────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    """Normalised token accounting returned by all LLM providers."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        # Tolerate providers that only report total (set completion = total - prompt).
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            object.__setattr__(
                self,
                "total_tokens",
                self.prompt_tokens + self.completion_tokens,
            )


# ── LLM response ──────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Complete, provider-agnostic response from an LLM generation call."""

    content: str
    """Full assistant text (empty string when ``finish_reason == "tool_calls"``)."""

    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    """Tool calls requested by the model (empty list when no tools were called)."""

    usage: TokenUsage | None = None
    """Token accounting; ``None`` when the provider does not expose usage."""

    model: str = ""
    """Model identifier as reported by the provider."""

    provider: str = ""
    """Provider name (matches :attr:`LLMProvider.provider_name`)."""

    finish_reason: str = "stop"
    """Provider finish reason string (``"stop"``, ``"tool_calls"``, ``"length"``, …)."""

    raw_metadata: dict[str, Any] = field(default_factory=dict)
    """Verbatim provider response metadata for debugging; never logged to disk."""


# ── Streaming chunk ───────────────────────────────────────────────────────────

@dataclass
class LLMStreamChunk:
    """A single incremental unit yielded during a streaming LLM generation."""

    delta_text: str = ""
    """Incremental text token(s) emitted in this chunk (empty for tool chunks)."""

    tool_call_delta: ToolCallRequest | None = None
    """Partial or complete tool call assembled during streaming (provider-dependent)."""

    is_final: bool = False
    """``True`` only for the last chunk in the stream."""

    usage: TokenUsage | None = None
    """Token usage reported on the final chunk (``None`` for all earlier chunks)."""

    finish_reason: str | None = None
    """Populated only on the final chunk."""


# ── Connection test result ────────────────────────────────────────────────────

@dataclass
class ConnectionTestResult:
    """Outcome of a provider connectivity and capability probe.

    Returned by :meth:`LLMProvider.test_connection`,
    :meth:`EmbeddingProvider.test_embedding`, and
    :meth:`RerankerProvider.test_reranker`.
    """

    success: bool
    """``True`` if the probe completed without error."""

    latency_ms: float
    """Round-trip latency for the probe request in milliseconds."""

    provider: str
    """Provider name."""

    model: str
    """Model used for the probe."""

    capabilities: ProviderCapabilities | None = None
    """Detected capabilities (populated on success, ``None`` on failure)."""

    detected_dimensions: int | None = None
    """Embedding vector length (populated by embedding probes only)."""

    error_code: str | None = None
    """Machine-readable error code (e.g. ``"auth_error"``) on failure."""

    error_message: str | None = None
    """Human-readable error description on failure."""
