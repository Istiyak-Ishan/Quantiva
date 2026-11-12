"""
app/providers/interfaces/exceptions.py
──────────────────────────────────────────────────────────────────────────────
Provider-layer error hierarchy for Quantiva.

All exceptions inherit from :class:`ProviderError` so callers can catch the
base class for broad handling or a specific subclass for fine-grained recovery.

Design rules:
* ``retriable`` signals whether an immediate retry (after a brief back-off)
  may succeed.  Non-retriable errors require user intervention (wrong key,
  wrong model name, context too long).
* ``user_message`` is a safe, non-sensitive string the UI may display directly.
  It must NEVER contain API keys, endpoint URLs, or raw stack traces.
* No external SDK is imported here — exceptions are raised by concrete provider
  implementations, not by this module itself.
"""
from __future__ import annotations


class ProviderError(Exception):
    """Base class for all AI provider errors.

    Parameters
    ----------
    provider:
        Name of the provider that raised the error (e.g. ``"gemini"``,
        ``"ollama"``).
    model:
        Model identifier in use when the error occurred, if known.
    retriable:
        ``True`` when a transient condition means retrying may succeed.
    user_message:
        A safe, human-readable message suitable for display in the UI.
    """

    def __init__(
        self,
        message: str = "An unexpected provider error occurred.",
        *,
        provider: str = "unknown",
        model: str | None = None,
        retriable: bool = False,
        user_message: str = "AI provider error. Please try again or check your settings.",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.retriable = retriable
        self.user_message = user_message

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"provider={self.provider!r}, model={self.model!r}, "
            f"retriable={self.retriable})"
        )


# ── Authentication errors ──────────────────────────────────────────────────────

class ProviderAuthenticationError(ProviderError):
    """API key is missing, expired, or rejected by the provider.

    retriable = False — the user must supply a valid key before retrying.
    """

    def __init__(
        self,
        message: str = "Provider authentication failed.",
        *,
        provider: str = "unknown",
        model: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            retriable=False,
            user_message=(
                "Authentication failed. Please check that your API key for "
                f"'{provider}' is correct and has not expired."
            ),
        )


# ── Connectivity errors ────────────────────────────────────────────────────────

class ProviderConnectionError(ProviderError):
    """Network-level failure: Ollama not running, cloud unreachable, timeout.

    retriable = True — transient conditions (server restart, brief outage).
    """

    def __init__(
        self,
        message: str = "Provider connection failed.",
        *,
        provider: str = "unknown",
        model: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            retriable=True,
            user_message=(
                f"Could not connect to '{provider}'. "
                "If using a local provider, ensure it is running. "
                "If using a cloud provider, check your internet connection."
            ),
        )


# ── Model not found ────────────────────────────────────────────────────────────

class ModelNotFoundError(ProviderError):
    """The specified model is not available on this provider.

    For Ollama this means the model has not been pulled yet.
    For cloud providers this means an invalid model identifier was supplied.

    retriable = False — the user must resolve the model availability.
    """

    def __init__(
        self,
        message: str = "Model not found.",
        *,
        provider: str = "unknown",
        model: str | None = None,
    ) -> None:
        model_hint = f" Model '{model}'" if model else ""
        super().__init__(
            message,
            provider=provider,
            model=model,
            retriable=False,
            user_message=(
                f"{model_hint} is not available on '{provider}'. "
                "For Ollama, run 'ollama pull <model>' first. "
                "For cloud providers, verify the model identifier."
            ),
        )


# ── Context window ─────────────────────────────────────────────────────────────

class ContextWindowExceededError(ProviderError):
    """The prompt (input + history) exceeds the model's context window.

    retriable = False — the caller must reduce context before retrying.
    """

    def __init__(
        self,
        message: str = "Context window exceeded.",
        *,
        provider: str = "unknown",
        model: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            retriable=False,
            user_message=(
                "The conversation or document context is too long for the "
                "selected model. Try starting a new session or selecting "
                "fewer documents."
            ),
        )


# ── Rate limiting ──────────────────────────────────────────────────────────────

class ProviderRateLimitError(ProviderError):
    """HTTP 429 — provider quota or rate limit exceeded.

    retriable = True — backing off and retrying will usually succeed.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded.",
        *,
        provider: str = "unknown",
        model: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            retriable=True,
            user_message=(
                f"The '{provider}' API rate limit was exceeded. "
                "Please wait a moment before trying again."
            ),
        )


# ── Capability not supported ───────────────────────────────────────────────────

class CapabilityNotSupportedError(ProviderError):
    """A requested capability is not available on this provider/model.

    Example: native tool calling on a model that only supports text generation.

    The orchestrator should catch this and fall back to ReAct prompt-based tool
    invocation rather than propagating the error to the user.

    retriable = False — the capability is structurally absent.
    """

    def __init__(
        self,
        capability: str,
        *,
        provider: str = "unknown",
        model: str | None = None,
    ) -> None:
        super().__init__(
            f"Capability '{capability}' is not supported by '{provider}'.",
            provider=provider,
            model=model,
            retriable=False,
            user_message=(
                f"The selected model does not support '{capability}'. "
                "The system will use an alternative approach automatically."
            ),
        )
        self.capability = capability


# ── Embedding dimension mismatch ───────────────────────────────────────────────

class EmbeddingDimensionMismatchError(ProviderError):
    """Active embedding model dimensions differ from the indexed collection.

    Raised before any vector-store operation so a mismatched query never
    silently corrupts or searches the wrong collection.

    retriable = False — dimension is locked at index creation time.
    """

    def __init__(
        self,
        expected: int,
        actual: int,
        *,
        provider: str = "unknown",
        model: str | None = None,
    ) -> None:
        super().__init__(
            f"Embedding dimension mismatch: index expects {expected}d, "
            f"active model produces {actual}d.",
            provider=provider,
            model=model,
            retriable=False,
            user_message=(
                f"The embedding model produces {actual}-dimensional vectors, "
                f"but the knowledge base was indexed with {expected}-dimensional "
                "vectors. Re-index the knowledge base with the current model "
                "before querying."
            ),
        )
        self.expected = expected
        self.actual = actual
