"""
tests/providers/test_interfaces.py
──────────────────────────────────────────────────────────────────────────────
Unit tests for the provider interfaces package.

Covers:
  • ProviderType enum values
  • ProviderCapabilities immutability
  • ChatRole enum values
  • ToolCallRequest / ToolCallResult / ChatMessage dataclasses
  • ToolDefinition dataclass
  • GenerationConfig defaults
  • TokenUsage total calculation
  • LLMResponse and LLMStreamChunk defaults
  • ConnectionTestResult fields
  • Full exception hierarchy: construction, attributes, repr, retriability
  • EmbeddingDimensionMismatchError expected/actual attributes
  • CapabilityNotSupportedError capability attribute
  • Abstract provider instantiation guards (cannot instantiate ABC directly)
  • LLMProvider.count_tokens_estimate default heuristic
  • EmbeddingProvider.validate_dimensions raises EmbeddingDimensionMismatchError
  • RerankDocument / RerankResult dataclasses
"""
from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import AsyncIterator

import pytest

# ── Types ──────────────────────────────────────────────────────────────────────
from app.providers.interfaces.types import (
    ChatMessage,
    ChatRole,
    ConnectionTestResult,
    GenerationConfig,
    LLMResponse,
    LLMStreamChunk,
    ProviderCapabilities,
    ProviderType,
    TokenUsage,
    ToolCallRequest,
    ToolCallResult,
    ToolDefinition,
)

# ── Exceptions ─────────────────────────────────────────────────────────────────
from app.providers.interfaces.exceptions import (
    CapabilityNotSupportedError,
    ContextWindowExceededError,
    EmbeddingDimensionMismatchError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)

# ── ABCs ───────────────────────────────────────────────────────────────────────
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.reranker import RerankerProvider, RerankDocument, RerankResult


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ProviderType enum
# ═══════════════════════════════════════════════════════════════════════════════

class TestProviderType:
    def test_all_values_present(self):
        names = {t.value for t in ProviderType}
        assert names == {
            "ollama",
            "gemini",
            "openai_compatible",
            "openai",
            "anthropic",
            "lm_studio",
            "llama_cpp",
            "sentence_transformers",
            "cross_encoder",
        }

    def test_string_subclass(self):
        assert isinstance(ProviderType.GEMINI, str)
        assert ProviderType.GEMINI == "gemini"

    def test_lookup_by_value(self):
        assert ProviderType("ollama") is ProviderType.OLLAMA


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ProviderCapabilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestProviderCapabilities:
    def _make(self, **overrides):
        defaults = dict(
            supports_streaming=True,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_vision=False,
            supports_system_prompt=True,
            max_context_window=128_000,
            is_local=False,
        )
        defaults.update(overrides)
        return ProviderCapabilities(**defaults)

    def test_construction(self):
        cap = self._make()
        assert cap.supports_streaming is True
        assert cap.is_local is False

    def test_frozen(self):
        cap = self._make()
        with pytest.raises((FrozenInstanceError, TypeError)):
            cap.is_local = True  # type: ignore[misc]

    def test_local_flag(self):
        cap = self._make(is_local=True)
        assert cap.is_local is True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ChatRole enum
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatRole:
    def test_all_values(self):
        assert {r.value for r in ChatRole} == {"system", "user", "assistant", "tool"}

    def test_string_subclass(self):
        assert isinstance(ChatRole.USER, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Tool call types
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolCallRequest:
    def test_construction(self):
        req = ToolCallRequest(id="call_1", name="retrieve_chunks", arguments={"query": "test"})
        assert req.id == "call_1"
        assert req.arguments == {"query": "test"}


class TestToolCallResult:
    def test_defaults(self):
        result = ToolCallResult(tool_call_id="call_1", name="retrieve_chunks", content="{}")
        assert result.is_error is False

    def test_error_flag(self):
        result = ToolCallResult(
            tool_call_id="call_1", name="retrieve_chunks", content="failed", is_error=True
        )
        assert result.is_error is True


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ChatMessage
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatMessage:
    def test_text_message(self):
        msg = ChatMessage(role=ChatRole.USER, content="Hello")
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_result is None

    def test_tool_call_message(self):
        calls = [ToolCallRequest(id="c1", name="fn", arguments={})]
        msg = ChatMessage(role=ChatRole.ASSISTANT, tool_calls=calls)
        assert msg.tool_calls == calls

    def test_tool_result_message(self):
        result = ToolCallResult(tool_call_id="c1", name="fn", content="ok")
        msg = ChatMessage(role=ChatRole.TOOL, tool_result=result)
        assert msg.tool_result.name == "fn"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ToolDefinition
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolDefinition:
    def test_construction(self):
        schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        td = ToolDefinition(name="search", description="Search docs", parameters_json_schema=schema)
        assert td.name == "search"
        assert td.parameters_json_schema["type"] == "object"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. GenerationConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 4096
        assert cfg.top_p == 0.95
        assert cfg.stop_sequences is None
        assert cfg.response_json_schema is None
        assert cfg.timeout_seconds == 120.0

    def test_custom_values(self):
        cfg = GenerationConfig(temperature=0.0, max_tokens=512, timeout_seconds=30.0)
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 512


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TokenUsage
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenUsage:
    def test_explicit_total(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.total_tokens == 150

    def test_auto_total_when_zero(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=0)
        assert u.total_tokens == 150

    def test_zero_all(self):
        u = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        assert u.total_tokens == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. LLMResponse
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse(content="Hello")
        assert r.tool_calls == []
        assert r.usage is None
        assert r.finish_reason == "stop"
        assert r.raw_metadata == {}

    def test_with_usage(self):
        u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        r = LLMResponse(content="ok", usage=u, provider="gemini", model="flash")
        assert r.usage.total_tokens == 15


# ═══════════════════════════════════════════════════════════════════════════════
# 10. LLMStreamChunk
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMStreamChunk:
    def test_defaults(self):
        chunk = LLMStreamChunk(delta_text="Hello")
        assert chunk.is_final is False
        assert chunk.usage is None
        assert chunk.finish_reason is None

    def test_final_chunk(self):
        u = TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        chunk = LLMStreamChunk(delta_text=".", is_final=True, usage=u, finish_reason="stop")
        assert chunk.is_final is True
        assert chunk.usage.total_tokens == 15


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ConnectionTestResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestConnectionTestResult:
    def test_success(self):
        r = ConnectionTestResult(
            success=True,
            latency_ms=42.5,
            provider="gemini",
            model="gemini-2.0-flash",
        )
        assert r.success is True
        assert r.error_code is None
        assert r.detected_dimensions is None

    def test_failure(self):
        r = ConnectionTestResult(
            success=False,
            latency_ms=0.0,
            provider="ollama",
            model="llama3.2",
            error_code="connection_refused",
            error_message="Ollama is not running.",
        )
        assert r.success is False
        assert r.error_code == "connection_refused"


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Exception hierarchy
# ═══════════════════════════════════════════════════════════════════════════════

class TestProviderError:
    def test_base_defaults(self):
        err = ProviderError()
        assert err.provider == "unknown"
        assert err.model is None
        assert err.retriable is False
        assert isinstance(err.user_message, str)

    def test_custom_attributes(self):
        err = ProviderError("bad", provider="gemini", model="flash", retriable=True)
        assert err.provider == "gemini"
        assert err.model == "flash"
        assert err.retriable is True

    def test_is_exception(self):
        with pytest.raises(ProviderError):
            raise ProviderError("boom")

    def test_repr(self):
        err = ProviderError(provider="gemini", model="flash")
        assert "gemini" in repr(err)
        assert "flash" in repr(err)


class TestAuthenticationError:
    def test_not_retriable(self):
        err = ProviderAuthenticationError(provider="gemini")
        assert err.retriable is False
        assert isinstance(err, ProviderError)

    def test_user_message_contains_provider(self):
        err = ProviderAuthenticationError(provider="openai")
        assert "openai" in err.user_message


class TestConnectionError:
    def test_retriable(self):
        err = ProviderConnectionError(provider="ollama")
        assert err.retriable is True

    def test_is_provider_error(self):
        assert isinstance(ProviderConnectionError(), ProviderError)


class TestModelNotFoundError:
    def test_not_retriable(self):
        err = ModelNotFoundError(provider="ollama", model="llama99")
        assert err.retriable is False
        assert err.model == "llama99"

    def test_user_message_mentions_model(self):
        err = ModelNotFoundError(provider="ollama", model="mistral:7b")
        assert "mistral:7b" in err.user_message


class TestContextWindowExceededError:
    def test_not_retriable(self):
        err = ContextWindowExceededError(provider="gemini", model="flash")
        assert err.retriable is False


class TestRateLimitError:
    def test_retriable(self):
        err = ProviderRateLimitError(provider="gemini")
        assert err.retriable is True


class TestCapabilityNotSupportedError:
    def test_capability_attribute(self):
        err = CapabilityNotSupportedError("function_calling", provider="ollama", model="llama3.2")
        assert err.capability == "function_calling"
        assert err.retriable is False
        assert "function_calling" in str(err)


class TestEmbeddingDimensionMismatchError:
    def test_expected_actual(self):
        err = EmbeddingDimensionMismatchError(expected=768, actual=1024, provider="gemini")
        assert err.expected == 768
        assert err.actual == 1024
        assert err.retriable is False
        assert "768" in err.user_message
        assert "1024" in err.user_message

    def test_is_provider_error(self):
        assert isinstance(
            EmbeddingDimensionMismatchError(768, 512),
            ProviderError,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Abstract provider instantiation guards
# ═══════════════════════════════════════════════════════════════════════════════

class TestAbstractGuards:
    def test_llm_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_embedding_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_reranker_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            RerankerProvider()  # type: ignore[abstract]


# ═══════════════════════════════════════════════════════════════════════════════
# 14. LLMProvider.count_tokens_estimate default heuristic
# ═══════════════════════════════════════════════════════════════════════════════

class _ConcreteMinimalLLM(LLMProvider):
    """Minimal concrete implementation used only for testing default methods."""

    def provider_name(self) -> str:
        return "test"

    def model_name(self) -> str:
        return "test-model"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=False,
            supports_function_calling=False,
            supports_structured_output=False,
            supports_vision=False,
            supports_system_prompt=True,
            max_context_window=4096,
            is_local=True,
        )

    async def generate(self, messages, config=None, tools=None):
        return LLMResponse(content="ok")

    async def stream(self, messages, config=None, tools=None):
        yield LLMStreamChunk(delta_text="ok", is_final=True)

    async def test_connection(self):
        return ConnectionTestResult(success=True, latency_ms=0.0, provider="test", model="test-model")


class TestCountTokensEstimate:
    provider = _ConcreteMinimalLLM()

    def test_empty_string_returns_zero(self):
        assert self.provider.count_tokens_estimate("") == 0

    def test_short_text(self):
        # "Hello" = 5 chars → 5 // 4 = 1
        result = self.provider.count_tokens_estimate("Hello")
        assert result >= 1

    def test_longer_text(self):
        text = "word " * 100  # 500 chars
        result = self.provider.count_tokens_estimate(text)
        # ~4 chars/token: expect ~100 ± generous margin
        assert 50 <= result <= 200

    def test_whitespace_collapsed(self):
        """Extra whitespace should not inflate the estimate significantly."""
        dense = self.provider.count_tokens_estimate("hello world " * 10)
        spaced = self.provider.count_tokens_estimate("hello   world   " * 10)
        # Should be close after whitespace collapsing
        assert abs(dense - spaced) <= 10


# ═══════════════════════════════════════════════════════════════════════════════
# 15. EmbeddingProvider.validate_dimensions
# ═══════════════════════════════════════════════════════════════════════════════

class _ConcreteMinimalEmbedding(EmbeddingProvider):
    def provider_name(self) -> str:
        return "test-embed"

    def model_name(self) -> str:
        return "test-embed-model"

    def get_dimensions(self) -> int:
        return 768

    async def embed_documents(self, texts, batch_size=32):
        return [[0.0] * 768 for _ in texts]

    async def embed_query(self, query):
        return [0.0] * 768

    async def test_embedding(self):
        return ConnectionTestResult(
            success=True,
            latency_ms=1.0,
            provider="test-embed",
            model="test-embed-model",
            detected_dimensions=768,
        )


class TestValidateDimensions:
    provider = _ConcreteMinimalEmbedding()

    def test_matching_dimensions_ok(self):
        # Should not raise
        self.provider.validate_dimensions([0.0] * 768)

    def test_mismatched_raises(self):
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            self.provider.validate_dimensions([0.0] * 512)
        assert exc_info.value.expected == 768
        assert exc_info.value.actual == 512


# ═══════════════════════════════════════════════════════════════════════════════
# 16. RerankDocument / RerankResult dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class TestRerankDocument:
    def test_defaults(self):
        doc = RerankDocument(id="uuid-1", content="Chunk text.")
        assert doc.initial_score == 0.0
        assert doc.metadata == {}

    def test_custom_score(self):
        doc = RerankDocument(id="uuid-2", content="Text", initial_score=0.87)
        assert doc.initial_score == 0.87


class TestRerankResult:
    def test_construction(self):
        result = RerankResult(
            id="uuid-1",
            content="Chunk text.",
            metadata={"page": 3},
            rerank_score=0.92,
            initial_score=0.5,
            rank=0,
        )
        assert result.rank == 0
        assert result.rerank_score == 0.92
        assert result.metadata["page"] == 3
