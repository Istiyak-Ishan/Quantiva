"""
tests/providers/test_gemini_provider.py
──────────────────────────────────────────────────────────────────────────────
Comprehensive unit tests for Gemini LLM and Embedding provider adapters.
All tests use mocked SDK responses so they run completely offline.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors, types

from app.providers.embeddings.gemini_embedding import GeminiEmbeddingProvider
from app.providers.interfaces.exceptions import (
    ContextWindowExceededError,
    EmbeddingDimensionMismatchError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from app.providers.interfaces.types import (
    ChatMessage,
    ChatRole,
    ConnectionTestResult,
    GenerationConfig,
    LLMResponse,
    LLMStreamChunk,
    TokenUsage,
    ToolCallRequest,
    ToolCallResult,
    ToolDefinition,
)
from app.providers.llm.gemini import GeminiLLMProvider


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers to build mock Gemini SDK responses
# ═══════════════════════════════════════════════════════════════════════════════

def make_gemini_response(
    text: str = "Test response",
    *,
    function_calls: list[Any] | None = None,
    finish_reason: types.FinishReason = types.FinishReason.STOP,
    prompt_tokens: int = 12,
    completion_tokens: int = 8,
    total_tokens: int = 20,
) -> types.GenerateContentResponse:
    parts: list[types.Part] = []
    if text:
        parts.append(types.Part.from_text(text=text))
    if function_calls:
        for fc in function_calls:
            parts.append(types.Part.from_function_call(name=fc["name"], args=fc.get("args", {})))

    cand = types.Candidate(
        content=types.Content(role="model", parts=parts),
        finish_reason=finish_reason,
    )
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
        thoughts_token_count=0,
        total_token_count=total_tokens,
    )
    return types.GenerateContentResponse(
        candidates=[cand],
        usage_metadata=usage,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GeminiLLMProvider Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeminiLLMProvider:
    @pytest.fixture
    def mock_sdk(self):
        sdk = MagicMock()
        sdk.aio.models.generate_content = AsyncMock()
        sdk.aio.models.generate_content_stream = AsyncMock()
        return sdk

    @pytest.fixture
    def provider(self, mock_sdk):
        return GeminiLLMProvider(
            api_key="test-api-key",
            model="gemini-2.0-flash",
            client=mock_sdk,
        )

    def test_identity_and_capabilities(self, provider):
        assert provider.provider_name() == "gemini"
        assert provider.model_name() == "gemini-2.0-flash"
        caps = provider.get_capabilities()
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_structured_output is True
        assert caps.is_local is False

    @pytest.mark.asyncio
    async def test_generate_text_success(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.return_value = make_gemini_response("Hello from Gemini!")

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=ChatRole.USER, content="Hello"),
        ]
        config = GenerationConfig(temperature=0.0, max_tokens=100)

        response = await provider.generate(messages, config=config)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from Gemini!"
        assert response.finish_reason == "stop"
        assert response.provider == "gemini"
        assert response.model == "gemini-2.0-flash"
        assert response.usage == TokenUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20)
        assert response.tool_calls == []

        # Verify SDK call arguments
        call_kwargs = mock_sdk.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.0-flash"
        gemini_cfg = call_kwargs["config"]
        assert gemini_cfg.system_instruction == "You are a helpful assistant."
        assert gemini_cfg.temperature == 0.0
        assert gemini_cfg.max_output_tokens == 100

    @pytest.mark.asyncio
    async def test_generate_with_tool_calling(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.return_value = make_gemini_response(
            text="",
            function_calls=[{"name": "search_docs", "args": {"query": "Quantiva architecture"}}],
        )

        tools = [
            ToolDefinition(
                name="search_docs",
                description="Search documents",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]

        messages = [ChatMessage(role=ChatRole.USER, content="Find architecture docs")]
        response = await provider.generate(messages, tools=tools)

        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search_docs"
        assert response.tool_calls[0].arguments == {"query": "Quantiva architecture"}

        # Verify tool declarations sent to Gemini
        gemini_cfg = mock_sdk.aio.models.generate_content.call_args.kwargs["config"]
        assert gemini_cfg.tools is not None
        assert gemini_cfg.automatic_function_calling.disable is True

    @pytest.mark.asyncio
    async def test_generate_with_tool_result_in_history(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.return_value = make_gemini_response("Based on the docs, here is the answer.")

        messages = [
            ChatMessage(role=ChatRole.USER, content="Search something"),
            ChatMessage(
                role=ChatRole.ASSISTANT,
                tool_calls=[ToolCallRequest(id="call_1", name="search_docs", arguments={"query": "test"})],
            ),
            ChatMessage(
                role=ChatRole.TOOL,
                tool_result=ToolCallResult(tool_call_id="call_1", name="search_docs", content='{"results": ["doc 1"]}'),
            ),
        ]

        response = await provider.generate(messages)
        assert response.content == "Based on the docs, here is the answer."

        contents = mock_sdk.aio.models.generate_content.call_args.kwargs["contents"]
        assert len(contents) == 3
        # Tool response turn is role='user' in Gemini Content
        assert contents[2].role == "user"
        assert contents[2].parts[0].function_response.name == "search_docs"
        assert contents[2].parts[0].function_response.response == {"result": '{"results": ["doc 1"]}'}

    @pytest.mark.asyncio
    async def test_generate_with_structured_output_schema(self, provider, mock_sdk):
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}
        mock_sdk.aio.models.generate_content.return_value = make_gemini_response('{"score": 0.95}')

        config = GenerationConfig(response_json_schema=schema)
        response = await provider.generate([ChatMessage(role=ChatRole.USER, content="score this")], config=config)

        assert response.content == '{"score": 0.95}'
        gemini_cfg = mock_sdk.aio.models.generate_content.call_args.kwargs["config"]
        assert gemini_cfg.response_mime_type == "application/json"
        assert gemini_cfg.response_schema == schema

    @pytest.mark.asyncio
    async def test_stream_chunks_and_final_sentinel(self, provider, mock_sdk):
        async def fake_stream():
            yield make_gemini_response("Quantiva ", prompt_tokens=5, completion_tokens=1, total_tokens=6)
            yield make_gemini_response("desktop ", prompt_tokens=5, completion_tokens=2, total_tokens=7)
            yield make_gemini_response("app.", prompt_tokens=5, completion_tokens=3, total_tokens=8)

        mock_sdk.aio.models.generate_content_stream.return_value = fake_stream()

        chunks: list[LLMStreamChunk] = []
        async for chunk in provider.stream([ChatMessage(role=ChatRole.USER, content="hi")]):
            chunks.append(chunk)

        assert len(chunks) == 4
        # Incremental text deltas
        assert chunks[0].delta_text == "Quantiva "
        assert chunks[0].is_final is False
        assert chunks[1].delta_text == "desktop "
        assert chunks[2].delta_text == "app."

        # Final sentinel chunk
        final_chunk = chunks[3]
        assert final_chunk.is_final is True
        assert final_chunk.delta_text == ""
        assert final_chunk.usage.total_tokens == 8

    # ── Error Translation Tests ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_error_translation_auth(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.side_effect = errors.APIError(403, "API key not valid. Please pass a valid API key.")
        with pytest.raises(ProviderAuthenticationError) as exc_info:
            await provider.generate([ChatMessage(role=ChatRole.USER, content="hi")])
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.retriable is False

    @pytest.mark.asyncio
    async def test_error_translation_rate_limit(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.side_effect = errors.APIError(429, "Resource has been exhausted (e.g. check quota).")
        with pytest.raises(ProviderRateLimitError) as exc_info:
            await provider.generate([ChatMessage(role=ChatRole.USER, content="hi")])
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.retriable is True

    @pytest.mark.asyncio
    async def test_error_translation_model_not_found(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.side_effect = errors.APIError(404, "models/gemini-invalid is not found")
        with pytest.raises(ModelNotFoundError) as exc_info:
            await provider.generate([ChatMessage(role=ChatRole.USER, content="hi")])
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.retriable is False

    @pytest.mark.asyncio
    async def test_error_translation_context_window(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.side_effect = errors.APIError(400, "Request exceeds maximum context token limit")
        with pytest.raises(ContextWindowExceededError) as exc_info:
            await provider.generate([ChatMessage(role=ChatRole.USER, content="hi")])
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.retriable is False

    @pytest.mark.asyncio
    async def test_error_translation_connection_error(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.side_effect = TimeoutError("Request timed out")
        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.generate([ChatMessage(role=ChatRole.USER, content="hi")])
        assert exc_info.value.provider == "gemini"
        assert exc_info.value.retriable is True

    # ── Connection Probe Tests ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_test_connection_success(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.return_value = make_gemini_response("pong")
        result = await provider.test_connection()
        assert isinstance(result, ConnectionTestResult)
        assert result.success is True
        assert result.latency_ms >= 0.0
        assert result.provider == "gemini"
        assert result.capabilities is not None

    @pytest.mark.asyncio
    async def test_test_connection_failure(self, provider, mock_sdk):
        mock_sdk.aio.models.generate_content.side_effect = errors.APIError(401, "API key expired")
        result = await provider.test_connection()
        assert isinstance(result, ConnectionTestResult)
        assert result.success is False
        assert result.error_code == "ProviderAuthenticationError"
        assert "API key" in result.error_message


# ═══════════════════════════════════════════════════════════════════════════════
# GeminiEmbeddingProvider Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeminiEmbeddingProvider:
    @pytest.fixture
    def mock_sdk(self):
        sdk = MagicMock()
        sdk.aio.models.embed_content = AsyncMock()
        return sdk

    @pytest.fixture
    def provider(self, mock_sdk):
        return GeminiEmbeddingProvider(
            api_key="test-api-key",
            model="gemini-embedding-001",
            dimensions=768,
            client=mock_sdk,
        )

    def _make_embed_response(self, count: int, dim: int = 768) -> types.EmbedContentResponse:
        embeddings = [
            types.ContentEmbedding(values=[1.0 / (i + 1)] * dim)
            for i in range(count)
        ]
        return types.EmbedContentResponse(embeddings=embeddings)

    def test_identity_and_dimensions(self, provider):
        assert provider.provider_name() == "gemini"
        assert provider.model_name() == "gemini-embedding-001"
        assert provider.get_dimensions() == 768

    @pytest.mark.asyncio
    async def test_embed_query_success(self, provider, mock_sdk):
        mock_sdk.aio.models.embed_content.return_value = self._make_embed_response(1, 768)

        vector = await provider.embed_query("What is Quantiva?")

        assert len(vector) == 768
        # Check that vector is L2-normalized (length should be close to 1.0)
        norm = sum(x * x for x in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-5

        # Verify task_type was set to RETRIEVAL_QUERY
        cfg = mock_sdk.aio.models.embed_content.call_args.kwargs["config"]
        assert cfg.task_type == "RETRIEVAL_QUERY"
        assert cfg.output_dimensionality == 768

    @pytest.mark.asyncio
    async def test_embed_documents_batching(self, provider, mock_sdk):
        # 5 documents with batch_size=2 -> 3 calls
        mock_sdk.aio.models.embed_content.side_effect = [
            self._make_embed_response(2, 768),
            self._make_embed_response(2, 768),
            self._make_embed_response(1, 768),
        ]

        texts = ["doc 1", "doc 2", "doc 3", "doc 4", "doc 5"]
        vectors = await provider.embed_documents(texts, batch_size=2)

        assert len(vectors) == 5
        assert all(len(v) == 768 for v in vectors)
        assert mock_sdk.aio.models.embed_content.call_count == 3

        # Verify task_type was set to RETRIEVAL_DOCUMENT
        cfg = mock_sdk.aio.models.embed_content.call_args.kwargs["config"]
        assert cfg.task_type == "RETRIEVAL_DOCUMENT"

    @pytest.mark.asyncio
    async def test_embed_dimension_mismatch_raises(self, provider, mock_sdk):
        # SDK returns 512d vectors when provider expects 768d
        mock_sdk.aio.models.embed_content.return_value = self._make_embed_response(1, 512)

        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            await provider.embed_query("test query")

        assert exc_info.value.expected == 768
        assert exc_info.value.actual == 512

    @pytest.mark.asyncio
    async def test_embed_invalid_inputs(self, provider):
        with pytest.raises(ValueError):
            await provider.embed_query("")
        with pytest.raises(ValueError):
            await provider.embed_query("   ")
        with pytest.raises(ValueError):
            await provider.embed_documents([])
        with pytest.raises(ValueError):
            await provider.embed_documents(["valid", ""])

    @pytest.mark.asyncio
    async def test_embed_error_translation(self, provider, mock_sdk):
        mock_sdk.aio.models.embed_content.side_effect = errors.APIError(429, "Quota exceeded")

        with pytest.raises(ProviderRateLimitError):
            await provider.embed_query("test query")

    @pytest.mark.asyncio
    async def test_test_embedding_success(self, provider, mock_sdk):
        mock_sdk.aio.models.embed_content.return_value = self._make_embed_response(1, 768)

        result = await provider.test_embedding()
        assert result.success is True
        assert result.detected_dimensions == 768
        assert result.provider == "gemini"
        assert result.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_test_embedding_failure(self, provider, mock_sdk):
        mock_sdk.aio.models.embed_content.side_effect = errors.APIError(403, "Invalid API key")

        result = await provider.test_embedding()
        assert result.success is False
        assert result.detected_dimensions is None
        assert result.error_code == "ProviderAuthenticationError"
