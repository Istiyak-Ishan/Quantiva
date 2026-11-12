"""
tests/providers/test_openai_and_anthropic_providers.py
──────────────────────────────────────────────────────────────────────────────
Offline unit tests for the generic OpenAI-compatible, LM Studio, llama.cpp,
OpenAI official, Anthropic, and API embedding adapters.

All HTTP traffic is served by :class:`httpx.MockTransport`; no network or
vendor SDK is required.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.providers.embeddings.api import (
    LlamaCppEmbeddingProvider,
    LMStudioEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
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
    GenerationConfig,
    ToolCallRequest,
    ToolCallResult,
    ToolDefinition,
)
from app.providers.llm.anthropic import AnthropicLLMProvider
from app.providers.llm.openai_compatible import (
    LlamaCppLLMProvider,
    LMStudioLLMProvider,
    OpenAICompatibleLLMProvider,
    OpenAILLMProvider,
)

OPENAI_BASE = "https://api.openai.com/v1"
LOCAL_BASE = "http://127.0.0.1:8080/v1"
MODEL = "test-model"


# ═══════════════════════════════════════════════════════════════════════════════
# Mock HTTP helpers
# ═══════════════════════════════════════════════════════════════════════════════

Route = Callable[[httpx.Request], httpx.Response]


def json_route(payload: Any, status_code: int = 200) -> Route:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return _handler


def error_route(status_code: int, message: str) -> Route:
    return json_route({"error": {"message": message}}, status_code=status_code)


def sse_route(data_payloads: list[str]) -> Route:
    body = "\n\n".join(f"data: {payload}" for payload in data_payloads) + "\n\n"

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    return _handler


def anthropic_sse_route(events: list[dict[str, Any]]) -> Route:
    chunks = [
        f"event: {event['type']}\ndata: {json.dumps(event)}" for event in events
    ]
    body = "\n\n".join(chunks) + "\n\n"

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    return _handler


def raise_route(exc_factory: Callable[[httpx.Request], Exception]) -> Route:
    def _handler(request: httpx.Request) -> httpx.Response:
        raise exc_factory(request)

    return _handler


class Recorder:
    def __init__(self, routes: dict[str, Route]) -> None:
        self.routes = routes
        self.requests: list[tuple[str, str, dict[str, Any], dict[str, str]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content) if request.content else {}
        except json.JSONDecodeError:
            body = {}
        headers = {key.lower(): value for key, value in request.headers.items()}
        self.requests.append((request.method, request.url.path, body, headers))

        route = self.routes.get(request.url.path)
        if route is None:
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return route(request)

    def paths(self) -> list[str]:
        return [path for _, path, _, _ in self.requests]

    def payload(self, path: str) -> dict[str, Any]:
        for _, request_path, body, _ in self.requests:
            if request_path == path:
                return body
        raise AssertionError(f"No request recorded for {path}")

    def headers(self, path: str) -> dict[str, str]:
        for _, request_path, _, headers in self.requests:
            if request_path == path:
                return headers
        raise AssertionError(f"No request recorded for {path}")


def make_client(recorder: Recorder, base_url: str = OPENAI_BASE) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(recorder),
        base_url=base_url,
    )


def make_openai_provider(
    routes: dict[str, Route],
    *,
    base_url: str = OPENAI_BASE,
    client: httpx.AsyncClient | None = None,
    **kwargs: Any,
) -> tuple[OpenAICompatibleLLMProvider, Recorder]:
    recorder = Recorder(routes)
    provider = OpenAICompatibleLLMProvider(
        base_url=base_url,
        model=MODEL,
        api_key="test-key",
        client=client or make_client(recorder, base_url),
        **kwargs,
    )
    return provider, recorder


def make_anthropic_provider(
    routes: dict[str, Route],
    *,
    client: httpx.AsyncClient | None = None,
    **kwargs: Any,
) -> tuple[AnthropicLLMProvider, Recorder]:
    recorder = Recorder(routes)
    provider = AnthropicLLMProvider(
        base_url="https://api.anthropic.com",
        model="claude-test",
        api_key="test-anthropic-key",
        client=client or make_client(recorder),
        **kwargs,
    )
    return provider, recorder


def make_embedding_provider(
    routes: dict[str, Route],
    *,
    base_url: str = OPENAI_BASE,
    client: httpx.AsyncClient | None = None,
    **kwargs: Any,
) -> tuple[OpenAICompatibleEmbeddingProvider, Recorder]:
    recorder = Recorder(routes)
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=base_url,
        model="text-embedding-3-small",
        api_key="embed-key",
        client=client or make_client(recorder, base_url),
        **kwargs,
    )
    return provider, recorder


def openai_chat_response(
    content: str = "Hello",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible LLM
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenAICompatibleLLM:
    def test_identity_and_capabilities(self) -> None:
        provider = OpenAICompatibleLLMProvider(
            base_url=LOCAL_BASE,
            model="qwen2.5-7b-instruct",
            provider_label="openai_compatible",
            is_local=True,
        )
        assert provider.provider_name() == "openai_compatible"
        assert provider.model_name() == "qwen2.5-7b-instruct"
        assert provider.base_url() == LOCAL_BASE
        caps = provider.get_capabilities()
        assert caps.is_local is True
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.supports_structured_output is True

    def test_preset_constructors(self) -> None:
        lm = LMStudioLLMProvider(model="local-llm")
        assert lm.provider_name() == "lm_studio"
        assert lm.base_url() == "http://127.0.0.1:1234/v1"
        assert lm.get_capabilities().is_local is True

        llama = LlamaCppLLMProvider(model="gguf-model")
        assert llama.provider_name() == "llama_cpp"
        assert llama.base_url() == "http://127.0.0.1:8080/v1"
        assert llama.get_capabilities().is_local is True

        official = OpenAILLMProvider(model="gpt-4o-mini", api_key="sk-test")
        assert official.provider_name() == "openai"
        assert official.base_url() == OPENAI_BASE
        assert official.get_capabilities().is_local is False

        assert isinstance(
            OpenAICompatibleLLMProvider.for_lm_studio("m"), LMStudioLLMProvider
        )
        assert isinstance(
            OpenAICompatibleLLMProvider.for_llama_cpp("m"), LlamaCppLLMProvider
        )
        assert isinstance(
            OpenAICompatibleLLMProvider.for_openai("gpt-4o-mini"),
            OpenAILLMProvider,
        )

    @pytest.mark.asyncio
    async def test_generate_text_payload_and_auth_headers(self) -> None:
        provider, recorder = make_openai_provider(
            {"/v1/chat/completions": json_route(openai_chat_response("Hello"))}
        )

        response = await provider.generate(
            [
                ChatMessage(role=ChatRole.SYSTEM, content="Be concise."),
                ChatMessage(role=ChatRole.USER, content="Hi"),
            ],
            config=GenerationConfig(
                temperature=0.0,
                top_p=0.5,
                max_tokens=64,
                stop_sequences=["END"],
            ),
        )

        assert response.content == "Hello"
        assert response.provider == "openai_compatible"
        assert response.model == MODEL
        assert response.finish_reason == "stop"
        assert response.usage is not None
        assert response.usage.total_tokens == 20
        assert response.raw_metadata["id"] == "chatcmpl-test"

        payload = recorder.payload("/v1/chat/completions")
        assert payload["model"] == MODEL
        assert payload["stream"] is False
        assert payload["messages"] == [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 0.5
        assert payload["max_tokens"] == 64
        assert payload["stop"] == ["END"]
        assert recorder.headers("/v1/chat/completions")["authorization"] == (
            "Bearer test-key"
        )

    @pytest.mark.asyncio
    async def test_generate_tool_calls_and_tool_history(self) -> None:
        provider, recorder = make_openai_provider(
            {
                "/v1/chat/completions": json_route(
                    openai_chat_response(
                        "",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_docs",
                                    "arguments": '{"query": "architecture"}',
                                },
                            }
                        ],
                        finish_reason="tool_calls",
                    )
                )
            }
        )
        tools = [
            ToolDefinition(
                name="search_docs",
                description="Search documents",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        ]

        response = await provider.generate(
            [ChatMessage(role=ChatRole.USER, content="find docs")], tools=tools
        )

        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_1"
        assert response.tool_calls[0].name == "search_docs"
        assert response.tool_calls[0].arguments == {"query": "architecture"}

        payload = recorder.payload("/v1/chat/completions")
        assert payload["tools"][0]["function"]["name"] == "search_docs"

        # Verify an assistant tool-call turn and tool result turn translate.
        await provider.generate(
            [
                ChatMessage(role=ChatRole.USER, content="search"),
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    tool_calls=[
                        ToolCallRequest(
                            id="call_1",
                            name="search_docs",
                            arguments={"query": "x"},
                        )
                    ],
                ),
                ChatMessage(
                    role=ChatRole.TOOL,
                    tool_result=ToolCallResult(
                        tool_call_id="call_1",
                        name="search_docs",
                        content="result",
                    ),
                ),
            ]
        )
        messages = recorder.requests[-1][2]["messages"]
        assert messages[1]["tool_calls"][0]["function"]["name"] == "search_docs"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_generate_json_schema_and_json_mode(self) -> None:
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}
        provider, recorder = make_openai_provider(
            {"/v1/chat/completions": json_route(openai_chat_response("{}"))}
        )

        await provider.generate(
            [ChatMessage(role=ChatRole.USER, content="score")],
            config=GenerationConfig(response_json_schema=schema),
        )
        response_format = recorder.payload("/v1/chat/completions")[
            "response_format"
        ]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"] == schema

        json_provider, json_recorder = make_openai_provider(
            {"/v1/chat/completions": json_route(openai_chat_response("{}"))},
            json_mode=True,
        )
        await json_provider.generate(
            [ChatMessage(role=ChatRole.USER, content="score")]
        )
        assert json_recorder.payload("/v1/chat/completions")[
            "response_format"
        ] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_stream_sse_text_and_usage(self) -> None:
        events = [
            json.dumps(
                {
                    "id": "chatcmpl-stream",
                    "choices": [
                        {"delta": {"role": "assistant", "content": "Quantiva "}}
                    ],
                }
            ),
            json.dumps(
                {"choices": [{"delta": {"content": "desktop"}, "finish_reason": None}]}
            ),
            json.dumps(
                {"choices": [{"delta": {"content": " app."}, "finish_reason": "stop"}]}
            ),
            json.dumps(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                }
            ),
            "[DONE]",
        ]
        provider, recorder = make_openai_provider(
            {"/v1/chat/completions": sse_route(events)}
        )

        chunks = [
            chunk
            async for chunk in provider.stream(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )
        ]

        visible = "".join(
            chunk.delta_text for chunk in chunks if not chunk.is_final
        )
        assert visible == "Quantiva desktop app."
        final = chunks[-1]
        assert final.is_final is True
        assert final.finish_reason == "stop"
        assert final.usage is not None
        assert final.usage.total_tokens == 8
        assert recorder.payload("/v1/chat/completions")["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_assembles_tool_call_across_chunks(self) -> None:
        events = [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_stream",
                                        "type": "function",
                                        "function": {
                                            "name": "search_docs",
                                            "arguments": '{"query":',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": '"quantiva"}'
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
            "[DONE]",
        ]
        provider, _ = make_openai_provider(
            {"/v1/chat/completions": sse_route(events)}
        )
        tools = [
            ToolDefinition(
                name="search_docs",
                description="Search",
                parameters_json_schema={"type": "object"},
            )
        ]

        chunks = [
            chunk
            async for chunk in provider.stream(
                [ChatMessage(role=ChatRole.USER, content="search")], tools=tools
            )
        ]

        tool_chunks = [chunk for chunk in chunks if chunk.tool_call_delta]
        assert len(tool_chunks) == 1
        call = tool_chunks[0].tool_call_delta
        assert call.id == "call_stream"
        assert call.name == "search_docs"
        assert call.arguments == {"query": "quantiva"}
        assert chunks[-1].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_tool_call_with_dict_arguments_fragment(self) -> None:
        events = [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_dict",
                                        "function": {
                                            "name": "chart",
                                            "arguments": {"series": [1, 2]},
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
            "[DONE]",
        ]
        provider, _ = make_openai_provider(
            {"/v1/chat/completions": sse_route(events)}
        )

        chunks = [
            chunk
            async for chunk in provider.stream(
                [ChatMessage(role=ChatRole.USER, content="chart")]
            )
        ]

        call = next(chunk.tool_call_delta for chunk in chunks if chunk.tool_call_delta)
        assert call.arguments == {"series": [1, 2]}

    @pytest.mark.asyncio
    async def test_http_error_normalisation(self) -> None:
        cases = [
            (401, "invalid api key", ProviderAuthenticationError),
            (404, "model not found", ModelNotFoundError),
            (429, "rate limit", ProviderRateLimitError),
            (400, "maximum context length exceeded", ContextWindowExceededError),
            (500, "internal error", ProviderError),
        ]
        for status, message, expected in cases:
            provider, _ = make_openai_provider(
                {
                    "/v1/chat/completions": error_route(status, message),
                }
            )
            with pytest.raises(expected):
                await provider.generate(
                    [ChatMessage(role=ChatRole.USER, content="hi")]
                )

    @pytest.mark.asyncio
    async def test_connect_error_for_local_runtime(self) -> None:
        provider, _ = make_openai_provider(
            {
                "/v1/chat/completions": raise_route(
                    lambda request: httpx.ConnectError(
                        "refused", request=request
                    )
                )
            },
            base_url="http://127.0.0.1:1234/v1",
            is_local=True,
            provider_label="lm_studio",
        )

        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )

        assert "lm_studio" in str(exc_info.value)
        assert "127.0.0.1:1234" in str(exc_info.value)
        assert exc_info.value.retriable is True

    @pytest.mark.asyncio
    async def test_openai_requires_api_key(self) -> None:
        recorder = Recorder({})
        provider = OpenAILLMProvider(
            model="gpt-4o-mini",
            api_key="",
            client=make_client(recorder),
        )

        with pytest.raises(ProviderAuthenticationError):
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )
        assert recorder.paths() == []

    @pytest.mark.asyncio
    async def test_connection_probe_success_and_failure(self) -> None:
        provider, _ = make_openai_provider(
            {"/v1/chat/completions": json_route(openai_chat_response("pong"))}
        )
        result = await provider.test_connection()
        assert result.success is True
        assert result.capabilities is not None

        failure, _ = make_openai_provider(
            {
                "/v1/chat/completions": raise_route(
                    lambda request: httpx.ConnectError(
                        "refused", request=request
                    )
                )
            }
        )
        failed = await failure.test_connection()
        assert failed.success is False
        assert failed.error_code == "ProviderConnectionError"


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnthropicLLM:
    def test_identity_and_capabilities(self) -> None:
        provider = AnthropicLLMProvider(
            model="claude-test", api_key="key"
        )
        assert provider.provider_name() == "anthropic"
        assert provider.model_name() == "claude-test"
        caps = provider.get_capabilities()
        assert caps.is_local is False
        assert caps.supports_streaming is True
        assert caps.supports_function_calling is True
        assert caps.max_context_window == 200_000

    @pytest.mark.asyncio
    async def test_generate_system_tools_and_tool_use_blocks(self) -> None:
        response_payload = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [
                {"type": "text", "text": "Let me search."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search_docs",
                    "input": {"query": "architecture"},
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
        provider, recorder = make_anthropic_provider(
            {"/v1/messages": json_route(response_payload)}
        )
        tools = [
            ToolDefinition(
                name="search_docs",
                description="Search documents",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        ]

        response = await provider.generate(
            [
                ChatMessage(role=ChatRole.SYSTEM, content="Be concise."),
                ChatMessage(role=ChatRole.USER, content="Find docs"),
            ],
            config=GenerationConfig(temperature=0.2, max_tokens=512),
            tools=tools,
        )

        assert response.content == "Let me search."
        assert response.finish_reason == "tool_calls"
        assert response.tool_calls[0].id == "toolu_1"
        assert response.tool_calls[0].arguments == {"query": "architecture"}
        assert response.usage is not None
        assert response.usage.total_tokens == 30
        assert response.raw_metadata["stop_reason"] == "tool_use"

        payload = recorder.payload("/v1/messages")
        assert payload["system"] == "Be concise."
        assert payload["max_tokens"] == 512
        assert payload["messages"] == [{"role": "user", "content": "Find docs"}]
        assert payload["tools"][0]["input_schema"]["type"] == "object"
        assert payload["stream"] is False
        headers = recorder.headers("/v1/messages")
        assert headers["x-api-key"] == "test-anthropic-key"
        assert headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    async def test_tool_result_turns_are_user_content_blocks(self) -> None:
        provider, recorder = make_anthropic_provider(
            {
                "/v1/messages": json_route(
                    {
                        "id": "msg_2",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Done"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                )
            }
        )

        response = await provider.generate(
            [
                ChatMessage(role=ChatRole.USER, content="search"),
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    tool_calls=[
                        ToolCallRequest(
                            id="toolu_1",
                            name="search_docs",
                            arguments={"query": "x"},
                        )
                    ],
                ),
                ChatMessage(
                    role=ChatRole.TOOL,
                    tool_result=ToolCallResult(
                        tool_call_id="toolu_1",
                        name="search_docs",
                        content="result",
                        is_error=True,
                    ),
                ),
            ]
        )

        assert response.content == "Done"
        assert response.finish_reason == "stop"
        messages = recorder.payload("/v1/messages")["messages"]
        assert messages[0] == {"role": "user", "content": "search"}
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"][0]["type"] == "tool_use"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["is_error"] is True

    @pytest.mark.asyncio
    async def test_stream_text_events(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stream",
                    "model": "claude-test",
                    "usage": {"input_tokens": 7, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Quantiva "},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "desktop"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            },
            {"type": "message_stop"},
        ]
        provider, recorder = make_anthropic_provider(
            {"/v1/messages": anthropic_sse_route(events)}
        )

        chunks = [
            chunk
            async for chunk in provider.stream(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )
        ]

        visible = "".join(
            chunk.delta_text for chunk in chunks if not chunk.is_final
        )
        assert visible == "Quantiva desktop"
        final = chunks[-1]
        assert final.is_final is True
        assert final.finish_reason == "stop"
        assert final.usage is not None
        assert final.usage.prompt_tokens == 7
        assert final.usage.completion_tokens == 4
        assert recorder.payload("/v1/messages")["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_tool_call_assembly(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 9, "output_tokens": 0}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_stream",
                    "name": "search_docs",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"query":',
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '"quantiva"}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 6},
            },
            {"type": "message_stop"},
        ]
        provider, _ = make_anthropic_provider(
            {"/v1/messages": anthropic_sse_route(events)}
        )
        tools = [
            ToolDefinition(
                name="search_docs",
                description="Search",
                parameters_json_schema={"type": "object"},
            )
        ]

        chunks = [
            chunk
            async for chunk in provider.stream(
                [ChatMessage(role=ChatRole.USER, content="search")], tools=tools
            )
        ]

        tool_chunks = [chunk for chunk in chunks if chunk.tool_call_delta]
        assert len(tool_chunks) == 1
        call = tool_chunks[0].tool_call_delta
        assert call.id == "toolu_stream"
        assert call.name == "search_docs"
        assert call.arguments == {"query": "quantiva"}
        assert chunks[-1].finish_reason == "tool_calls"
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_http_error_normalisation(self) -> None:
        cases = [
            (401, "authentication_error", ProviderAuthenticationError),
            (404, "not_found_error", ModelNotFoundError),
            (429, "rate_limit_error", ProviderRateLimitError),
            (400, "maximum context length exceeded", ContextWindowExceededError),
            (529, "overloaded_error", ProviderError),
        ]
        for status, message, expected in cases:
            provider, _ = make_anthropic_provider(
                {"/v1/messages": error_route(status, message)}
            )
            with pytest.raises(expected):
                await provider.generate(
                    [ChatMessage(role=ChatRole.USER, content="hi")]
                )

    @pytest.mark.asyncio
    async def test_requires_api_key(self) -> None:
        recorder = Recorder({})
        provider = AnthropicLLMProvider(
            model="claude-test", api_key="", client=make_client(recorder)
        )

        with pytest.raises(ProviderAuthenticationError):
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )
        assert recorder.paths() == []

    @pytest.mark.asyncio
    async def test_connection_probe_success_and_failure(self) -> None:
        provider, _ = make_anthropic_provider(
            {
                "/v1/messages": json_route(
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "pong"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                )
            }
        )
        result = await provider.test_connection()
        assert result.success is True
        assert result.capabilities is not None

        failure, _ = make_anthropic_provider(
            {
                "/v1/messages": raise_route(
                    lambda request: httpx.ConnectError(
                        "refused", request=request
                    )
                )
            }
        )
        failed = await failure.test_connection()
        assert failed.success is False
        assert failed.error_code == "ProviderConnectionError"


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI-compatible embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpenAICompatibleEmbeddings:
    def test_identity_and_presets(self) -> None:
        provider = OpenAICompatibleEmbeddingProvider(
            base_url=LOCAL_BASE, model="embed-model"
        )
        assert provider.provider_name() == "openai_compatible"
        assert provider.model_name() == "embed-model"
        assert provider.get_dimensions() == 0

        official = OpenAIEmbeddingProvider(api_key="sk-test")
        assert official.provider_name() == "openai"
        assert official.model_name() == "text-embedding-3-small"

        lm = LMStudioEmbeddingProvider(model="local-embed")
        assert lm.provider_name() == "lm_studio"
        assert lm.base_url() == "http://127.0.0.1:1234/v1"

        llama = LlamaCppEmbeddingProvider(model="local-embed")
        assert llama.provider_name() == "llama_cpp"
        assert llama.base_url() == "http://127.0.0.1:8080/v1"

        assert isinstance(
            OpenAICompatibleEmbeddingProvider.for_openai(), OpenAIEmbeddingProvider
        )

    @pytest.mark.asyncio
    async def test_batch_embedding_and_dimension_detection(self) -> None:
        def _embed(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": [3.0, 4.0],
                        }
                        for index, _text in enumerate(body["input"])
                    ],
                    "model": "text-embedding-3-small",
                },
            )

        recorder = Recorder({"/v1/embeddings": _embed})
        provider = OpenAICompatibleEmbeddingProvider(
            base_url=OPENAI_BASE,
            model="text-embedding-3-small",
            api_key="embed-key",
            client=make_client(recorder),
        )

        vectors = await provider.embed_documents(["alpha", "beta"])

        assert provider.get_dimensions() == 2
        assert vectors == [[0.6, 0.8], [0.6, 0.8]]
        payload = recorder.payload("/v1/embeddings")
        assert payload["input"] == ["alpha", "beta"]
        assert payload["model"] == "text-embedding-3-small"
        assert payload["encoding_format"] == "float"
        assert recorder.headers("/v1/embeddings")["authorization"] == (
            "Bearer embed-key"
        )

    @pytest.mark.asyncio
    async def test_batching_respects_batch_size(self) -> None:
        batches: list[list[str]] = []

        def _embed(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            batches.append(body["input"])
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "index": index,
                            "embedding": [float(len(text)), 1.0],
                        }
                        for index, text in enumerate(body["input"])
                    ]
                },
            )

        recorder = Recorder({"/v1/embeddings": _embed})
        provider = OpenAICompatibleEmbeddingProvider(
            base_url=OPENAI_BASE,
            model="text-embedding-3-small",
            api_key="k",
            client=make_client(recorder),
        )

        vectors = await provider.embed_documents(
            ["a", "bb", "ccc"], batch_size=2
        )

        assert batches == [["a", "bb"], ["ccc"]]
        assert len(vectors) == 3

    @pytest.mark.asyncio
    async def test_vectors_are_sorted_by_index(self) -> None:
        def _embed(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 5.0]},
                        {"index": 0, "embedding": [5.0, 0.0]},
                    ]
                },
            )

        recorder = Recorder({"/v1/embeddings": _embed})
        provider = OpenAICompatibleEmbeddingProvider(
            base_url=OPENAI_BASE,
            model="text-embedding-3-small",
            api_key="k",
            client=make_client(recorder),
        )

        vectors = await provider.embed_documents(["first", "second"])

        assert vectors == [[1.0, 0.0], [0.0, 1.0]]

    @pytest.mark.asyncio
    async def test_dimension_mismatch_and_empty_input(self) -> None:
        recorder = Recorder(
            {
                "/v1/embeddings": json_route(
                    {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}
                )
            }
        )
        provider = OpenAICompatibleEmbeddingProvider(
            base_url=OPENAI_BASE,
            model="text-embedding-3-small",
            api_key="k",
            dimensions=3,
            client=make_client(recorder),
        )

        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            await provider.embed_documents(["alpha"])
        assert exc_info.value.expected == 3
        assert exc_info.value.actual == 2

        with pytest.raises(ValueError):
            await provider.embed_documents([])
        with pytest.raises(ValueError):
            await provider.embed_documents(["ok", " "])


    @pytest.mark.asyncio
    async def test_http_error_normalisation(self) -> None:
        cases = [
            (401, "invalid api key", ProviderAuthenticationError),
            (404, "model not found", ModelNotFoundError),
            (429, "rate limit", ProviderRateLimitError),
            (500, "internal error", ProviderError),
        ]
        for status, message, expected in cases:
            provider, _ = make_embedding_provider(
                {
                    "/v1/embeddings": error_route(status, message),
                }
            )
            with pytest.raises(expected):
                await provider.embed_query("hello")

    @pytest.mark.asyncio
    async def test_openai_embedding_requires_api_key(self) -> None:
        recorder = Recorder({})
        provider = OpenAIEmbeddingProvider(
            api_key="", client=make_client(recorder)
        )

        with pytest.raises(ProviderAuthenticationError):
            await provider.embed_query("hello")
        assert recorder.paths() == []

    @pytest.mark.asyncio
    async def test_test_embedding_success_and_connection_failure(self) -> None:
        success_provider, _ = make_embedding_provider(
            {
                "/v1/embeddings": json_route(
                    {"data": [{"index": 0, "embedding": [1.0, 1.0, 1.0, 1.0]}]}
                )
            }
        )
        success = await success_provider.test_embedding()
        assert success.success is True
        assert success.detected_dimensions == 4
        assert success.provider == "openai_compatible"

        failure_provider, _ = make_embedding_provider(
            {
                "/v1/embeddings": raise_route(
                    lambda request: httpx.ConnectError(
                        "refused", request=request
                    )
                )
            },
            is_local=True,
            provider_label="lm_studio",
        )
        failure = await failure_provider.test_embedding()
        assert failure.success is False
        assert failure.error_code == "ProviderConnectionError"
        assert "lm_studio" in (failure.error_message or "")
