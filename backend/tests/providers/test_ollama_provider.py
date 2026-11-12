"""
tests/providers/test_ollama_provider.py
──────────────────────────────────────────────────────────────────────────────
Offline unit tests for the first-class Ollama LLM, embedding, and runtime
adapters.

Every HTTP interaction is served by :class:`httpx.MockTransport`, so the suite
never requires a running Ollama daemon.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.providers.embeddings.ollama import OllamaEmbeddingProvider
from app.providers.interfaces.exceptions import (
    CapabilityNotSupportedError,
    ContextWindowExceededError,
    EmbeddingDimensionMismatchError,
    ModelNotFoundError,
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
from app.providers.llm.ollama import OllamaLLMProvider, OllamaRuntimeManager

DEFAULT_BASE = "http://127.0.0.1:11434"
MODEL = "test-model"

DAEMON_MESSAGE = (
    "Cannot connect to Ollama at http://127.0.0.1:11434. "
    "Ensure Ollama is installed and running."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mock HTTP helpers
# ═══════════════════════════════════════════════════════════════════════════════

Route = Callable[[httpx.Request], httpx.Response]


def json_route(payload: Any, status_code: int = 200) -> Route:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return _handler


def ndjson_route(events: list[dict[str, Any]]) -> Route:
    body = "\n".join(json.dumps(event) for event in events) + "\n"

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body.encode("utf-8"),
            headers={"content-type": "application/x-ndjson"},
        )

    return _handler


def error_route(status_code: int, message: str) -> Route:
    return json_route({"error": message}, status_code=status_code)


def raise_route(exc_factory: Callable[[httpx.Request], Exception]) -> Route:
    def _handler(request: httpx.Request) -> httpx.Response:
        raise exc_factory(request)

    return _handler


class Recorder:
    """Records requests and dispatches them to path-based routes."""

    def __init__(self, routes: dict[str, Route]) -> None:
        self.routes = routes
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content) if request.content else {}
        except json.JSONDecodeError:
            body = {}
        self.requests.append((request.method, request.url.path, body))

        route = self.routes.get(request.url.path)
        if route is None:
            return httpx.Response(404, json={"error": "404 page not found"})
        return route(request)

    def paths(self) -> list[str]:
        return [path for _, path, _ in self.requests]

    def payload(self, path: str) -> dict[str, Any]:
        for _, request_path, body in self.requests:
            if request_path == path:
                return body
        raise AssertionError(f"No request recorded for {path}")


def make_client(recorder: Recorder) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(recorder),
        base_url=DEFAULT_BASE,
    )


def show_payload(
    *,
    capabilities: list[str] | None = None,
    context_length: int = 32_768,
    template: str = "{{ .System }}{{ .Prompt }}",
    parameters: str = "",
    family: str = "qwen2",
    model_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "template": template,
        "parameters": parameters,
        "details": {
            "family": family,
            "parameter_size": "7B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": model_info
        if model_info is not None
        else {"qwen2.context_length": context_length},
    }
    if capabilities is not None:
        payload["capabilities"] = capabilities
    return payload


def chat_payload(
    content: str = "Hello from Ollama",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    done_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    thinking: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if thinking is not None:
        message["thinking"] = thinking
    return {
        "model": MODEL,
        "created_at": "2026-01-01T00:00:00Z",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
        "total_duration": 123_456,
        "load_duration": 1_000,
        "prompt_eval_duration": 2_000,
        "eval_duration": 3_000,
    }


def make_provider(
    routes: dict[str, Route],
    model: str = MODEL,
    **kwargs: Any,
) -> tuple[OllamaLLMProvider, Recorder]:
    recorder = Recorder(routes)
    provider = OllamaLLMProvider(
        base_url=DEFAULT_BASE,
        model=model,
        client=make_client(recorder),
        **kwargs,
    )
    return provider, recorder


def make_embedding_provider(
    routes: dict[str, Route],
    **kwargs: Any,
) -> tuple[OllamaEmbeddingProvider, Recorder]:
    recorder = Recorder(routes)
    provider = OllamaEmbeddingProvider(
        base_url=DEFAULT_BASE,
        model="nomic-embed-text",
        client=make_client(recorder),
        **kwargs,
    )
    return provider, recorder


@pytest.fixture(autouse=True)
def clear_capability_cache() -> Any:
    OllamaLLMProvider.clear_capability_cache()
    yield
    OllamaLLMProvider.clear_capability_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# OllamaLLMProvider — identity & capabilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaLLMCapabilities:
    def test_identity_and_offline_capabilities(self) -> None:
        provider = OllamaLLMProvider(model="llama3.2")
        try:
            assert provider.provider_name() == "ollama"
            assert provider.model_name() == "llama3.2"
            assert provider.base_url() == DEFAULT_BASE
            caps = provider.get_capabilities()
            assert caps.is_local is True
            assert caps.supports_streaming is True
            assert caps.supports_function_calling is True
            assert caps.max_context_window == 131_072
        finally:
            # No client was ever created, so this is a no-op safety net.
            pass

    @pytest.mark.asyncio
    async def test_modern_capabilities_payload_is_cached(self) -> None:
        show = show_payload(
            capabilities=["completion", "tools", "vision"],
            context_length=65_536,
        )
        provider, recorder = make_provider({"/api/show": json_route(show)})

        first = await provider.detect_capabilities()
        second = await provider.detect_capabilities()

        assert first is second
        assert first.supports_function_calling is True
        assert first.supports_vision is True
        assert first.max_context_window == 65_536
        assert recorder.paths().count("/api/show") == 1
        assert provider.get_capabilities() is first

    @pytest.mark.asyncio
    async def test_template_and_num_ctx_detection(self) -> None:
        show = show_payload(
            capabilities=None,
            template="{{ if .Tools }}{{ .Tools }}{{ end }}",
            parameters="num_ctx 4096",
            context_length=131_072,
            family="qwen2",
        )
        provider, _ = make_provider({"/api/show": json_route(show)})

        caps = await provider.detect_capabilities()

        assert caps.supports_function_calling is True
        assert caps.supports_vision is False
        assert caps.max_context_window == 4096

    @pytest.mark.asyncio
    async def test_vision_metadata_detection(self) -> None:
        show = show_payload(
            capabilities=None,
            family="llava",
            model_info={
                "llama.context_length": 4096,
                "clip.has_vision_encoder": True,
            },
        )
        provider, _ = make_provider({"/api/show": json_route(show)})

        caps = await provider.detect_capabilities()

        assert caps.supports_vision is True
        assert caps.max_context_window == 4096

    @pytest.mark.asyncio
    async def test_show_failure_falls_back_to_heuristics(self) -> None:
        provider, _ = make_provider(
            {"/api/show": error_route(500, "internal server error")},
            model="qwen2.5-coder",
        )

        caps = await provider.detect_capabilities()

        assert caps.supports_function_calling is True
        assert caps.max_context_window == 32_768
        # Heuristic fallbacks are deliberately not cached.
        assert provider.get_capabilities().max_context_window == 32_768


# ═══════════════════════════════════════════════════════════════════════════════
# OllamaLLMProvider — generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaLLMGenerate:
    @pytest.mark.asyncio
    async def test_generate_text_and_request_translation(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": json_route(chat_payload("Hello from Ollama")),
        }
        provider, recorder = make_provider(routes)

        response = await provider.generate(
            [
                ChatMessage(role=ChatRole.SYSTEM, content="Be concise."),
                ChatMessage(role=ChatRole.USER, content="Hello"),
            ],
            config=GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                max_tokens=64,
                stop_sequences=["END"],
            ),
        )

        assert response.content == "Hello from Ollama"
        assert response.provider == "ollama"
        assert response.model == MODEL
        assert response.finish_reason == "stop"
        assert response.tool_calls == []
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.raw_metadata["total_duration"] == 123_456

        payload = recorder.payload("/api/chat")
        assert payload["model"] == MODEL
        assert payload["stream"] is False
        assert payload["messages"] == [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        assert payload["options"] == {
            "temperature": 0.1,
            "top_p": 0.8,
            "num_predict": 64,
            "stop": ["END"],
        }

    @pytest.mark.asyncio
    async def test_generate_extracts_complete_think_block(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": json_route(
                chat_payload("Visible answer<think>secret plan</think> done")
            ),
        }
        provider, _ = make_provider(routes)

        response = await provider.generate(
            [ChatMessage(role=ChatRole.USER, content="think")]
        )

        assert response.content == "Visible answer done"
        assert response.raw_metadata["reasoning"] == "secret plan"

    @pytest.mark.asyncio
    async def test_generate_preserves_native_thinking_field(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": json_route(
                chat_payload("Final answer", thinking="deep internal reasoning")
            ),
        }
        provider, _ = make_provider(routes)

        response = await provider.generate(
            [ChatMessage(role=ChatRole.USER, content="think")]
        )

        assert response.content == "Final answer"
        assert response.raw_metadata["reasoning"] == "deep internal reasoning"

    @pytest.mark.asyncio
    async def test_generate_unclosed_think_block_is_removed(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": json_route(
                chat_payload("Answer starts<think>never closed")
            ),
        }
        provider, _ = make_provider(routes)

        response = await provider.generate(
            [ChatMessage(role=ChatRole.USER, content="think")]
        )

        assert response.content == "Answer starts"
        assert response.raw_metadata["reasoning"] == "never closed"

    @pytest.mark.asyncio
    async def test_generate_native_tool_calls(self) -> None:
        routes = {
            "/api/show": json_route(
                show_payload(capabilities=["completion", "tools"])
            ),
            "/api/chat": json_route(
                chat_payload(
                    "",
                    tool_calls=[
                        {
                            "id": "call_abc",
                            "function": {
                                "name": "search_docs",
                                "arguments": '{"query": "architecture"}',
                            },
                        }
                    ],
                    done_reason="stop",
                )
            ),
        }
        provider, recorder = make_provider(routes)
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

        response = await provider.generate(
            [ChatMessage(role=ChatRole.USER, content="find docs")],
            tools=tools,
        )

        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_abc"
        assert response.tool_calls[0].name == "search_docs"
        assert response.tool_calls[0].arguments == {"query": "architecture"}

        payload = recorder.payload("/api/chat")
        assert payload["tools"][0]["function"]["name"] == "search_docs"

    @pytest.mark.asyncio
    async def test_generate_unsupported_tools_raises_by_default(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
        }
        provider, recorder = make_provider(routes)
        tools = [
            ToolDefinition(
                name="noop",
                description="No operation",
                parameters_json_schema={"type": "object"},
            )
        ]

        with pytest.raises(CapabilityNotSupportedError) as exc_info:
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="call a tool")],
                tools=tools,
            )

        assert exc_info.value.capability == "function_calling"
        assert exc_info.value.provider == "ollama"
        assert "/api/chat" not in recorder.paths()

    @pytest.mark.asyncio
    async def test_generate_react_shim_parses_json_tool_call(self) -> None:
        shim_response = (
            "```json\n"
            '{"tool": "search_docs", "arguments": {"query": "react"}}\n'
            "```"
        )
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": json_route(chat_payload(shim_response)),
        }
        provider, recorder = make_provider(routes, tool_fallback="react_shim")
        tools = [
            ToolDefinition(
                name="search_docs",
                description="Search documents",
                parameters_json_schema={"type": "object"},
            )
        ]

        response = await provider.generate(
            [ChatMessage(role=ChatRole.USER, content="find react")],
            tools=tools,
        )

        assert response.finish_reason == "tool_calls"
        assert response.content == ""
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search_docs"
        assert response.tool_calls[0].arguments == {"query": "react"}

        payload = recorder.payload("/api/chat")
        assert "tools" not in payload
        assert payload["messages"][0]["role"] == "system"
        assert "search_docs" in payload["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_generate_structured_output_schema_is_forwarded(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": json_route(chat_payload('{"score": 0.9}')),
        }
        provider, recorder = make_provider(routes)
        schema = {
            "type": "object",
            "properties": {"score": {"type": "number"}},
        }

        response = await provider.generate(
            [ChatMessage(role=ChatRole.USER, content="score it")],
            config=GenerationConfig(response_json_schema=schema),
        )

        assert response.content == '{"score": 0.9}'
        assert recorder.payload("/api/chat")["format"] == schema

    @pytest.mark.asyncio
    async def test_tool_result_history_is_translated(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": json_route(chat_payload("Answer")),
        }
        provider, recorder = make_provider(routes)

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
                        content='{"results": ["doc"]}',
                    ),
                ),
            ]
        )

        messages = recorder.payload("/api/chat")["messages"]
        assert messages[1]["tool_calls"][0]["function"]["name"] == "search_docs"
        assert messages[2] == {
            "role": "tool",
            "content": '{"results": ["doc"]}',
            "tool_name": "search_docs",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OllamaLLMProvider — streaming
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaLLMStream:
    @pytest.mark.asyncio
    async def test_stream_ndjson_chunks_and_final_usage(self) -> None:
        events = [
            {"message": {"role": "assistant", "content": "Quantiva "}, "done": False},
            {"message": {"content": "desktop"}, "done": False},
            {
                "message": {"content": " app."},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 4,
                "eval_count": 3,
            },
        ]
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": ndjson_route(events),
        }
        provider, recorder = make_provider(routes)

        chunks = [
            chunk
            async for chunk in provider.stream(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )
        ]

        deltas = "".join(chunk.delta_text for chunk in chunks if not chunk.is_final)
        assert deltas == "Quantiva desktop app."

        final = chunks[-1]
        assert final.is_final is True
        assert final.delta_text == ""
        assert final.finish_reason == "stop"
        assert final.usage is not None
        assert final.usage.total_tokens == 7
        assert recorder.payload("/api/chat")["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_extracts_split_think_tags(self) -> None:
        events = [
            {"message": {"content": "A<thi"}, "done": False},
            {"message": {"content": "nk>hidden"}, "done": False},
            {"message": {"content": "</thi"}, "done": False},
            {"message": {"content": "nk>B"}, "done": False},
            {"message": {"content": ""}, "done": True, "done_reason": "stop"},
        ]
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": ndjson_route(events),
        }
        provider, _ = make_provider(routes)

        chunks = [
            chunk
            async for chunk in provider.stream(
                [ChatMessage(role=ChatRole.USER, content="think")]
            )
        ]

        visible = "".join(chunk.delta_text for chunk in chunks if not chunk.is_final)
        assert visible == "AB"
        assert provider._last_stream_reasoning == "hidden"

    @pytest.mark.asyncio
    async def test_stream_native_tool_call(self) -> None:
        events = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search_docs",
                                "arguments": {"query": "stream"},
                            }
                        }
                    ],
                },
                "done": False,
            },
            {"message": {"content": ""}, "done": True, "done_reason": "stop"},
        ]
        routes = {
            "/api/show": json_route(
                show_payload(capabilities=["completion", "tools"])
            ),
            "/api/chat": ndjson_route(events),
        }
        provider, _ = make_provider(routes)
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
                [ChatMessage(role=ChatRole.USER, content="search")],
                tools=tools,
            )
        ]

        tool_chunks = [chunk for chunk in chunks if chunk.tool_call_delta]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call_delta.name == "search_docs"
        assert tool_chunks[0].tool_call_delta.arguments == {"query": "stream"}
        assert chunks[-1].is_final is True
        assert chunks[-1].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_stream_unsupported_tools_raises(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
        }
        provider, _ = make_provider(routes)
        tools = [
            ToolDefinition(
                name="noop",
                description="No operation",
                parameters_json_schema={"type": "object"},
            )
        ]

        with pytest.raises(CapabilityNotSupportedError):
            async for _chunk in provider.stream(
                [ChatMessage(role=ChatRole.USER, content="tool")],
                tools=tools,
            ):
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# OllamaLLMProvider — error normalisation & probes
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaLLMErrors:
    @pytest.mark.asyncio
    async def test_daemon_down_has_actionable_message(self) -> None:
        provider, _ = make_provider(
            {
                "/api/show": raise_route(
                    lambda request: httpx.ConnectError(
                        "connection refused", request=request
                    )
                )
            }
        )

        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )

        assert str(exc_info.value) == DAEMON_MESSAGE
        assert exc_info.value.user_message == DAEMON_MESSAGE
        assert exc_info.value.retriable is True
        assert exc_info.value.provider == "ollama"

    @pytest.mark.asyncio
    async def test_missing_model_on_show(self) -> None:
        provider, _ = make_provider(
            {
                "/api/show": error_route(
                    404, f"model '{MODEL}' not found"
                )
            }
        )

        with pytest.raises(ModelNotFoundError) as exc_info:
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )

        assert str(exc_info.value) == (
            f"Model '{MODEL}' is not installed in Ollama. "
            f"Pull it via Model Studio or run 'ollama pull {MODEL}'."
        )
        assert exc_info.value.user_message == str(exc_info.value)
        assert exc_info.value.retriable is False

    @pytest.mark.asyncio
    async def test_missing_model_on_chat(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": error_route(404, f"model '{MODEL}' not found"),
        }
        provider, _ = make_provider(routes)

        with pytest.raises(ModelNotFoundError) as exc_info:
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )

        assert f"Model '{MODEL}' is not installed in Ollama" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_context_window_error_translation(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": error_route(
                400, "context length exceeded: maximum context length is 4096"
            ),
        }
        provider, _ = make_provider(routes)

        with pytest.raises(ContextWindowExceededError):
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="too long")]
            )

    @pytest.mark.asyncio
    async def test_rate_limit_translation(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": error_route(429, "too many requests"),
        }
        provider, _ = make_provider(routes)

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )

        assert exc_info.value.retriable is True

    @pytest.mark.asyncio
    async def test_generic_server_error_is_provider_error(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": error_route(500, "model crashed"),
        }
        provider, _ = make_provider(routes)

        with pytest.raises(ProviderError) as exc_info:
            await provider.generate(
                [ChatMessage(role=ChatRole.USER, content="hi")]
            )

        assert exc_info.value.retriable is True

    @pytest.mark.asyncio
    async def test_connection_probe_success(self) -> None:
        routes = {
            "/api/show": json_route(show_payload(capabilities=["completion"])),
            "/api/chat": json_route(chat_payload("pong", prompt_tokens=1, completion_tokens=1)),
        }
        provider, _ = make_provider(routes)

        result = await provider.test_connection()

        assert result.success is True
        assert result.provider == "ollama"
        assert result.model == MODEL
        assert result.capabilities is not None
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_connection_probe_failure_never_raises(self) -> None:
        provider, _ = make_provider(
            {
                "/api/show": raise_route(
                    lambda request: httpx.ConnectError(
                        "refused", request=request
                    )
                )
            }
        )

        result = await provider.test_connection()

        assert result.success is False
        assert result.error_code == "ProviderConnectionError"
        assert result.capabilities is None


# ═══════════════════════════════════════════════════════════════════════════════
# OllamaRuntimeManager
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaRuntimeManager:
    @pytest.mark.asyncio
    async def test_is_daemon_running_true(self) -> None:
        recorder = Recorder({"/api/version": json_route({"version": "0.12.0"})})
        manager = OllamaRuntimeManager(
            base_url=DEFAULT_BASE, client=make_client(recorder)
        )

        assert await manager.is_daemon_running() is True

    @pytest.mark.asyncio
    async def test_is_daemon_running_falls_back_to_tags(self) -> None:
        recorder = Recorder(
            {
                "/api/version": error_route(404, "not found"),
                "/api/tags": json_route({"models": []}),
            }
        )
        manager = OllamaRuntimeManager(
            base_url=DEFAULT_BASE, client=make_client(recorder)
        )

        assert await manager.is_daemon_running() is True
        assert recorder.paths() == ["/api/version", "/api/tags"]

    @pytest.mark.asyncio
    async def test_is_daemon_running_false_when_unreachable(self) -> None:
        recorder = Recorder(
            {
                "/api/version": raise_route(
                    lambda request: httpx.ConnectError(
                        "refused", request=request
                    )
                )
            }
        )
        manager = OllamaRuntimeManager(
            base_url=DEFAULT_BASE, client=make_client(recorder)
        )

        assert await manager.is_daemon_running() is False

    @pytest.mark.asyncio
    async def test_list_installed_models_maps_details(self) -> None:
        payload = {
            "models": [
                {
                    "name": "llama3.2:latest",
                    "model": "llama3.2:latest",
                    "size": 2_019_393_189,
                    "digest": "sha256:abc",
                    "details": {
                        "family": "llama",
                        "parameter_size": "3.2B",
                        "quantization_level": "Q4_K_M",
                    },
                },
                {
                    "name": "nomic-embed-text:latest",
                    "size": 274_302_432,
                    "details": {
                        "family": "nomic-bert",
                        "parameter_size": "137M",
                        "quantization_level": "F16",
                    },
                },
            ]
        }
        recorder = Recorder({"/api/tags": json_route(payload)})
        manager = OllamaRuntimeManager(
            base_url=DEFAULT_BASE, client=make_client(recorder)
        )

        models = await manager.list_installed_models()

        assert models == [
            {
                "name": "llama3.2:latest",
                "size_bytes": 2_019_393_189,
                "family": "llama",
                "parameter_size": "3.2B",
                "quantization_level": "Q4_K_M",
            },
            {
                "name": "nomic-embed-text:latest",
                "size_bytes": 274_302_432,
                "family": "nomic-bert",
                "parameter_size": "137M",
                "quantization_level": "F16",
            },
        ]

    @pytest.mark.asyncio
    async def test_list_installed_models_connection_error(self) -> None:
        recorder = Recorder(
            {
                "/api/tags": raise_route(
                    lambda request: httpx.ConnectError(
                        "refused", request=request
                    )
                )
            }
        )
        manager = OllamaRuntimeManager(
            base_url=DEFAULT_BASE, client=make_client(recorder)
        )

        with pytest.raises(ProviderConnectionError) as exc_info:
            await manager.list_installed_models()

        assert str(exc_info.value) == DAEMON_MESSAGE

    @pytest.mark.asyncio
    async def test_pull_model_stream_reports_progress(self) -> None:
        events = [
            {"status": "pulling manifest"},
            {
                "status": "downloading digest",
                "digest": "sha256:abc",
                "total": 1_000,
                "completed": 250,
            },
            {
                "status": "downloading digest",
                "digest": "sha256:abc",
                "total": 1_000,
                "completed": 1_000,
            },
            {"status": "success"},
        ]
        recorder = Recorder({"/api/pull": ndjson_route(events)})
        manager = OllamaRuntimeManager(
            base_url=DEFAULT_BASE, client=make_client(recorder)
        )

        progress = [event async for event in manager.pull_model_stream("llama3.2")]

        assert [event["percent"] for event in progress] == [
            None,
            25.0,
            100.0,
            100.0,
        ]
        assert progress[0]["status"] == "pulling manifest"
        assert progress[-1]["status"] == "success"
        assert recorder.payload("/api/pull") == {
            "model": "llama3.2",
            "stream": True,
        }

    @pytest.mark.asyncio
    async def test_pull_model_missing_raises(self) -> None:
        recorder = Recorder(
            {"/api/pull": error_route(404, "model 'unknown' not found")}
        )
        manager = OllamaRuntimeManager(
            base_url=DEFAULT_BASE, client=make_client(recorder)
        )

        with pytest.raises(ModelNotFoundError):
            async for _event in manager.pull_model_stream("unknown"):
                pass

    @pytest.mark.asyncio
    async def test_empty_pull_model_name_rejected(self) -> None:
        manager = OllamaRuntimeManager(base_url=DEFAULT_BASE)
        with pytest.raises(ValueError):
            async for _event in manager.pull_model_stream("  "):
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# OllamaEmbeddingProvider
# ═══════════════════════════════════════════════════════════════════════════════

class TestOllamaEmbeddingProvider:
    @pytest.mark.asyncio
    async def test_batch_embedding_and_dimension_detection(self) -> None:
        routes = {
            "/api/embed": json_route(
                {
                    "model": "nomic-embed-text",
                    "embeddings": [[3.0, 4.0], [0.0, 1.0]],
                }
            )
        }
        provider, recorder = make_embedding_provider(routes)

        assert provider.provider_name() == "ollama"
        assert provider.model_name() == "nomic-embed-text"
        assert provider.get_dimensions() == 0

        vectors = await provider.embed_documents(["alpha", "beta"])

        assert provider.get_dimensions() == 2
        assert vectors == [[0.6, 0.8], [0.0, 1.0]]
        assert recorder.payload("/api/embed") == {
            "model": "nomic-embed-text",
            "input": ["alpha", "beta"],
        }

    @pytest.mark.asyncio
    async def test_batch_respects_batch_size_and_orders_results(self) -> None:
        calls: list[list[str]] = []

        def _embed(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body["input"])
            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        [float(len(text)), 1.0] for text in body["input"]
                    ]
                },
            )

        recorder = Recorder({"/api/embed": _embed})
        provider = OllamaEmbeddingProvider(
            base_url=DEFAULT_BASE,
            model="nomic-embed-text",
            client=make_client(recorder),
        )

        vectors = await provider.embed_documents(["a", "bb", "ccc"], batch_size=2)

        assert calls == [["a", "bb"], ["ccc"]]
        assert len(vectors) == 3
        assert vectors[0][0] == pytest.approx(1 / (2 ** 0.5))

    @pytest.mark.asyncio
    async def test_falls_back_to_legacy_endpoint_and_caches_choice(self) -> None:
        embed_calls = 0

        def _embed_route(request: httpx.Request) -> httpx.Response:
            nonlocal embed_calls
            embed_calls += 1
            return httpx.Response(
                404,
                json={"error": "404 page not found"},
            )

        def _legacy_route(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "nomic-embed-text"
            assert "prompt" in body
            return httpx.Response(
                200,
                json={"embedding": [1.0, 0.0, 0.0]},
            )

        recorder = Recorder(
            {
                "/api/embed": _embed_route,
                "/api/embeddings": _legacy_route,
            }
        )
        provider = OllamaEmbeddingProvider(
            base_url=DEFAULT_BASE,
            model="nomic-embed-text",
            client=make_client(recorder),
        )

        vectors = await provider.embed_documents(["one", "two"])

        assert vectors == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        assert provider.get_dimensions() == 3
        assert embed_calls == 1

        # The endpoint choice is cached: the next call goes straight to legacy.
        await provider.embed_query("three")
        assert embed_calls == 1
        assert recorder.paths().count("/api/embeddings") == 3

    @pytest.mark.asyncio
    async def test_model_not_found_on_embed_does_not_fallback(self) -> None:
        recorder = Recorder(
            {
                "/api/embed": error_route(
                    404, "model 'nomic-embed-text' not found"
                ),
            }
        )
        provider = OllamaEmbeddingProvider(
            base_url=DEFAULT_BASE,
            model="nomic-embed-text",
            client=make_client(recorder),
        )

        with pytest.raises(ModelNotFoundError):
            await provider.embed_query("hello")

        assert "/api/embeddings" not in recorder.paths()

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises_before_return(self) -> None:
        routes = {
            "/api/embed": json_route({"embeddings": [[1.0, 2.0]]}),
        }
        provider, _ = make_embedding_provider(routes, dimensions=3)

        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            await provider.embed_documents(["alpha"])

        assert exc_info.value.expected == 3
        assert exc_info.value.actual == 2

    @pytest.mark.asyncio
    async def test_empty_texts_rejected(self) -> None:
        provider, _ = make_embedding_provider({})
        with pytest.raises(ValueError):
            await provider.embed_documents([])

        with pytest.raises(ValueError):
            await provider.embed_documents(["ok", "   "])

    @pytest.mark.asyncio
    async def test_embed_query_returns_single_normalised_vector(self) -> None:
        routes = {
            "/api/embed": json_route({"embeddings": [[0.0, 5.0]]}),
        }
        provider, _ = make_embedding_provider(routes)

        vector = await provider.embed_query("hello")

        assert vector == [0.0, 1.0]
        assert provider.get_dimensions() == 2

    @pytest.mark.asyncio
    async def test_embedding_daemon_down_has_actionable_message(self) -> None:
        recorder = Recorder(
            {
                "/api/embed": raise_route(
                    lambda request: httpx.ConnectError(
                        "refused", request=request
                    )
                )
            }
        )
        provider = OllamaEmbeddingProvider(
            base_url=DEFAULT_BASE,
            model="nomic-embed-text",
            client=make_client(recorder),
        )

        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.embed_query("hello")

        assert str(exc_info.value) == DAEMON_MESSAGE

    @pytest.mark.asyncio
    async def test_test_embedding_success_and_failure(self) -> None:
        success_provider, _ = make_embedding_provider(
            {"/api/embed": json_route({"embeddings": [[1.0, 1.0, 1.0, 1.0]]})}
        )
        success = await success_provider.test_embedding()
        assert success.success is True
        assert success.detected_dimensions == 4
        assert success.provider == "ollama"

        failure_provider, _ = make_embedding_provider(
            {
                "/api/embed": raise_route(
                    lambda request: httpx.ConnectError(
                        "refused", request=request
                    )
                )
            }
        )
        failure = await failure_provider.test_embedding()
        assert failure.success is False
        assert failure.detected_dimensions is None
        assert failure.error_code == "ProviderConnectionError"

    @pytest.mark.asyncio
    async def test_malformed_embedding_payload_is_provider_error(self) -> None:
        routes = {
            "/api/embed": json_route({"embeddings": [[1.0], [2.0]]}),
        }
        provider, _ = make_embedding_provider(routes)

        with pytest.raises(ProviderError):
            await provider.embed_documents(["only-one-text"])

    @pytest.mark.asyncio
    async def test_zero_vector_is_rejected(self) -> None:
        routes = {
            "/api/embed": json_route({"embeddings": [[0.0, 0.0]]}),
        }
        provider, _ = make_embedding_provider(routes)

        with pytest.raises(ProviderError):
            await provider.embed_documents(["alpha"])
