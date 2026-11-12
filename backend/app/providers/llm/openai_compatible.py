"""
app/providers/llm/openai_compatible.py
──────────────────────────────────────────────────────────────────────────────
Generic OpenAI-compatible chat-completions adapter plus thin presets for
LM Studio, llama.cpp server, and OpenAI official.

Uses the plain ``POST {base_url}/chat/completions`` HTTP API through
:mod:`httpx`, including SSE streaming and incremental tool-call assembly.

Conforms to :class:`~app.providers.interfaces.llm.LLMProvider`.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import settings
from app.providers.interfaces.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
)
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.types import (
    ChatMessage,
    ChatRole,
    ConnectionTestResult,
    GenerationConfig,
    LLMResponse,
    LLMStreamChunk,
    ProviderCapabilities,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)
from app.providers.openai_common import (
    DEFAULT_LLAMA_CPP_BASE_URL,
    DEFAULT_LM_STUDIO_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    build_openai_headers,
    decode_tool_arguments,
    extract_openai_text,
    normalize_openai_base_url,
    raise_for_openai_status,
    translate_openai_transport_error,
    usage_from_openai,
)

_VISION_MODEL_TOKENS = (
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "o3",
    "o4",
    "vision",
    "llava",
    "qwen2.5-vl",
    "gemma3",
)


def _context_window_for_model(model: str) -> int:
    lowered = model.lower()
    if any(token in lowered for token in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4")):
        return 128_000
    if "gpt-4-turbo" in lowered or "gpt-4.5" in lowered:
        return 128_000
    if "gpt-4" in lowered:
        return 8_192
    if "gpt-3.5" in lowered:
        return 16_385
    return 32_768


def _supports_vision_for_model(model: str) -> bool:
    lowered = model.lower()
    return any(token in lowered for token in _VISION_MODEL_TOKENS)


def _map_finish_reason(value: Any, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    if not isinstance(value, str) or not value:
        return "stop"
    lowered = value.lower()
    if lowered in ("tool_calls", "function_call"):
        return "tool_calls"
    if lowered in ("length", "max_tokens"):
        return "length"
    if lowered in ("content_filter", "safety"):
        return "content_filter"
    return "stop" if lowered in ("stop", "end_turn", "eos", "stop_sequence") else lowered


class OpenAICompatibleLLMProvider(LLMProvider):
    """OpenAI-compatible chat adapter for cloud and local runtimes."""

    def __init__(
        self,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 10.0,
        is_local: bool = False,
        provider_label: str = "openai_compatible",
        requires_api_key: bool = False,
        supports_vision: bool | None = None,
        supports_function_calling: bool = True,
        max_context_window: int | None = None,
        json_mode: bool = False,
        include_stream_usage: bool = False,
    ) -> None:
        self._base_url = normalize_openai_base_url(base_url)
        self._model_name = model
        if api_key is not None:
            self._api_key = api_key
        elif requires_api_key:
            self._api_key = getattr(settings, "OPENAI_API_KEY", None)
        else:
            self._api_key = None
        self._custom_headers = dict(headers or {})
        self._client = client
        self._transport = transport
        self._owns_client = client is None
        self._timeout_seconds = float(timeout_seconds)
        self._connect_timeout_seconds = float(connect_timeout_seconds)
        self._is_local = is_local
        self._provider_label = provider_label
        self._requires_api_key = requires_api_key
        self._supports_vision = supports_vision
        self._supports_function_calling = supports_function_calling
        self._max_context_window = max_context_window
        self._json_mode = json_mode
        self._include_stream_usage = include_stream_usage
        self._last_stream_reasoning = ""

    # ── Identity & capabilities ───────────────────────────────────────────────

    def provider_name(self) -> str:
        return self._provider_label

    def model_name(self) -> str:
        return self._model_name

    def base_url(self) -> str:
        return self._base_url

    def get_capabilities(self) -> ProviderCapabilities:
        vision = (
            self._supports_vision
            if self._supports_vision is not None
            else _supports_vision_for_model(self._model_name)
        )
        context = (
            self._max_context_window
            if self._max_context_window is not None
            else _context_window_for_model(self._model_name)
        )
        return ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=self._supports_function_calling,
            supports_structured_output=True,
            supports_vision=vision,
            supports_system_prompt=True,
            max_context_window=context,
            is_local=self._is_local,
        )

    # ── HTTP lifecycle ────────────────────────────────────────────────────────

    def _chat_url(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                self._timeout_seconds,
                connect=self._connect_timeout_seconds,
            )
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=timeout,
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _request_headers(self, *, stream: bool = False) -> dict[str, str]:
        if self._requires_api_key and not self._api_key:
            raise ProviderAuthenticationError(
                (
                    f"An API key is required for '{self._provider_label}'. "
                    "Add it in Settings or set OPENAI_API_KEY."
                ),
                provider=self._provider_label,
                model=self._model_name,
            )
        return build_openai_headers(
            self._api_key,
            self._custom_headers,
            accept="text/event-stream" if stream else "application/json",
        )

    # ── Request construction ──────────────────────────────────────────────────

    def _translate_messages(
        self,
        messages: list[ChatMessage],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []

        for message in messages:
            if message.role == ChatRole.SYSTEM:
                converted.append(
                    {"role": "system", "content": message.content or ""}
                )
            elif message.role == ChatRole.USER:
                converted.append(
                    {"role": "user", "content": message.content or ""}
                )
            elif message.role == ChatRole.ASSISTANT:
                payload: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or "",
                }
                if message.tool_calls:
                    payload["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(
                                    call.arguments or {}, ensure_ascii=False
                                ),
                            },
                        }
                        for call in message.tool_calls
                    ]
                converted.append(payload)
            elif message.role == ChatRole.TOOL:
                result = message.tool_result
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            result.tool_call_id if result is not None else ""
                        ),
                        "content": (
                            result.content
                            if result is not None
                            else (message.content or "")
                        ),
                    }
                )

        return converted

    @staticmethod
    def _tool_payload(
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_json_schema,
                },
            }
            for tool in tools
        ]

    def _build_payload(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None,
        *,
        tools: list[ToolDefinition] | None,
        stream: bool,
    ) -> dict[str, Any]:
        effective = config or GenerationConfig()
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": self._translate_messages(messages),
            "stream": stream,
            "temperature": effective.temperature,
            "top_p": effective.top_p,
            "max_tokens": effective.max_tokens,
        }

        if effective.stop_sequences:
            payload["stop"] = list(effective.stop_sequences)

        if tools:
            payload["tools"] = self._tool_payload(tools)

        if effective.response_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "quantiva_response",
                    "schema": effective.response_json_schema,
                    "strict": True,
                },
            }
        elif self._json_mode:
            payload["response_format"] = {"type": "json_object"}

        if stream and self._include_stream_usage:
            payload["stream_options"] = {"include_usage": True}

        return payload

    # ── Response parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_tool_calls(raw_message: dict[str, Any]) -> list[ToolCallRequest]:
        calls: list[ToolCallRequest] = []
        for raw_call in raw_message.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            name = function.get("name") or raw_call.get("name")
            if not name:
                continue
            calls.append(
                ToolCallRequest(
                    id=raw_call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    name=str(name),
                    arguments=decode_tool_arguments(
                        function.get("arguments", raw_call.get("arguments", {}))
                    ),
                )
            )
        return calls

    @staticmethod
    def _raw_metadata(
        payload: dict[str, Any],
        reasoning: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in ("id", "created", "system_fingerprint", "object"):
            if key in payload:
                metadata[key] = payload[key]
        if reasoning:
            metadata["reasoning"] = reasoning
        if isinstance(payload.get("usage"), dict):
            metadata["usage"] = payload["usage"]
        return metadata

    async def _post_chat(
        self,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        headers = self._request_headers(stream=False)
        client = self._get_client()
        try:
            response = await client.post(
                self._chat_url(),
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise translate_openai_transport_error(
                exc,
                provider=self._provider_label,
                base_url=self._base_url,
                model=self._model_name,
                is_local=self._is_local,
            ) from exc

        if not response.is_success:
            raise_for_openai_status(
                response.status_code,
                response.text,
                provider=self._provider_label,
                model=self._model_name,
                base_url=self._base_url,
            )

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"{self._provider_label} returned invalid JSON from chat/completions.",
                provider=self._provider_label,
                model=self._model_name,
                retriable=True,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError(
                f"{self._provider_label} returned an unexpected chat response.",
                provider=self._provider_label,
                model=self._model_name,
                retriable=True,
            )
        return data

    # ── Generation ────────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(
            messages, config, tools=tools, stream=False
        )
        data = await self._post_chat(
            payload,
            timeout_seconds=(
                config.timeout_seconds
                if config is not None and config.timeout_seconds
                else self._timeout_seconds
            ),
        )

        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ProviderError(
                f"{self._provider_label} returned no completion choices.",
                provider=self._provider_label,
                model=self._model_name,
                retriable=True,
            )
        choice = choices[0]
        raw_message = choice.get("message") or {}
        if not isinstance(raw_message, dict):
            raw_message = {}

        content = extract_openai_text(raw_message.get("content"))
        tool_calls = self._extract_tool_calls(raw_message)
        reasoning = (
            raw_message.get("reasoning_content")
            or raw_message.get("reasoning")
            or ""
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage_from_openai(data),
            model=str(data.get("model") or self._model_name),
            provider=self.provider_name(),
            finish_reason=_map_finish_reason(
                choice.get("finish_reason"), bool(tool_calls)
            ),
            raw_metadata=self._raw_metadata(
                data, reasoning if isinstance(reasoning, str) else ""
            ),
        )

    @staticmethod
    async def _iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
        """Yield SSE data payloads, tolerating plain NDJSON lines."""
        data_lines: list[str] = []

        async for raw_line in response.aiter_lines():
            line = raw_line.rstrip("\r")
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
                continue
            if line.startswith(("event:", "id:", "retry:")):
                continue
            # Some local servers emit bare JSON lines instead of SSE.
            if data_lines:
                data_lines.append(line)
            else:
                yield line

        if data_lines:
            yield "\n".join(data_lines)

    async def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        payload = self._build_payload(
            messages, config, tools=tools, stream=True
        )
        headers = self._request_headers(stream=True)
        client = self._get_client()

        tool_buffers: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: TokenUsage | None = None
        reasoning_parts: list[str] = []
        saw_tool_calls = False

        try:
            async with client.stream(
                "POST",
                self._chat_url(),
                json=payload,
                headers=headers,
                timeout=(
                    config.timeout_seconds
                    if config is not None and config.timeout_seconds
                    else self._timeout_seconds
                ),
            ) as response:
                if not response.is_success:
                    try:
                        await response.aread()
                        error_body = response.text
                    except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                        error_body = ""
                    raise_for_openai_status(
                        response.status_code,
                        error_body,
                        provider=self._provider_label,
                        model=self._model_name,
                        base_url=self._base_url,
                    )

                async for data in self._iter_sse_data(response):
                    if data.strip() == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue

                    if event.get("error"):
                        raise ProviderError(
                            (
                                f"{self._provider_label} stream error: "
                                f"{event['error']}"
                            ),
                            provider=self._provider_label,
                            model=self._model_name,
                            retriable=True,
                        )

                    event_usage = usage_from_openai(event)
                    if event_usage is not None:
                        usage = event_usage

                    choices = event.get("choices") or []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        delta = {}

                    reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or ""
                    )
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_parts.append(reasoning)

                    text = extract_openai_text(delta.get("content"))
                    if text:
                        yield LLMStreamChunk(delta_text=text, is_final=False)

                    for raw_call in delta.get("tool_calls") or []:
                        if not isinstance(raw_call, dict):
                            continue
                        index = raw_call.get("index", 0)
                        if not isinstance(index, int):
                            index = 0
                        buffer = tool_buffers.setdefault(
                            index,
                            {
                                "id": None,
                                "name": "",
                                "arguments": "",
                                "arguments_obj": None,
                            },
                        )
                        if raw_call.get("id"):
                            buffer["id"] = raw_call["id"]
                        function = raw_call.get("function") or {}
                        if isinstance(function, dict):
                            name = function.get("name")
                            if isinstance(name, str) and name:
                                buffer["name"] += name
                            arguments = function.get("arguments")
                            if isinstance(arguments, str):
                                buffer["arguments"] += arguments
                            elif isinstance(arguments, dict):
                                buffer["arguments_obj"] = arguments

                    if choice.get("finish_reason"):
                        finish_reason = _map_finish_reason(
                            choice.get("finish_reason"), False
                        )

        except ProviderError:
            raise
        except Exception as exc:
            raise translate_openai_transport_error(
                exc,
                provider=self._provider_label,
                base_url=self._base_url,
                model=self._model_name,
                is_local=self._is_local,
            ) from exc

        for index in sorted(tool_buffers):
            buffer = tool_buffers[index]
            if not buffer["name"]:
                continue
            arguments = (
                buffer["arguments_obj"]
                if buffer["arguments_obj"] is not None
                else decode_tool_arguments(buffer["arguments"])
            )
            saw_tool_calls = True
            yield LLMStreamChunk(
                tool_call_delta=ToolCallRequest(
                    id=buffer["id"] or f"call_{uuid.uuid4().hex[:12]}",
                    name=buffer["name"],
                    arguments=arguments,
                ),
                is_final=False,
            )

        if saw_tool_calls:
            finish_reason = "tool_calls"

        self._last_stream_reasoning = "".join(reasoning_parts).strip()

        yield LLMStreamChunk(
            delta_text="",
            is_final=True,
            usage=usage,
            finish_reason=finish_reason,
        )

    # ── Connectivity probe ────────────────────────────────────────────────────

    async def test_connection(self) -> ConnectionTestResult:
        start_time = time.perf_counter()
        try:
            await self.generate(
                [ChatMessage(role=ChatRole.USER, content="ping")],
                config=GenerationConfig(max_tokens=1, timeout_seconds=10.0),
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=True,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                capabilities=self.get_capabilities(),
            )
        except ProviderError as err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                error_code=type(err).__name__,
                error_message=err.user_message,
            )
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                error_code="UnknownError",
                error_message=str(exc),
            )

    # ── Preset factory constructors ───────────────────────────────────────────

    @classmethod
    def for_lm_studio(
        cls,
        model: str = "local-model",
        **kwargs: Any,
    ) -> LMStudioLLMProvider:
        return LMStudioLLMProvider(model=model, **kwargs)

    @classmethod
    def for_llama_cpp(
        cls,
        model: str = "local-model",
        **kwargs: Any,
    ) -> LlamaCppLLMProvider:
        return LlamaCppLLMProvider(model=model, **kwargs)

    @classmethod
    def for_openai(
        cls,
        model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> OpenAILLMProvider:
        return OpenAILLMProvider(model=model, **kwargs)


class LMStudioLLMProvider(OpenAICompatibleLLMProvider):
    """Preset for LM Studio's local OpenAI-compatible server."""

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("is_local", True)
        kwargs.setdefault("provider_label", "lm_studio")
        kwargs.setdefault("requires_api_key", False)
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            **kwargs,
        )


class LlamaCppLLMProvider(OpenAICompatibleLLMProvider):
    """Preset for the llama.cpp OpenAI-compatible server."""

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = DEFAULT_LLAMA_CPP_BASE_URL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("is_local", True)
        kwargs.setdefault("provider_label", "llama_cpp")
        kwargs.setdefault("requires_api_key", False)
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            **kwargs,
        )


class OpenAILLMProvider(OpenAICompatibleLLMProvider):
    """Preset for OpenAI's official chat-completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("is_local", False)
        kwargs.setdefault("provider_label", "openai")
        kwargs.setdefault("requires_api_key", True)
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            **kwargs,
        )
