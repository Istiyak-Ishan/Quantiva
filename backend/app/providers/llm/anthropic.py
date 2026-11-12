"""
app/providers/llm/anthropic.py
──────────────────────────────────────────────────────────────────────────────
Anthropic Messages API adapter.

Talks to ``POST /v1/messages`` through :mod:`httpx` (the optional Anthropic
SDK is deliberately not required), normalising system prompts, tool-use
content blocks, SSE events, and API errors into the Quantiva LLM contract.

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
    decode_tool_arguments,
    raise_for_openai_status,
    translate_openai_transport_error,
)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5"


def _map_stop_reason(value: Any, has_tool_calls: bool = False) -> str:
    if has_tool_calls:
        return "tool_calls"
    if not isinstance(value, str) or not value:
        return "stop"
    lowered = value.lower()
    if lowered == "tool_use":
        return "tool_calls"
    if lowered == "max_tokens":
        return "length"
    if lowered in ("end_turn", "stop_sequence", "stop"):
        return "stop"
    return lowered


def _usage_from_anthropic(
    prompt_tokens: int,
    completion_tokens: int,
) -> TokenUsage | None:
    if not prompt_tokens and not completion_tokens:
        return None
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude adapter implementing the unified LLMProvider interface."""

    def __init__(
        self,
        base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 10.0,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        max_tokens: int = 4096,
        max_context_window: int = 200_000,
        supports_vision: bool = True,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._model_name = model
        self._api_key = (
            api_key
            if api_key is not None
            else getattr(settings, "ANTHROPIC_API_KEY", None)
        )
        self._custom_headers = dict(headers or {})
        self._client = client
        self._transport = transport
        self._owns_client = client is None
        self._timeout_seconds = float(timeout_seconds)
        self._connect_timeout_seconds = float(connect_timeout_seconds)
        self._anthropic_version = anthropic_version
        self._max_tokens = max_tokens
        self._max_context_window = max_context_window
        self._supports_vision = supports_vision
        self._last_stream_reasoning = ""

    # ── Identity & capabilities ───────────────────────────────────────────────

    def provider_name(self) -> str:
        return "anthropic"

    def model_name(self) -> str:
        return self._model_name

    def base_url(self) -> str:
        return self._base_url

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_vision=self._supports_vision,
            supports_system_prompt=True,
            max_context_window=self._max_context_window,
            is_local=False,
        )

    # ── HTTP lifecycle ────────────────────────────────────────────────────────

    def _messages_url(self) -> str:
        if self._base_url.endswith("/v1"):
            return f"{self._base_url}/messages"
        return f"{self._base_url}/v1/messages"

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
        if not self._api_key:
            raise ProviderAuthenticationError(
                (
                    "Anthropic API key is required. Add it in Settings or set "
                    "ANTHROPIC_API_KEY."
                ),
                provider="anthropic",
                model=self._model_name,
            )

        headers: dict[str, str] = {
            "accept": "text/event-stream" if stream else "application/json",
            "content-type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": self._anthropic_version,
        }
        headers.update(self._custom_headers)
        return headers

    # ── Request construction ──────────────────────────────────────────────────

    @staticmethod
    def _as_blocks(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        if isinstance(content, list):
            return list(content)
        return []

    def _append_message(
        self,
        converted: list[dict[str, Any]],
        role: str,
        content: Any,
    ) -> None:
        """Append a message, merging same-role turns as Anthropic requires."""
        if not converted or converted[-1]["role"] != role:
            converted.append({"role": role, "content": content})
            return

        previous = converted[-1]["content"]
        if isinstance(previous, str) and isinstance(content, str):
            converted[-1]["content"] = (
                f"{previous}\n\n{content}" if previous else content
            )
            return

        converted[-1]["content"] = (
            self._as_blocks(previous) + self._as_blocks(content)
        )

    def _translate_messages(
        self,
        messages: list[ChatMessage],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for message in messages:
            if message.role == ChatRole.SYSTEM:
                if message.content:
                    system_parts.append(message.content)
                continue

            if message.role == ChatRole.USER:
                self._append_message(
                    converted, "user", message.content or ""
                )
                continue

            if message.role == ChatRole.ASSISTANT:
                if message.tool_calls:
                    blocks: list[dict[str, Any]] = []
                    if message.content:
                        blocks.append(
                            {"type": "text", "text": message.content}
                        )
                    for call in message.tool_calls:
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": call.id,
                                "name": call.name,
                                "input": call.arguments or {},
                            }
                        )
                    self._append_message(converted, "assistant", blocks)
                else:
                    self._append_message(
                        converted, "assistant", message.content or ""
                    )
                continue

            if message.role == ChatRole.TOOL:
                result = message.tool_result
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": (
                        result.tool_call_id if result is not None else ""
                    ),
                    "content": (
                        result.content
                        if result is not None
                        else (message.content or "")
                    ),
                }
                if result is not None and result.is_error:
                    block["is_error"] = True
                self._append_message(converted, "user", [block])

        system = "\n\n".join(part for part in system_parts if part) or None
        return system, converted

    @staticmethod
    def _tool_payload(
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters_json_schema,
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
        system, converted = self._translate_messages(messages)

        payload: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": (
                effective.max_tokens if config is not None else self._max_tokens
            ),
            "messages": converted,
            "temperature": effective.temperature,
            "top_p": effective.top_p,
        }

        if system:
            payload["system"] = system

        if effective.stop_sequences:
            payload["stop_sequences"] = list(effective.stop_sequences)

        if tools:
            payload["tools"] = self._tool_payload(tools)

        if effective.response_json_schema:
            schema_instruction = (
                "Respond with only valid JSON that conforms to this JSON "
                "Schema, with no markdown fences or commentary:\n"
                + json.dumps(effective.response_json_schema, ensure_ascii=False)
            )
            payload["system"] = (
                f"{payload['system']}\n\n{schema_instruction}"
                if payload.get("system")
                else schema_instruction
            )

        payload["stream"] = stream

        return payload

    # ── Response parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_blocks(
        content_blocks: Any,
    ) -> tuple[str, list[ToolCallRequest], str]:
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        reasoning_parts: list[str] = []

        for block in content_blocks or []:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif block_type in ("tool_use", "server_tool_use"):
                name = block.get("name")
                if not name:
                    continue
                tool_calls.append(
                    ToolCallRequest(
                        id=block.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                        name=str(name),
                        arguments=decode_tool_arguments(block.get("input", {})),
                    )
                )
            elif block_type == "thinking":
                thinking = block.get("thinking")
                if isinstance(thinking, str) and thinking:
                    reasoning_parts.append(thinking)

        return (
            "".join(text_parts),
            tool_calls,
            "\n\n".join(reasoning_parts),
        )

    @staticmethod
    def _usage_from_response(payload: dict[str, Any]) -> TokenUsage | None:
        usage = payload.get("usage") or {}
        if not isinstance(usage, dict):
            return None
        return _usage_from_anthropic(
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )

    @staticmethod
    def _raw_metadata(
        payload: dict[str, Any],
        reasoning: str,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in (
            "id",
            "type",
            "role",
            "stop_reason",
            "stop_sequence",
            "model",
        ):
            if key in payload:
                metadata[key] = payload[key]
        if reasoning:
            metadata["reasoning"] = reasoning
        if isinstance(payload.get("usage"), dict):
            metadata["usage"] = payload["usage"]
        return metadata

    async def _post_messages(
        self,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        headers = self._request_headers(stream=False)
        client = self._get_client()
        try:
            response = await client.post(
                self._messages_url(),
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise translate_openai_transport_error(
                exc,
                provider="anthropic",
                base_url=self._base_url,
                model=self._model_name,
                is_local=False,
            ) from exc

        if not response.is_success:
            raise_for_openai_status(
                response.status_code,
                response.text,
                provider="anthropic",
                model=self._model_name,
                base_url=self._base_url,
            )

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "Anthropic returned invalid JSON from /v1/messages.",
                provider="anthropic",
                model=self._model_name,
                retriable=True,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError(
                "Anthropic returned an unexpected messages response.",
                provider="anthropic",
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
        data = await self._post_messages(
            payload,
            timeout_seconds=(
                config.timeout_seconds
                if config is not None and config.timeout_seconds
                else self._timeout_seconds
            ),
        )

        content, tool_calls, reasoning = self._extract_blocks(
            data.get("content")
        )
        stop_reason = data.get("stop_reason")

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=self._usage_from_response(data),
            model=str(data.get("model") or self._model_name),
            provider=self.provider_name(),
            finish_reason=_map_stop_reason(stop_reason, bool(tool_calls)),
            raw_metadata=self._raw_metadata(data, reasoning),
        )

    @staticmethod
    async def _iter_sse_events(
        response: httpx.Response,
    ) -> AsyncIterator[tuple[str | None, str]]:
        event_name: str | None = None
        data_lines: list[str] = []

        async for raw_line in response.aiter_lines():
            line = raw_line.rstrip("\r")
            if not line:
                if data_lines:
                    yield event_name, "\n".join(data_lines)
                    event_name = None
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            else:
                if data_lines:
                    data_lines.append(line)
                else:
                    yield event_name, line

        if data_lines:
            yield event_name, "\n".join(data_lines)

    @staticmethod
    def _stream_tool_call(buffer: dict[str, Any]) -> ToolCallRequest | None:
        name = buffer.get("name")
        if not name:
            return None
        if buffer.get("input_obj") is not None:
            arguments = decode_tool_arguments(buffer["input_obj"])
        else:
            arguments = decode_tool_arguments(buffer.get("json", ""))
        return ToolCallRequest(
            id=buffer.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            name=str(name),
            arguments=arguments,
        )

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

        prompt_tokens = 0
        completion_tokens = 0
        stop_reason: str | None = None
        reasoning_parts: list[str] = []
        tool_buffers: dict[int, dict[str, Any]] = {}
        saw_tool_calls = False

        try:
            async with client.stream(
                "POST",
                self._messages_url(),
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
                        provider="anthropic",
                        model=self._model_name,
                        base_url=self._base_url,
                    )

                async for event_name, data in self._iter_sse_events(response):
                    if not data.strip():
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue

                    event_type = event_name or event.get("type")

                    if event_type == "error" or event.get("type") == "error":
                        error = event.get("error") or {}
                        message = (
                            error.get("message")
                            if isinstance(error, dict)
                            else str(error)
                        )
                        raise ProviderError(
                            f"Anthropic stream error: {message}",
                            provider="anthropic",
                            model=self._model_name,
                            retriable=True,
                        )

                    if event_type == "message_start":
                        message = event.get("message") or {}
                        usage = message.get("usage") or {}
                        if isinstance(usage, dict):
                            prompt_tokens = int(
                                usage.get("input_tokens") or prompt_tokens
                            )
                        continue

                    if event_type == "content_block_start":
                        index = event.get("index", 0)
                        block = event.get("content_block") or {}
                        if not isinstance(block, dict):
                            block = {}
                        block_type = block.get("type")
                        if block_type in ("tool_use", "server_tool_use"):
                            tool_buffers[index] = {
                                "id": block.get("id"),
                                "name": block.get("name") or "",
                                "json": "",
                                "input_obj": block.get("input")
                                if isinstance(block.get("input"), dict)
                                and block.get("input")
                                else None,
                            }
                        elif block_type == "text":
                            text = block.get("text")
                            if isinstance(text, str) and text:
                                yield LLMStreamChunk(
                                    delta_text=text, is_final=False
                                )
                        elif block_type == "thinking":
                            thinking = block.get("thinking")
                            if isinstance(thinking, str) and thinking:
                                reasoning_parts.append(thinking)
                        continue

                    if event_type == "content_block_delta":
                        index = event.get("index", 0)
                        delta = event.get("delta") or {}
                        if not isinstance(delta, dict):
                            continue
                        delta_type = delta.get("type")
                        if delta_type == "text_delta":
                            text = delta.get("text")
                            if isinstance(text, str) and text:
                                yield LLMStreamChunk(
                                    delta_text=text, is_final=False
                                )
                        elif delta_type == "input_json_delta":
                            buffer = tool_buffers.setdefault(
                                index,
                                {
                                    "id": None,
                                    "name": "",
                                    "json": "",
                                    "input_obj": None,
                                },
                            )
                            partial = delta.get("partial_json")
                            if isinstance(partial, str):
                                buffer["json"] += partial
                        elif delta_type == "thinking_delta":
                            thinking = delta.get("thinking")
                            if isinstance(thinking, str) and thinking:
                                reasoning_parts.append(thinking)
                        continue

                    if event_type == "content_block_stop":
                        index = event.get("index", 0)
                        buffer = tool_buffers.pop(index, None)
                        if buffer is not None:
                            call = self._stream_tool_call(buffer)
                            if call is not None:
                                saw_tool_calls = True
                                yield LLMStreamChunk(
                                    tool_call_delta=call,
                                    is_final=False,
                                )
                        continue

                    if event_type == "message_delta":
                        delta = event.get("delta") or {}
                        if isinstance(delta, dict):
                            reason = delta.get("stop_reason")
                            if isinstance(reason, str) and reason:
                                stop_reason = reason
                        usage = event.get("usage") or {}
                        if isinstance(usage, dict):
                            completion_tokens = int(
                                usage.get("output_tokens")
                                or completion_tokens
                            )
                        continue

                    if event_type == "message_stop":
                        break

        except ProviderError:
            raise
        except Exception as exc:
            raise translate_openai_transport_error(
                exc,
                provider="anthropic",
                base_url=self._base_url,
                model=self._model_name,
                is_local=False,
            ) from exc

        # Flush any tool block that did not receive a content_block_stop event.
        for index in sorted(tool_buffers):
            call = self._stream_tool_call(tool_buffers[index])
            if call is not None:
                saw_tool_calls = True
                yield LLMStreamChunk(tool_call_delta=call, is_final=False)

        self._last_stream_reasoning = "\n\n".join(reasoning_parts).strip()

        yield LLMStreamChunk(
            delta_text="",
            is_final=True,
            usage=_usage_from_anthropic(prompt_tokens, completion_tokens),
            finish_reason=_map_stop_reason(stop_reason, saw_tool_calls),
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
