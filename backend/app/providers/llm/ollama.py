"""
app/providers/llm/ollama.py
──────────────────────────────────────────────────────────────────────────────
First-class local Ollama LLM adapter plus the desktop runtime manager.

The adapter uses Ollama's plain HTTP JSON API through :mod:`httpx` rather than
the Ollama Python SDK.  This keeps the packaged application lightweight and
avoids freezing unnecessary native dependencies with PyInstaller.

Conforms to :class:`~app.providers.interfaces.llm.LLMProvider`.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from app.config import settings
from app.providers.interfaces.exceptions import (
    CapabilityNotSupportedError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
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
from app.providers.ollama_common import (
    build_chat_options,
    daemon_unreachable_message,
    model_not_installed_message,
    normalize_base_url,
    raise_for_ollama_status,
    request_timeout,
    translate_transport_error,
)

# ── Detection constants ───────────────────────────────────────────────────────

_THINK_OPEN = "<" + "think>"
_THINK_CLOSE = "<" + "/think>"
_THINK_OPEN_PATTERN = "<" + r"think\b[^>]*>"

_TOOL_CAPABLE_MODEL_TOKENS = (
    "qwen2",
    "qwen3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "llama4",
    "mistral",
    "mixtral",
    "firefunction",
    "command-r",
    "hermes",
    "functionary",
    "deepseek-v3",
    "devstral",
    "magistral",
    "granite3.1",
    "granite3.2",
    "phi4",
)

_VISION_MODEL_TOKENS = (
    "llava",
    "bakllava",
    "moondream",
    "vision",
    "minicpm-v",
    "gemma3",
    "granite3.2-vision",
    "llama3.2-vision",
)

_REASONING_MODEL_TOKENS = (
    "deepseek-r1",
    "qwq",
    "qwen3",
    "moonlight",
    "magistral",
    "phi4-reasoning",
    "reasoning",
)

_CONTEXT_LENGTH_HINTS: tuple[tuple[str, int], ...] = (
    ("llama3.1", 131_072),
    ("llama3.2", 131_072),
    ("llama3.3", 131_072),
    ("llama4", 131_072),
    ("qwen2.5", 32_768),
    ("qwen3", 40_960),
    ("mixtral", 32_768),
    ("mistral", 32_768),
    ("deepseek-r1", 131_072),
    ("gemma3", 131_072),
    ("command-r", 131_072),
)

_DEFAULT_CONTEXT_WINDOW = 8_192

# Capabilities are immutable per (base_url, model) and cached process-wide.
_CAPABILITY_CACHE: dict[tuple[str, str], ProviderCapabilities] = {}


def _model_has_any_token(model: str, tokens: tuple[str, ...]) -> bool:
    lowered = model.lower()
    return any(token in lowered for token in tokens)


def _heuristic_capabilities(model: str) -> ProviderCapabilities:
    """Conservative, offline capabilities used before ``/api/show`` succeeds."""
    lowered = model.lower()
    context_window = _DEFAULT_CONTEXT_WINDOW
    for token, window in _CONTEXT_LENGTH_HINTS:
        if token in lowered:
            context_window = window
            break

    return ProviderCapabilities(
        supports_streaming=True,
        supports_function_calling=_model_has_any_token(
            model, _TOOL_CAPABLE_MODEL_TOKENS
        ),
        supports_structured_output=True,
        supports_vision=_model_has_any_token(model, _VISION_MODEL_TOKENS),
        supports_system_prompt=True,
        max_context_window=context_window,
        is_local=True,
    )


def _extract_context_window(data: dict[str, Any], model: str) -> int:
    """Detect context length from ``model_info`` or Modelfile parameters."""
    model_info = data.get("model_info") or {}
    detected = 0

    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if not str(key).lower().endswith("context_length"):
                continue
            if isinstance(value, (int, float)) and value > 0:
                detected = max(detected, int(value))

    # An explicit num_ctx in the model's parameters is the *effective* window
    # used by this Ollama model, so prefer it over the architecture maximum.
    parameters = data.get("parameters") or ""
    if isinstance(parameters, str):
        match = re.search(r"(?im)^\s*num_ctx\s+(\d+)\s*$", parameters)
        if match:
            detected = int(match.group(1))

    if detected > 0:
        return detected

    return _heuristic_capabilities(model).max_context_window


def _parse_show_capabilities(
    data: dict[str, Any],
    model: str,
) -> ProviderCapabilities:
    """Translate an Ollama ``/api/show`` payload into ProviderCapabilities."""
    if not isinstance(data, dict):
        return _heuristic_capabilities(model)

    capabilities = {
        str(item).strip().lower()
        for item in (data.get("capabilities") or [])
        if isinstance(item, (str, bytes))
    }
    template = data.get("template") or ""
    template_lower = template.lower() if isinstance(template, str) else ""
    parameters = data.get("parameters") or ""
    parameters_lower = parameters.lower() if isinstance(parameters, str) else ""
    details = data.get("details") or {}
    family = str(details.get("family") or "").lower() if isinstance(details, dict) else ""

    model_info = data.get("model_info") or {}

    # Modern Ollama exposes an authoritative capabilities array.  Only fall
    # back to template/model-name heuristics when it is absent.
    if capabilities:
        supports_tools = "tools" in capabilities
        supports_vision = "vision" in capabilities
    else:
        supports_tools = (
            ".tools" in template_lower
            or "{{if .tools" in template_lower
            or "{{ if .tools" in template_lower
            or "tool_calls" in template_lower
            or "tool" in parameters_lower
            or _model_has_any_token(model, _TOOL_CAPABLE_MODEL_TOKENS)
            or _model_has_any_token(family, _TOOL_CAPABLE_MODEL_TOKENS)
        )
        has_clip_metadata = isinstance(model_info, dict) and any(
            str(key).startswith("clip.") for key in model_info
        )
        supports_vision = (
            has_clip_metadata
            or _model_has_any_token(model, _VISION_MODEL_TOKENS)
            or _model_has_any_token(family, _VISION_MODEL_TOKENS)
            or "vision" in template_lower
        )

    supports_system_prompt = (
        not template_lower
        or ".system" in template_lower
        or ".prompt" in template_lower
        or "system" in template_lower
    )

    return ProviderCapabilities(
        supports_streaming=True,
        supports_function_calling=supports_tools,
        supports_structured_output=True,
        supports_vision=supports_vision,
        supports_system_prompt=supports_system_prompt,
        max_context_window=_extract_context_window(data, model),
        is_local=True,
    )


def _coerce_arguments(value: Any) -> dict[str, Any]:
    """Normalise tool arguments that Ollama may return as JSON text."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            decoded = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return {"_raw": value}
        return decoded if isinstance(decoded, dict) else {"value": decoded}
    return {"value": value}


def _extract_tool_calls(raw_message: dict[str, Any]) -> list[ToolCallRequest]:
    """Convert Ollama's message.tool_calls array into normalised requests."""
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
        arguments = function.get("arguments", raw_call.get("arguments", {}))
        calls.append(
            ToolCallRequest(
                id=raw_call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                name=str(name),
                arguments=_coerce_arguments(arguments),
            )
        )
    return calls


def _split_reasoning(text: str) -> tuple[str, str]:
    """Remove `` thinking`` blocks from *text* and return (content, reasoning).

    Handles complete and unclosed blocks (streaming models frequently emit the
    closing tag late or not at all when the context window is exhausted).
    """
    if not text:
        return "", ""

    reasoning_parts: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        reasoning_parts.append(match.group(1))
        return ""

    content = re.sub(
        _THINK_OPEN_PATTERN + r"(.*?)" + re.escape(_THINK_CLOSE),
        _capture,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    open_match = re.search(
        _THINK_OPEN_PATTERN, content, flags=re.IGNORECASE
    )
    if open_match is not None:
        reasoning_parts.append(content[open_match.end():])
        content = content[:open_match.start()]

    # Be defensive about stray closing tags left by malformed model output.
    content = re.sub(re.escape(_THINK_CLOSE), "", content, flags=re.IGNORECASE)

    reasoning = "\n\n".join(part.strip() for part in reasoning_parts if part.strip())
    return content.strip(), reasoning


class _ReasoningStreamParser:
    """Incrementally separates `` Thinking`` reasoning from visible content.

    Tags can be split across NDJSON chunks, so a small guard buffer keeps the
    trailing characters until the next chunk resolves whether they are text or
    part of a tag.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_reasoning = False
        self.reasoning_parts: list[str] = []

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []

        self._buffer += text
        emitted: list[tuple[str, str]] = []

        while True:
            if self._in_reasoning:
                lowered = self._buffer.lower()
                index = lowered.find(_THINK_CLOSE)
                if index == -1:
                    guard = len(_THINK_CLOSE) - 1
                    safe_length = max(0, len(self._buffer) - guard)
                    if safe_length:
                        emitted.append(("reasoning", self._buffer[:safe_length]))
                        self.reasoning_parts.append(self._buffer[:safe_length])
                        self._buffer = self._buffer[safe_length:]
                    break

                if index:
                    emitted.append(("reasoning", self._buffer[:index]))
                    self.reasoning_parts.append(self._buffer[:index])
                self._buffer = self._buffer[index + len(_THINK_CLOSE):]
                self._in_reasoning = False
                continue

            lowered = self._buffer.lower()
            index = lowered.find(_THINK_OPEN)
            if index == -1:
                guard = len(_THINK_OPEN) - 1
                safe_length = max(0, len(self._buffer) - guard)
                if safe_length:
                    emitted.append(("content", self._buffer[:safe_length]))
                    self._buffer = self._buffer[safe_length:]
                break

            if index:
                emitted.append(("content", self._buffer[:index]))
            self._buffer = self._buffer[index + len(_THINK_OPEN):]
            self._in_reasoning = True

        return emitted

    def flush(self) -> list[tuple[str, str]]:
        if not self._buffer:
            return []
        kind = "reasoning" if self._in_reasoning else "content"
        text = self._buffer
        self._buffer = ""
        if kind == "reasoning":
            self.reasoning_parts.append(text)
        return [(kind, text)]

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts).strip()


class OllamaLLMProvider(LLMProvider):
    """Local Ollama adapter implementing the unified LLMProvider interface."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float | None = None,
        connect_timeout_seconds: float | None = None,
        tool_fallback: Literal["raise", "react_shim"] = "raise",
    ) -> None:
        self._base_url = normalize_base_url(
            base_url or getattr(settings, "OLLAMA_BASE_URL", None)
        )
        self._model_name = (
            model
            or getattr(settings, "OLLAMA_CHAT_MODEL", None)
            or "llama3.2"
        )
        self._client = client
        self._transport = transport
        self._owns_client = client is None
        self._timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 120.0)
        )
        self._connect_timeout_seconds = float(
            connect_timeout_seconds
            if connect_timeout_seconds is not None
            else getattr(settings, "OLLAMA_CONNECT_TIMEOUT_SECONDS", 5.0)
        )
        self._tool_fallback = tool_fallback
        self._capability_lock: asyncio.Lock | None = None
        self._last_stream_reasoning = ""

    # ── Identity & capabilities ───────────────────────────────────────────────

    def provider_name(self) -> str:
        return "ollama"

    def model_name(self) -> str:
        return self._model_name

    def base_url(self) -> str:
        return self._base_url

    def get_capabilities(self) -> ProviderCapabilities:
        """Return cached capabilities, falling back to offline heuristics."""
        return _CAPABILITY_CACHE.get(
            (self._base_url, self._model_name)
        ) or _heuristic_capabilities(self._model_name)

    @classmethod
    def clear_capability_cache(cls) -> None:
        """Clear the process-wide capability cache (used by settings changes)."""
        _CAPABILITY_CACHE.clear()

    async def detect_capabilities(
        self,
        *,
        force: bool = False,
    ) -> ProviderCapabilities:
        """Query ``/api/show`` and cache detected capabilities.

        A missing model or unreachable daemon is surfaced as a typed provider
        error.  Other ``/api/show`` failures fall back to conservative offline
        heuristics so a healthy chat endpoint remains usable on older Ollama
        builds.
        """
        key = (self._base_url, self._model_name)
        if not force:
            cached = _CAPABILITY_CACHE.get(key)
            if cached is not None:
                return cached

        if self._capability_lock is None:
            self._capability_lock = asyncio.Lock()

        async with self._capability_lock:
            if not force:
                cached = _CAPABILITY_CACHE.get(key)
                if cached is not None:
                    return cached

            try:
                payload = await self._fetch_show()
            except (
                ModelNotFoundError,
                ProviderConnectionError,
                ProviderAuthenticationError,
                ProviderRateLimitError,
            ):
                raise
            except ProviderError:
                return _heuristic_capabilities(self._model_name)

            capabilities = _parse_show_capabilities(payload, self._model_name)
            _CAPABILITY_CACHE[key] = capabilities
            return capabilities

    async def _ensure_capabilities(self) -> ProviderCapabilities:
        return await self.detect_capabilities(force=False)

    async def _fetch_show(self) -> dict[str, Any]:
        client = self._get_client()
        try:
            response = await client.post(
                f"{self._base_url}/api/show",
                json={"model": self._model_name},
                timeout=self._connect_timeout_seconds * 2,
            )
        except Exception as exc:
            raise translate_transport_error(
                exc, base_url=self._base_url, model=self._model_name
            ) from exc

        if not response.is_success:
            raise_for_ollama_status(
                response.status_code,
                response.text,
                base_url=self._base_url,
                model=self._model_name,
            )

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"Ollama /api/show returned invalid JSON for model '{self._model_name}'.",
                provider="ollama",
                model=self._model_name,
                retriable=True,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError(
                f"Ollama /api/show returned an unexpected payload for model '{self._model_name}'.",
                provider="ollama",
                model=self._model_name,
                retriable=True,
            )
        return data

    # ── HTTP client lifecycle ─────────────────────────────────────────────────

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
        """Close the underlying HTTP client when this adapter owns it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ── Request construction ──────────────────────────────────────────────────

    def _to_ollama_messages(
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
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments or {},
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
                        "content": (
                            result.content
                            if result is not None
                            else (message.content or "")
                        ),
                        "tool_name": (
                            result.name if result is not None else "tool"
                        ),
                    }
                )

        return converted

    @staticmethod
    def _tool_definitions_payload(
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

    @staticmethod
    def _tool_shim_instructions(tools: list[ToolDefinition]) -> str:
        tool_lines = []
        for tool in tools:
            schema = json.dumps(tool.parameters_json_schema, ensure_ascii=False)
            tool_lines.append(
                f"- {tool.name}: {tool.description}\n  parameters: {schema}"
            )

        return (
            "You are an AI assistant with access to external tools.\n"
            "When you need to call a tool, respond with ONLY a single JSON "
            "object and no markdown fences:\n"
            '{"tool": "<tool_name>", "arguments": {<arguments>}}\n'
            "When no tool is needed, reply normally with plain text.\n\n"
            "Available tools:\n" + "\n".join(tool_lines)
        )

    def _inject_tool_shim(
        self,
        messages: list[ChatMessage],
        tools: list[ToolDefinition],
    ) -> list[ChatMessage]:
        shim_message = ChatMessage(
            role=ChatRole.SYSTEM,
            content=self._tool_shim_instructions(tools),
        )
        return [shim_message, *messages]

    def _build_payload(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None,
        *,
        tools: list[ToolDefinition] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": self._to_ollama_messages(messages),
            "stream": stream,
            "options": build_chat_options(config),
        }

        if config is not None and config.response_json_schema:
            # Ollama accepts either "json" or a full JSON schema in ``format``.
            payload["format"] = config.response_json_schema

        if tools:
            payload["tools"] = self._tool_definitions_payload(tools)

        return payload

    # ── Response parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _usage_from_payload(payload: dict[str, Any]) -> TokenUsage | None:
        prompt_tokens = int(payload.get("prompt_eval_count") or 0)
        completion_tokens = int(payload.get("eval_count") or 0)
        if not prompt_tokens and not completion_tokens:
            return None
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    @staticmethod
    def _raw_metadata(payload: dict[str, Any], reasoning: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key in (
            "created_at",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
            "prompt_eval_count",
            "eval_count",
        ):
            if key in payload:
                metadata[key] = payload[key]
        if reasoning:
            metadata["reasoning"] = reasoning
        return metadata

    @staticmethod
    def _find_json_tool_object(text: str) -> tuple[dict[str, Any], int, int] | None:
        """Find the first balanced JSON object that describes a tool call."""
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                decoded, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(decoded, dict):
                continue
            has_name = bool(
                decoded.get("tool")
                or decoded.get("name")
                or (
                    isinstance(decoded.get("function"), dict)
                    and decoded["function"].get("name")
                )
            )
            if has_name and (
                "arguments" in decoded
                or "args" in decoded
                or (
                    isinstance(decoded.get("function"), dict)
                    and "arguments" in decoded["function"]
                )
            ):
                return decoded, index, index + end
        return None

    def _parse_shim_response(
        self,
        text: str,
    ) -> tuple[str, ToolCallRequest | None]:
        found = self._find_json_tool_object(text)
        if found is None:
            return text, None

        payload, start, end = found
        function = payload.get("function")
        if not isinstance(function, dict):
            function = {}

        name = payload.get("tool") or payload.get("name") or function.get("name")
        if not name:
            return text, None

        arguments = payload.get(
            "arguments",
            payload.get("args", function.get("arguments", {})),
        )
        remaining = (text[:start] + text[end:]).strip()
        # Remove now-empty markdown fences left around the JSON tool call.
        fenceless = re.sub(r"```(?:json)?", "", remaining, flags=re.IGNORECASE).strip()
        if fenceless in ("", "```"):
            remaining = ""
        elif fenceless:
            remaining = fenceless
        return remaining, ToolCallRequest(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=str(name),
            arguments=_coerce_arguments(arguments),
        )

    def _finish_reason(
        self,
        payload: dict[str, Any],
        tool_calls: list[ToolCallRequest],
    ) -> str:
        if tool_calls:
            return "tool_calls"
        done_reason = payload.get("done_reason")
        if isinstance(done_reason, str) and done_reason:
            return done_reason.lower()
        return "stop"

    # ── Generation ────────────────────────────────────────────────────────────

    async def _post_chat(
        self,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        client = self._get_client()
        try:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise translate_transport_error(
                exc, base_url=self._base_url, model=self._model_name
            ) from exc

        if not response.is_success:
            raise_for_ollama_status(
                response.status_code,
                response.text,
                base_url=self._base_url,
                model=self._model_name,
            )

        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "Ollama returned invalid JSON from /api/chat.",
                provider="ollama",
                model=self._model_name,
                retriable=True,
            ) from exc

        if not isinstance(data, dict):
            raise ProviderError(
                "Ollama returned an unexpected /api/chat payload.",
                provider="ollama",
                model=self._model_name,
                retriable=True,
            )
        return data

    def _resolve_tool_strategy(
        self,
        tools: list[ToolDefinition] | None,
        capabilities: ProviderCapabilities,
        *,
        streaming: bool,
    ) -> bool:
        """Return True when the structured ReAct JSON shim is active.

        Streaming remains reserved for native tool calling: the Agent Engine
        catches :class:`CapabilityNotSupportedError` and switches to its own
        text-based ReAct parser, which has full control over the stream.
        """
        if not tools or capabilities.supports_function_calling:
            return False

        if self._tool_fallback == "react_shim" and not streaming:
            return True

        raise CapabilityNotSupportedError(
            "function_calling",
            provider="ollama",
            model=self._model_name,
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        capabilities = await self._ensure_capabilities()
        use_shim = self._resolve_tool_strategy(
            tools, capabilities, streaming=False
        )

        effective_messages = (
            self._inject_tool_shim(messages, tools)
            if use_shim and tools
            else messages
        )
        native_tools = tools if (tools and not use_shim) else None

        payload = self._build_payload(
            effective_messages,
            config,
            tools=native_tools,
            stream=False,
        )
        data = await self._post_chat(
            payload,
            timeout_seconds=request_timeout(config, self._timeout_seconds),
        )

        raw_message = data.get("message") or {}
        if not isinstance(raw_message, dict):
            raw_message = {}

        content, reasoning = _split_reasoning(
            str(raw_message.get("content") or "")
        )
        thinking = raw_message.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            reasoning = (
                f"{reasoning}\n\n{thinking.strip()}" if reasoning else thinking.strip()
            )

        tool_calls = _extract_tool_calls(raw_message)

        if use_shim and not tool_calls:
            content, shim_call = self._parse_shim_response(content)
            if shim_call is not None:
                tool_calls = [shim_call]

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=self._usage_from_payload(data),
            model=str(data.get("model") or self._model_name),
            provider=self.provider_name(),
            finish_reason=self._finish_reason(data, tool_calls),
            raw_metadata=self._raw_metadata(data, reasoning),
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        capabilities = await self._ensure_capabilities()
        self._resolve_tool_strategy(tools, capabilities, streaming=True)

        payload = self._build_payload(
            messages,
            config,
            tools=tools,
            stream=True,
        )
        reasoning_parser = _ReasoningStreamParser()
        finish_reason = "stop"
        usage: TokenUsage | None = None
        saw_tool_calls = False

        client = self._get_client()
        try:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=request_timeout(config, self._timeout_seconds),
            ) as response:
                if not response.is_success:
                    try:
                        await response.aread()
                        error_body = response.text
                    except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                        error_body = ""
                    raise_for_ollama_status(
                        response.status_code,
                        error_body,
                        base_url=self._base_url,
                        model=self._model_name,
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue

                    error = event.get("error")
                    if error:
                        message = str(error)
                        if "not found" in message.lower() or "not installed" in message.lower():
                            missing = model_not_installed_message(self._model_name)
                            model_error = ModelNotFoundError(
                                missing,
                                provider="ollama",
                                model=self._model_name,
                            )
                            model_error.user_message = missing
                            raise model_error
                        raise ProviderError(
                            f"Ollama stream error: {message}",
                            provider="ollama",
                            model=self._model_name,
                            retriable=True,
                        )

                    raw_message = event.get("message") or {}
                    if isinstance(raw_message, dict):
                        thinking = raw_message.get("thinking")
                        if isinstance(thinking, str) and thinking:
                            reasoning_parser.reasoning_parts.append(thinking)

                        for call in _extract_tool_calls(raw_message):
                            saw_tool_calls = True
                            finish_reason = "tool_calls"
                            yield LLMStreamChunk(
                                tool_call_delta=call,
                                is_final=False,
                            )

                        text = raw_message.get("content") or ""
                        if text:
                            for kind, piece in reasoning_parser.feed(str(text)):
                                if kind == "content" and piece:
                                    yield LLMStreamChunk(
                                        delta_text=piece,
                                        is_final=False,
                                    )

                    event_usage = self._usage_from_payload(event)
                    if event_usage is not None:
                        usage = event_usage

                    if event.get("done"):
                        finish_reason = (
                            "tool_calls"
                            if saw_tool_calls
                            else self._finish_reason(event, [])
                        )
                        break

        except ProviderError:
            raise
        except Exception as exc:
            raise translate_transport_error(
                exc, base_url=self._base_url, model=self._model_name
            ) from exc

        # Flush any buffered visible text left by the reasoning-tag guard.
        for kind, piece in reasoning_parser.flush():
            if kind == "content" and piece:
                yield LLMStreamChunk(delta_text=piece, is_final=False)

        self._last_stream_reasoning = reasoning_parser.reasoning

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
            capabilities = await self.detect_capabilities()
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
                capabilities=capabilities,
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


class OllamaRuntimeManager:
    """Desktop service helpers for discovering and installing Ollama models.

    All methods use the same plain HTTP transport as the inference adapters and
    never require an Ollama Python SDK.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._base_url = normalize_base_url(
            base_url or getattr(settings, "OLLAMA_BASE_URL", None)
        )
        self._client = client
        self._transport = transport
        self._owns_client = client is None
        self._timeout_seconds = float(timeout_seconds)

    def base_url(self) -> str:
        return self._base_url

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def is_daemon_running(self) -> bool:
        """Return True when an Ollama daemon answers on the configured port."""
        client = self._get_client()
        for path in ("/api/version", "/api/tags"):
            try:
                response = await client.get(
                    f"{self._base_url}{path}",
                    timeout=self._timeout_seconds,
                )
            except Exception:  # noqa: BLE001, S112 - offline probe must never raise
                continue

            if response.is_success:
                return True
            if response.status_code == 404:
                # Try the next well-known probe endpoint.
                continue
            # The port answered with a non-404 status: something is running,
            # even if this particular probe was rejected.
            return True
        return False

    async def list_installed_models(self) -> list[dict[str, Any]]:
        """Return installed models with UI-friendly size and quantisation data."""
        client = self._get_client()
        try:
            response = await client.get(
                f"{self._base_url}/api/tags",
                timeout=self._timeout_seconds * 3,
            )
        except Exception as exc:
            raise translate_transport_error(
                exc, base_url=self._base_url, model=None
            ) from exc

        if not response.is_success:
            raise ProviderError(
                (
                    f"Failed to list Ollama models (HTTP {response.status_code}): "
                    f"{response.text[:600]}"
                ),
                provider="ollama",
                retriable=response.status_code >= 500,
            )

        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "Ollama /api/tags returned invalid JSON.",
                provider="ollama",
                retriable=True,
            ) from exc

        if not isinstance(payload, dict):
            raise ProviderError(
                "Ollama /api/tags returned an unexpected payload.",
                provider="ollama",
                retriable=True,
            )

        results: list[dict[str, Any]] = []
        for model in payload.get("models") or []:
            if not isinstance(model, dict):
                continue
            name = model.get("name") or model.get("model")
            if not name:
                continue
            details = model.get("details") or {}
            if not isinstance(details, dict):
                details = {}
            try:
                size_bytes = int(model.get("size") or 0)
            except (TypeError, ValueError):
                size_bytes = 0

            results.append(
                {
                    "name": str(name),
                    "size_bytes": size_bytes,
                    "family": str(details.get("family") or ""),
                    "parameter_size": str(details.get("parameter_size") or ""),
                    "quantization_level": str(
                        details.get("quantization_level") or ""
                    ),
                }
            )
        return results

    async def pull_model_stream(
        self,
        model_name: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream ``/api/pull`` progress events enriched with a percentage.

        Each yielded dictionary is Ollama's native event plus a ``percent``
        field (0–100, or ``None`` when the total size is not yet known).
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        model_name = model_name.strip()
        client = self._get_client()

        try:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/pull",
                json={"model": model_name, "stream": True},
                timeout=None,
            ) as response:
                if not response.is_success:
                    try:
                        await response.aread()
                        error_body = response.text
                    except Exception:  # noqa: BLE001  # pragma: no cover - defensive
                        error_body = ""
                    raise_for_ollama_status(
                        response.status_code,
                        error_body,
                        base_url=self._base_url,
                        model=model_name,
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue

                    if event.get("error"):
                        message = str(event["error"])
                        if "not found" in message.lower():
                            missing = model_not_installed_message(model_name)
                            model_error = ModelNotFoundError(
                                missing,
                                provider="ollama",
                                model=model_name,
                            )
                            model_error.user_message = missing
                            raise model_error
                        raise ProviderError(
                            f"Ollama pull failed: {message}",
                            provider="ollama",
                            model=model_name,
                            retriable=True,
                        )

                    percent: float | None = None
                    total = event.get("total")
                    completed = event.get("completed")
                    if str(event.get("status") or "").lower() == "success":
                        percent = 100.0
                    elif isinstance(total, (int, float)) and total > 0:
                        completed_value = (
                            float(completed)
                            if isinstance(completed, (int, float))
                            else 0.0
                        )
                        percent = round(
                            max(0.0, min(100.0, completed_value / total * 100.0)),
                            2,
                        )

                    event["percent"] = percent
                    yield event

        except (ProviderError, ValueError):
            raise
        except Exception as exc:
            raise translate_transport_error(
                exc, base_url=self._base_url, model=model_name
            ) from exc

    # Convenience helper used by the UI/test layer and to surface the exact
    # actionable error message without duplicating the string.
    def daemon_error_message(self) -> str:
        return daemon_unreachable_message(self._base_url)
