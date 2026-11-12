"""
app/providers/llm/gemini.py
──────────────────────────────────────────────────────────────────────────────
Google Gemini LLM provider adapter.

Conforms to :class:`~app.providers.interfaces.llm.LLMProvider`.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from app.config import settings
from app.providers.gemini_common import build_gemini_client, translate_gemini_error
from app.providers.interfaces.exceptions import ProviderError
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


class GeminiLLMProvider(LLMProvider):
    """Google Gemini adapter implementing the unified LLMProvider interface."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: genai.Client | None = None,
    ) -> None:
        self._model_name = model or settings.GEMINI_CHAT_MODEL
        self._client = client or build_gemini_client(api_key=api_key)

    def provider_name(self) -> str:
        return "gemini"

    def model_name(self) -> str:
        return self._model_name

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_vision=True,
            supports_system_prompt=True,
            max_context_window=1_048_576,
            is_local=False,
        )

    # ── Message and config translation ────────────────────────────────────────

    def _normalize_messages(
        self,
        messages: list[ChatMessage],
    ) -> tuple[list[types.Content], str | None]:
        """Convert normalised ChatMessage list to Gemini Content objects and system prompt."""
        system_prompts: list[str] = []
        contents: list[types.Content] = []

        for msg in messages:
            if msg.role == ChatRole.SYSTEM:
                if msg.content:
                    system_prompts.append(msg.content)
            elif msg.role == ChatRole.USER:
                parts: list[types.Part] = [types.Part.from_text(text=msg.content or "")]
                contents.append(types.Content(role="user", parts=parts))
            elif msg.role == ChatRole.ASSISTANT:
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(
                            types.Part.from_function_call(
                                name=tc.name,
                                args=tc.arguments or {},
                            )
                        )
                if not parts:
                    parts.append(types.Part.from_text(text=""))
                contents.append(types.Content(role="model", parts=parts))
            elif msg.role == ChatRole.TOOL:
                if msg.tool_result:
                    resp_dict = {"result": msg.tool_result.content}
                    part = types.Part.from_function_response(
                        name=msg.tool_result.name,
                        response=resp_dict,
                    )
                else:
                    part = types.Part.from_function_response(
                        name="tool",
                        response={"result": msg.content or ""},
                    )
                contents.append(types.Content(role="user", parts=[part]))

        system_instruction = "\n\n".join(system_prompts) if system_prompts else None
        return contents, system_instruction

    def _build_config(
        self,
        config: GenerationConfig | None,
        system_instruction: str | None,
        tools: list[ToolDefinition] | None,
    ) -> types.GenerateContentConfig:
        """Construct a types.GenerateContentConfig for the Gemini SDK."""
        effective_cfg = config or GenerationConfig()
        gen_kwargs: dict[str, Any] = {
            "temperature": effective_cfg.temperature,
            "max_output_tokens": effective_cfg.max_tokens,
            "top_p": effective_cfg.top_p,
        }

        if effective_cfg.stop_sequences:
            gen_kwargs["stop_sequences"] = effective_cfg.stop_sequences

        if effective_cfg.response_json_schema:
            gen_kwargs["response_mime_type"] = "application/json"
            gen_kwargs["response_schema"] = effective_cfg.response_json_schema

        if system_instruction:
            gen_kwargs["system_instruction"] = system_instruction

        if tools:
            declarations = [
                types.FunctionDeclaration(
                    name=td.name,
                    description=td.description,
                    parameters=td.parameters_json_schema,
                )
                for td in tools
            ]
            gen_kwargs["tools"] = [types.Tool(function_declarations=declarations)]
            gen_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )

        return types.GenerateContentConfig(**gen_kwargs)

    def _extract_usage(self, metadata: Any) -> TokenUsage | None:
        """Convert Gemini usage_metadata to normalised TokenUsage."""
        if metadata is None:
            return None
        prompt_tokens = getattr(metadata, "prompt_token_count", 0) or 0
        candidates = getattr(metadata, "candidates_token_count", 0) or 0
        thoughts = getattr(metadata, "thoughts_token_count", 0) or 0
        completion_tokens = candidates + thoughts
        total = getattr(metadata, "total_token_count", 0) or (prompt_tokens + completion_tokens)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )

    # ── Generation API ────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        contents, system_instruction = self._normalize_messages(messages)
        gemini_cfg = self._build_config(config, system_instruction, tools)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=gemini_cfg,
            )
        except Exception as exc:
            raise translate_gemini_error(exc, model=self._model_name) from exc

        # Extract text content
        content = ""
        try:
            content = response.text or ""
        except Exception:
            # When finish_reason is tool call or blocked, response.text may raise
            if response.candidates and response.candidates[0].content:
                parts = response.candidates[0].content.parts or []
                text_parts = [p.text for p in parts if getattr(p, "text", None)]
                content = "".join(text_parts)

        # Extract tool calls
        tool_calls: list[ToolCallRequest] = []
        if getattr(response, "function_calls", None):
            for call in response.function_calls:
                call_id = getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:12]}"
                tool_calls.append(
                    ToolCallRequest(
                        id=call_id,
                        name=call.name,
                        arguments=dict(call.args or {}),
                    )
                )

        # Extract finish reason
        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif response.candidates and response.candidates[0].finish_reason:
            raw_fr = response.candidates[0].finish_reason
            if hasattr(raw_fr, "value") and isinstance(raw_fr.value, str):
                finish_reason = raw_fr.value.lower()
            elif hasattr(raw_fr, "name") and isinstance(raw_fr.name, str):
                finish_reason = raw_fr.name.lower()
            else:
                s = str(raw_fr).lower()
                finish_reason = s.split(".")[-1] if "." in s else s

        usage = self._extract_usage(getattr(response, "usage_metadata", None))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=self._model_name,
            provider=self.provider_name(),
            finish_reason=finish_reason,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        contents, system_instruction = self._normalize_messages(messages)
        gemini_cfg = self._build_config(config, system_instruction, tools)

        try:
            stream_resp = await self._client.aio.models.generate_content_stream(
                model=self._model_name,
                contents=contents,
                config=gemini_cfg,
            )
        except Exception as exc:
            raise translate_gemini_error(exc, model=self._model_name) from exc

        last_usage_metadata = None
        last_finish_reason = "stop"

        try:
            async for chunk in stream_resp:
                if getattr(chunk, "usage_metadata", None) is not None:
                    last_usage_metadata = chunk.usage_metadata

                if getattr(chunk, "candidates", None) and chunk.candidates:
                    cand = chunk.candidates[0]
                    if getattr(cand, "finish_reason", None):
                        raw_fr = cand.finish_reason
                        if hasattr(raw_fr, "value") and isinstance(raw_fr.value, str):
                            last_finish_reason = raw_fr.value.lower()
                        elif hasattr(raw_fr, "name") and isinstance(raw_fr.name, str):
                            last_finish_reason = raw_fr.name.lower()
                        else:
                            s = str(raw_fr).lower()
                            last_finish_reason = s.split(".")[-1] if "." in s else s

                # Yield function calls if present in chunk
                if getattr(chunk, "function_calls", None):
                    last_finish_reason = "tool_calls"
                    for call in chunk.function_calls:
                        call_id = getattr(call, "id", None) or f"call_{uuid.uuid4().hex[:12]}"
                        yield LLMStreamChunk(
                            tool_call_delta=ToolCallRequest(
                                id=call_id,
                                name=call.name,
                                arguments=dict(call.args or {}),
                            ),
                            is_final=False,
                        )

                # Yield text delta
                text = ""
                try:
                    text = chunk.text or ""
                except Exception:
                    pass
                if text:
                    yield LLMStreamChunk(delta_text=text, is_final=False)

        except Exception as exc:
            raise translate_gemini_error(exc, model=self._model_name) from exc

        # Emit final sentinel chunk
        final_usage = self._extract_usage(last_usage_metadata)
        yield LLMStreamChunk(
            delta_text="",
            is_final=True,
            usage=final_usage,
            finish_reason=last_finish_reason,
        )

    # ── Connectivity probe ────────────────────────────────────────────────────

    async def test_connection(self) -> ConnectionTestResult:
        start_time = time.perf_counter()
        try:
            probe_messages = [ChatMessage(role=ChatRole.USER, content="ping")]
            probe_config = GenerationConfig(max_tokens=1, timeout_seconds=10.0)
            await self.generate(probe_messages, config=probe_config)
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
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                error_code="UnknownError",
                error_message=str(exc),
            )
