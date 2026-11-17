"""Provider-decoupled dual-mode ReAct reasoning agent."""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.agents.tools import AgentTool, AgentToolRegistry
from app.providers.interfaces.exceptions import CapabilityNotSupportedError
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.types import (
    ChatMessage,
    ChatRole,
    GenerationConfig,
    LLMResponse,
    ToolCallRequest,
    ToolCallResult,
)

TokenCallback = Callable[[str], Awaitable[None] | None]
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"Final\s+Answer\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


class AgentIterationLimitError(RuntimeError):
    """Raised when an agent does not produce a final answer within its budget."""


@dataclass(slots=True)
class AgentRunResult:
    answer: str
    tool_calls_made: int
    iterations: int
    used_native_tool_calling: bool


@dataclass(frozen=True, slots=True)
class ParsedReActResponse:
    thought: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    final_answer: str | None = None


def _extract_json_object(text: str, start: int = 0) -> dict[str, Any] | None:
    """Decode the first balanced JSON object after *start*."""
    decoder = json.JSONDecoder()
    for position in range(start, len(text)):
        if text[position] != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_react_response(text: str) -> ParsedReActResponse:
    """Parse robust text ReAct output, including code fences and think tags."""
    without_think = _THINK_RE.sub("", text or "").strip()
    final_match = _FINAL_RE.search(without_think)
    if final_match:
        answer = final_match.group(1).strip()
        answer = re.sub(r"^```(?:markdown|text)?\s*|\s*```$", "", answer).strip()
        return ParsedReActResponse(final_answer=answer)

    action_match = re.search(r"Action\s*:", without_think, re.IGNORECASE)
    payload = _extract_json_object(
        without_think,
        action_match.end() if action_match else 0,
    )
    if payload is not None:
        tool = payload.get("tool") or payload.get("name")
        arguments = payload.get("arguments", payload.get("args", {}))
        if isinstance(tool, str) and isinstance(arguments, Mapping):
            thought_match = re.search(
                r"Thought\s*:\s*(.*?)(?=\s*Action\s*:)",
                without_think,
                re.IGNORECASE | re.DOTALL,
            )
            return ParsedReActResponse(
                thought=thought_match.group(1).strip() if thought_match else None,
                tool=tool,
                arguments=dict(arguments),
            )

    # A plain response is treated as final rather than forcing a local model
    # into another loop merely because it omitted the protocol label.
    if without_think:
        return ParsedReActResponse(final_answer=without_think)
    return ParsedReActResponse()


class ReActAgent:
    """Multi-step agent using native tools when available and text ReAct otherwise."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are Quantiva, a document reasoning agent. Decompose complex queries, "
        "retrieve evidence, compare documents, and use calculation tools for "
        "numerical synthesis. Never invent evidence or tool results."
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: (
            AgentToolRegistry
            | Sequence[AgentTool]
            | Mapping[str, Callable[..., Any]]
            | None
        ) = None,
        *,
        system_prompt: str | None = None,
        max_iterations: int = 5,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        if llm_provider is None:
            raise ValueError("An injected LLMProvider is required")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.llm_provider = llm_provider
        if isinstance(tools, AgentToolRegistry):
            self.tools = tools
        else:
            self.tools = AgentToolRegistry(tools)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_iterations = int(max_iterations)
        self.generation_config = generation_config or GenerationConfig(
            temperature=0.0,
            max_tokens=4096,
        )

    async def _emit(self, callback: TokenCallback | None, text: str) -> None:
        if callback is None or not text:
            return
        result = callback(text)
        if result is not None:
            await result

    async def _generate(
        self,
        messages: list[ChatMessage],
        *,
        native_tools: bool,
        on_token: TokenCallback | None,
    ) -> LLMResponse:
        definitions = self.tools.definitions if native_tools else None
        capabilities = self.llm_provider.get_capabilities()
        if on_token is None or not capabilities.supports_streaming:
            return await self.llm_provider.generate(
                messages,
                config=self.generation_config,
                tools=definitions,
            )

        text_parts: list[str] = []
        calls: dict[str, ToolCallRequest] = {}
        usage = None
        finish_reason = "stop"
        async for chunk in self.llm_provider.stream(
            messages,
            config=self.generation_config,
            tools=definitions,
        ):
            if chunk.delta_text:
                text_parts.append(chunk.delta_text)
                await self._emit(on_token, chunk.delta_text)
            if chunk.tool_call_delta is not None:
                calls[chunk.tool_call_delta.id] = chunk.tool_call_delta
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
        return LLMResponse(
            content="".join(text_parts),
            tool_calls=list(calls.values()),
            usage=usage,
            model=self.llm_provider.model_name(),
            provider=self.llm_provider.provider_name(),
            finish_reason=finish_reason,
        )

    async def run(
        self,
        query: str,
        *,
        history: Sequence[ChatMessage] | None = None,
        on_token: TokenCallback | None = None,
    ) -> AgentRunResult:
        if not query.strip():
            raise ValueError("query must be non-empty")
        native = bool(
            self.tools.definitions
            and self.llm_provider.get_capabilities().supports_function_calling
        )
        if native:
            try:
                return await self._run_native(query, history=history, on_token=on_token)
            except CapabilityNotSupportedError:
                # Capability descriptors may be stale or over-optimistic.
                pass
        return await self._run_text_react(query, history=history, on_token=on_token)

    async def _run_native(
        self,
        query: str,
        *,
        history: Sequence[ChatMessage] | None,
        on_token: TokenCallback | None,
    ) -> AgentRunResult:
        messages = [ChatMessage(ChatRole.SYSTEM, self.system_prompt)]
        messages.extend(history or ())
        messages.append(ChatMessage(ChatRole.USER, query.strip()))
        tool_calls_made = 0

        for iteration in range(1, self.max_iterations + 1):
            response = await self._generate(
                messages,
                native_tools=True,
                on_token=on_token,
            )
            if response.tool_calls:
                messages.append(
                    ChatMessage(
                        ChatRole.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                for call in response.tool_calls:
                    tool_calls_made += 1
                    observation = await self.tools.execute(call.name, call.arguments)
                    messages.append(
                        ChatMessage(
                            ChatRole.TOOL,
                            tool_result=ToolCallResult(
                                tool_call_id=call.id,
                                name=call.name,
                                content=observation,
                                is_error='"success": false' in observation.lower(),
                            ),
                        )
                    )
                continue
            answer = response.content.strip()
            if answer:
                if on_token is not None and not self.llm_provider.get_capabilities().supports_streaming:
                    await self._emit(on_token, answer)
                return AgentRunResult(answer, tool_calls_made, iteration, True)

        raise AgentIterationLimitError(
            f"Agent exceeded the configured maximum of {self.max_iterations} iterations"
        )

    def _react_system_prompt(self) -> str:
        schemas = [
            {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.parameters_json_schema,
            }
            for definition in self.tools.definitions
        ]
        return (
            f"{self.system_prompt}\n\n"
            "Native function calling is unavailable. Follow this protocol exactly.\n"
            "To use one tool, output:\n"
            'Thought: brief reason\nAction: {"tool":"tool_name","arguments":{...}}\n'
            "After an Observation, either call another tool or output:\n"
            "Final Answer: your evidence-grounded answer\n"
            "Never invent an Observation. Call at most one tool per turn.\n"
            f"Available tools (JSON): {json.dumps(schemas, sort_keys=True)}"
        )

    async def _run_text_react(
        self,
        query: str,
        *,
        history: Sequence[ChatMessage] | None,
        on_token: TokenCallback | None,
    ) -> AgentRunResult:
        messages = [ChatMessage(ChatRole.SYSTEM, self._react_system_prompt())]
        messages.extend(history or ())
        messages.append(ChatMessage(ChatRole.USER, query.strip()))
        tool_calls_made = 0

        for iteration in range(1, self.max_iterations + 1):
            response = await self._generate(
                messages,
                native_tools=False,
                on_token=None,  # do not expose private Thought/Action text
            )
            parsed = parse_react_response(response.content)
            if parsed.tool is not None:
                tool_calls_made += 1
                observation = await self.tools.execute(
                    parsed.tool,
                    parsed.arguments or {},
                )
                messages.append(ChatMessage(ChatRole.ASSISTANT, response.content))
                messages.append(
                    ChatMessage(
                        ChatRole.USER,
                        f"Observation: {observation}",
                    )
                )
                continue
            if parsed.final_answer is not None:
                await self._emit(on_token, parsed.final_answer)
                return AgentRunResult(
                    parsed.final_answer,
                    tool_calls_made,
                    iteration,
                    False,
                )

            messages.append(ChatMessage(ChatRole.ASSISTANT, response.content))
            messages.append(
                ChatMessage(
                    ChatRole.USER,
                    "Your response was empty or malformed. Use Action JSON or Final Answer.",
                )
            )

        raise AgentIterationLimitError(
            f"Agent exceeded the configured maximum of {self.max_iterations} iterations"
        )


__all__ = [
    "AgentIterationLimitError",
    "AgentRunResult",
    "ParsedReActResponse",
    "ReActAgent",
    "parse_react_response",
]
