from typing import Any

import logging

from google.genai import types
from pydantic import BaseModel, Field

from app.services.llm.gemini_client import gemini_client, TokenUsage
from app.services.llm.errors import AgentStepLimitError, ModelResponseError
from app.services.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentResult(BaseModel):
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls_made: int = 0
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class BaseAgent:
    def __init__(self, system_prompt: str, tools: ToolRegistry | None = None):
        self.client = gemini_client
        self.model_name = gemini_client.model_name
        self.system_prompt = system_prompt
        self.tools = tools
        self.MAX_ITERATIONS = 10

    def _observe(self, messages, system_prompt=None):
        return [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part.from_text(text=m["content"])]) for m in messages]

    async def run_loop(self, query: str, context=None, history=None, on_text=None) -> AgentResult:
        contents = self._observe(list(history or []) + [{"role": "user", "content": query}])
        usage, calls = TokenUsage(), 0
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt, temperature=0.0,
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**schema) for schema in self.tools.get_schemas()])] if self.tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=8192,
        )
        for iteration in range(self.MAX_ITERATIONS):
            if on_text is None:
                response = await self.client.generate(contents, config, usage=usage)
            else:
                parts: list[types.Part] = []
                raw_text = ""
                async for chunk in self.client.generate_stream(contents, config, usage=usage):
                    if chunk.candidates and chunk.candidates[0].content:
                        parts.extend(chunk.candidates[0].content.parts or [])
                    if chunk.text:
                        raw_text += chunk.text
                        await on_text(raw_text)
                response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])
            if not response.candidates or not response.candidates[0].content:
                logger.warning(
                    "Model returned no candidate content at iteration %d (candidates=%d)",
                    iteration + 1,
                    len(response.candidates or []),
                )
                raise ModelResponseError()
            # Keep the SDK's exact model content, including Gemini thought signatures.
            # Rebuilding function-call parts loses these and breaks later tool turns.
            contents.append(response.candidates[0].content)
            if response.function_calls:
                results = []
                for call in response.function_calls:
                    calls += 1
                    result = await self.tools.execute(call.name, call.args or {})
                    results.append(types.Part.from_function_response(name=call.name, response={"result": result}))
                contents.append(types.Content(role="user", parts=results))
            elif response.text:
                return AgentResult(answer=response.text, tool_calls_made=calls, iterations=iteration + 1, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, total_tokens=usage.total_tokens)
            else:
                logger.warning("Model returned neither text nor a tool call at iteration %d", iteration + 1)
                raise ModelResponseError()
        # Iteration exhaustion is failure, never a successfully completed answer.
        logger.warning("Agent exhausted %d iterations without a final answer", self.MAX_ITERATIONS)
        raise AgentStepLimitError()
