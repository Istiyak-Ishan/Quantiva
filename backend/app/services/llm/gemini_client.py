"""Shared google-genai adapter. No mock answers, token estimates, or silent success."""
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
import logging
import time

from google import genai
from google.genai import types

from app.config import settings
from app.core.metrics import LLM_TOKENS, LLM_LATENCY, LLM_COST
from app.services.llm.errors import LLMUnavailableError, ModelResponseError

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, metadata):
        if metadata is None or metadata.prompt_token_count is None or metadata.total_token_count is None:
            raise ModelResponseError()
        self.input_tokens += metadata.prompt_token_count
        # Thinking tokens are billable output too, not just visible candidate text.
        self.output_tokens += (metadata.candidates_token_count or 0) + (metadata.thoughts_token_count or 0)
        self.total_tokens += metadata.total_token_count


class GeminiClient:
    def __init__(self, client=None):
        self.client = client  # Explicit injection for contract tests, not runtime test detection.
        self.model_name = settings.GEMINI_CHAT_MODEL

    def _new_client(self):
        return genai.Client(api_key=settings.GOOGLE_API_KEY, http_options=types.HttpOptions(timeout=settings.GOOGLE_TIMEOUT_MS))

    @contextmanager
    def _sync_api(self):
        if self.client is not None:
            yield self.client
        else:
            with self._new_client() as client:
                yield client

    @asynccontextmanager
    async def _async_api(self):
        if self.client is not None:
            yield self.client.aio
        else:
            # Each call owns its async transport; safe across Celery event loops.
            async with self._new_client().aio as client:
                yield client

    def _format_messages(self, messages):
        return [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part.from_text(text=m["content"])]) for m in messages]

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        if input_tokens == output_tokens == 0:
            return 0.0
        if settings.LLM_INPUT_USD_PER_MILLION is None or settings.LLM_OUTPUT_USD_PER_MILLION is None:
            return None
        return (input_tokens * settings.LLM_INPUT_USD_PER_MILLION + output_tokens * settings.LLM_OUTPUT_USD_PER_MILLION) / 1_000_000

    def _trace(self):
        if not (settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY):
            return None
        try:
            from langfuse import Langfuse
            client = Langfuse(secret_key=settings.LANGFUSE_SECRET_KEY, public_key=settings.LANGFUSE_PUBLIC_KEY, host=settings.LANGFUSE_HOST)
            # Do not send tenant documents/prompts to a tracing vendor by default.
            return client.start_observation(name="quantiva-llm", as_type="generation", model=self.model_name)
        except Exception:
            logger.warning("Langfuse tracing unavailable", exc_info=False)
            return None

    def _record(self, metadata, usage, start, trace):
        call_usage = TokenUsage()
        call_usage.add(metadata)
        usage.input_tokens += call_usage.input_tokens
        usage.output_tokens += call_usage.output_tokens
        usage.total_tokens += call_usage.total_tokens
        LLM_TOKENS.labels(model=self.model_name, direction="input").inc(call_usage.input_tokens)
        LLM_TOKENS.labels(model=self.model_name, direction="output").inc(call_usage.output_tokens)
        LLM_LATENCY.labels(model=self.model_name).observe(time.perf_counter() - start)
        cost = self.calculate_cost(call_usage.input_tokens, call_usage.output_tokens)
        if cost is not None:
            LLM_COST.labels(model=self.model_name).inc(cost)
        if trace is not None:
            try:
                trace.update(usage_details={"input": call_usage.input_tokens, "output": call_usage.output_tokens}, cost_details={"total": cost} if cost is not None else None)
                trace.end()
            except Exception:
                logger.warning("Langfuse recording failed", exc_info=False)

    def _failed_trace(self, trace):
        if trace is not None:
            try:
                trace.update(level="ERROR", status_message="Generation failed")
                trace.end()
            except Exception:
                logger.warning("Langfuse error recording failed", exc_info=False)

    def complete(self, messages, system_prompt, **kwargs) -> str:
        start, trace = time.perf_counter(), self._trace()
        try:
            with self._sync_api() as api:
                response = api.models.generate_content(model=self.model_name, contents=self._format_messages(messages), config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.0))
            if not response.text:
                raise ModelResponseError()
            self._record(response.usage_metadata, TokenUsage(), start, trace)
            return response.text
        except ModelResponseError:
            self._failed_trace(trace)
            raise
        except Exception as exc:
            self._failed_trace(trace)
            raise LLMUnavailableError() from exc

    async def generate(self, contents, config, *, usage: TokenUsage):
        start, trace = time.perf_counter(), self._trace()
        try:
            async with self._async_api() as api:
                response = await api.models.generate_content(model=self.model_name, contents=contents, config=config)
            self._record(response.usage_metadata, usage, start, trace)
            return response
        except ModelResponseError:
            self._failed_trace(trace)
            raise
        except Exception as exc:
            self._failed_trace(trace)
            raise LLMUnavailableError() from exc

    async def generate_stream(self, contents, config, *, usage: TokenUsage):
        start, trace, metadata = time.perf_counter(), self._trace(), None
        try:
            async with self._async_api() as api:
                stream = await api.models.generate_content_stream(model=self.model_name, contents=contents, config=config)
                async for chunk in stream:
                    if chunk.usage_metadata is not None:
                        metadata = chunk.usage_metadata
                    yield chunk
            self._record(metadata, usage, start, trace)
        except ModelResponseError:
            self._failed_trace(trace)
            raise
        except Exception as exc:
            self._failed_trace(trace)
            raise LLMUnavailableError() from exc

    async def complete_stream(self, messages, system_prompt, *, usage: TokenUsage | None = None, **kwargs):
        async for chunk in self.generate_stream(self._format_messages(messages), types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.0), usage=usage if usage is not None else TokenUsage()):
            if chunk.text:
                yield chunk.text


gemini_client = GeminiClient()
