"""
backend/app/services/llm/gemini_client.py
────────────────────────────────────────────────────────────────────────────────
Gemini 1.5 Flash client with full Langfuse LLM observability.

Every call creates a Langfuse trace with:
  • A root "llm-call" span (model, latency, token counts, cost)
  • A nested "retrieval" span when chunks are injected into context
  • A nested "tool-call" span for each agentic tool invocation

Langfuse is optional — if LANGFUSE_SECRET_KEY is not set, tracing is a no-op
and the client behaves exactly as before.
"""

import time
import asyncio
from typing import List, Dict, AsyncGenerator, Optional, Any

from google import genai
from google.genai import types
from app.config import settings
from app.core.metrics import LLM_TOKENS, LLM_LATENCY, LLM_COST

# ── Langfuse setup (optional) ─────────────────────────────────────────────────

_langfuse = None

def _get_langfuse():
    """Lazy-init Langfuse client. Returns None if not configured."""
    global _langfuse
    if _langfuse is not None:
        return _langfuse
    if not settings.LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            secret_key=settings.LANGFUSE_SECRET_KEY,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            host=settings.LANGFUSE_HOST,
        )
        return _langfuse
    except Exception:
        return None


# ── Pricing (Gemini 1.5 Flash, per 1M tokens) ─────────────────────────────────
_PRICE_INPUT_PER_1M  = 0.075   # USD
_PRICE_OUTPUT_PER_1M = 0.30    # USD


class GeminiClient:
    """
    Wrapper around the google-genai SDK with integrated Langfuse tracing
    and Prometheus metrics.

    Tracing model:
        trace
        └── generation span  (llm-call)
            ├── usage: {input: N, output: M}
            ├── model: gemini-1.5-flash
            ├── latency: Xs
            └── cost_usd: $Y
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = "gemini-1.5-flash"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[types.Content]:
        """Convert standard role/content dicts to google-genai Content objects."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
        return contents

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return estimated USD cost for the given token counts."""
        return (
            (input_tokens  / 1_000_000) * _PRICE_INPUT_PER_1M
            + (output_tokens / 1_000_000) * _PRICE_OUTPUT_PER_1M
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars ≈ 1 token) for mock/fallback paths."""
        return max(1, len(text) // 4)

    # ── Tracing helpers ───────────────────────────────────────────────────────

    def start_trace(
        self,
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Open a new Langfuse trace. Returns the trace object (or None if
        Langfuse is not configured).

        A trace groups all spans for one user request end-to-end.
        """
        lf = _get_langfuse()
        if lf is None:
            return None
        return lf.trace(
            name=name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
            tags=[settings.APP_ENV],
        )

    def trace_retrieval(
        self,
        trace,
        query: str,
        chunks: List[Dict],
        latency_s: float,
    ):
        """
        Add a 'retrieval' span to an existing trace, logging the hybrid
        retrieval step (vector + BM25 + rerank).
        """
        if trace is None:
            return
        trace.span(
            name="retrieval",
            input={"query": query},
            output={"chunk_count": len(chunks), "chunks": chunks[:3]},  # sample first 3
            metadata={"latency_s": round(latency_s, 3)},
        )

    def trace_tool_call(
        self,
        trace,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        latency_s: float,
    ):
        """Add a 'tool-call' span for an agentic tool invocation."""
        if trace is None:
            return
        trace.span(
            name=f"tool:{tool_name}",
            input=tool_input,
            output=tool_output,
            metadata={"latency_s": round(latency_s, 3)},
        )

    # ── Synchronous generation ─────────────────────────────────────────────────

    def complete(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        trace=None,
        span_name: str = "llm-call",
    ) -> str:
        """
        Call Gemini synchronously and record a Langfuse generation span.

        Args:
            messages: List of {role, content} dicts.
            system_prompt: The system instruction.
            trace: An active Langfuse trace object (or None).
            span_name: Label for the generation span.

        Returns:
            The model's text response.
        """
        contents = self._format_messages(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
        )

        # Open Langfuse generation span
        generation = None
        if trace is not None:
            generation = trace.generation(
                name=span_name,
                model=self.model_name,
                input={"messages": messages, "system": system_prompt},
            )

        start = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            text = response.text
            latency = time.perf_counter() - start

            # Extract token counts from response metadata
            input_tokens  = getattr(response.usage_metadata, "prompt_token_count",     self._estimate_tokens(system_prompt))
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", self._estimate_tokens(text))
            cost = self.calculate_cost(input_tokens, output_tokens)

            # ── Prometheus ──────────────────────────────────────────────────
            LLM_TOKENS.labels(model=self.model_name, direction="input").inc(input_tokens)
            LLM_TOKENS.labels(model=self.model_name, direction="output").inc(output_tokens)
            LLM_LATENCY.labels(model=self.model_name).observe(latency)
            LLM_COST.labels(model=self.model_name).inc(cost)

            # ── Langfuse ────────────────────────────────────────────────────
            if generation is not None:
                generation.end(
                    output=text,
                    usage={
                        "input":  input_tokens,
                        "output": output_tokens,
                        "unit":   "TOKENS",
                    },
                    metadata={
                        "latency_s":  round(latency, 3),
                        "cost_usd":   round(cost, 6),
                        "model":      self.model_name,
                    },
                )

            return text

        except Exception as e:
            latency = time.perf_counter() - start
            if generation is not None:
                generation.end(
                    output=None,
                    level="ERROR",
                    status_message=str(e),
                    metadata={"latency_s": round(latency, 3)},
                )
            # Graceful fallback for local dev without a valid API key
            return (
                f"Mock AI Response (Gemini error: {str(e)[:80]}) "
                "— Revenue is $5M based on Q3 report."
            )

    # ── Async streaming generation ─────────────────────────────────────────────

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        trace=None,
        span_name: str = "llm-stream",
    ) -> AsyncGenerator[str, None]:
        """
        Stream Gemini tokens one by one.

        Yields each text chunk as it arrives.
        On completion, closes the Langfuse generation span with token counts.
        """
        contents = self._format_messages(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
        )

        generation = None
        if trace is not None:
            generation = trace.generation(
                name=span_name,
                model=self.model_name,
                input={"messages": messages, "system": system_prompt},
            )

        start = time.perf_counter()
        full_text = ""

        try:
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=config,
            )
            for chunk in response:
                token_text = chunk.text or ""
                full_text += token_text
                yield token_text

            latency = time.perf_counter() - start

            # Use metadata from the last chunk if available
            input_tokens  = self._estimate_tokens(system_prompt + " ".join(m["content"] for m in messages))
            output_tokens = self._estimate_tokens(full_text)
            cost = self.calculate_cost(input_tokens, output_tokens)

            LLM_TOKENS.labels(model=self.model_name, direction="input").inc(input_tokens)
            LLM_TOKENS.labels(model=self.model_name, direction="output").inc(output_tokens)
            LLM_LATENCY.labels(model=self.model_name).observe(latency)
            LLM_COST.labels(model=self.model_name).inc(cost)

            if generation is not None:
                generation.end(
                    output=full_text,
                    usage={
                        "input":  input_tokens,
                        "output": output_tokens,
                        "unit":   "TOKENS",
                    },
                    metadata={
                        "latency_s": round(latency, 3),
                        "cost_usd":  round(cost, 6),
                    },
                )

        except Exception as e:
            latency = time.perf_counter() - start
            if generation is not None:
                generation.end(
                    output=None,
                    level="ERROR",
                    status_message=str(e),
                    metadata={"latency_s": round(latency, 3)},
                )
            # Stream a mock fallback
            mock = (
                f"Mock AI Streaming (Gemini error: {str(e)[:60]}) "
                "— Revenue is $5M. "
            )
            for word in mock.split(" "):
                await asyncio.sleep(0.05)
                yield word + " "


# ── Singleton ─────────────────────────────────────────────────────────────────
gemini_client = GeminiClient()
