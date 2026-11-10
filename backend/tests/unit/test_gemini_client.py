from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from app.config import settings
from app.services.llm.errors import LLMUnavailableError, ModelResponseError
from app.services.llm.gemini_client import GeminiClient, TokenUsage


def response(text="Hello", *, metadata=True):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))], usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=10, candidates_token_count=3, thoughts_token_count=2, total_token_count=15) if metadata else None)


@pytest.fixture
def client():
    sdk = MagicMock()
    sdk.aio.models.generate_content = AsyncMock(return_value=response())
    return GeminiClient(client=sdk)


def test_returns_real_model_text(client):
    client.client.models.generate_content.return_value = response()
    assert client.complete([{"role": "user", "content": "hi"}], "system") == "Hello"
    assert client.client.models.generate_content.call_args.kwargs["model"] == settings.GEMINI_CHAT_MODEL


def test_api_error_never_fabricates_an_answer(client):
    client.client.models.generate_content.side_effect = RuntimeError("private provider details")
    with pytest.raises(LLMUnavailableError, match="unavailable") as error:
        client.complete([{"role": "user", "content": "hi"}], "system")
    assert "private" not in str(error.value)


async def test_usage_comes_from_metadata_including_thinking(client):
    usage = TokenUsage()
    await client.generate([], types.GenerateContentConfig(), usage=usage)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (10, 5, 15)


def test_missing_usage_is_not_estimated(client):
    client.client.models.generate_content.return_value = response(metadata=False)
    with pytest.raises(ModelResponseError):
        client.complete([], "system")


async def test_real_async_stream_and_final_metadata(client):
    async def chunks():
        yield response("Hello ", metadata=False)
        yield response("world")
    client.client.aio.models.generate_content_stream = AsyncMock(return_value=chunks())
    usage = TokenUsage()
    output = [text async for text in client.complete_stream([], "sys", usage=usage)]
    assert output == ["Hello ", "world"]
    assert usage.total_tokens == 15


async def test_stream_failure_is_typed_not_fake_text(client):
    client.client.aio.models.generate_content_stream = AsyncMock(side_effect=RuntimeError("network"))
    with pytest.raises(LLMUnavailableError):
        _ = [text async for text in client.complete_stream([], "sys")]


def test_cost_unknown_until_prices_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "LLM_INPUT_USD_PER_MILLION", None)
    assert client.calculate_cost(1, 1) is None
    assert client.calculate_cost(0, 0) == 0
    monkeypatch.setattr(settings, "LLM_INPUT_USD_PER_MILLION", 0.5)
    monkeypatch.setattr(settings, "LLM_OUTPUT_USD_PER_MILLION", 2.0)
    assert client.calculate_cost(1_000_000, 1_000_000) == 2.5


def test_langfuse_uses_current_observation_api(client, monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "test-secret")
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "test-public")
    client.client.models.generate_content.return_value = response()
    with patch("langfuse.Langfuse") as langfuse:
        client.complete([], "system")
        langfuse.return_value.start_observation.assert_called_once_with(name="quantiva-llm", as_type="generation", model=settings.GEMINI_CHAT_MODEL)
        langfuse.return_value.start_observation.return_value.end.assert_called_once()
