import pytest
from unittest.mock import MagicMock, patch
from app.services.llm.gemini_client import GeminiClient

@pytest.fixture
def client():
    with patch("app.services.llm.gemini_client.genai.Client"):
        return GeminiClient()

class TestComplete:
    def test_returns_model_text(self, client):
        mock_response = MagicMock()
        mock_response.text = "The revenue was $5M in Q3."
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 20
        client.client.models.generate_content.return_value = mock_response

        result = client.complete([{"role": "user", "content": "What is revenue?"}], "You are helpful.")
        assert "revenue" in result.lower() or "$5M" in result

    def test_fallback_on_api_error(self, client):
        client.client.models.generate_content.side_effect = Exception("API quota exceeded")
        result = client.complete([{"role": "user", "content": "hello"}], "system")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_langfuse_trace_none_safe(self, client):
        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        client.client.models.generate_content.return_value = mock_response

        result = client.complete([{"role": "user", "content": "hi"}], "sys", trace=None)
        assert isinstance(result, str)

class TestCompleteStream:
    @pytest.mark.asyncio
    async def test_yields_tokens(self, client):
        mock_chunk1 = MagicMock(); mock_chunk1.text = "Hello "
        mock_chunk2 = MagicMock(); mock_chunk2.text = "world."
        client.client.models.generate_content_stream.return_value = iter([mock_chunk1, mock_chunk2])

        tokens = []
        async for token in client.complete_stream([{"role": "user", "content": "hi"}], "sys"):
            tokens.append(token)

        assert len(tokens) == 2
        assert "".join(tokens) == "Hello world."

    @pytest.mark.asyncio
    async def test_fallback_stream_on_error(self, client):
        client.client.models.generate_content_stream.side_effect = Exception("network error")

        tokens = []
        async for token in client.complete_stream([{"role": "user", "content": "hi"}], "sys"):
            tokens.append(token)

        full = "".join(tokens)
        assert len(full) > 0

class TestCostCalculation:
    def test_zero_tokens_zero_cost(self):
        c = GeminiClient.__new__(GeminiClient)
        assert c.calculate_cost(0, 0) == 0.0

    def test_1m_input_tokens_costs_0_075_usd(self):
        c = GeminiClient.__new__(GeminiClient)
        cost = c.calculate_cost(1_000_000, 0)
        assert abs(cost - 0.075) < 0.001

    def test_1m_output_tokens_costs_0_30_usd(self):
        c = GeminiClient.__new__(GeminiClient)
        cost = c.calculate_cost(0, 1_000_000)
        assert abs(cost - 0.30) < 0.001

