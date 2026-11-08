import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from app.schemas.agent import AgentAnswer, Citation

MOCK_ANSWER = AgentAnswer(
    answer="The Q3 revenue was $5 million, representing a 15% increase over Q2.",
    citations=[Citation(
        doc_id="doc-abc",
        filename="annual_report.pdf",
        page_or_row=12,
        text_excerpt="Q3 revenue totalled $5M",
        relevance_score=0.93,
    )],
    confidence=0.93,
    tokens_used=280,
)

MOCK_AGENT_RESULT = MagicMock()
MOCK_AGENT_RESULT.answer = MOCK_ANSWER
MOCK_AGENT_RESULT.tool_calls_made = 1
MOCK_AGENT_RESULT.iterations = 2

class TestSessions:
    async def test_create_session(self, client, auth_headers):
        resp = await client.post("/api/v1/chat/sessions", json={"document_ids": []}, headers=auth_headers)
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    async def test_create_session_unauthenticated(self, client):
        resp = await client.post("/api/v1/chat/sessions", json={"document_ids": []})
        assert resp.status_code == 401

class TestChatMessage:
    async def test_send_message_returns_answer(self, client, auth_headers):
        sess = await client.post("/api/v1/chat/sessions", json={"document_ids": []}, headers=auth_headers)
        session_id = sess.json()["session_id"]

        with (
            patch("app.api.v1.chat.ReasoningAgent") as MockAgent,
            patch("app.services.memory.session_memory.session_memory.load_history", new_callable=AsyncMock, return_value=[]),
            patch("app.services.memory.session_memory.session_memory.save_turn", new_callable=AsyncMock),
            patch("app.services.memory.semantic_cache.semantic_cache.get", new_callable=AsyncMock, return_value=None),
            patch("app.services.memory.semantic_cache.semantic_cache.set", new_callable=AsyncMock),
        ):
            MockAgent.return_value.run = AsyncMock(return_value=MOCK_AGENT_RESULT)
            resp = await client.post(
                "/api/v1/chat/message",
                json={"session_id": session_id, "query": "What is Q3 revenue?"},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert data["confidence"] > 0

