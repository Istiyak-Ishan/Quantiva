import asyncio
import json
import uuid
from unittest.mock import AsyncMock, patch

from google.genai import types
from sqlalchemy import select

from app.api.v1.chat import stream_message
from app.models.user import User
from app.models.message import Message
from app.schemas.agent import AgentAnswer, AgentResult
from app.services.llm.errors import LLMUnavailableError


async def session(client, headers):
    response = await client.post("/api/v1/chat/sessions", headers=headers, json={"document_ids": []})
    assert response.status_code == 200
    return response.json()["session_id"]


async def test_stream_emits_before_completion_and_saves_once(client, auth_headers, db_session):
    session_id = await session(client, auth_headers)
    user = await db_session.scalar(select(User))
    release = asyncio.Event()
    answer = AgentAnswer(answer="First. Second.", confidence=0, tokens_used=32, input_tokens=20, output_tokens=12)

    async def run(**kwargs):
        await kwargs["on_token"]("First. ")
        await release.wait()
        await kwargs["on_token"]("Second.")
        return AgentResult(answer=answer, iterations=1, tool_calls_made=0)

    with patch("app.api.v1.chat.orchestrator.run", side_effect=run), patch("app.api.v1.chat.session_memory.save_turn", new_callable=AsyncMock) as memory:
        response = await stream_message(uuid.UUID(session_id), "hello", user, db_session)
        iterator = response.body_iterator
        first = await asyncio.wait_for(anext(iterator), timeout=1)
        assert json.loads(first["data"])["token"] == "First. "
        assert not release.is_set()  # The producer has not returned an answer yet.
        release.set()
        rest = [event async for event in iterator]
        final = json.loads(rest[-1]["data"])
        assert final["done"] and final["answer"] == answer.answer
        assert final["confidence"] == 0
        assert final["tokens_used"] == 32
        assert "chart_url" in final and "cost_usd" in final
        memory.assert_awaited_once()
    history = (await client.get(f"/api/v1/chat/sessions/{session_id}/history", headers=auth_headers)).json()
    assert len(history) == 2
    assert history[1]["confidence"] == 0
    assert history[1]["cost_usd"] is None


async def test_provider_failure_is_not_saved_or_reported_as_success(client, auth_headers, db_session):
    session_id = await session(client, auth_headers)
    with patch("app.api.v1.chat.orchestrator.run", new_callable=AsyncMock, side_effect=LLMUnavailableError()):
        response = await client.post("/api/v1/chat/message", headers=auth_headers, json={"session_id": session_id, "query": "hello"})
        assert response.status_code == 503
        response = await client.get("/api/v1/chat/message/stream", headers=auth_headers, params={"session_id": session_id, "query": "hello"})
        assert "event: error" in response.text
        assert "provider_unavailable" in response.text
    assert list((await db_session.scalars(select(Message))).all()) == []


async def test_real_agent_stream_buffers_split_email_before_redaction(client, auth_headers, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "PII_ENGINE", "patterns")
    session_id = await session(client, auth_headers)

    async def stream(contents, config, *, usage):
        for text in ['{"answer":"Contact ali', 'ce@example.com.\\n', 'A second line.","confidence":0.9}']:
            yield types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part.from_text(text=text)]))])
        usage.input_tokens, usage.output_tokens, usage.total_tokens = 15, 20, 35

    with patch("app.services.llm.gemini_client.gemini_client.generate_stream", side_effect=stream), patch("app.api.v1.chat.session_memory.save_turn", new_callable=AsyncMock):
        response = await client.get("/api/v1/chat/message/stream", headers=auth_headers, params={"session_id": session_id, "query": "hello"})
    assert "alice@example.com" not in response.text
    assert "[REDACTED_EMAIL_ADDRESS]" in response.text
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1]["done"]
    assert events[-1]["tokens_used"] == 35
    assert events[-1]["confidence"] == 0  # No sources, not fake model confidence.
