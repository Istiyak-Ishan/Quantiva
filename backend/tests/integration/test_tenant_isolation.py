import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from app.models.document import Document
from app.models.eval_result import EvalRun
from app.models.message import Message
from app.models.user import User


async def test_cannot_attach_another_tenants_document(client, tenants):
    _, bob, doc, _, _ = tenants
    response = await client.post("/api/v1/chat/sessions", headers=bob, json={"document_ids": [str(doc.id)]})
    assert response.status_code == 404


async def test_history_authorizes_before_memory_or_messages(client, tenants):
    alice, bob, _, _, session_id = tenants
    with patch("app.services.memory.session_memory.session_memory.load_history", new_callable=AsyncMock, return_value=[{"content": "PRIVATE REDIS"}]) as memory:
        response = await client.get(f"/api/v1/chat/sessions/{session_id}/history", headers=bob)
        assert response.status_code == 404
        memory.assert_not_awaited()
    response = await client.get(f"/api/v1/chat/sessions/{session_id}/history", headers=alice)
    assert response.status_code == 200
    assert response.json()[0]["content"] == "ALICE PRIVATE HISTORY"


async def test_foreign_session_cannot_be_queried_streamed_or_deleted(client, tenants):
    _, bob, _, _, session_id = tenants
    with patch("app.api.v1.chat.orchestrator.run", new_callable=AsyncMock) as agent:
        response = await client.post("/api/v1/chat/message", headers=bob, json={"session_id": session_id, "query": "secret"})
        assert response.status_code == 404
        response = await client.get("/api/v1/chat/message/stream", headers=bob, params={"session_id": session_id, "query": "secret"})
        assert response.status_code == 404
        response = await client.delete(f"/api/v1/chat/sessions/{session_id}", headers=bob)
        assert response.status_code == 404
        agent.assert_not_awaited()


async def test_documents_and_chunks_are_scoped(client, tenants):
    _, bob, doc, _, _ = tenants
    for suffix in ("", "/chunks"):
        response = await client.get(f"/api/v1/documents/{doc.id}{suffix}", headers=bob)
        assert response.status_code == 404
    response = await client.delete(f"/api/v1/documents/{doc.id}", headers=bob)
    assert response.status_code == 404


async def test_evaluation_reads_lists_and_trigger_are_scoped(client, tenants):
    alice, bob, doc, run, _ = tenants
    response = await client.get(f"/api/v1/eval/{run.id}", headers=bob)
    assert response.status_code == 404
    response = await client.get("/api/v1/eval/history/list", headers=bob)
    assert response.json() == []
    response = await client.get(f"/api/v1/eval/{run.id}", headers=alice)
    assert response.status_code == 200
    response = await client.post("/api/v1/eval/run", params={"doc_id": str(doc.id)}, headers=bob)
    assert response.status_code == 404


@pytest.mark.parametrize("bad_id", ["fallback-session-123", str(uuid.uuid4())])
async def test_invalid_or_unknown_session_never_substitutes_latest(client, tenants, bad_id):
    alice, _, _, _, _ = tenants
    expected = 422 if bad_id.startswith("fallback") else 404
    response = await client.get(f"/api/v1/chat/sessions/{bad_id}/history", headers=alice)
    assert response.status_code == expected
    response = await client.post("/api/v1/chat/message", headers=alice, json={"session_id": bad_id, "query": "hello"})
    assert response.status_code == expected


async def test_invalid_document_uuid_is_validation_error(client, auth_headers):
    response = await client.post("/api/v1/chat/sessions", headers=auth_headers, json={"document_ids": ["invalid"]})
    assert response.status_code == 422
