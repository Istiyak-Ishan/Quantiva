import json
from unittest.mock import AsyncMock, patch

import pytest

from mcp_server import server


def test_mcp_never_supplies_mock_credentials(monkeypatch):
    for token in ("", "mock-token"):
        monkeypatch.setattr(server, "API_TOKEN", token)
        with pytest.raises(ValueError, match="real access token"):
            server._headers()


async def test_mcp_search_payload_matches_real_route():
    with patch.object(server, "_post", new_callable=AsyncMock, return_value={"chunks": []}) as post:
        result = json.loads(await server.search_documents("question", ["doc"]))
    assert result["status"] == "no_results"
    post.assert_awaited_once_with("/retrieval/search", {"query": "question", "doc_ids": ["doc"], "top_k": 5})


async def test_mcp_summary_uses_session_document_scope_not_ignored_fields():
    with patch.object(server, "_post", new_callable=AsyncMock, side_effect=[{"session_id": "session"}, {"answer": "Summary"}]) as post:
        result = json.loads(await server.summarize_document("doc"))
    assert result["status"] == "ok"
    assert post.call_args_list[0].args == ("/chat/sessions", {"document_ids": ["doc"]})
    assert set(post.call_args_list[1].args[1]) == {"session_id", "query"}


async def test_mcp_rejects_overriding_existing_session_scope():
    with patch.object(server, "_get", new_callable=AsyncMock, return_value={"document_ids": ["other-doc"]}), patch.object(server, "_post", new_callable=AsyncMock) as post:
        result = json.loads(await server.ask_question("question", ["doc"], "existing-session"))
    assert result["status"] == "error"
    post.assert_not_awaited()
