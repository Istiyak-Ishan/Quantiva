from unittest.mock import AsyncMock, patch
from app.models.chunk import Chunk


async def test_search_rejects_foreign_document_before_retrieval(client, tenants):
    _, bob, doc, _, _ = tenants
    with patch("app.api.v1.retrieval.hybrid_retriever.search", new_callable=AsyncMock) as search:
        response = await client.post("/api/v1/retrieval/search", headers=bob, json={"query": "private", "doc_ids": [str(doc.id)]})
    assert response.status_code == 404
    search.assert_not_awaited()


async def test_empty_library_never_searches_global_vectors(client, auth_headers):
    response = await client.post("/api/v1/retrieval/search", headers=auth_headers, json={"query": "private"})
    assert response.status_code == 200
    assert response.json() == {"chunks": []}


async def test_caller_cannot_supply_another_user_identity(client, auth_headers):
    response = await client.post("/api/v1/retrieval/search", headers=auth_headers, json={"query": "private", "user_id": "another-user"})
    assert response.status_code == 422


async def test_changed_embedding_configuration_requires_reindex_before_any_provider_call(client, tenants, db_session):
    alice, _, doc, _, session_id = tenants
    doc.metadata_json = {"embedding_collection": "old-incompatible-index"}
    db_session.add(Chunk(doc_id=doc.id, chunk_index=0, text_preview="source", full_text="complete source", page_number=1))
    await db_session.commit()
    with patch("app.api.v1.retrieval.hybrid_retriever.search", new_callable=AsyncMock) as search:
        response = await client.post("/api/v1/retrieval/search", headers=alice, json={"query": "question", "doc_ids": [str(doc.id)]})
        assert response.status_code == 409
        search.assert_not_awaited()
    with patch("app.api.v1.chat.orchestrator.run", new_callable=AsyncMock) as agent:
        response = await client.post("/api/v1/chat/message", headers=alice, json={"session_id": session_id, "query": "question"})
        assert response.status_code == 409
        agent.assert_not_awaited()
