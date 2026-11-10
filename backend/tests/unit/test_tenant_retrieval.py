import json
import uuid
from unittest.mock import AsyncMock, patch

from qdrant_client import AsyncQdrantClient

from app.schemas.agent import AgentAnswer
from app.services.memory.semantic_cache import SemanticCache
from app.services.retrieval.vector_store import VectorStore


async def test_qdrant_filters_both_tenant_and_document_with_real_local_engine():
    store = VectorStore.__new__(VectorStore)
    store.client = AsyncQdrantClient(location=":memory:")
    store.collection_name = "tenant_test"
    store.vector_size = 3
    try:
        await store.create_collection()
        doc_id = str(uuid.uuid4())
        # Same document ID deliberately inserted for two owners proves tenant
        # filtering independently of the SQL ownership boundary.
        for user in ("alice", "bob"):
            await store.upsert_chunks([{"text": f"{user} private", "metadata": {"page": 1}}], [[1., 0., 0.]], doc_id, user)
        results = await store.search([1., 0., 0.], [doc_id], user_id="bob")
        assert len(results) == 1
        assert results[0]["payload"]["text"] == "bob private"
        assert await store.search([1., 0., 0.], [], user_id="bob") == []
        await store.delete_document(doc_id, user_id="bob")
        assert await store.search([1., 0., 0.], [doc_id], user_id="bob") == []
        assert len(await store.search([1., 0., 0.], [doc_id], user_id="alice")) == 1
    finally:
        await store.client.close()


async def test_semantic_cache_keys_cannot_cross_users():
    cache = SemanticCache()
    cache.client = AsyncMock()
    answer = AgentAnswer(answer="Alice private answer", confidence=0.8)
    entries = {cache._generate_key(["doc"], user_id="alice"): {"question": json.dumps({"embedding": [1., 0.], "answer": answer.model_dump()})}}
    cache.client.hgetall.side_effect = lambda key: entries.get(key, {})
    with patch("app.services.memory.semantic_cache.embedding_service.embed_text", return_value=[1., 0.]):
        assert (await cache.get("question", ["doc"], user_id="alice")).answer == "Alice private answer"
        assert await cache.get("question", ["doc"], user_id="bob") is None
        assert await cache.get("question", [], user_id="alice") is None


async def test_deleting_a_failed_first_upload_does_not_require_an_index_to_exist():
    store = VectorStore.__new__(VectorStore)
    store.client = AsyncQdrantClient(location=":memory:")
    store.collection_name = "not_created"
    try:
        await store.delete_document(str(uuid.uuid4()), user_id="owner")
        assert not await store.client.collection_exists(store.collection_name)
    finally:
        await store.client.close()


async def test_cache_invalidates_only_affected_tenant_document():
    cache = SemanticCache()
    cache.client = AsyncMock()
    cache.client.smembers.return_value = {"alice-corpus-1", "alice-corpus-2"}
    await cache.invalidate("doc", user_id="alice")
    cache.client.smembers.assert_awaited_once_with(cache._doc_index("doc", "alice"))
    deleted = cache.client.delete.call_args_list
    assert set(deleted[0].args) == {"alice-corpus-1", "alice-corpus-2"}
    assert deleted[1].args == (cache._doc_index("doc", "alice"),)


def test_cache_corpus_and_history_changes_cannot_reuse_answers():
    cache = SemanticCache()
    first = cache._generate_key(["doc"], user_id="alice", corpus_version="v1", context_fingerprint="history1")
    assert first != cache._generate_key(["doc"], user_id="alice", corpus_version="v2", context_fingerprint="history1")
    assert first != cache._generate_key(["doc"], user_id="alice", corpus_version="v1", context_fingerprint="history2")
