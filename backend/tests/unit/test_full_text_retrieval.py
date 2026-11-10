from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.retrieval.hybrid_retriever import HybridRetriever


async def test_keyword_beyond_preview_is_searchable_by_bm25():
    store = SimpleNamespace(search=AsyncMock(return_value=[]))
    retriever = HybridRetriever(store)
    text = "introduction " * 30 + " rareallowancekeyword is 4800 USD"
    chunks = [SimpleNamespace(doc_id="doc", text=text, text_preview=text[:200], qdrant_id="match", page_number=1), SimpleNamespace(doc_id="doc", text="unrelated policies", text_preview="unrelated policies", qdrant_id="other", page_number=2), SimpleNamespace(doc_id="doc", text="another unrelated section", text_preview="another unrelated section", qdrant_id="third", page_number=3)]
    with patch("app.services.retrieval.hybrid_retriever.embedding_service.embed_text", return_value=[1., 0., 0.]), patch("app.services.retrieval.hybrid_retriever.reranker.rerank", side_effect=lambda query, candidates, top_k: candidates[:top_k]):
        results = await retriever.search("rareallowancekeyword", ["doc"], chunks, user_id="owner")
    assert results and results[0]["id"] == "match"
    assert results[0]["payload"]["text"] == text
