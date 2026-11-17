"""Deterministic tests for the provider-agnostic retrieval pipeline.

No network, provider SDK, vector database, or reranker model is used.  Every
collaborator is a small in-memory fake implementing the public abstract
interfaces.
"""
from __future__ import annotations

import pytest

from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.exceptions import EmbeddingDimensionMismatchError
from app.providers.interfaces.reranker import (
    RerankDocument,
    RerankResult,
    RerankerProvider,
)
from app.providers.interfaces.types import ConnectionTestResult
from app.providers.registry import IndexMetadata
from app.retrieval import (
    RetrievalPipeline,
    SparseRetriever,
    reciprocal_rank_fusion,
)
from app.retrieval.types import RetrievedChunk
from app.storage.interfaces import (
    CollectionInfo,
    SearchHit,
    SparseIndexRepository,
    VectorStoreRepository,
)


# ── Fakes ────────────────────────────────────────────────────────────────────

class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider with controllable output length."""

    def __init__(self, dimensions: int = 8, vector_length: int | None = None) -> None:
        self._dimensions = dimensions
        self._vector_length = dimensions if vector_length is None else vector_length
        self.queries: list[str] = []

    def provider_name(self) -> str:
        return "mock"

    def model_name(self) -> str:
        return "mock-embedding"

    def get_dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts, batch_size=32):
        return [await self.embed_query(text) for text in texts]

    async def embed_query(self, query):
        self.queries.append(query)
        return [float(i + 1) for i in range(self._vector_length)]

    async def test_embedding(self):
        return ConnectionTestResult(
            success=True,
            latency_ms=0.0,
            provider=self.provider_name(),
            model=self.model_name(),
            detected_dimensions=self._vector_length,
        )


class FakeVectorStore(VectorStoreRepository):
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = list(hits or [])
        self.search_calls: list[dict] = []
        self.upsert_calls: list[dict] = []

    async def upsert_chunks(self, collection_name, chunks, embeddings, *, metadata=None):
        self.upsert_calls.append(
            {
                "collection_name": collection_name,
                "chunks": list(chunks),
                "embeddings": [list(vector) for vector in embeddings],
                "metadata": metadata,
            }
        )
        return [f"point-{index}" for index, _ in enumerate(chunks)]

    async def search_dense(self, collection_name, query_vector, top_k, filters=None):
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "query_vector": list(query_vector),
                "top_k": top_k,
                "filters": filters,
            }
        )
        return self.hits[:top_k]

    async def delete_by_document_id(self, collection_name, document_id, *, filters=None):
        return None

    async def delete_collection(self, collection_name):
        return None

    async def get_collection_info(self, collection_name):
        return CollectionInfo(name=collection_name, dimensions=8)


class FakeSparseIndex(SparseIndexRepository):
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = list(hits or [])
        self.search_calls: list[dict] = []

    async def index_chunks(self, knowledge_base_id, chunks, *, metadata=None):
        return None

    async def search_sparse(self, knowledge_base_id, query_text, top_k, filters=None):
        self.search_calls.append(
            {
                "knowledge_base_id": knowledge_base_id,
                "query_text": query_text,
                "top_k": top_k,
                "filters": filters,
            }
        )
        return self.hits[:top_k]

    async def delete_by_document_id(self, knowledge_base_id, document_id, *, filters=None):
        return None


class FakeReranker(RerankerProvider):
    def __init__(
        self,
        order: list[str] | None = None,
        scores: dict[str, float] | None = None,
    ) -> None:
        self.order = order
        self.scores = scores or {}
        self.calls: list[dict] = []

    def provider_name(self) -> str:
        return "fake-reranker"

    def model_name(self) -> str:
        return "fake-cross-encoder"

    async def rerank(self, query, documents, top_k=5):
        self.calls.append(
            {
                "query": query,
                "documents": list(documents),
                "top_k": top_k,
            }
        )
        by_id = {document.id: document for document in documents}
        order = self.order or [document.id for document in documents]
        results: list[RerankResult] = []
        for position, chunk_id in enumerate(order):
            document = by_id.get(chunk_id)
            if document is None:
                continue
            score = self.scores.get(
                chunk_id,
                float(len(order) - position),
            )
            results.append(
                RerankResult(
                    id=chunk_id,
                    content=document.content,
                    metadata=dict(document.metadata),
                    rerank_score=score,
                    initial_score=document.initial_score,
                    rank=position,
                )
            )
        return results[:top_k]

    async def test_reranker(self):
        return ConnectionTestResult(
            success=True,
            latency_ms=0.0,
            provider=self.provider_name(),
            model=self.model_name(),
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_metadata(dimensions: int = 8) -> IndexMetadata:
    return IndexMetadata(
        knowledge_base_id="kb-1",
        embedding_provider="mock",
        embedding_model_name="mock-embedding",
        embedding_dimensions=dimensions,
        collection_name="kb-1-collection",
    )


def make_hit(
    chunk_id: str,
    score: float,
    content: str | None = None,
    *,
    source_file: str = "report.pdf",
    page_number: int = 2,
    chunk_index: int = 0,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=score,
        content=content or f"content for {chunk_id}",
        metadata={
            "source_file": source_file,
            "page_number": page_number,
            "chunk_index": chunk_index,
        },
    )


# ── RRF ──────────────────────────────────────────────────────────────────────

def test_weighted_rrf_formula_and_provenance():
    dense = [
        RetrievedChunk("d1", "dense one", dense_rank=1, dense_score=0.91),
        RetrievedChunk("d2", "dense two", dense_rank=2, dense_score=0.82),
    ]
    sparse = [
        RetrievedChunk("d2", "dense two with a longer sparse preview", sparse_rank=1, sparse_score=12.5),
        RetrievedChunk("d3", "sparse three", sparse_rank=2, sparse_score=8.0),
    ]

    fused = reciprocal_rank_fusion(
        dense,
        sparse,
        k=60,
        dense_weight=2.0,
        sparse_weight=1.0,
    )

    assert [chunk.chunk_id for chunk in fused] == ["d2", "d1", "d3"]
    by_id = {chunk.chunk_id: chunk for chunk in fused}

    assert by_id["d1"].rrf_score == pytest.approx(2.0 / 61.0)
    assert by_id["d2"].rrf_score == pytest.approx(2.0 / 62.0 + 1.0 / 61.0)
    assert by_id["d3"].rrf_score == pytest.approx(1.0 / 62.0)

    assert by_id["d2"].dense_rank == 2
    assert by_id["d2"].sparse_rank == 1
    assert by_id["d2"].dense_score == pytest.approx(0.82)
    assert by_id["d2"].sparse_score == pytest.approx(12.5)
    assert by_id["d2"].content == "dense two with a longer sparse preview"
    assert by_id["d2"].dense_score is not None
    assert by_id["d1"].sparse_rank is None
    assert by_id["d3"].dense_rank is None


def test_default_k_and_deterministic_tie_break():
    fused = reciprocal_rank_fusion(
        [RetrievedChunk("b", "b", dense_rank=1)],
        [RetrievedChunk("a", "a", sparse_rank=1)],
    )

    assert [chunk.chunk_id for chunk in fused] == ["a", "b"]
    assert fused[0].rrf_score == pytest.approx(1.0 / 61.0)
    assert fused[1].rrf_score == pytest.approx(1.0 / 61.0)


def test_fusion_accepts_legacy_mapping_results():
    fused = reciprocal_rank_fusion(
        [{"id": "a", "score": 0.9, "payload": {"text": "alpha"}}],
        [
            {
                "chunk": {
                    "qdrant_id": "a",
                    "text": "alpha with more terms",
                    "payload": {"text": "alpha with more terms"},
                },
                "score": 4.0,
            }
        ],
    )

    assert len(fused) == 1
    assert fused[0].chunk_id == "a"
    assert fused[0].content == "alpha with more terms"
    assert fused[0].dense_score == pytest.approx(0.9)
    assert fused[0].sparse_score == pytest.approx(4.0)


def test_rrf_deduplicates_and_respects_top_k():
    dense = [
        RetrievedChunk("same", "short", dense_rank=1, dense_score=0.9),
        RetrievedChunk("other", "other", dense_rank=2, dense_score=0.8),
    ]
    sparse = [
        RetrievedChunk("same", "a much longer sparse version", sparse_rank=1, sparse_score=4.0),
    ]

    fused = reciprocal_rank_fusion(dense, sparse, top_k=1)

    assert len(fused) == 1
    assert fused[0].chunk_id == "same"
    assert fused[0].content == "a much longer sparse version"
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 1


# ── Pipeline modes ───────────────────────────────────────────────────────────

async def test_hybrid_mode_fuses_and_builds_citations():
    embeddings = MockEmbeddingProvider(dimensions=8)
    vector_store = FakeVectorStore(
        [
            make_hit("d1", 0.91, "dense one", page_number=1, chunk_index=0),
            make_hit("d2", 0.82, "dense two", page_number=2, chunk_index=3),
        ]
    )
    sparse_index = FakeSparseIndex(
        [
            make_hit("d2", 12.0, "dense two", page_number=2, chunk_index=3),
            make_hit("s1", 9.0, "sparse one", page_number=5, chunk_index=7),
        ]
    )
    pipeline = RetrievalPipeline(
        "hybrid",
        embedding_provider=embeddings,
        vector_store=vector_store,
        sparse_index=sparse_index,
        index_metadata=make_metadata(),
        rerank_enabled=False,
        final_top_k=3,
    )

    result = await pipeline.retrieve("  Revenue GROWTH  ", filters={"user_id": "u1"})

    assert result.mode == "hybrid"
    assert result.candidates_considered == 3
    assert [chunk.chunk_id for chunk in result.chunks] == ["d2", "d1", "s1"]
    assert result.reranked is False
    assert result.context_blocks[0].citation == (
        "[Doc: report.pdf, Page: 2, Chunk: 3]"
    )
    assert "dense two" in result.context
    assert result.chunks[0].dense_rank == 2
    assert result.chunks[0].sparse_rank == 1
    assert [chunk.final_rank for chunk in result.chunks] == [1, 2, 3]

    assert embeddings.queries == ["Revenue GROWTH"]
    assert vector_store.search_calls[0]["collection_name"] == "kb-1-collection"
    assert len(vector_store.search_calls[0]["query_vector"]) == 8
    assert vector_store.search_calls[0]["top_k"] == 20
    assert vector_store.search_calls[0]["filters"] == {"user_id": "u1"}
    assert sparse_index.search_calls[0] == {
        "knowledge_base_id": "kb-1",
        "query_text": "revenue growth",
        "top_k": 20,
        "filters": {"user_id": "u1"},
    }


async def test_dense_only_mode_skips_sparse_index():
    vector_store = FakeVectorStore([make_hit("d1", 0.9)])
    sparse_index = FakeSparseIndex([make_hit("s1", 5.0)])
    pipeline = RetrievalPipeline(
        "dense_only",
        embedding_provider=MockEmbeddingProvider(dimensions=8),
        vector_store=vector_store,
        sparse_index=sparse_index,
        index_metadata=make_metadata(),
        rerank_enabled=False,
        final_top_k=5,
    )

    result = await pipeline.retrieve("query", knowledge_base_id="kb-1")

    assert result.mode == "dense_only"
    assert [chunk.chunk_id for chunk in result.chunks] == ["d1"]
    assert vector_store.search_calls
    assert sparse_index.search_calls == []


async def test_sparse_only_mode_skips_embedding_and_vector_store():
    embeddings = MockEmbeddingProvider(dimensions=8)
    vector_store = FakeVectorStore([make_hit("d1", 0.9)])
    sparse_index = FakeSparseIndex(
        [
            make_hit("s1", 5.0, page_number=4, chunk_index=9),
            make_hit("s2", 4.0, page_number=4, chunk_index=10),
        ]
    )
    pipeline = RetrievalPipeline(
        "sparse_only",
        embedding_provider=embeddings,
        vector_store=vector_store,
        sparse_index=sparse_index,
        index_metadata=make_metadata(),
        rerank_enabled=False,
        final_top_k=1,
    )

    result = await pipeline.retrieve("  Revenue, GROWTH!!  ")

    assert result.mode == "sparse_only"
    assert [chunk.chunk_id for chunk in result.chunks] == ["s1"]
    assert result.context_blocks[0].citation == (
        "[Doc: report.pdf, Page: 4, Chunk: 9]"
    )
    assert embeddings.queries == []
    assert vector_store.search_calls == []
    assert sparse_index.search_calls[0]["query_text"] == "revenue growth"
    assert sparse_index.search_calls[0]["knowledge_base_id"] == "kb-1"


# ── Reranker toggling ────────────────────────────────────────────────────────

async def test_reranker_receives_initial_candidates_and_selects_final_top_k():
    vector_store = FakeVectorStore(
        [
            make_hit("d1", 0.91, "dense one"),
            make_hit("d2", 0.82, "dense two"),
        ]
    )
    sparse_index = FakeSparseIndex(
        [
            make_hit("s1", 12.0, "sparse one"),
            make_hit("d2", 9.0, "dense two"),
        ]
    )
    reranker = FakeReranker(
        order=["s1", "d1", "d2"],
        scores={"d1": 0.5, "s1": 0.99, "d2": 0.1},
    )
    pipeline = RetrievalPipeline(
        "hybrid",
        embedding_provider=MockEmbeddingProvider(dimensions=8),
        vector_store=vector_store,
        sparse_index=sparse_index,
        index_metadata=make_metadata(),
        reranker=reranker,
        rerank_enabled=True,
        initial_top_k=3,
        final_top_k=2,
    )

    result = await pipeline.retrieve("query", knowledge_base_id="kb-1")

    assert result.reranked is True
    assert result.candidates_considered == 3
    assert len(reranker.calls) == 1
    assert reranker.calls[0]["top_k"] == 2
    assert len(reranker.calls[0]["documents"]) == 3
    assert all(isinstance(doc, RerankDocument) for doc in reranker.calls[0]["documents"])
    assert [chunk.chunk_id for chunk in result.chunks] == ["s1", "d1"]
    assert result.chunks[0].rerank_score == pytest.approx(0.99)
    assert result.chunks[0].final_rank == 1
    assert result.chunks[1].final_rank == 2


async def test_disabled_reranker_preserves_upstream_order():
    vector_store = FakeVectorStore(
        [
            make_hit("d1", 0.91, "dense one"),
            make_hit("d2", 0.82, "dense two"),
        ]
    )
    sparse_index = FakeSparseIndex([make_hit("s1", 12.0, "sparse one")])
    reranker = FakeReranker(order=["s1", "d1", "d2"])
    pipeline = RetrievalPipeline(
        "hybrid",
        embedding_provider=MockEmbeddingProvider(dimensions=8),
        vector_store=vector_store,
        sparse_index=sparse_index,
        index_metadata=make_metadata(),
        reranker=reranker,
        rerank_enabled=False,
        final_top_k=3,
    )

    result = await pipeline.retrieve("query", knowledge_base_id="kb-1")

    assert result.reranked is False
    assert reranker.calls == []
    assert [chunk.chunk_id for chunk in result.chunks] == ["d1", "s1", "d2"]
    assert all(chunk.rerank_score is None for chunk in result.chunks)
    assert [chunk.final_rank for chunk in result.chunks] == [1, 2, 3]


# ── Dimension verification ───────────────────────────────────────────────────

async def test_declared_embedding_dimension_mismatch_raises_before_search():
    vector_store = FakeVectorStore([make_hit("d1", 0.9)])
    embeddings = MockEmbeddingProvider(dimensions=4)
    pipeline = RetrievalPipeline(
        "dense_only",
        embedding_provider=embeddings,
        vector_store=vector_store,
        index_metadata=make_metadata(dimensions=8),
        rerank_enabled=False,
    )

    with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
        await pipeline.retrieve("query", knowledge_base_id="kb-1")

    assert excinfo.value.expected == 8
    assert excinfo.value.actual == 4
    assert excinfo.value.provider == "mock"
    assert excinfo.value.model == "mock-embedding"
    assert embeddings.queries == []
    assert vector_store.search_calls == []


async def test_query_vector_length_mismatch_raises_before_search():
    vector_store = FakeVectorStore([make_hit("d1", 0.9)])
    embeddings = MockEmbeddingProvider(dimensions=8, vector_length=5)
    pipeline = RetrievalPipeline(
        "dense_only",
        embedding_provider=embeddings,
        vector_store=vector_store,
        index_metadata=make_metadata(dimensions=8),
        rerank_enabled=False,
    )

    with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
        await pipeline.retrieve("query", knowledge_base_id="kb-1")

    assert excinfo.value.expected == 8
    assert excinfo.value.actual == 5
    assert embeddings.queries == ["query"]
    assert vector_store.search_calls == []


async def test_sparse_retriever_short_circuits_empty_term_query():
    sparse_index = FakeSparseIndex([make_hit("s1", 1.0)])

    result = await SparseRetriever(sparse_index).retrieve(
        "!!! ???",
        knowledge_base_id="kb-1",
    )

    assert result == []
    assert sparse_index.search_calls == []


# ── Validation ───────────────────────────────────────────────────────────────

def test_pipeline_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported retrieval mode"):
        RetrievalPipeline("semantic_magic")  # type: ignore[arg-type]


def test_pipeline_rejects_zero_initial_top_k():
    with pytest.raises(ValueError, match="initial_top_k"):
        RetrievalPipeline("sparse_only", sparse_index=FakeSparseIndex(), initial_top_k=0)
