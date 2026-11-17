"""
tests/core/test_rag_orchestrator.py
──────────────────────────────────────────────────────────────────────────────
End-to-end integration and unit tests for the provider-independent RAGOrchestrator.

Verifies:
1. Provider Swapping: Runs the exact same document ingestion, hybrid retrieval,
   reranking, and question-answering flow first with a mocked Gemini provider and
   then with a mocked Ollama provider without changing a single line of orchestrator
   or retrieval code.
2. Dual RAG Modes:
   - Direct RAG mode (Retrieve -> Rerank -> Context Build -> Stream/Generate -> Validate Citations).
   - Agentic RAG mode (ReAct reasoning + Data Analyst tools -> Final Answer -> Validate Citations).
3. Normalized SSE Streaming Events:
   - status (retrieving, reranking, thinking, executing_tool, generating)
   - sources (retrieved chunks with metadata)
   - reasoning_delta (think tag reasoning or ReAct thought)
   - token_delta (generation tokens)
   - citations (validated citations with marker, quote, overlap, validity)
   - done (usage and latency_ms)
   - error (code and message)
4. Non-streaming query convenience returning RAGResponse.
5. Invariant 3: Embedding dimension mismatch raises before hitting vector store.
"""
from __future__ import annotations

import math
from collections.abc import AsyncIterator, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.application_context import get_provider_registry, reset_application_context
from app.core.orchestrator import (
    RAGOrchestrator,
    RAGResponse,
    StreamingThinkParser,
    format_sse,
)
from app.ingestion.types import DocumentChunkPayload
from app.providers.embeddings.gemini_embedding import GeminiEmbeddingProvider
from app.providers.embeddings.ollama import OllamaEmbeddingProvider
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.exceptions import EmbeddingDimensionMismatchError
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.reranker import (
    RerankDocument,
    RerankResult,
    RerankerProvider,
)
from app.providers.interfaces.types import (
    ChatMessage,
    ChatRole,
    ConnectionTestResult,
    GenerationConfig,
    LLMResponse,
    LLMStreamChunk,
    ProviderCapabilities,
    ProviderType,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.llm.ollama import OllamaLLMProvider
from app.providers.registry import (
    EmbeddingConfig,
    IndexMetadata,
    LLMConfig,
    ProviderRegistry,
    RerankerConfig,
)
from app.retrieval.types import RetrievedChunk
from app.storage.interfaces import (
    CollectionInfo,
    SearchHit,
    SparseIndexRepository,
    VectorStoreRepository,
)


# ── In-Memory Repositories ────────────────────────────────────────────────────

def _cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class InMemoryVectorStore(VectorStoreRepository):
    """In-memory vector store repository for deterministic tests."""

    def __init__(self) -> None:
        self.collections: dict[str, list[tuple[dict[str, Any], list[float]]]] = {}

    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[DocumentChunkPayload],
        embeddings: Sequence[Sequence[float]],
        *,
        metadata: Any = None,
    ) -> list[str]:
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        for chunk, emb in zip(chunks, embeddings):
            self.collections[collection_name].append((dict(chunk), [float(x) for x in emb]))
        return [str(c["chunk_id"]) for c in chunks]

    async def search_dense(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        top_k: int,
        filters: Any = None,
    ) -> list[SearchHit]:
        items = self.collections.get(collection_name, [])
        scored: list[SearchHit] = []
        for chunk, emb in items:
            score = _cosine_similarity(query_vector, emb)
            scored.append(
                SearchHit(
                    chunk_id=str(chunk["chunk_id"]),
                    score=score,
                    content=str(chunk.get("content", "")),
                    metadata=dict(chunk.get("metadata", {})),
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    async def delete_by_document_id(
        self,
        collection_name: str,
        document_id: str,
        *,
        filters: Any = None,
    ) -> None:
        if collection_name in self.collections:
            self.collections[collection_name] = [
                (c, e)
                for c, e in self.collections[collection_name]
                if c.get("document_id") != document_id
            ]

    async def delete_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)

    async def get_collection_info(self, collection_name: str) -> CollectionInfo | None:
        if collection_name not in self.collections:
            return None
        items = self.collections[collection_name]
        dim = len(items[0][1]) if items else 8
        return CollectionInfo(name=collection_name, dimensions=dim, point_count=len(items))


class InMemorySparseIndex(SparseIndexRepository):
    """In-memory BM25-like sparse index repository for deterministic tests."""

    def __init__(self) -> None:
        self.indices: dict[str, list[dict[str, Any]]] = {}

    async def index_chunks(
        self,
        knowledge_base_id: str,
        chunks: Sequence[DocumentChunkPayload],
        *,
        metadata: Any = None,
    ) -> None:
        if knowledge_base_id not in self.indices:
            self.indices[knowledge_base_id] = []
        for chunk in chunks:
            self.indices[knowledge_base_id].append(dict(chunk))

    async def search_sparse(
        self,
        knowledge_base_id: str,
        query_text: str,
        top_k: int,
        filters: Any = None,
    ) -> list[SearchHit]:
        chunks = self.indices.get(knowledge_base_id, [])
        query_words = set(query_text.lower().split())
        scored: list[SearchHit] = []
        for chunk in chunks:
            content_words = set(chunk.get("content", "").lower().split())
            matches = len(query_words & content_words)
            score = matches / (len(query_words) + 1.0)
            scored.append(
                SearchHit(
                    chunk_id=str(chunk["chunk_id"]),
                    score=score,
                    content=str(chunk.get("content", "")),
                    metadata=dict(chunk.get("metadata", {})),
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    async def delete_by_document_id(
        self,
        knowledge_base_id: str,
        document_id: str,
        *,
        filters: Any = None,
    ) -> None:
        if knowledge_base_id in self.indices:
            self.indices[knowledge_base_id] = [
                c
                for c in self.indices[knowledge_base_id]
                if c.get("document_id") != document_id
            ]


class FakeReranker(RerankerProvider):
    """Predictable reranker that orders chunks and computes normalized scores."""

    def provider_name(self) -> str:
        return "fake-cross-encoder"

    def model_name(self) -> str:
        return "fake-reranker-v1"

    async def rerank(
        self,
        query: str,
        documents: Sequence[RerankDocument],
        top_k: int = 5,
    ) -> list[RerankResult]:
        results: list[RerankResult] = []
        for idx, doc in enumerate(documents):
            score = 0.95 - (idx * 0.05)
            results.append(
                RerankResult(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    rerank_score=score,
                    initial_score=doc.initial_score,
                    rank=idx,
                )
            )
        return results[:top_k]

    async def test_reranker(self) -> ConnectionTestResult:
        return ConnectionTestResult(
            success=True,
            latency_ms=0.0,
            provider=self.provider_name(),
            model=self.model_name(),
        )


@pytest.fixture(autouse=True)
def clean_context():
    """Reset ApplicationContext before and after each test and register FakeReranker."""
    reset_application_context()
    registry = get_provider_registry()
    registry.register_custom_reranker_factory("cross_encoder", lambda cfg: FakeReranker())
    yield
    reset_application_context()


# ── Embedding Helper ──────────────────────────────────────────────────────────

def _make_deterministic_embedding(text: str, dimensions: int = 8) -> list[float]:
    vec = [0.0] * dimensions
    for i, char in enumerate(text.lower()):
        vec[i % dimensions] += float(ord(char))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestProviderSwappingEndToEnd:
    """Verifies that RAGOrchestrator runs the exact same flow across swapped providers."""

    @pytest.mark.asyncio
    async def test_end_to_end_provider_swapping_direct_rag(self):
        """Proof of provider independence: runs identical direct RAG with Gemini, then Ollama."""
        registry = ProviderRegistry()
        registry.register_custom_reranker_factory("cross_encoder", lambda cfg: FakeReranker())
        vector_store = InMemoryVectorStore()
        sparse_index = InMemorySparseIndex()

        orchestrator = RAGOrchestrator(
            provider_registry=registry,
            vector_store=vector_store,
            sparse_index=sparse_index,
        )

        document_bytes = (
            b"Quantiva Enterprise Overview\n\n"
            b"Quantiva Q4 revenue reached $12.5M, representing 45% year-over-year growth.\n"
            b"The operating profit margin increased to 28% for the fiscal quarter."
        )

        # ──────────────────────────────────────────────────────────────────────
        # STEP 1: Run with Mocked GEMINI Provider
        # ──────────────────────────────────────────────────────────────────────
        gemini_llm_cfg = LLMConfig(
            provider=ProviderType.GEMINI,
            model="gemini-3.5-flash-lite",
            api_key="mock-gemini-key",
        )
        gemini_embed_cfg = EmbeddingConfig(
            provider=ProviderType.GEMINI,
            model="gemini-embedding-001",
            dimensions=8,
            api_key="mock-gemini-key",
        )
        gemini_rerank_cfg = RerankerConfig(
            provider=ProviderType.CROSS_ENCODER,
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            enabled=True,
        )

        registry.set_llm_config(gemini_llm_cfg)
        registry.set_embedding_config(gemini_embed_cfg)
        registry.set_reranker_config(gemini_rerank_cfg)

        # Confirm registry instantiated Gemini
        active_llm_gemini = registry.get_llm_provider()
        active_emb_gemini = registry.get_embedding_provider()
        assert isinstance(active_llm_gemini, GeminiLLMProvider)
        assert isinstance(active_emb_gemini, GeminiEmbeddingProvider)
        assert active_llm_gemini.provider_name() == "gemini"
        assert active_emb_gemini.provider_name() == "gemini"

        # Mock network methods of the active Gemini instances
        async def mock_gemini_embed_docs(texts, batch_size=32):
            return [_make_deterministic_embedding(t, 8) for t in texts]

        async def mock_gemini_embed_query(text):
            return _make_deterministic_embedding(text, 8)

        async def mock_gemini_stream(messages, config=None, tools=None):
            tokens = [
                "Quantiva reported Q4 revenue of $12.5M ",
                "with an operating profit margin of 28% [1].",
            ]
            for t in tokens:
                yield LLMStreamChunk(delta_text=t)
            yield LLMStreamChunk(
                delta_text="",
                is_final=True,
                usage=TokenUsage(prompt_tokens=40, completion_tokens=22, total_tokens=62),
            )

        with patch.object(active_emb_gemini, "embed_documents", side_effect=mock_gemini_embed_docs), \
             patch.object(active_emb_gemini, "embed_query", side_effect=mock_gemini_embed_query), \
             patch.object(active_llm_gemini, "stream", side_effect=mock_gemini_stream):

            # 1. Ingest Document
            ingest_result = await orchestrator.ingest_document(
                document_bytes,
                document_id="doc-gemini-1",
                knowledge_base_id="kb-finance",
                file_name="quantiva_overview.txt",
                chunk_size=128,
                chunk_overlap=20,
            )
            assert ingest_result.document_id == "doc-gemini-1"
            assert len(ingest_result.chunks) > 0

            # 2. Run Direct RAG Stream
            events_gemini = []
            async for event in orchestrator.stream_query(
                "What was the Q4 revenue and operating margin?",
                knowledge_base_id="kb-finance",
                mode="direct",
            ):
                events_gemini.append(event)

            # 3. Assert All Normalized SSE Events
            event_types = [e["event"] for e in events_gemini]
            assert "status" in event_types
            assert "sources" in event_types
            assert "token_delta" in event_types
            assert "citations" in event_types
            assert "done" in event_types

            # Verify Stages
            stages = [e["data"]["stage"] for e in events_gemini if e["event"] == "status"]
            assert "retrieving" in stages
            assert "reranking" in stages
            assert "generating" in stages

            # Verify Sources
            sources_event = next(e for e in events_gemini if e["event"] == "sources")
            assert len(sources_event["data"]) > 0
            first_source = sources_event["data"][0]
            assert "chunk_id" in first_source
            assert "score" in first_source
            assert "metadata" in first_source

            # Verify Answer
            full_answer = "".join(
                e["data"]["text"] for e in events_gemini if e["event"] == "token_delta"
            )
            assert "$12.5M" in full_answer
            assert "[1]" in full_answer

            # Verify Citations
            citations_event = next(e for e in events_gemini if e["event"] == "citations")
            assert len(citations_event["data"]) > 0
            citation = citations_event["data"][0]
            assert citation["marker"] == "[1]"
            assert citation["index"] == 1
            assert citation["valid"] is True
            assert citation["lexical_overlap"] > 0.1

            # Verify Done event
            done_event = next(e for e in events_gemini if e["event"] == "done")
            assert done_event["data"]["latency_ms"] >= 0.0
            assert done_event["data"]["usage"]["total_tokens"] == 62

        # ──────────────────────────────────────────────────────────────────────
        # STEP 2: Swap to Mocked OLLAMA Provider (Without changing orchestrator/retrieval code)
        # ──────────────────────────────────────────────────────────────────────
        ollama_llm_cfg = LLMConfig(
            provider=ProviderType.OLLAMA,
            model="llama3.2",
            base_url="http://127.0.0.1:11434",
        )
        ollama_embed_cfg = EmbeddingConfig(
            provider=ProviderType.OLLAMA,
            model="nomic-embed-text",
            dimensions=8,
            base_url="http://127.0.0.1:11434",
        )

        registry.set_llm_config(ollama_llm_cfg)
        registry.set_embedding_config(ollama_embed_cfg)

        # Confirm registry instantiated Ollama
        active_llm_ollama = registry.get_llm_provider()
        active_emb_ollama = registry.get_embedding_provider()
        assert isinstance(active_llm_ollama, OllamaLLMProvider)
        assert isinstance(active_emb_ollama, OllamaEmbeddingProvider)
        assert active_llm_ollama.provider_name() == "ollama"
        assert active_emb_ollama.provider_name() == "ollama"
        assert active_llm_ollama is not active_llm_gemini
        assert active_emb_ollama is not active_emb_gemini

        # Mock network methods of the active Ollama instances
        async def mock_ollama_embed_docs(texts, batch_size=32):
            return [_make_deterministic_embedding(t, 8) for t in texts]

        async def mock_ollama_embed_query(text):
            return _make_deterministic_embedding(text, 8)

        async def mock_ollama_stream(messages, config=None, tools=None):
            tokens = [
                "Based on the documents, Quantiva revenue for Q4 was $12.5M ",
                "with 28% margin [1].",
            ]
            for t in tokens:
                yield LLMStreamChunk(delta_text=t)
            yield LLMStreamChunk(
                delta_text="",
                is_final=True,
                usage=TokenUsage(prompt_tokens=42, completion_tokens=20, total_tokens=62),
            )

        with patch.object(active_emb_ollama, "embed_documents", side_effect=mock_ollama_embed_docs), \
             patch.object(active_emb_ollama, "embed_query", side_effect=mock_ollama_embed_query), \
             patch.object(active_llm_ollama, "stream", side_effect=mock_ollama_stream):

            # 1. Ingest Document into a new KB with Ollama embeddings
            ingest_result_ollama = await orchestrator.ingest_document(
                document_bytes,
                document_id="doc-ollama-1",
                knowledge_base_id="kb-ollama-finance",
                file_name="quantiva_overview.txt",
                chunk_size=128,
                chunk_overlap=20,
            )
            assert ingest_result_ollama.document_id == "doc-ollama-1"
            assert len(ingest_result_ollama.chunks) > 0

            # 2. Run Direct RAG Stream with Ollama
            events_ollama = []
            async for event in orchestrator.stream_query(
                "What was the Q4 revenue and operating margin?",
                knowledge_base_id="kb-ollama-finance",
                mode="direct",
            ):
                events_ollama.append(event)

            # 3. Assert All Normalized SSE Events for Ollama
            event_types_ollama = [e["event"] for e in events_ollama]
            assert "status" in event_types_ollama
            assert "sources" in event_types_ollama
            assert "token_delta" in event_types_ollama
            assert "citations" in event_types_ollama
            assert "done" in event_types_ollama

            # Verify Stages
            stages_ollama = [e["data"]["stage"] for e in events_ollama if e["event"] == "status"]
            assert "retrieving" in stages_ollama
            assert "reranking" in stages_ollama
            assert "generating" in stages_ollama

            # Verify Answer
            full_answer_ollama = "".join(
                e["data"]["text"] for e in events_ollama if e["event"] == "token_delta"
            )
            assert "$12.5M" in full_answer_ollama
            assert "[1]" in full_answer_ollama

            # Verify Citations
            citations_ollama = next(e for e in events_ollama if e["event"] == "citations")
            assert len(citations_ollama["data"]) > 0
            assert citations_ollama["data"][0]["valid"] is True

    @pytest.mark.asyncio
    async def test_end_to_end_provider_swapping_agentic_rag(self):
        """Verifies Agentic RAG mode across both native tool calling (Gemini) and text ReAct (Ollama)."""
        registry = ProviderRegistry()
        registry.register_custom_reranker_factory("cross_encoder", lambda cfg: FakeReranker())
        vector_store = InMemoryVectorStore()
        sparse_index = InMemorySparseIndex()

        orchestrator = RAGOrchestrator(
            provider_registry=registry,
            vector_store=vector_store,
            sparse_index=sparse_index,
        )

        document_bytes = (
            b"Product Launch Analysis\n\n"
            b"Model A achieved 94.2% accuracy on benchmark test.\n"
            b"Model B achieved 89.1% accuracy on benchmark test."
        )

        # ── 1. Agentic RAG with Gemini (Native Tool Calling) ──────────────────
        gemini_llm_cfg = LLMConfig(
            provider=ProviderType.GEMINI,
            model="gemini-3.5-flash-lite",
            api_key="mock-gemini-key",
        )
        gemini_embed_cfg = EmbeddingConfig(
            provider=ProviderType.GEMINI,
            model="gemini-embedding-001",
            dimensions=8,
            api_key="mock-gemini-key",
        )
        registry.set_llm_config(gemini_llm_cfg)
        registry.set_embedding_config(gemini_embed_cfg)

        active_llm_gemini = registry.get_llm_provider()
        active_emb_gemini = registry.get_embedding_provider()

        async def mock_gemini_embed_docs(texts, batch_size=32):
            return [_make_deterministic_embedding(t, 8) for t in texts]

        async def mock_gemini_embed_query(text):
            return _make_deterministic_embedding(text, 8)

        # Native tool calling sequence:
        # Turn 1: tool call chunk for search_knowledge_base
        # Turn 2: final answer tokens
        gemini_agent_turns = [
            # Turn 1: stream yielding tool call
            [
                LLMStreamChunk(
                    tool_call_delta=ToolCallRequest(
                        id="call_001",
                        name="search_knowledge_base",
                        arguments={"query": "Model A benchmark accuracy", "top_k": 2},
                    )
                ),
                LLMStreamChunk(is_final=True, finish_reason="tool_calls"),
            ],
            # Turn 2: stream yielding answer
            [
                LLMStreamChunk(delta_text="Model A achieved 94.2% accuracy on the benchmark test [1]."),
                LLMStreamChunk(is_final=True, finish_reason="stop"),
            ],
        ]

        async def mock_gemini_agent_stream(messages, config=None, tools=None):
            turn_chunks = gemini_agent_turns.pop(0)
            for c in turn_chunks:
                yield c

        with patch.object(active_emb_gemini, "embed_documents", side_effect=mock_gemini_embed_docs), \
             patch.object(active_emb_gemini, "embed_query", side_effect=mock_gemini_embed_query), \
             patch.object(active_llm_gemini, "stream", side_effect=mock_gemini_agent_stream):

            await orchestrator.ingest_document(
                document_bytes,
                document_id="doc-agent-gemini",
                knowledge_base_id="kb-benchmarks",
                file_name="benchmark_report.txt",
            )

            events_agentic_gemini = []
            async for event in orchestrator.stream_query(
                "Compare Model A accuracy against the benchmark",
                knowledge_base_id="kb-benchmarks",
                mode="agentic",
            ):
                events_agentic_gemini.append(event)

            stages_gemini = [
                e["data"]["stage"] for e in events_agentic_gemini if e["event"] == "status"
            ]
            assert "thinking" in stages_gemini
            assert "retrieving" in stages_gemini
            assert "generating" in stages_gemini

            sources_gemini = next(e for e in events_agentic_gemini if e["event"] == "sources")
            assert len(sources_gemini["data"]) > 0

            answer_gemini = "".join(
                e["data"]["text"] for e in events_agentic_gemini if e["event"] == "token_delta"
            )
            assert "94.2%" in answer_gemini
            assert "[1]" in answer_gemini

            citations_gemini = next(e for e in events_agentic_gemini if e["event"] == "citations")
            assert len(citations_gemini["data"]) > 0
            assert citations_gemini["data"][0]["valid"] is True

        # ── 2. Swap to Ollama (Text ReAct Mode) ───────────────────────────────
        ollama_llm_cfg = LLMConfig(
            provider=ProviderType.OLLAMA,
            model="llama3.2",
            base_url="http://127.0.0.1:11434",
        )
        ollama_embed_cfg = EmbeddingConfig(
            provider=ProviderType.OLLAMA,
            model="nomic-embed-text",
            dimensions=8,
            base_url="http://127.0.0.1:11434",
        )
        registry.set_llm_config(ollama_llm_cfg)
        registry.set_embedding_config(ollama_embed_cfg)

        active_llm_ollama = registry.get_llm_provider()
        active_emb_ollama = registry.get_embedding_provider()

        # Text ReAct responses:
        # Turn 1: Thought + Action JSON
        # Turn 2: Final Answer
        responses_ollama = [
            LLMResponse(
                content=(
                    "Thought: I should search for the Model A benchmark numbers.\n"
                    'Action: {"tool": "search_knowledge_base", "arguments": {"query": "Model A benchmark"}}\n'
                ),
                tool_calls=[],
                model="llama3.2",
                provider="ollama",
            ),
            LLMResponse(
                content="Final Answer: Model A achieved 94.2% accuracy on the benchmark test [1].",
                tool_calls=[],
                model="llama3.2",
                provider="ollama",
            ),
        ]

        async def mock_ollama_generate(messages, config=None, tools=None):
            return responses_ollama.pop(0)

        ollama_caps = ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=False,
            supports_structured_output=False,
            supports_vision=False,
            supports_system_prompt=True,
            max_context_window=4096,
            is_local=True,
        )

        with patch.object(active_emb_ollama, "embed_documents", side_effect=mock_gemini_embed_docs), \
             patch.object(active_emb_ollama, "embed_query", side_effect=mock_gemini_embed_query), \
             patch.object(active_llm_ollama, "get_capabilities", return_value=ollama_caps), \
             patch.object(active_llm_ollama, "generate", side_effect=mock_ollama_generate):

            await orchestrator.ingest_document(
                document_bytes,
                document_id="doc-agent-ollama",
                knowledge_base_id="kb-benchmarks-ollama",
                file_name="benchmark_report.txt",
            )

            events_agentic_ollama = []
            async for event in orchestrator.stream_query(
                "What was Model A benchmark score?",
                knowledge_base_id="kb-benchmarks-ollama",
                mode="agentic",
            ):
                events_agentic_ollama.append(event)

            # Check that reasoning_delta was emitted for Thought
            reasoning_events = [
                e["data"]["text"]
                for e in events_agentic_ollama
                if e["event"] == "reasoning_delta"
            ]
            assert any("Model A benchmark numbers" in r for r in reasoning_events)

            # Check answer and citations
            answer_ollama = "".join(
                e["data"]["text"] for e in events_agentic_ollama if e["event"] == "token_delta"
            )
            assert "94.2%" in answer_ollama

            citations_ollama = next(e for e in events_agentic_ollama if e["event"] == "citations")
            assert len(citations_ollama["data"]) > 0
            assert citations_ollama["data"][0]["valid"] is True


class TestStreamingEventsAndThinkTags:
    """Verifies that think tags (<think>...</think>) are properly separated into reasoning_delta."""

    @pytest.mark.asyncio
    async def test_streaming_think_parser_unit(self):
        """Unit test for the StreamingThinkParser state machine."""
        parser = StreamingThinkParser()

        # Feed across split chunks
        tokens_in = [
            "<thi",
            "nk>Analyzing user query step by step.",
            " Finding key evidence.</think>The final ",
            "result is 42.",
        ]
        parsed: list[tuple[str, str]] = []
        for t in tokens_in:
            parsed.extend(parser.feed(t))
        parsed.extend(parser.flush())

        reasoning = "".join(text for kind, text in parsed if kind == "reasoning")
        generation = "".join(text for kind, text in parsed if kind == "token")

        assert "Analyzing user query step by step." in reasoning
        assert "Finding key evidence." in reasoning
        assert "The final result is 42." in generation

    @pytest.mark.asyncio
    async def test_direct_rag_streams_think_tags_as_reasoning_delta(self):
        """Verifies reasoning models outputting <think> emit reasoning_delta then token_delta."""
        registry = ProviderRegistry()
        registry.register_custom_reranker_factory("cross_encoder", lambda cfg: FakeReranker())
        vector_store = InMemoryVectorStore()
        sparse_index = InMemorySparseIndex()

        registry.set_llm_config(LLMConfig(
            provider=ProviderType.GEMINI,
            model="gemini-3.5-flash-lite",
            api_key="mock-key",
        ))
        registry.set_embedding_config(EmbeddingConfig(
            provider=ProviderType.GEMINI,
            model="gemini-embedding-001",
            dimensions=8,
            api_key="mock-key",
        ))

        orchestrator = RAGOrchestrator(
            provider_registry=registry,
            vector_store=vector_store,
            sparse_index=sparse_index,
        )

        # Seed storage with a chunk
        await vector_store.upsert_chunks(
            "kb-think",
            [
                {
                    "chunk_id": "c1",
                    "content": "Deep reasoning fact: Quantum entanglement verified.",
                    "document_id": "d1",
                    "metadata": {"source_file": "physics.txt", "page_number": 3},
                }
            ],
            [[1.0] * 8],
        )
        await sparse_index.index_chunks(
            "kb-think",
            [
                {
                    "chunk_id": "c1",
                    "content": "Deep reasoning fact: Quantum entanglement verified.",
                    "document_id": "d1",
                    "metadata": {"source_file": "physics.txt", "page_number": 3},
                }
            ],
        )

        mock_llm = registry.get_llm_provider()
        mock_emb = registry.get_embedding_provider()

        async def mock_stream(messages, config=None, tools=None):
            chunks = [
                LLMStreamChunk(delta_text="<think>I need to check the physics fact."),
                LLMStreamChunk(delta_text=" Entanglement is verified.</think>"),
                LLMStreamChunk(delta_text="Quantum entanglement is verified [1]."),
                LLMStreamChunk(delta_text="", is_final=True, usage=TokenUsage(20, 15, 35)),
            ]
            for c in chunks:
                yield c

        with patch.object(mock_emb, "embed_query", return_value=[1.0] * 8), \
             patch.object(mock_llm, "stream", side_effect=mock_stream):

            events = []
            async for ev in orchestrator.stream_query(
                "Is entanglement verified?",
                knowledge_base_id="kb-think",
                mode="direct",
            ):
                events.append(ev)

            reasoning = "".join(
                e["data"]["text"] for e in events if e["event"] == "reasoning_delta"
            )
            answer = "".join(
                e["data"]["text"] for e in events if e["event"] == "token_delta"
            )

            assert "Entanglement is verified" in reasoning
            assert "Quantum entanglement is verified [1]." in answer

            # Stages must include thinking and generating
            stages = [e["data"]["stage"] for e in events if e["event"] == "status"]
            assert "thinking" in stages
            assert "generating" in stages


class TestNonStreamingQueryConvenience:
    """Verifies the non-streaming orchestrator.query convenience method."""

    @pytest.mark.asyncio
    async def test_non_streaming_query_returns_rag_response(self):
        registry = ProviderRegistry()
        registry.register_custom_reranker_factory("cross_encoder", lambda cfg: FakeReranker())
        vector_store = InMemoryVectorStore()
        sparse_index = InMemorySparseIndex()

        registry.set_llm_config(LLMConfig(
            provider=ProviderType.GEMINI,
            model="gemini-3.5-flash-lite",
            api_key="mock-key",
        ))
        registry.set_embedding_config(EmbeddingConfig(
            provider=ProviderType.GEMINI,
            model="gemini-embedding-001",
            dimensions=8,
            api_key="mock-key",
        ))

        orchestrator = RAGOrchestrator(
            provider_registry=registry,
            vector_store=vector_store,
            sparse_index=sparse_index,
        )

        await vector_store.upsert_chunks(
            "kb-conv",
            [
                {
                    "chunk_id": "c1",
                    "content": "Direct query convenience answer text.",
                    "document_id": "d1",
                    "metadata": {"source_file": "info.txt", "page_number": 1},
                }
            ],
            [[1.0] * 8],
        )
        await sparse_index.index_chunks(
            "kb-conv",
            [
                {
                    "chunk_id": "c1",
                    "content": "Direct query convenience answer text.",
                    "document_id": "d1",
                    "metadata": {"source_file": "info.txt", "page_number": 1},
                }
            ],
        )

        mock_llm = registry.get_llm_provider()
        mock_emb = registry.get_embedding_provider()

        async def mock_stream(messages, config=None, tools=None):
            yield LLMStreamChunk(delta_text="Direct query answer [1].")
            yield LLMStreamChunk(
                delta_text="",
                is_final=True,
                usage=TokenUsage(10, 10, 20),
            )

        with patch.object(mock_emb, "embed_query", return_value=[1.0] * 8), \
             patch.object(mock_llm, "stream", side_effect=mock_stream):

            response: RAGResponse = await orchestrator.query(
                "Give me the info",
                knowledge_base_id="kb-conv",
                mode="direct",
            )

            assert isinstance(response, RAGResponse)
            assert "Direct query answer [1]." in response.answer
            assert len(response.sources) > 0
            assert len(response.citations) > 0
            assert response.citations[0].valid is True
            assert response.usage is not None
            assert response.usage.total_tokens == 20
            assert response.latency_ms >= 0.0


class TestErrorHandlingAndValidation:
    """Verifies robust error events and invariant enforcement."""

    @pytest.mark.asyncio
    async def test_error_event_on_retrieval_failure(self):
        """When an unhandled exception occurs, an error event is yielded."""
        registry = ProviderRegistry()
        orchestrator = RAGOrchestrator(provider_registry=registry)

        # Querying hybrid mode without vector_store / sparse_index raises ValueError
        events = []
        async for ev in orchestrator.stream_query("Query that fails", mode="direct"):
            events.append(ev)

        assert any(ev["event"] == "error" for ev in events)
        error_ev = next(ev for ev in events if ev["event"] == "error")
        assert "ValueError" in error_ev["data"]["code"]

    @pytest.mark.asyncio
    async def test_embedding_dimension_mismatch_error_enforced(self):
        """Invariant 3: querying with mismatched dimension raises EmbeddingDimensionMismatchError."""
        registry = ProviderRegistry()
        registry.register_custom_reranker_factory("cross_encoder", lambda cfg: FakeReranker())
        vector_store = InMemoryVectorStore()
        sparse_index = InMemorySparseIndex()

        # Metadata specifies 768 dimensions
        metadata = IndexMetadata(
            knowledge_base_id="kb-locked",
            embedding_provider="gemini",
            embedding_model_name="gemini-embedding-001",
            embedding_dimensions=768,
        )

        orchestrator = RAGOrchestrator(
            provider_registry=registry,
            vector_store=vector_store,
            sparse_index=sparse_index,
            index_metadata=metadata,
        )

        # Retrieve the provider that will be used for this locked metadata
        kb_emb = registry.get_knowledge_base_embedding_provider(metadata)
        # Patch declared dimensions to report mismatch (384 vs expected 768)
        with patch.object(kb_emb, "get_dimensions", return_value=384):
            events = []
            async for ev in orchestrator.stream_query(
                "Query with dimension mismatch",
                knowledge_base_id="kb-locked",
                mode="direct",
            ):
                events.append(ev)

            error_ev = next(ev for ev in events if ev["event"] == "error")
            assert "EmbeddingDimensionMismatchError" in error_ev["data"]["code"]

    def test_format_sse_helper(self):
        """Verifies format_sse correctly encodes event dict to standard SSE string."""
        payload = {"event": "status", "data": {"stage": "retrieving"}}
        sse_str = format_sse(payload)
        assert sse_str == 'event: status\ndata: {"stage": "retrieving"}\n\n'
