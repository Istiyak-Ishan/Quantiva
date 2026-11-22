"""
app/core/orchestrator.py
──────────────────────────────────────────────────────────────────────────────
Unified Provider-Independent End-to-End RAG Orchestrator for Quantiva.

Ties together:
  - RetrievalPipeline (Dense + Sparse + RRF Fusion + Reranking)
  - ReActAgent / DataAnalystAgent (Tool-using reasoning agents)
  - CitationValidator (Lexical and numerical citation verification)
  - ProviderRegistry (Dynamic swapping of LLM, Embedding, and Reranker providers)

Key Invariants:
1. Strict Provider Decoupling: No concrete provider SDK (google.genai, openai,
   anthropic, ollama) is imported here. All model interactions pass through
   app.providers.interfaces.*.
2. Dual RAG Execution Modes:
   - Direct RAG: Retrieve -> Rerank -> Context Build -> Stream/Generate -> Validate Citations.
   - Agentic RAG: ReAct + Data Analyst tools -> Final Answer -> Validate Citations.
3. Normalized SSE Streaming Events:
   - {"event": "status", "data": {"stage": "retrieving" | "reranking" | "thinking" | "executing_tool" | "generating"}}
   - {"event": "sources", "data": [ ...retrieved chunks with metadata... ]}
   - {"event": "reasoning_delta", "data": {"text": "..."}}
   - {"event": "token_delta", "data": {"text": "..."}}
   - {"event": "citations", "data": [ ...validated citations... ]}
   - {"event": "done", "data": {"usage": {...}, "latency_ms": ...}}
   - {"event": "error", "data": {"code": "...", "message": "..."}}
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from app.agents.citations import (
    CitationValidationResult,
    CitationValidator,
    StructuredCitation,
    _normalise_chunk,
)
from app.agents.data_analyst import DataAnalystAgent
from app.agents.reasoning import (
    AgentRunResult,
    TokenCallback,
    parse_react_response,
)
from app.agents.tools import (
    AgentTool,
    AgentToolRegistry,
    ControlledAgentTools,
)
from app.core.application_context import get_provider_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.types import (
    DenseIndexer,
    DocumentChunkPayload,
    IngestionResult,
    ProgressCallback,
    SparseIndexer,
)
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.reranker import RerankerProvider
from app.providers.interfaces.types import (
    ChatMessage,
    ChatRole,
    GenerationConfig,
    TokenUsage,
)
from app.providers.registry import IndexMetadata, ProviderRegistry
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.types import (
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
)
from app.security.sandbox import LocalPythonSandbox
from app.storage.interfaces import SparseIndexRepository, VectorStoreRepository

logger = logging.getLogger(__name__)

StageType = Literal["retrieving", "reranking", "thinking", "executing_tool", "generating"]
OrchestratorMode = Literal["direct", "agentic"]

_DONE_SENTINEL = object()

DEFAULT_DIRECT_RAG_SYSTEM_PROMPT = (
    "You are Quantiva, an enterprise document intelligence assistant. "
    "Answer the user's question using ONLY the provided context blocks. "
    "Every factual assertion must cite its source chunk using [index] notation "
    "(e.g., [1], [2]). Never invent facts or citations not grounded in the context."
)


# ── Ingestion Indexer Adapters ────────────────────────────────────────────────

class VectorStoreDenseIndexer:
    """Adapts a VectorStoreRepository to the IngestionPipeline DenseIndexer protocol."""

    def __init__(self, vector_store: VectorStoreRepository, collection_name: str) -> None:
        self.vector_store = vector_store
        self.collection_name = collection_name

    async def index_dense(
        self,
        chunks: Sequence[DocumentChunkPayload],
        embeddings: Sequence[list[float]],
    ) -> list[str]:
        return await self.vector_store.upsert_chunks(
            self.collection_name,
            chunks,
            embeddings,
        )


class SparseIndexIndexer:
    """Adapts a SparseIndexRepository to the IngestionPipeline SparseIndexer protocol."""

    def __init__(self, sparse_index: SparseIndexRepository, knowledge_base_id: str) -> None:
        self.sparse_index = sparse_index
        self.knowledge_base_id = knowledge_base_id

    async def index_sparse(
        self,
        chunks: Sequence[DocumentChunkPayload],
    ) -> Any:
        return await self.sparse_index.index_chunks(
            self.knowledge_base_id,
            chunks,
        )


# ── Streaming Think Parser ────────────────────────────────────────────────────

class StreamingThinkParser:
    """Separates <think>...</think> reasoning deltas from answer token deltas."""

    def __init__(self) -> None:
        self.in_think: bool = False
        self._buffer: str = ""

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """Feed an incoming text chunk and return [(kind, text), ...] where kind is 'reasoning' or 'token'."""
        self._buffer += delta
        emitted: list[tuple[str, str]] = []

        while self._buffer:
            if not self.in_think:
                pos = self._buffer.find("<think>")
                if pos != -1:
                    pre = self._buffer[:pos]
                    if pre:
                        emitted.append(("token", pre))
                    self.in_think = True
                    self._buffer = self._buffer[pos + len("<think>"):]
                else:
                    prefix_match = False
                    for i in range(1, len("<think>")):
                        if self._buffer.endswith("<think>"[:i]):
                            prefix_match = True
                            safe = self._buffer[:-i]
                            if safe:
                                emitted.append(("token", safe))
                            self._buffer = self._buffer[-i:]
                            break
                    if not prefix_match:
                        emitted.append(("token", self._buffer))
                        self._buffer = ""
                    else:
                        break
            else:
                pos = self._buffer.find("</think>")
                if pos != -1:
                    pre = self._buffer[:pos]
                    if pre:
                        emitted.append(("reasoning", pre))
                    self.in_think = False
                    self._buffer = self._buffer[pos + len("</think>"):]
                else:
                    prefix_match = False
                    for i in range(1, len("</think>")):
                        if self._buffer.endswith("</think>"[:i]):
                            prefix_match = True
                            safe = self._buffer[:-i]
                            if safe:
                                emitted.append(("reasoning", safe))
                            self._buffer = self._buffer[-i:]
                            break
                    if not prefix_match:
                        emitted.append(("reasoning", self._buffer))
                        self._buffer = ""
                    else:
                        break

        return emitted

    def flush(self) -> list[tuple[str, str]]:
        """Flush any remaining buffered text at stream end."""
        if not self._buffer:
            return []
        kind = "reasoning" if self.in_think else "token"
        res = [(kind, self._buffer)]
        self._buffer = ""
        return res


# ── Response Dataclass ────────────────────────────────────────────────────────

@dataclass(slots=True)
class RAGResponse:
    """Non-streaming result returned by :meth:`RAGOrchestrator.query`."""

    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    citations: list[StructuredCitation] = field(default_factory=list)
    mode: str = "direct"
    usage: TokenUsage | None = None
    latency_ms: float = 0.0
    reasoning: str | None = None


# ── Orchestrated Agent Subclass ───────────────────────────────────────────────

class OrchestratedDataAnalystAgent(DataAnalystAgent):
    """DataAnalystAgent with callback hook for parsed reasoning thoughts in text ReAct mode."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        retrieval_pipeline: RetrievalPipeline | None = None,
        sandbox: LocalPythonSandbox | None = None,
        table_extractor: Any = None,
        knowledge_base_id: str | None = None,
        tools: AgentToolRegistry | Sequence[AgentTool] | None = None,
        max_iterations: int = 5,
        generation_config: GenerationConfig | None = None,
        on_thought: TokenCallback | None = None,
    ) -> None:
        super().__init__(
            llm_provider,
            retrieval_pipeline=retrieval_pipeline,
            sandbox=sandbox,
            table_extractor=table_extractor,
            knowledge_base_id=knowledge_base_id,
            tools=tools,
            max_iterations=max_iterations,
            generation_config=generation_config,
        )
        self.on_thought = on_thought

    async def _run_text_react(
        self,
        query: str,
        *,
        history: Sequence[ChatMessage] | None,
        on_token: TokenCallback | None,
    ) -> AgentRunResult:
        messages = [ChatMessage(ChatRole.SYSTEM, self._react_system_prompt())]
        messages.extend(history or ())
        messages.append(ChatMessage(ChatRole.USER, query.strip()))
        tool_calls_made = 0

        for iteration in range(1, self.max_iterations + 1):
            response = await self._generate(
                messages,
                native_tools=False,
                on_token=None,
            )
            parsed = parse_react_response(response.content)

            if parsed.thought and self.on_thought is not None:
                await self._emit(self.on_thought, parsed.thought)

            if parsed.tool is not None:
                tool_calls_made += 1
                observation = await self.tools.execute(
                    parsed.tool,
                    parsed.arguments or {},
                )
                messages.append(ChatMessage(ChatRole.ASSISTANT, response.content))
                messages.append(
                    ChatMessage(
                        ChatRole.USER,
                        f"Observation: {observation}",
                    )
                )
                continue

            if parsed.final_answer is not None:
                await self._emit(on_token, parsed.final_answer)
                return AgentRunResult(
                    parsed.final_answer,
                    tool_calls_made,
                    iteration,
                    False,
                )

            messages.append(ChatMessage(ChatRole.ASSISTANT, response.content))
            messages.append(
                ChatMessage(
                    ChatRole.USER,
                    "Your response was empty or malformed. Use Action JSON or Final Answer.",
                )
            )

        from app.agents.reasoning import AgentIterationLimitError

        raise AgentIterationLimitError(
            f"Agent exceeded the configured maximum of {self.max_iterations} iterations"
        )


# ── Unified RAG Orchestrator ──────────────────────────────────────────────────

class RAGOrchestrator:
    """Unified provider-independent orchestrator connecting retrieval, agents, and citations.

    Supports both:
    1. Direct RAG mode: Retrieve -> Rerank -> Context Build -> Stream/Generate -> Validate Citations.
    2. Agentic RAG mode: ReAct reasoning + Data Analyst tools -> Final Answer -> Validate Citations.
    """

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry | None = None,
        retrieval_pipeline: RetrievalPipeline | None = None,
        citation_validator: CitationValidator | None = None,
        vector_store: VectorStoreRepository | None = None,
        sparse_index: SparseIndexRepository | None = None,
        index_metadata: IndexMetadata | None = None,
        sandbox: LocalPythonSandbox | None = None,
        table_extractor: Any = None,
        default_mode: OrchestratorMode = "direct",
        retrieval_mode: RetrievalMode = "hybrid",
        system_prompt: str | None = None,
        max_agent_iterations: int = 5,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.provider_registry = provider_registry or get_provider_registry()
        self._retrieval_pipeline = retrieval_pipeline
        self.citation_validator = citation_validator or CitationValidator()
        self.vector_store = vector_store or (
            retrieval_pipeline.vector_store if retrieval_pipeline else None
        )
        self.sparse_index = sparse_index or (
            retrieval_pipeline.sparse_index if retrieval_pipeline else None
        )
        self.index_metadata = index_metadata or (
            retrieval_pipeline.index_metadata if retrieval_pipeline else None
        )
        self.sandbox = sandbox
        self.table_extractor = table_extractor
        self.default_mode: OrchestratorMode = default_mode
        self.retrieval_mode: RetrievalMode = retrieval_mode
        self.system_prompt = system_prompt
        self.max_agent_iterations = int(max_agent_iterations)
        self.generation_config = generation_config or GenerationConfig(
            temperature=0.2,
            max_tokens=4096,
        )

    # ── Provider Resolution ───────────────────────────────────────────────────

    def get_llm_provider(self) -> LLMProvider:
        """Return the active LLM provider from the registry."""
        return self.provider_registry.get_llm_provider()

    def get_embedding_provider(
        self,
        knowledge_base_id: str | None = None,
        index_metadata: IndexMetadata | None = None,
    ) -> EmbeddingProvider:
        """Return the appropriate embedding provider (KB-locked if metadata present)."""
        meta = index_metadata or self.index_metadata
        if meta is not None:
            return self.provider_registry.get_knowledge_base_embedding_provider(meta)
        return self.provider_registry.get_embedding_provider()

    def get_reranker_provider(self) -> RerankerProvider:
        """Return the active reranker provider from the registry."""
        return self.provider_registry.get_reranker_provider()

    def get_retrieval_pipeline(
        self,
        *,
        knowledge_base_id: str | None = None,
        index_metadata: IndexMetadata | None = None,
        mode: RetrievalMode | None = None,
    ) -> RetrievalPipeline:
        """Resolve or dynamically build a RetrievalPipeline using current active providers."""
        target_mode = mode or self.retrieval_mode
        active_embedding = self.get_embedding_provider(
            knowledge_base_id=knowledge_base_id,
            index_metadata=index_metadata,
        )
        active_reranker = self.get_reranker_provider()

        meta = index_metadata or self.index_metadata
        if meta is None and self.vector_store is not None:
            # Auto-construct IndexMetadata so dense retrieval has locked dimensions
            kb_id = knowledge_base_id or "default"
            col_name = kb_id
            dim = (
                int(active_embedding.get_dimensions())
                if hasattr(active_embedding, "get_dimensions")
                else 8
            )
            meta = IndexMetadata(
                knowledge_base_id=kb_id,
                embedding_provider=active_embedding.provider_name(),
                embedding_model_name=active_embedding.model_name(),
                embedding_dimensions=dim,
                collection_name=col_name,
            )

        if self._retrieval_pipeline is not None:
            # Check if active providers match the cached pipeline
            if (
                self._retrieval_pipeline.embedding_provider is active_embedding
                and getattr(self._retrieval_pipeline, "_reranking", None)
                and getattr(self._retrieval_pipeline._reranking, "reranker", None) is active_reranker
                and (mode is None or self._retrieval_pipeline.mode == target_mode)
            ):
                return self._retrieval_pipeline

        # Construct a pipeline bound to current registry providers and repositories
        return RetrievalPipeline(
            mode=target_mode,
            embedding_provider=active_embedding,
            vector_store=self.vector_store,
            sparse_index=self.sparse_index,
            index_metadata=meta,
            reranker=active_reranker,
            rerank_enabled=True,
        )

    # ── Document Ingestion ────────────────────────────────────────────────────

    async def ingest_document(
        self,
        file_bytes: bytes,
        *,
        document_id: str,
        knowledge_base_id: str,
        file_name: str | None = None,
        mime_type: str | None = None,
        collection_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
        allow_duplicates: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        index_metadata: IndexMetadata | None = None,
    ) -> IngestionResult:
        """Ingest a document using the active embedding provider and configured stores."""
        embedding_provider = self.get_embedding_provider(
            knowledge_base_id=knowledge_base_id,
            index_metadata=index_metadata,
        )

        dense_indexer: DenseIndexer | None = None
        if self.vector_store is not None:
            resolved_col = (
                collection_name
                or (self.index_metadata.collection_name if self.index_metadata else None)
                or knowledge_base_id
            )
            dense_indexer = VectorStoreDenseIndexer(self.vector_store, resolved_col)

        sparse_indexer: SparseIndexer | None = None
        if self.sparse_index is not None:
            sparse_indexer = SparseIndexIndexer(self.sparse_index, knowledge_base_id)

        pipeline = IngestionPipeline(
            embedding_provider=embedding_provider,
            dense_indexer=dense_indexer,
            sparse_indexer=sparse_indexer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return await pipeline.run(
            file_bytes,
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            file_name=file_name,
            mime_type=mime_type,
            progress_callback=progress_callback,
            allow_duplicates=allow_duplicates,
        )

    # ── Serializers ───────────────────────────────────────────────────────────

    @staticmethod
    def _serialize_chunk(chunk: RetrievedChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "score": chunk.score,
            "metadata": dict(chunk.metadata),
            "dense_rank": chunk.dense_rank,
            "sparse_rank": chunk.sparse_rank,
            "rrf_score": chunk.rrf_score,
            "rerank_score": chunk.rerank_score,
            "final_rank": chunk.final_rank,
        }

    @staticmethod
    def _serialize_citation(c: StructuredCitation) -> dict[str, Any]:
        return {
            "marker": c.marker,
            "index": c.index,
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "source_file": c.source_file,
            "page_number": c.page_number,
            "quote_snippet": c.quote_snippet,
            "lexical_overlap": c.lexical_overlap,
            "quote_verified": c.quote_verified,
            "valid": c.valid,
        }

    @staticmethod
    def _serialize_usage(usage: TokenUsage | None) -> dict[str, int]:
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": int(usage.prompt_tokens),
            "completion_tokens": int(usage.completion_tokens),
            "total_tokens": int(usage.total_tokens),
        }

    @staticmethod
    def _build_context_prompt(chunks: Sequence[RetrievedChunk]) -> str:
        parts: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata or {}
            source_file = (
                meta.get("source_file")
                or meta.get("file_name")
                or meta.get("filename")
                or "unknown"
            )
            page = meta.get("page_number", meta.get("page", 1))
            parts.append(
                f"[{idx}] (Source: {source_file}, Page: {page})\n{chunk.content}"
            )
        return "\n\n".join(parts)

    # ── Main Entry Points ─────────────────────────────────────────────────────

    async def stream_query(
        self,
        query: str,
        *,
        mode: OrchestratorMode | None = None,
        knowledge_base_id: str | None = None,
        collection_name: str | None = None,
        filters: Mapping[str, Any] | None = None,
        retrieval_mode: RetrievalMode | None = None,
        history: Sequence[ChatMessage] | None = None,
        generation_config: GenerationConfig | None = None,
        agent_tools: Sequence[AgentTool] | AgentToolRegistry | None = None,
        sandbox: LocalPythonSandbox | None = None,
        table_extractor: Any = None,
        max_iterations: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Normalize streaming events for the frontend SSE stream."""
        start_time = time.perf_counter()
        target_mode = mode or self.default_mode

        try:
            if target_mode == "direct":
                async for event in self._stream_direct_rag(
                    query,
                    knowledge_base_id=knowledge_base_id,
                    collection_name=collection_name,
                    filters=filters,
                    retrieval_mode=retrieval_mode,
                    generation_config=generation_config,
                    system_prompt=self.system_prompt,
                    start_time=start_time,
                ):
                    yield event
            elif target_mode == "agentic":
                async for event in self._stream_agentic_rag(
                    query,
                    knowledge_base_id=knowledge_base_id,
                    collection_name=collection_name,
                    filters=filters,
                    retrieval_mode=retrieval_mode,
                    generation_config=generation_config,
                    system_prompt=self.system_prompt,
                    history=history,
                    agent_tools=agent_tools,
                    sandbox=sandbox,
                    table_extractor=table_extractor,
                    max_iterations=max_iterations,
                    start_time=start_time,
                ):
                    yield event
            else:
                raise ValueError(f"Unsupported orchestrator mode: '{target_mode}'")
        except Exception as exc:
            logger.exception("Error during RAG orchestration: %s", exc)
            yield {
                "event": "error",
                "data": {
                    "code": type(exc).__name__,
                    "message": str(exc),
                },
            }

    async def query(
        self,
        query: str,
        *,
        mode: OrchestratorMode | None = None,
        knowledge_base_id: str | None = None,
        collection_name: str | None = None,
        filters: Mapping[str, Any] | None = None,
        retrieval_mode: RetrievalMode | None = None,
        history: Sequence[ChatMessage] | None = None,
        generation_config: GenerationConfig | None = None,
        agent_tools: Sequence[AgentTool] | AgentToolRegistry | None = None,
        sandbox: LocalPythonSandbox | None = None,
        table_extractor: Any = None,
        max_iterations: int | None = None,
    ) -> RAGResponse:
        """Run non-streaming query and return a unified :class:`RAGResponse`."""
        target_mode = mode or self.default_mode
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        sources: list[RetrievedChunk] = []
        citations: list[StructuredCitation] = []
        usage: TokenUsage | None = None
        latency_ms = 0.0

        async for event in self.stream_query(
            query,
            mode=target_mode,
            knowledge_base_id=knowledge_base_id,
            collection_name=collection_name,
            filters=filters,
            retrieval_mode=retrieval_mode,
            history=history,
            generation_config=generation_config,
            agent_tools=agent_tools,
            sandbox=sandbox,
            table_extractor=table_extractor,
            max_iterations=max_iterations,
        ):
            event_type = event.get("event")
            data = event.get("data", {})
            if event_type == "token_delta":
                answer_parts.append(data.get("text", ""))
            elif event_type == "reasoning_delta":
                reasoning_parts.append(data.get("text", ""))
            elif event_type == "sources":
                for item in data:
                    sources.append(_normalise_chunk(item))
            elif event_type == "citations":
                for item in data:
                    citations.append(
                        StructuredCitation(
                            marker=item.get("marker", ""),
                            index=item.get("index", 0),
                            chunk_id=item.get("chunk_id", ""),
                            document_id=item.get("document_id", ""),
                            source_file=item.get("source_file", ""),
                            page_number=item.get("page_number"),
                            quote_snippet=item.get("quote_snippet", ""),
                            lexical_overlap=float(item.get("lexical_overlap", 0.0)),
                            quote_verified=bool(item.get("quote_verified", False)),
                            valid=bool(item.get("valid", False)),
                        )
                    )
            elif event_type == "done":
                usage_data = data.get("usage", {})
                usage = TokenUsage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )
                latency_ms = float(data.get("latency_ms", 0.0))
            elif event_type == "error":
                raise RuntimeError(f"Orchestrator error [{data.get('code')}]: {data.get('message')}")

        return RAGResponse(
            answer="".join(answer_parts),
            sources=sources,
            citations=citations,
            mode=target_mode,
            usage=usage,
            latency_ms=latency_ms,
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
        )

    # ── Direct RAG Mode Implementation ────────────────────────────────────────

    async def _stream_direct_rag(
        self,
        query: str,
        *,
        knowledge_base_id: str | None,
        collection_name: str | None,
        filters: Mapping[str, Any] | None,
        retrieval_mode: RetrievalMode | None,
        generation_config: GenerationConfig | None,
        system_prompt: str | None,
        start_time: float,
    ) -> AsyncIterator[dict[str, Any]]:
        # 1. Retrieve
        yield {"event": "status", "data": {"stage": "retrieving"}}
        pipeline = self.get_retrieval_pipeline(
            knowledge_base_id=knowledge_base_id,
            mode=retrieval_mode,
        )
        retrieval_result: RetrievalResult = await pipeline.retrieve(
            query,
            knowledge_base_id=knowledge_base_id,
            collection_name=collection_name,
            filters=filters,
            mode=retrieval_mode,
        )

        # 2. Reranking stage notification if reranking occurred
        if retrieval_result.reranked:
            yield {"event": "status", "data": {"stage": "reranking"}}

        # 3. Emit sources
        serialized_sources = [self._serialize_chunk(c) for c in retrieval_result.chunks]
        yield {"event": "sources", "data": serialized_sources}

        # 4. Context Build
        context_text = self._build_context_prompt(retrieval_result.chunks)
        effective_sys = system_prompt or DEFAULT_DIRECT_RAG_SYSTEM_PROMPT
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        messages = [
            ChatMessage(ChatRole.SYSTEM, effective_sys),
            ChatMessage(ChatRole.USER, user_prompt),
        ]

        # 5. Stream / Generate
        llm = self.get_llm_provider()
        capabilities = llm.get_capabilities()
        gen_cfg = generation_config or self.generation_config
        parser = StreamingThinkParser()
        answer_parts: list[str] = []
        usage: TokenUsage | None = None
        current_stage: StageType = "generating"

        yield {"event": "status", "data": {"stage": "generating"}}

        if capabilities.supports_streaming:
            async for chunk in llm.stream(messages, config=gen_cfg):
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.delta_text:
                    items = parser.feed(chunk.delta_text)
                    for kind, text in items:
                        if kind == "reasoning":
                            if current_stage != "thinking":
                                current_stage = "thinking"
                                yield {"event": "status", "data": {"stage": "thinking"}}
                            yield {"event": "reasoning_delta", "data": {"text": text}}
                        else:
                            if current_stage != "generating":
                                current_stage = "generating"
                                yield {"event": "status", "data": {"stage": "generating"}}
                            answer_parts.append(text)
                            yield {"event": "token_delta", "data": {"text": text}}

            for kind, text in parser.flush():
                if kind == "reasoning":
                    if current_stage != "thinking":
                        current_stage = "thinking"
                        yield {"event": "status", "data": {"stage": "thinking"}}
                    yield {"event": "reasoning_delta", "data": {"text": text}}
                else:
                    if current_stage != "generating":
                        current_stage = "generating"
                        yield {"event": "status", "data": {"stage": "generating"}}
                    answer_parts.append(text)
                    yield {"event": "token_delta", "data": {"text": text}}
        else:
            response = await llm.generate(messages, config=gen_cfg)
            usage = response.usage
            items = parser.feed(response.content)
            items.extend(parser.flush())
            for kind, text in items:
                if kind == "reasoning":
                    if current_stage != "thinking":
                        current_stage = "thinking"
                        yield {"event": "status", "data": {"stage": "thinking"}}
                    yield {"event": "reasoning_delta", "data": {"text": text}}
                else:
                    if current_stage != "generating":
                        current_stage = "generating"
                        yield {"event": "status", "data": {"stage": "generating"}}
                    answer_parts.append(text)
                    yield {"event": "token_delta", "data": {"text": text}}

        # 6. Validate Citations
        full_answer = "".join(answer_parts)
        validation: CitationValidationResult = self.citation_validator.validate_citations(
            full_answer,
            retrieval_result.chunks,
        )
        serialized_citations = [self._serialize_citation(c) for c in validation.citations]
        yield {"event": "citations", "data": serialized_citations}

        # 7. Done
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        yield {
            "event": "done",
            "data": {
                "usage": self._serialize_usage(usage),
                "latency_ms": latency_ms,
            },
        }

    # ── Agentic RAG Mode Implementation ───────────────────────────────────────

    async def _stream_agentic_rag(
        self,
        query: str,
        *,
        knowledge_base_id: str | None,
        collection_name: str | None,
        filters: Mapping[str, Any] | None,
        retrieval_mode: RetrievalMode | None,
        generation_config: GenerationConfig | None,
        system_prompt: str | None,
        history: Sequence[ChatMessage] | None,
        agent_tools: Sequence[AgentTool] | AgentToolRegistry | None,
        sandbox: LocalPythonSandbox | None,
        table_extractor: Any,
        max_iterations: int | None,
        start_time: float,
    ) -> AsyncIterator[dict[str, Any]]:
        llm = self.get_llm_provider()
        pipeline = self.get_retrieval_pipeline(
            knowledge_base_id=knowledge_base_id,
            mode=retrieval_mode,
        )
        sb = sandbox or self.sandbox or LocalPythonSandbox()
        extractor = table_extractor or self.table_extractor

        # Build tools
        if agent_tools is not None:
            base_registry = (
                agent_tools
                if isinstance(agent_tools, AgentToolRegistry)
                else AgentToolRegistry(agent_tools)
            )
        else:
            controlled = ControlledAgentTools(
                retrieval_pipeline=pipeline,
                sandbox=sb,
                table_extractor=extractor,
                knowledge_base_id=knowledge_base_id,
            )
            base_registry = controlled.registry()

        queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue()
        accumulated_chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        # Wrap tools to emit status, reranking, and sources events
        wrapped_tools: list[AgentTool] = []
        for defn in base_registry.definitions:
            tool_name = defn.name
            orig_tool = base_registry._tools[tool_name]
            orig_handler = orig_tool.handler

            def _create_handler(t_name: str, h: Callable[..., Any]) -> Callable[..., Any]:
                async def _wrapped(**kwargs: Any) -> Any:
                    is_retrieval_tool = t_name in ("search_knowledge_base", "compare_documents")
                    if is_retrieval_tool:
                        await queue.put({"event": "status", "data": {"stage": "retrieving"}})
                    else:
                        await queue.put({"event": "status", "data": {"stage": "executing_tool"}})

                    result = h(**kwargs)
                    if inspect.isawaitable(result):
                        result = await result

                    if is_retrieval_tool:
                        if getattr(pipeline, "_reranking", None) and pipeline._reranking.active:
                            await queue.put({"event": "status", "data": {"stage": "reranking"}})

                        new_chunks: list[RetrievedChunk] = []
                        if isinstance(result, dict) and "chunks" in result:
                            for item in result["chunks"]:
                                chunk_obj = _normalise_chunk(item)
                                if chunk_obj.chunk_id not in seen_chunk_ids:
                                    seen_chunk_ids.add(chunk_obj.chunk_id)
                                    accumulated_chunks.append(chunk_obj)
                                    new_chunks.append(chunk_obj)
                        elif isinstance(result, dict) and "documents" in result:
                            for doc_chunks in result["documents"].values():
                                for item in doc_chunks:
                                    chunk_obj = _normalise_chunk(item)
                                    if chunk_obj.chunk_id not in seen_chunk_ids:
                                        seen_chunk_ids.add(chunk_obj.chunk_id)
                                        accumulated_chunks.append(chunk_obj)
                                        new_chunks.append(chunk_obj)

                        if new_chunks:
                            serialized = [self._serialize_chunk(c) for c in new_chunks]
                            await queue.put({"event": "sources", "data": serialized})

                    await queue.put({"event": "status", "data": {"stage": "thinking"}})
                    return result

                return _wrapped

            wrapped_tools.append(
                AgentTool(
                    definition=defn,
                    handler=_create_handler(tool_name, orig_handler),
                )
            )

        wrapped_registry = AgentToolRegistry(wrapped_tools)
        parser = StreamingThinkParser()
        is_generating = False

        async def on_token(delta_text: str) -> None:
            nonlocal is_generating
            items = parser.feed(delta_text)
            for kind, text in items:
                if kind == "reasoning":
                    await queue.put({"event": "reasoning_delta", "data": {"text": text}})
                else:
                    if not is_generating:
                        is_generating = True
                        await queue.put({"event": "status", "data": {"stage": "generating"}})
                    await queue.put({"event": "token_delta", "data": {"text": text}})

        async def on_thought(thought_text: str) -> None:
            await queue.put({"event": "reasoning_delta", "data": {"text": thought_text}})

        agent = OrchestratedDataAnalystAgent(
            llm_provider=llm,
            retrieval_pipeline=pipeline,
            sandbox=sb,
            table_extractor=extractor,
            knowledge_base_id=knowledge_base_id,
            tools=wrapped_registry,
            max_iterations=max_iterations or self.max_agent_iterations,
            generation_config=generation_config or self.generation_config,
            on_thought=on_thought,
        )
        if system_prompt:
            agent.system_prompt = system_prompt

        await queue.put({"event": "status", "data": {"stage": "thinking"}})

        async def worker() -> None:
            try:
                run_res: AgentRunResult = await agent.run(
                    query,
                    history=history,
                    on_token=on_token,
                )

                for kind, text in parser.flush():
                    if kind == "reasoning":
                        await queue.put({"event": "reasoning_delta", "data": {"text": text}})
                    else:
                        await queue.put({"event": "token_delta", "data": {"text": text}})

                # Validate citations on final answer
                validation: CitationValidationResult = self.citation_validator.validate_citations(
                    run_res.answer,
                    accumulated_chunks,
                )
                serialized_citations = [self._serialize_citation(c) for c in validation.citations]
                await queue.put({"event": "citations", "data": serialized_citations})

                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                await queue.put(
                    {
                        "event": "done",
                        "data": {
                            "usage": self._serialize_usage(None),
                            "latency_ms": latency_ms,
                        },
                    }
                )
            except Exception as exc:
                await queue.put(exc)
            finally:
                await queue.put(_DONE_SENTINEL)

        worker_task = asyncio.create_task(worker())

        while True:
            item = await queue.get()
            if item is _DONE_SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield cast(dict[str, Any], item)

        await worker_task


# ── SSE Utility ───────────────────────────────────────────────────────────────

def format_sse(event: dict[str, Any]) -> str:
    """Format an event dictionary as a standard SSE payload string."""
    name = event.get("event", "message")
    data = json.dumps(event.get("data", {}))
    return f"event: {name}\ndata: {data}\n\n"


__all__ = [
    "DEFAULT_DIRECT_RAG_SYSTEM_PROMPT",
    "OrchestratedDataAnalystAgent",
    "OrchestratorMode",
    "RAGOrchestrator",
    "RAGResponse",
    "SparseIndexIndexer",
    "StageType",
    "StreamingThinkParser",
    "VectorStoreDenseIndexer",
    "format_sse",
]
