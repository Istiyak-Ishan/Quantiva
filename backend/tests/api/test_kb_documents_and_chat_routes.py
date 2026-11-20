"""
backend/tests/api/test_kb_documents_and_chat_routes.py
────────────────────────────────────────────────────────────────────────────────
Integration tests for Knowledge Base, Document, Job & Chat FastAPI Routes.
Prompt P5.3 verification suite.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
import zipfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.api.routes.dependencies import verify_desktop_token
from app.config import settings
from app.core.application_context import get_application_context, reset_application_context
from app.core.knowledge_bases import reset_knowledge_base_service
from app.core.orchestrator import RAGOrchestrator
from app.ingestion.types import DocumentChunkPayload
from app.jobs.manager import configure_default_job_manager, get_local_job_manager, LocalJobManager
from app.main import app
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.reranker import RerankDocument, RerankResult, RerankerProvider
from app.providers.interfaces.types import (
    ChatMessage,
    ConnectionTestResult,
    LLMResponse,
    LLMStreamChunk,
    ProviderCapabilities,
    TokenUsage,
)
from app.storage.database import (
    SessionLocal as LocalSessionLocal,
    engine as local_db_engine,
    initialize_database as init_local_db,
)
from app.storage.models import (
    Conversation,
    Document,
    DocumentChunk,
    EmbeddingConfiguration,
    IndexMetadata,
    KnowledgeBase,
    Message,
    ModelConfiguration,
    ProcessingJob,
    RetrievalConfiguration,
)
from app.storage.sparse_store import SQLiteSparseIndex
from app.storage.vector_store import LocalVectorStore, VectorStoreManager

AUTH_HEADER = {"X-Quantiva-Token": "quantiva-test-token"}


# ── Test Providers ────────────────────────────────────────────────────────────

class FakeLLM(LLMProvider):
    def provider_name(self) -> str:
        return "test-llm-provider"

    def model_name(self) -> str:
        return "test-llm-model"

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_vision=False,
            supports_system_prompt=True,
            max_context_window=8192,
            is_local=True,
        )

    async def generate(self, messages: list[ChatMessage], **kwargs) -> LLMResponse:
        return LLMResponse(
            content="Q4 net revenue grew 18% based on the report [1].",
            usage=TokenUsage(prompt_tokens=42, completion_tokens=14, total_tokens=56),
        )

    async def stream(self, messages: list[ChatMessage], **kwargs):
        yield LLMStreamChunk(delta_text="Q4 net revenue grew 18% ")
        yield LLMStreamChunk(delta_text="based on the report [1].")

    async def test_connection(self) -> ConnectionTestResult:
        return ConnectionTestResult(success=True, latency_ms=1.0, provider=self.provider_name(), model=self.model_name())


class FakeEmbedding(EmbeddingProvider):
    def provider_name(self) -> str:
        return "test-embedding-provider"

    def model_name(self) -> str:
        return "test-embed-model"

    def get_dimensions(self) -> int:
        return 384

    async def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * 384
        vec[0] = 1.0
        return vec

    async def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        results: list[list[float]] = []
        for _ in texts:
            vec = [0.0] * 384
            vec[0] = 1.0
            results.append(vec)
        return results

    def validate_dimensions(self, embedding: list[float]) -> None:
        if len(embedding) != 384:
            raise ValueError(f"Expected 384, got {len(embedding)}")

    async def test_embedding(self) -> ConnectionTestResult:
        return ConnectionTestResult(success=True, latency_ms=1.0, provider=self.provider_name(), model=self.model_name())


class FakeReranker(RerankerProvider):
    def provider_name(self) -> str:
        return "test-reranker"

    def model_name(self) -> str:
        return "test-rerank-model"

    async def rerank(self, query: str, documents: list[RerankDocument], top_k: int = 5) -> list[RerankResult]:
        results: list[RerankResult] = []
        for idx, doc in enumerate(documents):
            results.append(
                RerankResult(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    rerank_score=0.98 - (idx * 0.05),
                    initial_score=doc.initial_score,
                    rank=idx,
                )
            )
        return results[:top_k]

    async def test_reranker(self) -> ConnectionTestResult:
        return ConnectionTestResult(success=True, latency_ms=1.0, provider=self.provider_name(), model=self.model_name())


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def desktop_client(tmp_path: Path):
    """AsyncClient wired with clean local DB, singleton stores, and preset configurations."""
    init_local_db(local_db_engine)
    reset_application_context()
    reset_knowledge_base_service()

    # Start VectorStoreManager for test process
    vec_manager = VectorStoreManager(data_dir=settings.QUANTIVA_DATA_DIR)
    vec_manager.start()

    # Configure default mock LocalJobManager for desktop tests
    job_manager = MagicMock(spec=LocalJobManager)
    job_manager.cancel = AsyncMock(return_value=True)
    job_manager.retry_job = AsyncMock(return_value="retry-job-99")
    job_manager.submit = AsyncMock(return_value="reindex-job-42")
    configure_default_job_manager(job_manager)

    # Seed configurations
    with LocalSessionLocal() as session:
        # Explicit 384d embedding configuration
        emb = session.get(EmbeddingConfiguration, "default-test-emb-384")
        if not emb:
            emb = EmbeddingConfiguration(
                id="default-test-emb-384",
                name="Default Desktop Embedding 384",
                provider_type="sentence_transformers",
                model_name="all-MiniLM-L6-v2",
                dimensions=384,
                is_default=False,
            )
            session.add(emb)

        # Default retrieval configuration
        ret = session.scalar(select(RetrievalConfiguration).where(RetrievalConfiguration.is_default == True))
        if not ret:
            ret = RetrievalConfiguration(
                id="default-test-ret",
                name="Default Desktop Retrieval",
                search_mode="hybrid",
                dense_top_k=10,
                sparse_top_k=10,
                is_default=True,
            )
            session.add(ret)

        # Default model configuration
        model = session.scalar(select(ModelConfiguration).where(ModelConfiguration.is_default == True))
        if not model:
            model = ModelConfiguration(
                id="default-test-llm",
                name="Default Desktop LLM",
                provider_type="ollama",
                model_name="qwen2.5:7b",
                is_default=True,
            )
            session.add(model)

        session.commit()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    reset_knowledge_base_service()


# ── Knowledge Base Route Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_knowledge_base_crud_and_reindex(desktop_client: AsyncClient):
    kb_name = f"Financial Documents {uuid.uuid4().hex[:8]}"
    # 1. Create Knowledge Base
    create_resp = await desktop_client.post(
        "/api/v1/knowledge-bases",
        json={
            "name": kb_name,
            "description": "10-K and 10-Q filings",
            "embedding_config_id": "default-test-emb-384",
        },
        headers=AUTH_HEADER,
    )
    assert create_resp.status_code == 201, create_resp.text
    kb_data = create_resp.json()
    kb_id = kb_data["id"]
    assert kb_data["name"] == kb_name
    assert kb_data["embedding_dimensions"] == 384
    assert kb_data["vector_collection_name"] == f"kb_{kb_id}"
    assert kb_data["is_healthy"] is True

    # 2. List Knowledge Bases
    list_resp = await desktop_client.get("/api/v1/knowledge-bases", headers=AUTH_HEADER)
    assert list_resp.status_code == 200
    all_kbs = list_resp.json()
    assert any(k["id"] == kb_id for k in all_kbs)

    # 3. Get Details
    get_resp = await desktop_client.get(f"/api/v1/knowledge-bases/{kb_id}", headers=AUTH_HEADER)
    assert get_resp.status_code == 200
    details = get_resp.json()
    assert details["id"] == kb_id
    assert details["document_count"] == 0
    assert details["chunk_count"] == 0

    # 4. Update KB
    updated_name = f"Updated Filings {uuid.uuid4().hex[:8]}"
    update_resp = await desktop_client.put(
        f"/api/v1/knowledge-bases/{kb_id}",
        json={"name": updated_name, "description": "Updated description"},
        headers=AUTH_HEADER,
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["name"] == updated_name

    # 5. Reindex KB
    with patch("app.jobs.indexing.submit_knowledge_base_reindex_job", new_callable=AsyncMock, return_value="reindex-job-42"):
        reindex_resp = await desktop_client.post(
            f"/api/v1/knowledge-bases/{kb_id}/reindex",
            json={},
            headers=AUTH_HEADER,
        )
        assert reindex_resp.status_code == 200
        assert reindex_resp.json()["job_id"] == "reindex-job-42"
        assert reindex_resp.json()["status"] == "QUEUED"

    # 6. Delete KB
    del_resp = await desktop_client.delete(f"/api/v1/knowledge-bases/{kb_id}", headers=AUTH_HEADER)
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # 7. Confirm 404 after deletion
    get_del = await desktop_client.get(f"/api/v1/knowledge-bases/{kb_id}", headers=AUTH_HEADER)
    assert get_del.status_code == 404


@pytest.mark.asyncio
async def test_knowledge_base_export_and_import(desktop_client: AsyncClient):
    kb_name = f"Export Import KB {uuid.uuid4().hex[:8]}"
    # Create KB
    create_resp = await desktop_client.post(
        "/api/v1/knowledge-bases",
        json={"name": kb_name, "description": "Portable KB"},
        headers=AUTH_HEADER,
    )
    assert create_resp.status_code == 201
    kb_id = create_resp.json()["id"]

    # Upload document
    upload_resp = await desktop_client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/upload",
        files={"files": ("annual_report.txt", b"Annual Report content for export", "text/plain")},
        headers=AUTH_HEADER,
    )
    assert upload_resp.status_code == 201

    # Export KB -> ZIP archive
    export_resp = await desktop_client.post(
        f"/api/v1/knowledge-bases/{kb_id}/export",
        headers=AUTH_HEADER,
    )
    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"] == "application/zip"
    zip_bytes = export_resp.content
    assert zipfile.is_zipfile(io.BytesIO(zip_bytes))

    # Import KB from the exported ZIP
    import_resp = await desktop_client.post(
        "/api/v1/knowledge-bases/import",
        files={"file": ("export_package.zip", zip_bytes, "application/zip")},
        headers=AUTH_HEADER,
    )
    assert import_resp.status_code == 201
    imported_kb = import_resp.json()
    assert "Imported" in imported_kb["name"]
    assert imported_kb["document_count"] >= 1


# ── Document & Job Route Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_document_routes_and_job_lifecycle(desktop_client: AsyncClient):
    kb_name = f"Doc & Job Test KB {uuid.uuid4().hex[:8]}"
    # Create KB
    create_resp = await desktop_client.post(
        "/api/v1/knowledge-bases",
        json={"name": kb_name},
        headers=AUTH_HEADER,
    )
    assert create_resp.status_code == 201, create_resp.text
    kb_id = create_resp.json()["id"]

    # 1. Multi-file upload
    file1 = ("q1.txt", b"Q1 Revenue was $10M", "text/plain")
    file2 = ("q2.txt", b"Q2 Revenue was $12M", "text/plain")
    upload_resp = await desktop_client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/upload",
        files=[("files", file1), ("files", file2)],
        headers=AUTH_HEADER,
    )
    assert upload_resp.status_code == 201
    uploaded_data = upload_resp.json()
    assert uploaded_data["total_uploaded"] == 2
    doc1_id = uploaded_data["uploaded"][0]["id"]
    doc2_id = uploaded_data["uploaded"][1]["id"]

    # 2. Duplicate upload detection without allow_overwrite
    dup_resp = await desktop_client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/upload",
        files=[("files", ("q1_duplicate.txt", b"Q1 Revenue was $10M", "text/plain"))],
        headers=AUTH_HEADER,
    )
    assert dup_resp.json()["total_failed"] == 1
    assert dup_resp.json()["failed"][0]["error"] == "DUPLICATE_DOCUMENT"

    # 3. List documents in KB
    list_docs_resp = await desktop_client.get(
        f"/api/v1/knowledge-bases/{kb_id}/documents",
        headers=AUTH_HEADER,
    )
    assert list_docs_resp.status_code == 200
    docs = list_docs_resp.json()
    assert len(docs) == 2

    # 4. Get Document Metadata
    doc_meta_resp = await desktop_client.get(f"/api/v1/documents/{doc1_id}", headers=AUTH_HEADER)
    assert doc_meta_resp.status_code == 200
    assert doc_meta_resp.json()["filename"] == "q1.txt"

    # 5. Stream Original Document File
    file_resp = await desktop_client.get(f"/api/v1/documents/{doc1_id}/file", headers=AUTH_HEADER)
    assert file_resp.status_code == 200
    assert file_resp.content == b"Q1 Revenue was $10M"

    # 6. Chunks endpoint
    chunks_resp = await desktop_client.get(f"/api/v1/documents/{doc1_id}/chunks", headers=AUTH_HEADER)
    assert chunks_resp.status_code == 200
    assert "chunks" in chunks_resp.json()

    # 7. Job Inspection and Control
    with LocalSessionLocal() as session:
        job = ProcessingJob(
            id="test-job-inspection-1",
            job_type="DOCUMENT_INGESTION",
            knowledge_base_id=kb_id,
            document_id=doc1_id,
            status="PENDING",
            stage="VALIDATING",
            progress_percent=10.0,
            message="Validating document",
        )
        session.add(job)
        session.commit()

    jobs_resp = await desktop_client.get(f"/api/v1/jobs?knowledge_base_id={kb_id}", headers=AUTH_HEADER)
    assert jobs_resp.status_code == 200
    assert any(j["id"] == "test-job-inspection-1" for j in jobs_resp.json())

    single_job_resp = await desktop_client.get("/api/v1/jobs/test-job-inspection-1", headers=AUTH_HEADER)
    assert single_job_resp.status_code == 200
    assert single_job_resp.json()["status"] == "PENDING"

    # 8. Delete document
    del_doc_resp = await desktop_client.delete(f"/api/v1/documents/{doc1_id}", headers=AUTH_HEADER)
    assert del_doc_resp.status_code == 200
    assert del_doc_resp.json()["success"] is True

    # Confirm doc1 is gone
    get_gone = await desktop_client.get(f"/api/v1/documents/{doc1_id}", headers=AUTH_HEADER)
    assert get_gone.status_code == 404


# ── Complete End-to-End Workflow: KB -> Doc -> Ingestion -> Chat -> Citations ─

@pytest.mark.asyncio
async def test_complete_e2e_flow_kb_doc_chat_and_citations(desktop_client: AsyncClient):
    """Complete workflow: create KB -> upload document -> index chunk -> ask question -> verify citations."""
    kb_name = f"Earnings Intelligence KB {uuid.uuid4().hex[:8]}"
    # Step 1: Create Knowledge Base
    kb_resp = await desktop_client.post(
        "/api/v1/knowledge-bases",
        json={
            "name": kb_name,
            "description": "Q4 financial intelligence",
            "embedding_config_id": "default-test-emb-384",
        },
        headers=AUTH_HEADER,
    )
    assert kb_resp.status_code == 201, kb_resp.text
    kb_id = kb_resp.json()["id"]

    # Step 2: Upload Document
    doc_content = b"Q4 net revenue grew 18% to $100M driven by software subscription expansion."
    upload_resp = await desktop_client.post(
        f"/api/v1/knowledge-bases/{kb_id}/documents/upload",
        files={"files": ("q4_earnings.txt", doc_content, "text/plain")},
        headers=AUTH_HEADER,
    )
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["uploaded"][0]["id"]

    # Step 3: Index chunk into SQLite DocumentChunk, Vector Store, and SQLiteSparseIndex
    chunk_id = f"chunk-e2e-{uuid.uuid4().hex[:8]}"
    chunk_payload: DocumentChunkPayload = {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "knowledge_base_id": kb_id,
        "page_number": 1,
        "chunk_index": 0,
        "content": "Q4 net revenue grew 18% to $100M driven by software subscription expansion.",
        "source_file": "q4_earnings.txt",
        "section_title": "Executive Summary",
        "sheet_name": None,
        "is_table": False,
        "content_hash": f"hash-{chunk_id}",
        "char_count": len(doc_content),
    }

    with LocalSessionLocal() as session:
        session.add(
            DocumentChunk(
                id=chunk_id,
                document_id=doc_id,
                knowledge_base_id=kb_id,
                chunk_index=0,
                page_number=1,
                section_title="Executive Summary",
                content=chunk_payload["content"],
                content_hash=chunk_payload["content_hash"],
                char_count=chunk_payload["char_count"],
            )
        )
        kb_row = session.get(KnowledgeBase, kb_id)
        if kb_row:
            kb_row.chunk_count = 1
        doc_row = session.get(Document, doc_id)
        if doc_row:
            doc_row.chunk_count = 1
            doc_row.status = "READY"
        session.commit()

    # Upsert chunk vector into embedded vector store & sparse index
    v_store = LocalVectorStore()
    dummy_vec = [0.0] * 384
    dummy_vec[0] = 1.0
    await v_store.upsert_chunks(f"kb_{kb_id}", [chunk_payload], [dummy_vec])

    s_store = SQLiteSparseIndex(local_db_engine)
    await s_store.index_chunks(kb_id, [chunk_payload])

    # Step 4: Create Conversation
    conv_resp = await desktop_client.post(
        "/api/v1/conversations",
        json={"knowledge_base_id": kb_id, "title": "Q4 Revenue Inquiry"},
        headers=AUTH_HEADER,
    )
    assert conv_resp.status_code == 201
    conv_id = conv_resp.json()["id"]

    # Step 5: Ask Question (Non-Streaming) with Mock Providers
    test_llm = FakeLLM()
    test_emb = FakeEmbedding()
    test_reranker = FakeReranker()

    with (
        patch.object(RAGOrchestrator, "get_llm_provider", return_value=test_llm),
        patch.object(RAGOrchestrator, "get_embedding_provider", return_value=test_emb),
        patch.object(RAGOrchestrator, "get_reranker_provider", return_value=test_reranker),
    ):
        ask_resp = await desktop_client.post(
            f"/api/v1/conversations/{conv_id}/ask",
            json={"query": "What was Q4 revenue growth?", "stream": False},
            headers=AUTH_HEADER,
        )
        assert ask_resp.status_code == 200
        data = ask_resp.json()
        assert "revenue grew 18%" in data["answer"]
        assert len(data["citations"]) >= 1
        citation = data["citations"][0]
        assert citation["chunk_id"] == chunk_id
        assert citation["source_file"] == "q4_earnings.txt"
        assert citation["valid"] is True

    # Step 6: Verify Message History Persisted in Database
    msgs_resp = await desktop_client.get(f"/api/v1/conversations/{conv_id}/messages", headers=AUTH_HEADER)
    assert msgs_resp.status_code == 200
    messages = msgs_resp.json()
    assert len(messages) == 2  # user + assistant
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What was Q4 revenue growth?"
    assert messages[1]["role"] == "assistant"
    assert len(messages[1]["citations"]) >= 1

    # Step 7: Streaming Ask (SSE Stream)
    with (
        patch.object(RAGOrchestrator, "get_llm_provider", return_value=test_llm),
        patch.object(RAGOrchestrator, "get_embedding_provider", return_value=test_emb),
        patch.object(RAGOrchestrator, "get_reranker_provider", return_value=test_reranker),
    ):
        stream_resp = await desktop_client.post(
            f"/api/v1/conversations/{conv_id}/ask",
            json={"query": "Give me another summary", "stream": True},
            headers=AUTH_HEADER,
        )
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers["content-type"]
        sse_text = stream_resp.text
        assert "event: token_delta" in sse_text
        assert "event: done" in sse_text

    # Verify conversation now has 4 messages
    msgs_resp2 = await desktop_client.get(f"/api/v1/conversations/{conv_id}/messages", headers=AUTH_HEADER)
    assert len(msgs_resp2.json()) == 4

    # Step 8: Delete Conversation
    del_conv = await desktop_client.delete(f"/api/v1/conversations/{conv_id}", headers=AUTH_HEADER)
    assert del_conv.status_code == 200
    assert del_conv.json()["success"] is True
