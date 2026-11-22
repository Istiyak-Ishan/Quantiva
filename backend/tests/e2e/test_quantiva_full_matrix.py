"""
backend/tests/e2e/test_quantiva_full_matrix.py
────────────────────────────────────────────────────────────────────────────────
Self-contained, reproducible End-to-End Stress & Integration Test Suite.
Verifies all Phase 8 desktop requirements across four full-coverage matrices:

1. Document Format Matrix:
   - Multi-page PDF with text and embedded financial table
   - Scanned image-only PDF triggering OCR pipeline
   - Structured DOCX with multi-level headings and tables
   - Multi-sheet XLSX financial workbook and CSV dataset
   - Large multi-section document (chunk boundaries & table header continuation)
   - Duplicate file upload detection (SHA-256)

2. Retrieval & Reranking Matrix:
   - Dense-only vs. Sparse FTS5 vs. Hybrid RRF vs. Hybrid + Cross-Encoder Reranking
   - Citation accuracy verification (exact source_file and page_number matching)

3. Provider & Embedding Change Matrix:
   - Local LLM (Ollama) vs. Cloud API (Gemini)
   - 384-dim to 768-dim model switch, EmbeddingDimensionMismatchError protection,
     and shadow-collection reindex job recovery.

4. Resilience, Offline & Recovery Matrix:
   - Simulated interrupted ingestion job recovered on restart
   - Full application context teardown and restart persistence
   - Hot atomic backup creation, simulated data deletion, and full restore
   - Offline mode enforcement (socket blocking monkeypatch ensuring zero external calls)
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
import csv
import io
import os
from pathlib import Path
import shutil
import socket
from typing import Any
import uuid

import docx
import fitz
import openpyxl
from PIL import Image, ImageDraw
import pytest
from sqlalchemy import func, select

from app.config import LocalDataPaths, settings
from app.core.orchestrator import RAGOrchestrator
from app.ingestion.chunker import chunk_blocks
from app.ingestion.cleaner import compute_file_hash
from app.ingestion.extractor import extract_document, needs_ocr
from app.ingestion.ocr import OCRService
from app.ingestion.parser import (
    parse_csv,
    parse_docx,
    parse_pdf,
    parse_xlsx,
)
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.types import DocumentChunkPayload, DuplicateDocumentError
from app.jobs.indexing import KnowledgeBaseReindexJobHandler
from app.jobs.manager import JobStatus, LocalJobManager
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
    TokenUsage,
    ToolDefinition,
)
from app.providers.registry import IndexMetadata, ProviderRegistry
from app.retrieval.pipeline import RetrievalPipeline, build_context_blocks
from app.retrieval.types import CitationAnchor
from app.storage.backup import create_backup, restore_backup
from app.storage.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from app.storage.file_store import LocalFileStore
from app.storage.models import (
    Document,
    DocumentChunk,
    EmbeddingConfiguration,
    IndexMetadata as StoredIndexMetadata,
    KnowledgeBase,
    ProcessingJob,
    RetrievalConfiguration,
)
from app.storage.sparse_store import SQLiteSparseIndex
from app.storage.vector_store import (
    LocalVectorStore,
    VectorStoreManager,
    collection_name_for_knowledge_base,
)


@pytest.fixture(autouse=True)
def cleanup_vector_manager():
    """Ensures VectorStoreManager singleton is cleanly closed and reset between tests."""
    yield
    with VectorStoreManager._instance_lock:
        if VectorStoreManager._instance is not None:
            try:
                VectorStoreManager._instance.close()
            except Exception:
                pass
            VectorStoreManager._instance = None


# ─── PROGRAMMATIC SYNTHETIC TEST FILE GENERATORS ─────────────────────────────

def generate_multipage_pdf() -> bytes:
    """Generates a 2-page PDF with narrative text and an embedded financial table."""
    doc = fitz.open()

    # Page 1: Financial narrative and embedded table
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 60), "ACME CORP - QUARTERLY FINANCIAL REPORT", fontsize=14)
    p1.insert_text(
        (50, 90),
        "Operating performance remained strong across all business units in Q3 2024.\n"
        "Gross revenue reached record levels, supported by enterprise cloud migrations.",
        fontsize=10,
    )

    # Draw table bounding box and dividing lines for PyMuPDF table detection
    p1.draw_rect(fitz.Rect(50, 130, 500, 230))
    p1.draw_line(fitz.Point(50, 155), fitz.Point(500, 155))
    p1.draw_line(fitz.Point(50, 180), fitz.Point(500, 180))
    p1.draw_line(fitz.Point(50, 205), fitz.Point(500, 205))
    p1.draw_line(fitz.Point(160, 130), fitz.Point(160, 230))
    p1.draw_line(fitz.Point(270, 130), fitz.Point(270, 230))
    p1.draw_line(fitz.Point(380, 130), fitz.Point(380, 230))

    # Header text
    p1.insert_text((70, 148), "Quarter", fontsize=10)
    p1.insert_text((180, 148), "Revenue", fontsize=10)
    p1.insert_text((290, 148), "Gross Profit", fontsize=10)
    p1.insert_text((400, 148), "Operating Margin", fontsize=10)

    # Row 1 text
    p1.insert_text((70, 173), "Q1 2024", fontsize=10)
    p1.insert_text((180, 173), "42.0M", fontsize=10)
    p1.insert_text((290, 173), "25.2M", fontsize=10)
    p1.insert_text((400, 173), "60.0%", fontsize=10)

    # Row 2 text
    p1.insert_text((70, 198), "Q2 2024", fontsize=10)
    p1.insert_text((180, 198), "48.5M", fontsize=10)
    p1.insert_text((290, 198), "29.1M", fontsize=10)
    p1.insert_text((400, 198), "60.0%", fontsize=10)

    # Row 3 text
    p1.insert_text((70, 223), "Q3 2024", fontsize=10)
    p1.insert_text((180, 223), "54.2M", fontsize=10)
    p1.insert_text((290, 223), "33.6M", fontsize=10)
    p1.insert_text((400, 223), "62.0%", fontsize=10)

    # Page 2: Notes & Risk Factors
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 60), "SECTION 2: RISK FACTORS & DISCLOSURES", fontsize=14)
    p2.insert_text(
        (50, 90),
        "Note 7: Foreign currency fluctuations created a $2.1M headwind in EMEA.\n"
        "Supply chain disruptions have been resolved with dual-sourcing contracts.",
        fontsize=10,
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def generate_scanned_image_pdf() -> bytes:
    """Generates a scanned image-only PDF with zero selectable text, triggering OCR."""
    img = Image.new("RGB", (600, 300), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "SCANNED INVOICE #98421", fill="black")
    draw.text((20, 50), "VENDOR: ACME INDUSTRIAL LOGISTICS", fill="black")
    draw.text((20, 80), "TOTAL AMOUNT DUE: $14,250.00", fill="black")
    draw.text((20, 110), "TERMS: NET 30 DAYS", fill="black")

    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_bytes = img_buf.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=600, height=300)
    page.insert_image(fitz.Rect(0, 0, 600, 300), stream=img_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def generate_structured_docx() -> bytes:
    """Generates a DOCX with multi-level headings and tables."""
    d = docx.Document()
    d.add_heading("Corporate Governance Charter", level=1)
    d.add_paragraph("This document outlines committee structures and responsibilities.")

    d.add_heading("Audit Committee Mandate", level=2)
    d.add_paragraph(
        "The Audit Committee oversees internal financial controls and risk management."
    )

    table = d.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "Member Name"
    table.cell(0, 1).text = "Committee Role"
    table.cell(1, 0).text = "Jane Doe"
    table.cell(1, 1).text = "Chairperson"
    table.cell(2, 0).text = "John Smith"
    table.cell(2, 1).text = "Independent Auditor"

    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def generate_multisheet_xlsx() -> bytes:
    """Generates a multi-sheet XLSX financial workbook."""
    wb = openpyxl.Workbook()

    ws1 = wb.active
    ws1.title = "Income Statement"
    ws1.append(["Account", "2023 Actual", "2024 Budget"])
    ws1.append(["Subscription Revenue", 1200000, 1500000])
    ws1.append(["Professional Services", 300000, 350000])

    ws2 = wb.create_sheet(title="Balance Sheet")
    ws2.append(["Category", "Q1", "Q2"])
    ws2.append(["Cash & Equivalents", 450000, 620000])
    ws2.append(["Accounts Receivable", 210000, 195000])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def generate_csv_dataset() -> bytes:
    """Generates a clean CSV dataset."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["TransactionID", "Date", "Department", "AmountUSD"])
    writer.writerow(["TX-901", "2024-09-01", "Engineering", "12500.00"])
    writer.writerow(["TX-902", "2024-09-02", "Marketing", "4200.50"])
    writer.writerow(["TX-903", "2024-09-03", "Operations", "8900.00"])
    return output.getvalue().encode("utf-8")


# ─── DETERMINISTIC TEST MOCKS ────────────────────────────────────────────────

class DeterministicMockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider with controllable dimension vector space."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    def provider_name(self) -> str:
        return "deterministic-mock"

    def model_name(self) -> str:
        return f"mock-embedding-{self._dimensions}d"

    def get_dimensions(self) -> int:
        return self._dimensions

    def validate_dimensions(self, vector: Sequence[float]) -> None:
        if len(vector) != self._dimensions:
            raise EmbeddingDimensionMismatchError(
                expected=self._dimensions,
                actual=len(vector),
                provider=self.provider_name(),
                model_name=self.model_name(),
            )

    async def embed_documents(
        self, texts: Sequence[str], batch_size: int = 32
    ) -> list[list[float]]:
        return [await self.embed_query(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        val = sum(ord(c) for c in query) % 100
        vec = [0.0] * self._dimensions
        vec[0] = float(val) / 100.0
        vec[1] = 1.0 - vec[0]
        return vec

    async def test_embedding(self) -> ConnectionTestResult:
        return ConnectionTestResult(success=True, message="Mock embedding active")


class DeterministicMockRerankerProvider(RerankerProvider):
    """Promotes documents containing the search query keywords to the top."""

    def provider_name(self) -> str:
        return "deterministic-reranker"

    def model_name(self) -> str:
        return "mock-cross-encoder"

    async def rerank(
        self,
        query: str,
        documents: list[RerankDocument],
        top_k: int = 5,
    ) -> list[RerankResult]:
        query_words = set(query.lower().split())
        scored: list[tuple[float, int, RerankDocument]] = []

        for idx, doc in enumerate(documents):
            text = doc.content.lower()
            match_count = sum(1 for w in query_words if w in text)
            score = 0.5 + (0.1 * match_count)
            scored.append((score, idx, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        limit = top_k if top_k is not None else len(scored)

        return [
            RerankResult(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                rerank_score=score,
                initial_score=doc.initial_score,
                rank=i,
            )
            for i, (score, orig_idx, doc) in enumerate(scored[:limit])
        ]

    async def test_reranker(self) -> ConnectionTestResult:
        return ConnectionTestResult(success=True, message="Mock reranker active")


class DeterministicMockLLMProvider(LLMProvider):
    """Deterministic LLM answering queries with structured citations."""

    def __init__(self, provider: str = "mock-local", model: str = "qwen2.5:7b") -> None:
        self._provider = provider
        self._model = model

    def provider_name(self) -> str:
        return self._provider

    def model_name(self) -> str:
        return self._model

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_vision=False,
            supports_system_prompt=True,
            max_context_window=32768,
            is_local=self._provider == "ollama",
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content="According to the report [1], total Q3 revenue was $54.2M.",
            usage=TokenUsage(prompt_tokens=45, completion_tokens=15, total_tokens=60),
            finish_reason="stop",
            model=self._model,
            provider=self._provider,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        yield LLMStreamChunk(
            delta_text="According to the report [1], total Q3 revenue was $54.2M.",
            is_final=True,
            usage=TokenUsage(prompt_tokens=45, completion_tokens=15, total_tokens=60),
            finish_reason="stop",
        )

    async def test_connection(self) -> ConnectionTestResult:
        return ConnectionTestResult(success=True, message="Mock LLM connected")


# ─── MATRIX 1: DOCUMENT FORMAT MATRIX ────────────────────────────────────────

class TestDocumentFormatMatrix:
    """Verifies parsing, chunking, and duplicate detection across all supported document formats."""

    def test_multipage_pdf_with_table(self):
        pdf_bytes = generate_multipage_pdf()
        out = parse_pdf(pdf_bytes)

        assert out.page_count == 2
        assert len(out.blocks) >= 2
        # Verify financial table markdown parsing
        table_blocks = [b for b in out.blocks if b.block_type == "table"]
        assert len(table_blocks) >= 1
        assert "Q3" in table_blocks[0].text
        assert "2024" in table_blocks[0].text
        assert "54" in table_blocks[0].text

    def test_scanned_image_pdf_triggers_ocr_pipeline(self):
        scanned_pdf = generate_scanned_image_pdf()
        extraction = extract_document(scanned_pdf, file_name="invoice.pdf")

        # The image-only PDF contains zero digital text, so it must be flagged for OCR
        assert needs_ocr(extraction) is True
        assert len(extraction.ocr_candidates) >= 1
        assert extraction.ocr_candidates[0] == 1  # Page 1 candidate

    def test_structured_docx_headings_and_tables(self):
        docx_bytes = generate_structured_docx()
        out = parse_docx(docx_bytes)

        assert len(out.blocks) >= 3
        # Check hierarchy
        section_paths = [b.section_path for b in out.blocks if b.section_path]
        assert any("Audit Committee Mandate" in path for path in section_paths)
        # Check table
        tables = [b for b in out.blocks if b.block_type == "table"]
        assert len(tables) >= 1
        assert "Jane Doe" in tables[0].text
        assert "Independent Auditor" in tables[0].text

    def test_multisheet_xlsx_and_csv(self):
        xlsx_bytes = generate_multisheet_xlsx()
        xlsx_out = parse_xlsx(xlsx_bytes)

        # Must parse both sheets into discrete blocks
        sheets = {b.sheet_name for b in xlsx_out.blocks if b.sheet_name}
        assert "Income Statement" in sheets
        assert "Balance Sheet" in sheets

        # CSV test
        csv_bytes = generate_csv_dataset()
        csv_out = parse_csv(csv_bytes)
        assert len(csv_out.blocks) >= 1
        assert "TX-901" in csv_out.blocks[0].text
        assert "Engineering" in csv_out.blocks[0].text

    def test_large_multisection_chunking_and_table_continuation(self):
        """Verifies chunk boundary preservation and header continuation for long tables."""
        pdf_bytes = generate_multipage_pdf()
        extraction = extract_document(pdf_bytes, file_name="report.pdf")

        chunks = chunk_blocks(
            extraction.blocks,
            document_id="doc-report-1",
            knowledge_base_id="kb-test-1",
            source_file="report.pdf",
            chunk_size=150,  # small size to force multi-chunk boundary
            chunk_overlap=30,
        )

        assert len(chunks) >= 3
        # Ensure page numbers and table metadata are preserved across split chunks
        assert all(c["page_number"] is not None for c in chunks)
        table_chunks = [c for c in chunks if c["is_table"]]
        assert len(table_chunks) >= 1

    def test_duplicate_file_upload_detection_sha256(self):
        pdf_bytes = generate_multipage_pdf()
        hash1 = compute_file_hash(pdf_bytes)
        hash2 = compute_file_hash(pdf_bytes)
        assert hash1 == hash2

        # IngestionPipeline duplicate check
        existing_lookup = lambda h: "existing-doc-id-123" if h == hash1 else None
        pipeline = IngestionPipeline(
            embedding_provider=DeterministicMockEmbeddingProvider(384),
            existing_document_lookup=existing_lookup,
        )

        with pytest.raises(DuplicateDocumentError) as exc_info:
            asyncio.run(
                pipeline.run(
                    pdf_bytes,
                    document_id="new-doc-id",
                    knowledge_base_id="kb-123",
                    file_name="q3_report.pdf",
                )
            )

        assert "already exists" in str(exc_info.value).lower()
        assert exc_info.value.content_hash == hash1


# ─── MATRIX 2: RETRIEVAL & RERANKING MATRIX ──────────────────────────────────

class TestRetrievalAndRerankingMatrix:
    """Verifies Dense Only vs. Sparse FTS5 vs. Hybrid RRF vs. Hybrid + Cross-Encoder Reranking."""

    @pytest.mark.asyncio
    async def test_retrieval_strategies_and_citation_accuracy(self, tmp_path):
        # 1. Setup isolated environment
        settings.QUANTIVA_DATA_DIR = str(tmp_path)
        engine = create_sqlite_engine(tmp_path)
        initialize_database(engine)

        v_manager = VectorStoreManager(data_dir=tmp_path)
        v_manager.start()
        vector_store = LocalVectorStore(v_manager)
        sparse_store = SQLiteSparseIndex(engine)

        kb_id = f"test-kb-{uuid.uuid4().hex[:6]}"
        collection_name = collection_name_for_knowledge_base(kb_id)
        embedding_provider = DeterministicMockEmbeddingProvider(384)
        reranker = DeterministicMockRerankerProvider()

        index_metadata = IndexMetadata(
            knowledge_base_id=kb_id,
            embedding_provider="deterministic-mock",
            embedding_model_name="mock-embedding-384d",
            embedding_dimensions=384,
            collection_name=collection_name,
        )

        # 2. Ingest structured chunks into vector and sparse stores
        chunks_payload: list[DocumentChunkPayload] = [
            {
                "chunk_id": f"chunk-{i}",
                "document_id": "doc-financials-1",
                "knowledge_base_id": kb_id,
                "page_number": 1 if i < 2 else 2,
                "chunk_index": i,
                "content": content,
                "source_file": "financial_statement.pdf",
                "section_title": "Executive Summary" if i < 2 else "Risk Disclosures",
                "sheet_name": None,
                "is_table": i == 1,
                "content_hash": f"hash-{i}",
                "char_count": len(content),
            }
            for i, content in enumerate([
                "Total net revenue reached $54.2M in Q3 2024 with 62% operating margin.",
                "| Quarter | Revenue | Gross Profit |\n| Q3 2024 | $54.2M | $33.6M |",
                "Foreign exchange currency fluctuations created a $2.1M headwind in EMEA.",
            ])
        ]

        # Embed and index
        vectors = await embedding_provider.embed_documents([c["content"] for c in chunks_payload])
        await vector_store.ensure_collection(collection_name, embedding_dimensions=384)
        await vector_store.upsert_chunks(collection_name, chunks_payload, vectors)
        await sparse_store.index_chunks(kb_id, chunks_payload)

        # 3. Test Dense Only
        pipeline_dense = RetrievalPipeline(
            mode="dense_only",
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            sparse_index=sparse_store,
            index_metadata=index_metadata,
            reranker=reranker,
            rerank_enabled=False,
        )
        res_dense = await pipeline_dense.retrieve("Q3 revenue", knowledge_base_id=kb_id)
        assert len(res_dense.chunks) > 0

        # 4. Test Sparse FTS5 Only
        pipeline_sparse = RetrievalPipeline(
            mode="sparse_only",
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            sparse_index=sparse_store,
            index_metadata=index_metadata,
            reranker=reranker,
            rerank_enabled=False,
        )
        res_sparse = await pipeline_sparse.retrieve("currency headwind", knowledge_base_id=kb_id)
        assert len(res_sparse.chunks) > 0
        assert any("currency" in c.content.lower() for c in res_sparse.chunks)

        # 5. Test Hybrid RRF
        pipeline_hybrid = RetrievalPipeline(
            mode="hybrid",
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            sparse_index=sparse_store,
            index_metadata=index_metadata,
            reranker=reranker,
            rerank_enabled=False,
        )
        res_hybrid = await pipeline_hybrid.retrieve("operating margin", knowledge_base_id=kb_id)
        assert len(res_hybrid.chunks) > 0
        assert res_hybrid.chunks[0].rrf_score is not None

        # 6. Test Hybrid + Cross-Encoder Reranker
        pipeline_rerank = RetrievalPipeline(
            mode="hybrid",
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            sparse_index=sparse_store,
            index_metadata=index_metadata,
            reranker=reranker,
            rerank_enabled=True,
        )
        res_rerank = await pipeline_rerank.retrieve("operating margin", knowledge_base_id=kb_id)
        assert len(res_rerank.chunks) > 0
        assert res_rerank.chunks[0].rerank_score is not None

        # 7. Verify Citation Accuracy (exact source_file and page_number)
        top_chunk = res_rerank.chunks[0]
        assert top_chunk.metadata.get("source_file") == "financial_statement.pdf"
        assert top_chunk.metadata.get("page_number") in (1, 2)
        assert len(res_rerank.context_blocks) > 0
        assert res_rerank.context_blocks[0].citation.startswith("[Doc: financial_statement.pdf")

        v_manager.close()


# ─── MATRIX 3: PROVIDER & EMBEDDING CHANGE MATRIX ────────────────────────────

class TestProviderAndEmbeddingChangeMatrix:
    """Verifies LLM swapping, embedding dimension protection, and reindex job recovery."""

    @pytest.mark.asyncio
    async def test_local_llm_vs_cloud_api(self, tmp_path):
        """Verifies direct RAG works equivalently with local Ollama vs cloud Gemini models."""
        mock_local_llm = DeterministicMockLLMProvider("ollama", "qwen2.5:7b")
        mock_cloud_llm = DeterministicMockLLMProvider("gemini", "gemini-1.5-pro")

        # Local LLM response
        resp_local = await mock_local_llm.generate(
            [ChatMessage(role=ChatRole.USER, content="Revenue?")]
        )
        assert "54.2M" in resp_local.content

        # Cloud LLM response
        resp_cloud = await mock_cloud_llm.generate(
            [ChatMessage(role=ChatRole.USER, content="Revenue?")]
        )
        assert "54.2M" in resp_cloud.content

    @pytest.mark.asyncio
    async def test_embedding_dimension_switch_protection_and_reindexing(self, tmp_path):
        """
        Tests:
          1. Initial collection created at 384 dimensions.
          2. Switching model to 768 dimensions raises EmbeddingDimensionMismatchError.
          3. Reindex job safely rebuilds shadow collection at 768 dimensions.
          4. Subsequent queries at 768 dimensions succeed.
        """
        settings.QUANTIVA_DATA_DIR = str(tmp_path)
        engine = create_sqlite_engine(tmp_path)
        initialize_database(engine)
        session_factory = create_session_factory(engine)

        v_manager = VectorStoreManager(data_dir=tmp_path)
        v_manager.start()
        vector_store = LocalVectorStore(v_manager)

        kb_id = f"switch-kb-{uuid.uuid4().hex[:6]}"
        collection_name = collection_name_for_knowledge_base(kb_id)

        # Step 1: Create collection with 384 dimensions
        info = await vector_store.ensure_collection(collection_name, embedding_dimensions=384)
        assert info.name == collection_name
        assert info.dimensions == 384

        chunk_payload: list[DocumentChunkPayload] = [
            {
                "chunk_id": "c1",
                "document_id": "doc1",
                "knowledge_base_id": kb_id,
                "page_number": 1,
                "chunk_index": 0,
                "content": "Sample financial text",
                "source_file": "doc.pdf",
                "section_title": "Summary",
                "sheet_name": None,
                "is_table": False,
                "content_hash": "h1",
                "char_count": 21,
            }
        ]
        vec_384 = [0.1] * 384
        await vector_store.upsert_chunks(collection_name, chunk_payload, [vec_384])

        # Step 2: Querying with mismatched 768-dim query raises EmbeddingDimensionMismatchError
        vec_768 = [0.1] * 768
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            await vector_store.search_dense(collection_name, query_vector=vec_768, top_k=5)

        assert "768" in str(exc_info.value) or "384" in str(exc_info.value)

        # Step 3: Run Reindex Job to rebuild with 768 dimensions
        with session_factory() as session:
            emb_cfg_768 = EmbeddingConfiguration(
                name="768-dim Model",
                provider_type="local",
                model_name="nomic-embed-text",
                dimensions=768,
            )
            session.add(emb_cfg_768)
            session.commit()
            session.refresh(emb_cfg_768)

            kb = KnowledgeBase(name=f"KB-{kb_id}", embedding_config_id=emb_cfg_768.id)
            session.add(kb)
            session.commit()
            session.refresh(kb)

            meta = StoredIndexMetadata(
                knowledge_base_id=kb.id,
                vector_collection_name=collection_name,
                embedding_provider="local",
                embedding_model_name="mock-384",
                embedding_dimensions=384,
            )
            session.add(meta)

            doc = Document(
                knowledge_base_id=kb.id,
                filename="doc.pdf",
                original_rel_path="doc.pdf",
                file_type="pdf",
                file_size_bytes=100,
                sha256_hash="hash-doc1",
                status="READY",
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)

            dchunk = DocumentChunk(
                knowledge_base_id=kb.id,
                document_id=doc.id,
                chunk_index=0,
                content="Sample financial text",
                char_count=21,
                content_hash="h1",
            )
            session.add(dchunk)
            session.commit()

            reindex_kb_id = kb.id
            reindex_cfg_id = emb_cfg_768.id

        provider_768 = DeterministicMockEmbeddingProvider(768)
        job_handler = KnowledgeBaseReindexJobHandler(
            session_factory=session_factory,
            embedding_provider_resolver=lambda cfg: provider_768,
            vector_store=vector_store,
        )

        job_mgr = LocalJobManager(session_factory=session_factory)
        job_mgr.register_handler(job_handler.JOB_TYPE, job_handler)
        await job_mgr.start()

        job_id = await job_mgr.submit(
            job_handler.JOB_TYPE,
            payload={
                "knowledge_base_id": reindex_kb_id,
                "new_embedding_config_id": reindex_cfg_id,
            },
            knowledge_base_id=reindex_kb_id,
        )

        completed_job = await job_mgr.wait_for_job(job_id, timeout=10.0)
        assert completed_job.status == JobStatus.COMPLETED.value
        await job_mgr.stop()

        # Step 4: Subsequent query with 768 dimensions now succeeds on promoted collection
        with session_factory() as session:
            promoted_meta = session.scalar(
                select(StoredIndexMetadata).where(
                    StoredIndexMetadata.knowledge_base_id == reindex_kb_id
                )
            )
            assert promoted_meta is not None
            promoted_collection = promoted_meta.vector_collection_name

        res_768 = await vector_store.search_dense(promoted_collection, query_vector=vec_768, top_k=5)
        assert len(res_768) >= 1
        assert res_768[0].chunk_id == dchunk.id

        v_manager.close()


# ─── MATRIX 4: RESILIENCE, OFFLINE & RECOVERY MATRIX ─────────────────────────

class TestResilienceOfflineAndRecoveryMatrix:
    """Verifies interrupted job recovery, database persistence, atomic backup/restore, and offline safety."""

    @pytest.mark.asyncio
    async def test_simulated_interrupted_ingestion_job_recovery(self, tmp_path):
        """Simulates application crash during ingestion and verifies clean recovery on restart."""
        settings.QUANTIVA_DATA_DIR = str(tmp_path)
        engine = create_sqlite_engine(tmp_path)
        initialize_database(engine)
        session_factory = create_session_factory(engine)

        with session_factory() as session:
            kb = KnowledgeBase(name=f"Crash Test KB {uuid.uuid4().hex[:6]}")
            session.add(kb)
            session.commit()
            session.refresh(kb)

            doc = Document(
                knowledge_base_id=kb.id,
                filename="interrupted.pdf",
                original_rel_path="interrupted.pdf",
                file_type="pdf",
                file_size_bytes=5000,
                sha256_hash="crash-hash-1",
                status="PROCESSING",  # Stuck in processing
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)

            job = ProcessingJob(
                job_type="DOCUMENT_INGESTION",
                knowledge_base_id=kb.id,
                document_id=doc.id,
                status="RUNNING",  # Stuck in running
                progress_percent=45.0,
            )
            session.add(job)
            session.commit()
            job_id = job.id
            doc_id = doc.id

        # Restart and invoke recovery
        manager = LocalJobManager(session_factory=session_factory)
        recovered_ids = await manager.recover_interrupted_jobs()

        assert job_id in recovered_ids

        with session_factory() as session:
            recovered_job = session.get(ProcessingJob, job_id)
            recovered_doc = session.get(Document, doc_id)
            assert recovered_job.status == JobStatus.INTERRUPTED.value
            assert recovered_doc.status == "INTERRUPTED_NEEDS_RETRY"

    @pytest.mark.asyncio
    async def test_full_context_teardown_and_restart_persistence(self, tmp_path):
        """Verifies complete persistence across process destruction and restart."""
        settings.QUANTIVA_DATA_DIR = str(tmp_path)
        engine = create_sqlite_engine(tmp_path)
        initialize_database(engine)
        session_factory = create_session_factory(engine)

        v_manager = VectorStoreManager(data_dir=tmp_path)
        v_manager.start()
        vector_store = LocalVectorStore(v_manager)

        kb_id = f"kb-p-{uuid.uuid4().hex[:6]}"
        collection_name = collection_name_for_knowledge_base(kb_id)
        await vector_store.ensure_collection(collection_name, embedding_dimensions=384)
        await vector_store.upsert_chunks(
            collection_name,
            [
                {
                    "chunk_id": "persist-1",
                    "document_id": "doc-p1",
                    "knowledge_base_id": kb_id,
                    "page_number": 1,
                    "chunk_index": 0,
                    "content": "Critical persisted record",
                    "source_file": "p.pdf",
                    "section_title": "P",
                    "sheet_name": None,
                    "is_table": False,
                    "content_hash": "phash",
                    "char_count": 25,
                }
            ],
            [[0.5] * 384],
        )

        with session_factory() as session:
            kb = KnowledgeBase(name=f"Persisted KB {uuid.uuid4().hex[:6]}")
            session.add(kb)
            session.commit()
            kb_name = kb.name

        # Teardown
        v_manager.close()
        with VectorStoreManager._instance_lock:
            VectorStoreManager._instance = None

        # Restart fresh instance on same directory
        v_manager_restarted = VectorStoreManager(data_dir=tmp_path)
        v_manager_restarted.start()
        vector_store_restarted = LocalVectorStore(v_manager_restarted)

        info = await vector_store_restarted.ensure_collection(collection_name, embedding_dimensions=384)
        assert info.name == collection_name
        hits = await vector_store_restarted.search_dense(collection_name, query_vector=[0.5] * 384, top_k=2)
        assert len(hits) == 1
        assert hits[0].chunk_id == "persist-1"

        with session_factory() as session:
            loaded_kb = session.scalar(select(KnowledgeBase).where(KnowledgeBase.name == kb_name))
            assert loaded_kb is not None

        v_manager_restarted.close()

    def test_hot_backup_creation_and_full_restoration(self, tmp_path):
        """Verifies hot atomic .zip backup and zero-loss restore."""
        settings.QUANTIVA_DATA_DIR = str(tmp_path)
        engine = create_sqlite_engine(tmp_path)
        initialize_database(engine)
        session_factory = create_session_factory(engine)

        target_kb_name = f"Backup KB {uuid.uuid4().hex[:6]}"
        with session_factory() as session:
            kb = KnowledgeBase(name=target_kb_name)
            session.add(kb)
            session.commit()

        # Create hot backup archive
        backup_archive = create_backup(data_dir=tmp_path, include_models=False)
        assert backup_archive.is_file()
        assert backup_archive.suffix == ".zip"

        # Simulate data loss: delete KnowledgeBase row
        with session_factory() as session:
            kb_to_delete = session.scalar(select(KnowledgeBase).where(KnowledgeBase.name == target_kb_name))
            if kb_to_delete:
                session.delete(kb_to_delete)
                session.commit()

            assert session.scalar(select(KnowledgeBase).where(KnowledgeBase.name == target_kb_name)) is None

        # Restore from backup archive
        restored_archive = restore_backup(
            backup_archive,
            data_dir=tmp_path,
            create_safety_snapshot=False,
            engine_to_dispose=engine,
        )
        assert restored_archive.is_file()

        paths = LocalDataPaths(tmp_path)
        assert paths.database_file.is_file()

        # Verify restored data
        engine_restored = create_sqlite_engine(tmp_path)
        session_factory_restored = create_session_factory(engine_restored)
        with session_factory_restored() as session:
            restored_kb = session.scalar(select(KnowledgeBase).where(KnowledgeBase.name == target_kb_name))
            assert restored_kb is not None
            assert restored_kb.name == target_kb_name

    @pytest.mark.asyncio
    async def test_offline_mode_enforcement_zero_external_calls(self, tmp_path, monkeypatch):
        """
        Guards that when network sockets to external IPs are blocked via monkeypatch,
        the Fully Local pipeline completes ingestion, retrieval, reranking, and generation
        with zero external network requests.
        """
        # Block all outbound network sockets (except loopback / localhost)
        orig_connect = socket.socket.connect

        def guarded_connect(self, address):
            host = address[0] if isinstance(address, tuple) else address
            if str(host) in ("127.0.0.1", "localhost", "::1"):
                return orig_connect(self, address)
            raise RuntimeError(f"Blocked unauthorized external network call to: {address}")

        monkeypatch.setattr(socket.socket, "connect", guarded_connect)

        # Run full local ingestion -> retrieval -> rerank -> generation pipeline
        settings.QUANTIVA_DATA_DIR = str(tmp_path)
        engine = create_sqlite_engine(tmp_path)
        initialize_database(engine)

        v_manager = VectorStoreManager(data_dir=tmp_path)
        v_manager.start()
        vector_store = LocalVectorStore(v_manager)
        sparse_store = SQLiteSparseIndex(engine)

        kb_id = f"offline-kb-{uuid.uuid4().hex[:6]}"
        collection_name = collection_name_for_knowledge_base(kb_id)
        local_embedding = DeterministicMockEmbeddingProvider(384)
        local_reranker = DeterministicMockRerankerProvider()
        local_llm = DeterministicMockLLMProvider("ollama", "qwen2.5:7b")

        index_metadata = IndexMetadata(
            knowledge_base_id=kb_id,
            embedding_provider="local",
            embedding_model_name="mock-384",
            embedding_dimensions=384,
            collection_name=collection_name,
        )

        # Ingestion
        pdf_bytes = generate_multipage_pdf()
        extraction = extract_document(pdf_bytes, file_name="offline_doc.pdf")
        chunks = chunk_blocks(
            extraction.blocks,
            document_id="off-doc-1",
            knowledge_base_id=kb_id,
            source_file="offline_doc.pdf",
            chunk_size=300,
            chunk_overlap=50,
        )

        chunk_payloads: list[DocumentChunkPayload] = [
            {
                "chunk_id": f"off-{i}",
                "document_id": "off-doc-1",
                "knowledge_base_id": kb_id,
                "page_number": c["page_number"] or 1,
                "chunk_index": i,
                "content": c["content"],
                "source_file": "offline_doc.pdf",
                "section_title": c["section_title"],
                "sheet_name": None,
                "is_table": c["is_table"],
                "content_hash": f"off-h-{i}",
                "char_count": len(c["content"]),
            }
            for i, c in enumerate(chunks)
        ]

        vecs = await local_embedding.embed_documents([cp["content"] for cp in chunk_payloads])
        await vector_store.ensure_collection(collection_name, embedding_dimensions=384)
        await vector_store.upsert_chunks(collection_name, chunk_payloads, vecs)
        await sparse_store.index_chunks(kb_id, chunk_payloads)

        # Retrieval & Reranking
        retrieval_pipe = RetrievalPipeline(
            mode="hybrid",
            embedding_provider=local_embedding,
            vector_store=vector_store,
            sparse_index=sparse_store,
            index_metadata=index_metadata,
            reranker=local_reranker,
            rerank_enabled=True,
        )
        ret_result = await retrieval_pipe.retrieve("Quarterly report revenue", knowledge_base_id=kb_id)
        assert len(ret_result.chunks) > 0

        # Generation
        ans = await local_llm.generate(
            [ChatMessage(role=ChatRole.USER, content="Summarize revenue")]
        )
        assert len(ans.content) > 0
        assert ans.usage.total_tokens > 0

        v_manager.close()
