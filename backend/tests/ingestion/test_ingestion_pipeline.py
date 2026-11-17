"""End-to-end and stage-level tests for ``app.ingestion``.

Coverage:
* PDF page numbers + embedded Markdown tables
* DOCX heading hierarchy + tables
* XLSX multi-chunk header repetition + summary metadata
* SHA-256 file duplicate detection
* chunk-level duplicate suppression
* canonical ``DocumentChunkPayload`` metadata contract
* OCR graceful fallback when Tesseract is unavailable
* TXT / Markdown / CSV support and structure-aware chunk boundaries
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import docx
import fitz
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from app.ingestion import (
    DocumentChunkPayload,
    DocumentHashRegistry,
    DuplicateDocumentError,
    ExtractionError,
    ExtractedBlock,
    IngestionPipeline,
    OCRService,
    chunk_blocks,
    clean_blocks,
    compute_content_hash,
    compute_file_hash,
    extract_document,
)
from app.ingestion.types import DocumentValidationError
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.types import ConnectionTestResult

CANONICAL_CHUNK_KEYS = {
    "chunk_id",
    "document_id",
    "knowledge_base_id",
    "page_number",
    "chunk_index",
    "content",
    "source_file",
    "section_title",
    "sheet_name",
    "is_table",
    "content_hash",
    "char_count",
}


# ── Fakes ────────────────────────────────────────────────────────────────────

class MockEmbeddingProvider(EmbeddingProvider):
    """No-network, deterministic embedding provider for pipeline tests."""

    def __init__(self, dimensions: int = 8) -> None:
        self._dimensions = dimensions
        self.calls: list[list[str]] = []

    def provider_name(self) -> str:
        return "mock"

    def model_name(self) -> str:
        return "mock-embedding"

    def get_dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts, batch_size=32):
        if not texts:
            raise ValueError("texts must be non-empty")
        if any(not text.strip() for text in texts):
            raise ValueError("blank texts are not embeddable")
        self.calls.append(list(texts))
        vectors = []
        for position, text in enumerate(texts):
            raw = [
                float((position + 1) * (channel + 1) + len(text) % 7)
                for channel in range(self._dimensions)
            ]
            norm = sum(value * value for value in raw) ** 0.5 or 1.0
            vectors.append([value / norm for value in raw])
        return vectors

    async def embed_query(self, query):
        return (await self.embed_documents([query]))[0]

    async def test_embedding(self):
        return ConnectionTestResult(
            success=True,
            latency_ms=0.1,
            provider="mock",
            model="mock-embedding",
            detected_dimensions=self._dimensions,
        )


class FakeDenseIndexer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[DocumentChunkPayload], list[list[float]]]] = []

    async def index_dense(self, chunks, embeddings):
        self.calls.append((list(chunks), [list(vector) for vector in embeddings]))
        return [f"dense-{chunk['chunk_index']}" for chunk in chunks]


class FakeSparseIndexer:
    def __init__(self) -> None:
        self.calls: list[list[DocumentChunkPayload]] = []

    async def index_sparse(self, chunks):
        self.calls.append(list(chunks))


# ── Fixtures / document builders ─────────────────────────────────────────────

def build_pdf() -> bytes:
    """Two-page PDF with prose and a ruled table on page 1."""
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 72), "Annual report introduction for page one.")
    x0, y0, x1, y1 = 72, 100, 400, 190
    for x in (72, 220, 400):
        page.draw_line((x, y0), (x, y1))
    for y in (100, 130, 160, 190):
        page.draw_line((x0, y), (x1, y))
    page.insert_text((82, 120), "Metric")
    page.insert_text((230, 120), "Value")
    page.insert_text((82, 150), "Revenue")
    page.insert_text((230, 150), "$10M")
    page.insert_text((82, 180), "Profit")
    page.insert_text((230, 180), "$2M")

    page_two = document.new_page(width=612, height=792)
    page_two.insert_text((72, 72), "Page two text goes here.")
    data = document.tobytes()
    document.close()
    return data


def build_docx() -> bytes:
    document = docx.Document()
    document.add_heading("Financial Overview", level=1)
    document.add_paragraph("Introductory paragraph.")
    document.add_heading("Q3 Revenue", level=2)
    document.add_paragraph("Q3 revenue text.")
    document.add_heading("Q4 Revenue", level=2)
    document.add_paragraph("Q4 revenue grew strongly.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Revenue"
    table.cell(1, 1).text = "$10M"
    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()


def build_xlsx(rows: int = 60) -> bytes:
    frame = pd.DataFrame(
        [
            {
                "Quarter": f"Q{index}",
                "Revenue": index * 100,
                "Profit": index * 10,
                "Note": f"note {index}",
            }
            for index in range(1, rows + 1)
        ]
    )
    stream = io.BytesIO()
    frame.to_excel(stream, index=False, sheet_name="Financials")
    return stream.getvalue()


def build_scanned_pdf() -> bytes:
    image = Image.new("RGB", (1000, 1300), "white")
    ImageDraw.Draw(image).text((80, 100), "SCANNED INVOICE TOTAL 42", fill="black")
    image_stream = io.BytesIO()
    image.save(image_stream, format="PNG")

    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(40, 40, 570, 750), stream=image_stream.getvalue())
    data = document.tobytes()
    document.close()
    return data


def table_header(content: str) -> list[str]:
    return [line for line in content.splitlines() if line.strip()][:2]


# ── PDF ──────────────────────────────────────────────────────────────────────

class TestPdfIngestion:
    def test_pdf_preserves_page_numbers_and_markdown_tables(self):
        result = extract_document(
            build_pdf(),
            file_name="annual-report.pdf",
            mime_type="application/pdf",
        )

        assert result.page_count == 2
        assert result.blocks
        page_numbers = {block.page_number for block in result.blocks}
        assert page_numbers == {1, 2}, "PDF page numbers must be 1-indexed and preserved"

        table_blocks = [
            block
            for block in result.blocks
            if block.block_type == "table" and block.page_number == 1
        ]
        assert table_blocks, "embedded PDF table was not extracted"
        table = table_blocks[0]
        assert table.text.startswith("|")
        assert "| --- |" in table.text
        assert "Revenue" in table.text and "$10M" in table.text
        assert table.table_headers and "Metric" in table.table_headers

        chunks = chunk_blocks(
            result.blocks,
            document_id="doc-pdf",
            knowledge_base_id="kb-pdf",
            source_file="annual-report.pdf",
        )
        table_chunks = [chunk for chunk in chunks if chunk["is_table"]]
        assert table_chunks
        assert all(chunk["page_number"] == 1 for chunk in table_chunks)
        assert any("Page two text" in chunk["content"] for chunk in chunks)

    def test_scanned_pdf_is_marked_for_ocr(self):
        result = extract_document(
            build_scanned_pdf(),
            file_name="scan.pdf",
            mime_type="application/pdf",
        )
        assert result.ocr_candidates == [1]


# ── DOCX ─────────────────────────────────────────────────────────────────────

class TestDocxIngestion:
    def test_docx_preserves_heading_hierarchy_and_tables(self):
        result = extract_document(
            build_docx(),
            file_name="report.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        q4_blocks = [
            block
            for block in result.blocks
            if block.section_path
            and "Q4 Revenue" in block.section_path
        ]
        assert q4_blocks
        assert q4_blocks[0].section_path == (
            "1. Financial Overview > 1.2 Q4 Revenue"
        )

        table_blocks = [block for block in result.blocks if block.block_type == "table"]
        assert len(table_blocks) == 1
        table = table_blocks[0]
        assert table.section_path == "1. Financial Overview > 1.2 Q4 Revenue"
        assert table.text.splitlines()[0] == "| Metric | Value |"
        assert "Revenue" in table.text and "$10M" in table.text


# ── XLSX / CSV ───────────────────────────────────────────────────────────────

class TestSpreadsheetIngestion:
    def test_xlsx_multi_chunk_repeats_header_and_keeps_summary_metadata(self):
        result = extract_document(
            build_xlsx(rows=60),
            file_name="finance.xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        assert result.sheet_names == ["Financials"]
        summary = result.metadata["sheets"]["Financials"]
        assert summary["row_count"] == 60
        assert summary["column_names"] == ["Quarter", "Revenue", "Profit", "Note"]
        assert summary["numeric_columns"] == ["Revenue", "Profit"]
        assert result.metadata["row_count"] == 60

        block = result.blocks[0]
        assert block.sheet_name == "Financials"
        assert block.metadata["row_count"] == 60
        assert block.metadata["column_names"] == ["Quarter", "Revenue", "Profit", "Note"]
        assert block.metadata["numeric_columns"] == ["Revenue", "Profit"]

        chunks = chunk_blocks(
            result.blocks,
            document_id="doc-xlsx",
            knowledge_base_id="kb-xlsx",
            source_file="finance.xlsx",
            chunk_size=220,
        )
        assert len(chunks) > 1, "large spreadsheet must span multiple chunks"
        assert all(chunk["is_table"] for chunk in chunks)
        assert all(chunk["sheet_name"] == "Financials" for chunk in chunks)
        for chunk in chunks:
            header, divider = table_header(chunk["content"])
            assert header == "| Quarter | Revenue | Profit | Note |", (
                "every spreadsheet chunk must repeat the column header row"
            )
            assert divider == "| --- | --- | --- | --- |"
        combined = "\n".join(chunk["content"] for chunk in chunks)
        assert "| Q1 | 100 | 10 | note 1 |" in combined
        assert "| Q60 | 6000 | 600 | note 60 |" in combined

    def test_csv_extraction_keeps_headers_and_numeric_metadata(self):
        payload = b"Quarter,Revenue,Profit\nQ1,100,20\nQ2,150,35\n"
        result = extract_document(
            payload,
            file_name="revenue.csv",
            mime_type="text/csv",
        )
        block = result.blocks[0]
        assert block.block_type == "table"
        assert block.table_headers == ("Quarter", "Revenue", "Profit")
        assert block.metadata["row_count"] == 2
        assert block.metadata["numeric_columns"] == ["Revenue", "Profit"]
        assert result.metadata["sheets"][""]["column_names"] == [
            "Quarter",
            "Revenue",
            "Profit",
        ]


# ── Hashing / duplicates ─────────────────────────────────────────────────────

class TestHashingAndDuplicates:
    def test_sha256_file_hash_is_stable_and_content_sensitive(self):
        first = compute_file_hash(b"same bytes")
        second = compute_file_hash(b"same bytes")
        other = compute_file_hash(b"different bytes")
        assert first == second
        assert first != other
        assert re.fullmatch(r"[0-9a-f]{64}", first)

    async def test_pipeline_rejects_duplicate_document_by_sha256(self):
        csv_bytes = b"Quarter,Revenue\nQ1,100\n"
        content_hash = compute_file_hash(csv_bytes)
        registry = DocumentHashRegistry()
        registry.add(content_hash, "existing-doc-id")

        pipeline = IngestionPipeline(
            MockEmbeddingProvider(),
            existing_document_lookup=registry,
        )
        with pytest.raises(DuplicateDocumentError) as error:
            await pipeline.run(
                file_bytes=csv_bytes,
                document_id="new-doc-id",
                knowledge_base_id="kb",
                file_name="duplicate.csv",
                mime_type="text/csv",
            )
        assert error.value.existing_document_id == "existing-doc-id"
        assert error.value.content_hash == content_hash

    def test_chunk_level_duplicates_are_deduplicated(self):
        blocks = [
            ExtractedBlock(text="The same paragraph appears twice."),
            ExtractedBlock(text="  the same   paragraph appears twice.  "),
        ]
        assert len(clean_blocks(blocks)) == 1

        chunks = chunk_blocks(
            [
                ExtractedBlock(text="Identical chunk content."),
                ExtractedBlock(text="Identical chunk content."),
            ],
            document_id="doc",
            knowledge_base_id="kb",
            source_file="x.txt",
        )
        assert len(chunks) == 1
        assert chunks[0]["content_hash"] == compute_content_hash("Identical chunk content.")


# ── Pipeline contract ────────────────────────────────────────────────────────

class TestPipelineContract:
    async def test_metadata_contract_and_stage_order(self):
        csv_bytes = (
            b"Quarter,Revenue,Profit\n"
            b"Q1,100,20\nQ2,150,35\nQ3,175,40\n"
        )
        embeddings = MockEmbeddingProvider(dimensions=8)
        dense = FakeDenseIndexer()
        sparse = FakeSparseIndexer()
        progress: list[tuple[str, float, str]] = []

        pipeline = IngestionPipeline(
            embeddings,
            dense_indexer=dense,
            sparse_indexer=sparse,
            chunk_size=160,
            chunk_overlap=20,
            embed_batch_size=2,
        )
        result = await pipeline.run(
            file_bytes=csv_bytes,
            document_id="doc-1",
            knowledge_base_id="kb-1",
            file_name="revenue.csv",
            mime_type="text/csv",
            progress_callback=lambda stage, percent, message: progress.append(
                (stage, percent, message)
            ),
        )

        expected_stages = [
            "validate",
            "extract",
            "ocr_if_needed",
            "clean",
            "chunk",
            "embed",
            "index_dense",
            "index_sparse",
        ]
        assert result.stages_completed == expected_stages
        assert [entry[0] for entry in progress][0] == "validate"
        assert [entry[0] for entry in progress][-1] == "index_sparse"
        assert progress[-1][1] == 100.0
        percents = [entry[1] for entry in progress]
        assert percents == sorted(percents), "progress must be monotonic"

        assert result.chunks
        assert len(result.embeddings) == len(result.chunks)
        assert all(len(vector) == 8 for vector in result.embeddings)
        assert dense.calls and dense.calls[0][0] == result.chunks
        assert len(dense.calls[0][1]) == len(result.chunks)
        assert sparse.calls and sparse.calls[0] == result.chunks
        assert result.sparse_indexed is True
        assert result.dense_point_ids == [
            f"dense-{chunk['chunk_index']}" for chunk in result.chunks
        ]
        assert result.file_hash == compute_file_hash(csv_bytes)

        for position, chunk in enumerate(result.chunks):
            assert set(chunk) == CANONICAL_CHUNK_KEYS
            assert chunk["chunk_index"] == position
            assert chunk["document_id"] == "doc-1"
            assert chunk["knowledge_base_id"] == "kb-1"
            assert chunk["source_file"] == "revenue.csv"
            assert chunk["content"]
            assert chunk["char_count"] == len(chunk["content"])
            assert re.fullmatch(r"[0-9a-f]{64}", chunk["content_hash"])
            assert chunk["chunk_id"]
            assert chunk["is_table"] is True
            assert chunk["sheet_name"] is None
        assert len({chunk["chunk_id"] for chunk in result.chunks}) == len(result.chunks)

    async def test_empty_document_is_rejected(self):
        pipeline = IngestionPipeline(MockEmbeddingProvider())
        with pytest.raises(ExtractionError):
            await pipeline.run(
                file_bytes=b"",
                document_id="doc",
                knowledge_base_id="kb",
                file_name="empty.txt",
                mime_type="text/plain",
            )

    def test_invalid_pdf_signature_is_rejected(self):
        with pytest.raises(DocumentValidationError):
            extract_document(
                b"not really a pdf",
                file_name="fake.pdf",
                mime_type="application/pdf",
            )


# ── OCR fallback ─────────────────────────────────────────────────────────────

class TestOcrFallback:
    def test_missing_tesseract_adds_warning_metadata_without_failing(self, monkeypatch):
        result = extract_document(
            build_scanned_pdf(),
            file_name="scan.pdf",
            mime_type="application/pdf",
        )
        assert result.ocr_candidates == [1]

        monkeypatch.setattr(OCRService, "_locate_tesseract", lambda self: None)
        service = OCRService(enabled=True)
        updated = service.apply(b"fake-pdf-bytes", result)

        assert updated is result
        assert updated.ocr_applied is False
        assert updated.metadata["ocr"]["status"] == "unavailable"
        assert updated.metadata["ocr"]["tesseract_path"] is None
        assert any("OCR unavailable" in warning for warning in updated.warnings)

    def test_disabled_ocr_adds_warning_metadata(self):
        result = extract_document(
            build_scanned_pdf(),
            file_name="scan.pdf",
            mime_type="application/pdf",
        )
        service = OCRService(enabled=False)
        service.apply(b"fake-pdf-bytes", result)
        assert result.ocr_applied is False
        assert result.metadata["ocr"]["status"] == "disabled"
        assert any("OCR is disabled" in warning for warning in result.warnings)

    def test_ocr_success_uses_pypdfium2_rendering(self, monkeypatch):
        result = extract_document(
            build_scanned_pdf(),
            file_name="scan.pdf",
            mime_type="application/pdf",
        )
        # Pretend a bundled Tesseract exists; avoid invoking the real binary.
        monkeypatch.setattr(
            OCRService,
            "_locate_tesseract",
            lambda self: Path(sys.executable),
        )
        monkeypatch.setattr(
            OCRService,
            "ocr_image",
            lambda self, image: f"OCR INVOICE TOTAL 42 ({image.width}x{image.height})",
        )
        service = OCRService(enabled=True)
        service.apply(build_scanned_pdf(), result)

        assert result.ocr_applied is True
        assert result.metadata["ocr"]["status"] == "applied"
        assert result.metadata["ocr"]["pages_ocr"] == [1]
        ocr_blocks = [
            block
            for block in result.blocks
            if block.metadata.get("ocr") is True
        ]
        assert len(ocr_blocks) == 1
        assert ocr_blocks[0].page_number == 1
        assert "INVOICE TOTAL 42" in ocr_blocks[0].text


# ── Structure-aware chunking / remaining formats ─────────────────────────────

class TestChunkBoundariesAndRemainingFormats:
    def test_small_code_block_is_not_split(self):
        code = '```python\nprint("hello")\n```'
        chunks = chunk_blocks(
            [ExtractedBlock(text=code, block_type="code")],
            document_id="doc",
            knowledge_base_id="kb",
            source_file="note.md",
            chunk_size=512,
        )
        assert len(chunks) == 1
        assert chunks[0]["content"] == code

    def test_large_code_block_split_keeps_fences_balanced(self):
        body = "\n".join(f"value_{index} = {index}" for index in range(80))
        code = f"```python\n{body}\n```"
        chunks = chunk_blocks(
            [ExtractedBlock(text=code, block_type="code")],
            document_id="doc",
            knowledge_base_id="kb",
            source_file="note.md",
            chunk_size=120,
        )
        assert len(chunks) > 1
        for chunk in chunks:
            content = chunk["content"]
            assert content.startswith("```python")
            assert content.endswith("```")
            assert len(content) <= 120

    def test_small_markdown_table_inside_prose_is_not_split(self):
        text = (
            "The table below summarises the result.\n\n"
            "| Metric | Value |\n| --- | --- |\n| Revenue | 10 |\n"
        )
        chunks = chunk_blocks(
            [ExtractedBlock(text=text)],
            document_id="doc",
            knowledge_base_id="kb",
            source_file="note.md",
            chunk_size=512,
        )
        table_chunks = [chunk for chunk in chunks if "| Metric | Value |" in chunk["content"]]
        assert table_chunks
        assert "| --- | --- |" in table_chunks[0]["content"]
        assert "| Revenue | 10 |" in table_chunks[0]["content"]

    def test_markdown_extraction_preserves_headings_code_and_tables(self):
        markdown = (
            b"# Overview\n\nSome prose.\n\n"
            b"## Revenue\n\n"
            b"| Quarter | Revenue |\n| --- | --- |\n| Q1 | 10 |\n\n"
            b"```python\nprint('hello')\n```\n"
        )
        result = extract_document(
            markdown,
            file_name="notes.md",
            mime_type="text/markdown",
        )
        block_types = [block.block_type for block in result.blocks]
        assert "table" in block_types
        assert "code" in block_types
        assert any(
            block.section_path == "1. Overview > 1.1 Revenue"
            for block in result.blocks
        )

    def test_plain_text_extraction_returns_single_page_block(self):
        result = extract_document(
            b"Plain text document.\n\nSecond paragraph.",
            file_name="notes.txt",
            mime_type="text/plain",
        )
        assert len(result.blocks) == 1
        assert result.blocks[0].page_number == 1
        assert "Second paragraph" in result.blocks[0].text
