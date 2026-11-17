"""Quantiva document ingestion pipeline.

Public entry points::

    from app.ingestion import IngestionPipeline, OCRService

    pipeline = IngestionPipeline(embedding_provider, dense_indexer=..., sparse_indexer=...)
    result = await pipeline.run(
        file_bytes=raw,
        document_id=doc_id,
        knowledge_base_id=kb_id,
        file_name="report.pdf",
    )

The package is intentionally SDK-free except for parser/OCR libraries and
never imports a concrete LLM/embedding provider.
"""
from app.ingestion.chunker import chunk_blocks
from app.ingestion.cleaner import (
    DocumentHashRegistry,
    clean_blocks,
    clean_text,
    compute_content_hash,
    compute_file_hash,
    compute_file_hash_from_path,
    deduplicate_chunks,
)
from app.ingestion.extractor import extract_document, needs_ocr, validate_document
from app.ingestion.ocr import OCRService, render_pdf_page_to_image
from app.ingestion.parser import (
    CSV_MIME,
    DOCX_MIME,
    MARKDOWN_MIME,
    PDF_MIME,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MIME_TYPES,
    TXT_MIME,
    XLSX_MIME,
    detect_file_type,
)
from app.ingestion.pipeline import IngestionPipeline, ingest_document
from app.ingestion.types import (
    DocumentChunkPayload,
    DocumentValidationError,
    DuplicateDocumentError,
    ExtractionError,
    ExtractionResult,
    ExtractedBlock,
    IngestionError,
    IngestionResult,
    ProgressCallback,
    UnsupportedDocumentTypeError,
)

__all__ = [
    "CSV_MIME",
    "DOCX_MIME",
    "DocumentChunkPayload",
    "DocumentHashRegistry",
    "DocumentValidationError",
    "DuplicateDocumentError",
    "ExtractionError",
    "ExtractionResult",
    "ExtractedBlock",
    "IngestionError",
    "IngestionPipeline",
    "IngestionResult",
    "MARKDOWN_MIME",
    "OCRService",
    "PDF_MIME",
    "ProgressCallback",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_MIME_TYPES",
    "TXT_MIME",
    "UnsupportedDocumentTypeError",
    "XLSX_MIME",
    "chunk_blocks",
    "clean_blocks",
    "clean_text",
    "compute_content_hash",
    "compute_file_hash",
    "compute_file_hash_from_path",
    "deduplicate_chunks",
    "detect_file_type",
    "extract_document",
    "ingest_document",
    "needs_ocr",
    "render_pdf_page_to_image",
    "validate_document",
]
