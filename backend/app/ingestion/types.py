"""Shared, SDK-free types for the Quantiva document ingestion pipeline.

This module deliberately imports no parser, OCR, embedding, or vector-store
library.  Everything in :mod:`app.ingestion` exchanges these small dataclasses
so the individual stages stay independently testable and provider-agnostic.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict


# ── Canonical chunk contract ──────────────────────────────────────────────────

class DocumentChunkPayload(TypedDict):
    """The canonical metadata contract emitted by ``chunker.py``.

    Every value stored in a vector collection or SQLite chunk row must be
    derived from one of these payloads.  The keys are intentionally stable:
    retrieval, citations, and the local job manager all consume them.
    """

    chunk_id: str
    document_id: str
    knowledge_base_id: str
    page_number: int | None
    chunk_index: int
    content: str
    source_file: str
    section_title: str | None
    sheet_name: str | None
    is_table: bool
    content_hash: str
    char_count: int


BlockType = Literal["text", "table", "code"]


@dataclass(slots=True)
class ExtractedBlock:
    """One structured unit produced by a format parser.

    ``text`` is always the exact payload that will be chunked.  For tables it
    is a GitHub-flavoured Markdown table.  Structural fields are kept beside
    the text rather than embedded inside it so the chunker can repeat table
    headers and preserve section paths without re-parsing prose.
    """

    text: str
    block_type: BlockType = "text"
    page_number: int | None = None
    section_path: str | None = None
    section_title: str | None = None
    sheet_name: str | None = None
    row_range: str | None = None
    table_headers: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""


@dataclass(slots=True)
class ExtractionResult:
    """Normalised output of one document extraction (before cleaning/chunking)."""

    blocks: list[ExtractedBlock]
    file_name: str
    file_type: str
    file_hash: str
    byte_size: int
    page_count: int | None = None
    sheet_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    #: 1-based PDF pages that contain images but too little extracted text.
    ocr_candidates: list[int] = field(default_factory=list)
    ocr_applied: bool = False


@dataclass(slots=True)
class IngestionResult:
    """End-to-end result returned by :class:`app.ingestion.pipeline.IngestionPipeline`."""

    document_id: str
    knowledge_base_id: str
    file_name: str
    file_type: str
    file_hash: str
    chunks: list[DocumentChunkPayload]
    embeddings: list[list[float]]
    dense_point_ids: list[str] = field(default_factory=list)
    sparse_indexed: bool = False
    warnings: list[str] = field(default_factory=list)
    stages_completed: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Progress reporting ────────────────────────────────────────────────────────

ProgressCallback = Callable[[str, float, str], None]
"""Receives ``(stage, percent, message)``; percent is 0..100."""


# ── Injected indexing interfaces ─────────────────────────────────────────────

class DenseIndexer(Protocol):
    """Minimal dense-vector indexing contract.

    The pipeline never imports a concrete vector database.  The application
    context supplies an adapter around the singleton embedded Qdrant/LanceDB
    client.  Implementations may be sync or async.
    """

    def index_dense(
        self,
        chunks: Sequence[DocumentChunkPayload],
        embeddings: Sequence[list[float]],
    ) -> Awaitable[list[str] | None] | list[str] | None:
        ...


class SparseIndexer(Protocol):
    """Minimal sparse/keyword indexing contract (FTS5 or BM25)."""

    def index_sparse(
        self,
        chunks: Sequence[DocumentChunkPayload],
    ) -> Awaitable[Any] | Any:
        ...


# ── Error hierarchy ───────────────────────────────────────────────────────────

class IngestionError(Exception):
    """Base class for all document-ingestion failures."""


class UnsupportedDocumentTypeError(IngestionError):
    """Raised when no parser exists for the supplied extension/MIME type."""


class DocumentValidationError(IngestionError):
    """Raised when bytes do not match the declared document format."""


class ExtractionError(IngestionError):
    """Raised when a valid document cannot yield any usable content."""


class DuplicateDocumentError(IngestionError):
    """Raised when the SHA-256 of the file already belongs to a document.

    File-level hashing is the cheap first line of duplicate detection.  The
    caller supplies an existing-document lookup to the pipeline; this error is
    only raised after that lookup returns a document id.
    """

    def __init__(
        self,
        content_hash: str,
        existing_document_id: str,
        *,
        file_name: str | None = None,
    ) -> None:
        self.content_hash = content_hash
        self.existing_document_id = existing_document_id
        self.file_name = file_name
        label = f" '{file_name}'" if file_name else ""
        super().__init__(
            f"Document{label} already exists with SHA-256 {content_hash} "
            f"(existing document id: {existing_document_id})."
        )
