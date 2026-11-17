"""End-to-end ingestion orchestration.

Stage order is fixed and observable:

``validate -> extract -> ocr_if_needed -> clean -> chunk -> embed ->
index_dense -> index_sparse``

The pipeline owns no provider SDKs and no vector-store clients.  The embedding
provider is injected (an :class:`~app.providers.interfaces.embedding.EmbeddingProvider`)
and dense/sparse indexers are injected adapters.  Progress callbacks receive
``(stage, percent, message)`` so the local job manager can surface progress in
the desktop UI.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from app.ingestion.chunker import chunk_blocks
from app.ingestion.cleaner import clean_blocks, compute_file_hash
from app.ingestion.extractor import extract_document
from app.ingestion.ocr import OCRService
from app.ingestion.types import (
    DenseIndexer,
    DocumentChunkPayload,
    DuplicateDocumentError,
    ExtractionError,
    IngestionResult,
    ProgressCallback,
    SparseIndexer,
)
from app.providers.interfaces.embedding import EmbeddingProvider

__all__ = ["IngestionPipeline", "ingest_document"]

logger = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class IngestionPipeline:
    """Provider-agnostic document ingestion pipeline.

    Parameters
    ----------
    embedding_provider:
        Injected embedding implementation.  Never constructed here.
    dense_indexer / sparse_indexer:
        Optional adapters around the singleton embedded vector store and the
        SQLite FTS5/BM25 index.  Calls may be sync or async.
    ocr_service:
        Optional OCR service.  A default Tesseract-backed instance is used
        when omitted; it degrades to warnings when Tesseract is unavailable.
    existing_document_lookup:
        Optional ``Callable[[sha256_hex], document_id | None]`` used for
        file-level duplicate detection.  Sync and async callables are accepted.
        The pipeline never queries the DB directly.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        dense_indexer: DenseIndexer | None = None,
        sparse_indexer: SparseIndexer | None = None,
        ocr_service: OCRService | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embed_batch_size: int = 32,
        max_file_bytes: int | None = None,
        existing_document_lookup: (
            Callable[[str], str | None | Awaitable[str | None]] | None
        ) = None,
    ) -> None:
        if embedding_provider is None:
            raise ValueError("An injected EmbeddingProvider is required")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if embed_batch_size <= 0:
            raise ValueError("embed_batch_size must be positive")

        self.embedding_provider = embedding_provider
        self.dense_indexer = dense_indexer
        self.sparse_indexer = sparse_indexer
        self.ocr_service = ocr_service or self._default_ocr_service()
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.embed_batch_size = int(embed_batch_size)
        self.max_file_bytes = max_file_bytes
        self.existing_document_lookup = existing_document_lookup

    # ── Defaults ──────────────────────────────────────────────────────────

    @staticmethod
    def _default_ocr_service() -> OCRService:
        """Build OCRService from desktop settings when available."""
        try:
            from app.config import settings

            return OCRService(
                enabled=bool(getattr(settings, "OCR_ENABLED", True)),
                tesseract_cmd=getattr(settings, "OCR_TESSERACT_CMD", None),
                languages=getattr(settings, "OCR_LANGUAGES", "eng"),
                dpi=getattr(settings, "OCR_DPI", 200),
                min_text_chars=getattr(settings, "OCR_MIN_TEXT_CHARS", 30),
            )
        except Exception:  # pragma: no cover - settings optional for library use
            return OCRService()

    # ── Progress ──────────────────────────────────────────────────────────

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        stages: list[str],
        stage: str,
        percent: float,
        message: str,
    ) -> None:
        if stage not in stages:
            stages.append(stage)
        if callback is None:
            return
        try:
            callback(stage, round(float(percent), 2), message)
        except Exception:  # pragma: no cover - callbacks must not abort ingestion
            logger.warning("Ingestion progress callback failed at stage %s", stage, exc_info=True)

    # ── Embedding ─────────────────────────────────────────────────────────

    async def _embed_chunks(
        self,
        chunks: Sequence[DocumentChunkPayload],
        callback: ProgressCallback | None,
        stages: list[str],
    ) -> list[list[float]]:
        texts = [chunk["content"] for chunk in chunks]
        if not texts:
            return []

        dimensions = int(self.embedding_provider.get_dimensions())
        if dimensions <= 0:
            raise ValueError("Embedding provider reported invalid dimensions")

        embeddings: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, self.embed_batch_size):
            batch = texts[start : start + self.embed_batch_size]
            vectors = await self.embedding_provider.embed_documents(
                batch,
                batch_size=len(batch),
            )
            if len(vectors) != len(batch):
                raise ValueError(
                    "Embedding provider returned "
                    f"{len(vectors)} vectors for {len(batch)} texts"
                )
            for vector in vectors:
                self.embedding_provider.validate_dimensions(vector)
                if len(vector) != dimensions:
                    raise ValueError("Embedding vector dimension mismatch")
                embeddings.append(list(vector))

            completed = min(total, start + len(batch))
            self._emit(
                callback,
                stages,
                "embed",
                60.0 + 25.0 * (completed / total),
                f"Embedded {completed}/{total} chunks",
            )

        if len(embeddings) != total:
            raise ValueError(
                f"Embedding stage produced {len(embeddings)} vectors for {total} chunks"
            )
        return embeddings

    # ── Indexing ──────────────────────────────────────────────────────────

    async def _index_dense(
        self,
        chunks: Sequence[DocumentChunkPayload],
        embeddings: Sequence[list[float]],
        callback: ProgressCallback | None,
        stages: list[str],
    ) -> list[str]:
        if self.dense_indexer is None:
            self._emit(
                callback,
                stages,
                "index_dense",
                90.0,
                "No dense indexer configured; skipped dense indexing",
            )
            return []
        result = await _maybe_await(
            self.dense_indexer.index_dense(list(chunks), [list(v) for v in embeddings])
        )
        self._emit(
            callback,
            stages,
            "index_dense",
            90.0,
            f"Dense index updated for {len(chunks)} chunks",
        )
        if result is None:
            return []
        return [str(point_id) for point_id in result]

    async def _index_sparse(
        self,
        chunks: Sequence[DocumentChunkPayload],
        callback: ProgressCallback | None,
        stages: list[str],
    ) -> bool:
        if self.sparse_indexer is None:
            self._emit(
                callback,
                stages,
                "index_sparse",
                97.0,
                "No sparse indexer configured; skipped sparse indexing",
            )
            return False
        await _maybe_await(self.sparse_indexer.index_sparse(list(chunks)))
        self._emit(
            callback,
            stages,
            "index_sparse",
            97.0,
            f"Sparse index updated for {len(chunks)} chunks",
        )
        return True

    # ── Main entry point ──────────────────────────────────────────────────

    async def run(
        self,
        file_bytes: bytes,
        *,
        document_id: str,
        knowledge_base_id: str,
        file_name: str | None = None,
        mime_type: str | None = None,
        progress_callback: ProgressCallback | None = None,
        allow_duplicates: bool = False,
    ) -> IngestionResult:
        """Run all ingestion stages for one document.

        Raises
        ------
        DuplicateDocumentError
            When the supplied lookup reports the file SHA-256 already exists.
        DocumentValidationError / UnsupportedDocumentTypeError / ExtractionError
            When the file cannot be read as a supported document.
        """
        if not document_id:
            raise ValueError("document_id is required")
        if not knowledge_base_id:
            raise ValueError("knowledge_base_id is required")
        if not file_bytes:
            raise ExtractionError("Empty documents are not supported")

        stages: list[str] = []
        warnings: list[str] = []

        # ── validate ─────────────────────────────────────────────────────
        self._emit(progress_callback, stages, "validate", 2.0, "Validating document")
        file_hash = compute_file_hash(file_bytes)
        if self.existing_document_lookup is not None and not allow_duplicates:
            existing_id = await _maybe_await(self.existing_document_lookup(file_hash))
            if existing_id:
                raise DuplicateDocumentError(
                    file_hash,
                    str(existing_id),
                    file_name=file_name,
                )
        self._emit(progress_callback, stages, "validate", 5.0, "Document validated")

        # ── extract ──────────────────────────────────────────────────────
        self._emit(progress_callback, stages, "extract", 8.0, "Extracting content")
        extraction = extract_document(
            file_bytes,
            file_name=file_name,
            mime_type=mime_type,
            file_hash=file_hash,
            max_bytes=self.max_file_bytes,
            ocr_min_text_chars=self.ocr_service.min_text_chars,
        )
        warnings.extend(extraction.warnings)
        self._emit(
            progress_callback,
            stages,
            "extract",
            20.0,
            f"Extracted {len(extraction.blocks)} blocks",
        )

        # ── ocr_if_needed ────────────────────────────────────────────────
        if extraction.ocr_candidates:
            self._emit(
                progress_callback,
                stages,
                "ocr_if_needed",
                22.0,
                f"OCR fallback for page(s) {extraction.ocr_candidates}",
            )
            await asyncio.to_thread(self.ocr_service.apply, file_bytes, extraction)
            self._emit(
                progress_callback,
                stages,
                "ocr_if_needed",
                35.0,
                "OCR stage complete"
                if extraction.ocr_applied
                else "OCR was not applied; see warnings",
            )
        else:
            self._emit(
                progress_callback,
                stages,
                "ocr_if_needed",
                35.0,
                "OCR not required",
            )

        # ── clean ────────────────────────────────────────────────────────
        self._emit(progress_callback, stages, "clean", 40.0, "Cleaning extracted text")
        extraction.blocks = clean_blocks(extraction.blocks)
        if not extraction.blocks:
            detail = "; ".join(extraction.warnings)
            raise ExtractionError(
                "Document produced no usable text after cleaning"
                + (f": {detail}" if detail else "")
            )
        self._emit(
            progress_callback,
            stages,
            "clean",
            50.0,
            f"Kept {len(extraction.blocks)} non-duplicate blocks",
        )

        # ── chunk ────────────────────────────────────────────────────────
        self._emit(progress_callback, stages, "chunk", 52.0, "Chunking document")
        chunks = chunk_blocks(
            extraction.blocks,
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            source_file=extraction.file_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if not chunks:
            raise ExtractionError("Document produced no chunkable content")
        self._emit(
            progress_callback,
            stages,
            "chunk",
            60.0,
            f"Created {len(chunks)} chunks",
        )

        # ── embed ────────────────────────────────────────────────────────
        self._emit(progress_callback, stages, "embed", 60.0, "Generating embeddings")
        embeddings = await self._embed_chunks(chunks, progress_callback, stages)
        self._emit(
            progress_callback,
            stages,
            "embed",
            85.0,
            f"Generated {len(embeddings)} embeddings",
        )

        # ── index_dense ──────────────────────────────────────────────────
        self._emit(progress_callback, stages, "index_dense", 86.0, "Indexing dense vectors")
        dense_ids = await self._index_dense(
            chunks,
            embeddings,
            progress_callback,
            stages,
        )

        # ── index_sparse ─────────────────────────────────────────────────
        self._emit(progress_callback, stages, "index_sparse", 94.0, "Indexing keywords")
        sparse_indexed = await self._index_sparse(
            chunks,
            progress_callback,
            stages,
        )

        # Copy OCR warnings that were appended to the extraction result.
        warnings.extend(
            warning
            for warning in extraction.warnings
            if warning not in warnings
        )
        metadata = {
            **extraction.metadata,
            "file_name": extraction.file_name,
            "file_type": extraction.file_type,
            "byte_size": extraction.byte_size,
            "page_count": extraction.page_count,
            "sheet_names": extraction.sheet_names,
            "ocr": extraction.metadata.get("ocr"),
            "chunk_count": len(chunks),
            "embedding_provider": self.embedding_provider.provider_name(),
            "embedding_model": self.embedding_provider.model_name(),
            "embedding_dimensions": self.embedding_provider.get_dimensions(),
        }
        self._emit(
            progress_callback,
            stages,
            "index_sparse",
            100.0,
            "Ingestion complete",
        )

        return IngestionResult(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            file_name=extraction.file_name,
            file_type=extraction.file_type,
            file_hash=file_hash,
            chunks=chunks,
            embeddings=embeddings,
            dense_point_ids=dense_ids,
            sparse_indexed=sparse_indexed,
            warnings=warnings,
            stages_completed=stages,
            metadata=metadata,
        )


async def ingest_document(
    file_bytes: bytes,
    *,
    document_id: str,
    knowledge_base_id: str,
    embedding_provider: EmbeddingProvider,
    file_name: str | None = None,
    mime_type: str | None = None,
    dense_indexer: DenseIndexer | None = None,
    sparse_indexer: SparseIndexer | None = None,
    progress_callback: ProgressCallback | None = None,
    **pipeline_options: Any,
) -> IngestionResult:
    """Small functional wrapper around :class:`IngestionPipeline`."""
    pipeline = IngestionPipeline(
        embedding_provider,
        dense_indexer=dense_indexer,
        sparse_indexer=sparse_indexer,
        **pipeline_options,
    )
    return await pipeline.run(
        file_bytes=file_bytes,
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        file_name=file_name,
        mime_type=mime_type,
        progress_callback=progress_callback,
    )
