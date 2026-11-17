"""Validation, format dispatch, and normalisation of extracted documents.

``extractor.py`` sits between the raw bytes and the cleaning/chunking stages.
It performs cheap structural validation (magic bytes / ZIP members), chooses the
right parser, and returns one :class:`ExtractionResult` carrying file hashes,
warnings, OCR candidates, and sheet/page metadata.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import PurePath

from app.ingestion.cleaner import compute_file_hash
from app.ingestion.parser import (
    CSV_MIME,
    DOCX_MIME,
    MARKDOWN_MIME,
    PDF_MIME,
    TXT_MIME,
    XLSX_MIME,
    ParserOutput,
    detect_file_type,
    parse_csv,
    parse_docx,
    parse_markdown,
    parse_pdf,
    parse_text,
    parse_xlsx,
)
from app.ingestion.types import (
    DocumentValidationError,
    ExtractionError,
    ExtractionResult,
)

__all__ = ["extract_document", "needs_ocr", "validate_document"]


def _max_upload_bytes() -> int | None:
    try:
        from app.config import settings

        return int(settings.MAX_UPLOAD_BYTES)
    except Exception:  # pragma: no cover - extraction remains usable standalone
        return None


def validate_document(
    file_bytes: bytes,
    *,
    file_name: str | None = None,
    mime_type: str | None = None,
    max_bytes: int | None = None,
) -> str:
    """Validate upload bytes and return the resolved MIME type.

    Checks are intentionally format-structural rather than content-semantic:
    the parsers themselves raise ``ExtractionError`` for damaged internals.
    """
    if not isinstance(file_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("validate_document expects bytes")
    data = bytes(file_bytes)
    if not data:
        raise DocumentValidationError("Empty documents are not supported")

    limit = _max_upload_bytes() if max_bytes is None else max_bytes
    if limit is not None and len(data) > limit:
        raise DocumentValidationError(
            f"Document exceeds the {limit} byte ingestion limit"
        )

    file_type = detect_file_type(file_name, mime_type)

    if file_type == PDF_MIME:
        if b"%PDF-" not in data[:1024]:
            raise DocumentValidationError("Invalid PDF signature")
        return file_type

    if file_type in {DOCX_MIME, XLSX_MIME}:
        expected_member = (
            "word/document.xml" if file_type == DOCX_MIME else "xl/workbook.xml"
        )
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
                if "[Content_Types].xml" not in names or expected_member not in names:
                    raise DocumentValidationError(
                        "Office document does not contain the expected XML parts"
                    )
                if any(info.flag_bits & 1 for info in archive.infolist()):
                    raise DocumentValidationError("Encrypted Office documents are not supported")
        except zipfile.BadZipFile as exc:
            raise DocumentValidationError("Invalid or corrupted Office document archive") from exc
        return file_type

    if file_type in {CSV_MIME, TXT_MIME, MARKDOWN_MIME}:
        if b"\x00" in data:
            # UTF-16 text legitimately contains NUL bytes; only reject binary
            # content when it does not look like a UTF-16 BOM-prefixed stream.
            if not data.startswith((b"\xff\xfe", b"\xfe\xff")):
                raise DocumentValidationError("Text documents must not contain binary NUL bytes")
        return file_type

    raise DocumentValidationError(f"Unsupported document type: {file_type}")


def _dispatch(
    file_bytes: bytes,
    file_type: str,
    file_name: str,
    *,
    ocr_min_text_chars: int,
) -> ParserOutput:
    if file_type == PDF_MIME:
        return parse_pdf(file_bytes, min_text_chars=ocr_min_text_chars)
    if file_type == DOCX_MIME:
        return parse_docx(file_bytes)
    if file_type == XLSX_MIME:
        return parse_xlsx(file_bytes)
    if file_type == CSV_MIME:
        return parse_csv(file_bytes)
    if file_type == MARKDOWN_MIME:
        return parse_markdown(file_bytes, file_name=file_name)
    if file_type == TXT_MIME:
        return parse_text(file_bytes, file_name=file_name)
    raise ExtractionError(f"No parser registered for '{file_type}'")


def extract_document(
    file_bytes: bytes,
    *,
    file_name: str | None = None,
    mime_type: str | None = None,
    file_hash: str | None = None,
    max_bytes: int | None = None,
    ocr_min_text_chars: int = 30,
) -> ExtractionResult:
    """Validate and extract a supported document into structured blocks."""
    file_type = validate_document(
        file_bytes,
        file_name=file_name,
        mime_type=mime_type,
        max_bytes=max_bytes,
    )
    source_name = PurePath(file_name).name if file_name else "document"
    content_hash = file_hash or compute_file_hash(file_bytes)

    parsed = _dispatch(
        bytes(file_bytes),
        file_type,
        source_name,
        ocr_min_text_chars=ocr_min_text_chars,
    )
    ocr_candidates = sorted(set(parsed.ocr_candidates))
    if not parsed.blocks and not ocr_candidates:
        detail = "; ".join(parsed.warnings) or "no readable content was found"
        raise ExtractionError(f"Document produced no extractable blocks: {detail}")
    if not parsed.blocks and ocr_candidates:
        parsed.warnings.append(
            "No text layer was found on PDF page(s) "
            f"{ocr_candidates}; OCR fallback is required."
        )

    return ExtractionResult(
        blocks=parsed.blocks,
        file_name=source_name,
        file_type=file_type,
        file_hash=content_hash,
        byte_size=len(file_bytes),
        page_count=parsed.page_count,
        sheet_names=parsed.sheet_names,
        warnings=list(parsed.warnings),
        metadata=dict(parsed.metadata),
        ocr_candidates=ocr_candidates,
    )


def needs_ocr(result: ExtractionResult) -> bool:
    """True when extraction marked at least one PDF page for OCR fallback."""
    return bool(result.ocr_candidates)
