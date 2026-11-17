"""Format-specific document parsers.

Every parser converts raw bytes into :class:`ExtractedBlock` objects and keeps
structural metadata beside the text:

* PDF  → exact 1-based ``page_number``, Markdown tables, OCR candidates.
* DOCX → heading hierarchy in ``section_path`` plus Markdown tables.
* XLSX → one table per sheet, sheet name, headers, summary statistics.
* CSV  → one table, headers and summary statistics.
* TXT  → plain text blocks.
* MD   → headings, fenced code blocks, and Markdown tables.

The parsers contain no LLM/embedding provider imports and never read from
network storage.  They are pure ``bytes`` → ``blocks`` functions.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Iterable, Sequence

from app.ingestion.types import (
    DocumentValidationError,
    ExtractionError,
    ExtractedBlock,
    UnsupportedDocumentTypeError,
)

__all__ = [
    "CSV_MIME",
    "DOCX_MIME",
    "MARKDOWN_MIME",
    "PDF_MIME",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_MIME_TYPES",
    "TXT_MIME",
    "XLSX_MIME",
    "ParserOutput",
    "detect_file_type",
    "parse_csv",
    "parse_docx",
    "parse_markdown",
    "parse_pdf",
    "parse_text",
    "parse_xlsx",
    "rows_to_markdown",
]

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
CSV_MIME = "text/csv"
TXT_MIME = "text/plain"
MARKDOWN_MIME = "text/markdown"

SUPPORTED_MIME_TYPES = frozenset(
    {PDF_MIME, DOCX_MIME, XLSX_MIME, CSV_MIME, TXT_MIME, MARKDOWN_MIME}
)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": PDF_MIME,
    ".docx": DOCX_MIME,
    ".xlsx": XLSX_MIME,
    ".csv": CSV_MIME,
    ".txt": TXT_MIME,
    ".md": MARKDOWN_MIME,
    ".markdown": MARKDOWN_MIME,
}

_MIME_ALIASES = {
    "application/x-pdf": PDF_MIME,
    "application/csv": CSV_MIME,
    "text/x-csv": CSV_MIME,
    "text/txt": TXT_MIME,
    "text/x-markdown": MARKDOWN_MIME,
    "text/md": MARKDOWN_MIME,
    "application/markdown": MARKDOWN_MIME,
    "application/x-markdown": MARKDOWN_MIME,
    "application/docx": DOCX_MIME,
    "application/x-docx": DOCX_MIME,
    "application/xlsx": XLSX_MIME,
    "application/vnd.ms-excel": XLSX_MIME,
}

_HEADING_MD_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_DELIM_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)+\|?\s*$"
)
_EXISTING_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)*[\s.)]")
_WORDS_PER_PAGE_ESTIMATE = 450
_OCR_MIN_TEXT_CHARS = 30


def detect_file_type(
    file_name: str | None = None,
    mime_type: str | None = None,
) -> str:
    """Resolve a supported MIME type from the filename and/or declared MIME.

    The extension is authoritative when it is known because browsers often
    send generic ``application/octet-stream`` values for CSV/Markdown uploads.
    """
    suffix = PurePath(file_name or "").suffix.lower()
    normalized_mime = (mime_type or "").split(";")[0].strip().lower()
    normalized_mime = _MIME_ALIASES.get(normalized_mime, normalized_mime)

    if suffix in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[suffix]
    if normalized_mime in SUPPORTED_MIME_TYPES:
        return normalized_mime
    if suffix:
        raise UnsupportedDocumentTypeError(
            f"Unsupported document extension '{suffix}' (MIME '{mime_type or 'unknown'}')"
        )
    raise UnsupportedDocumentTypeError(
        f"Unsupported document type '{mime_type or 'unknown'}'; "
        "supply a .pdf, .docx, .xlsx, .csv, .txt, or .md file."
    )


# ── Parser output container ───────────────────────────────────────────────────

@dataclass(slots=True)
class ParserOutput:
    """Everything a parser knows beyond the block list."""

    blocks: list[ExtractedBlock]
    page_count: int | None = None
    sheet_names: list[str] = field(default_factory=list)
    ocr_candidates: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Shared Markdown helpers ───────────────────────────────────────────────────

def _escape_cell(value: Any) -> str:
    if value is None:
        return ""
    # pandas.NA / NaT are not ``None`` but should render as empty cells.
    if value.__class__.__name__ in {"NAType", "NaTType"}:
        return ""
    if isinstance(value, float):
        try:
            if value != value:  # NaN
                return ""
        except Exception:  # pragma: no cover - defensive
            pass
        if value.is_integer():
            return str(int(value))
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = text.replace("|", r"\|")
    return re.sub(r"\s+", " ", text).strip()


def _unique_headers(columns: Iterable[Any]) -> list[str]:
    headers: list[str] = []
    counts: dict[str, int] = {}
    for index, column in enumerate(columns, start=1):
        header = _escape_cell(column) or f"Column {index}"
        counts[header] = counts.get(header, 0) + 1
        if counts[header] > 1:
            header = f"{header} ({counts[header]})"
        headers.append(header)
    return headers


def rows_to_markdown(
    rows: Sequence[Sequence[Any]],
    *,
    header: Sequence[Any] | None = None,
) -> str:
    """Render rows as a clean GitHub-flavoured Markdown table.

    The header is always repeated in the returned string, which is what makes
    later table splitting safe: every sub-chunk can start with this header.
    """
    normalized = [list(row) for row in rows]
    if header is None:
        if not normalized:
            return ""
        header_cells = [_escape_cell(cell) for cell in normalized.pop(0)]
    else:
        header_cells = [_escape_cell(cell) for cell in header]

    width = max([len(header_cells), *(len(row) for row in normalized)], default=0)
    if width == 0:
        return ""
    header_cells = [cell or f"Column {i + 1}" for i, cell in enumerate(header_cells)]
    header_cells += [f"Column {i + 1}" for i in range(len(header_cells), width)]

    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in normalized:
        cells = [_escape_cell(cell) for cell in row]
        cells += [""] * (width - len(cells))
        lines.append("| " + " | ".join(cells[:width]) + " |")
    return "\n".join(lines)


def _markdown_header_cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    if not stripped:
        return []
    return [cell.strip() for cell in stripped.split("|")]


def _table_start(lines: Sequence[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and bool(_TABLE_DELIM_RE.match(lines[index + 1]))
    )


# ── Heading hierarchy ────────────────────────────────────────────────────────

class _HeadingTracker:
    """Builds numbered section paths such as ``1. Overview > 1.2 Revenue``."""

    def __init__(self) -> None:
        self._counters: dict[int, int] = {}
        self._labels: dict[int, str] = {}

    def add(self, level: int, title: str) -> tuple[str, str]:
        level = max(1, min(int(level), 6))
        for deeper in [key for key in self._counters if key > level]:
            del self._counters[deeper]
        for deeper in [key for key in self._labels if key > level]:
            del self._labels[deeper]
        self._counters[level] = self._counters.get(level, 0) + 1
        for ancestor in range(1, level):
            self._counters.setdefault(ancestor, 1)

        number = ".".join(
            str(self._counters.get(depth, 1)) for depth in range(1, level + 1)
        )
        clean_title = title.strip()
        if _EXISTING_NUMBER_RE.match(clean_title):
            label = clean_title
        elif level == 1:
            label = f"{number}. {clean_title}"
        else:
            label = f"{number} {clean_title}"
        self._labels[level] = label
        path = " > ".join(self._labels[depth] for depth in sorted(self._labels))
        return label, path


class _PageEstimator:
    """Approximate page numbers for formats without physical page boundaries."""

    def __init__(self, words_per_page: int = _WORDS_PER_PAGE_ESTIMATE) -> None:
        self._words_per_page = words_per_page
        self._page = 1
        self._words = 0

    @property
    def page(self) -> int:
        return self._page

    def consume(self, text: str) -> None:
        words = len(text.split())
        if self._words and self._words + words > self._words_per_page:
            self._page += 1
            self._words = words
        else:
            self._words += words


# ── PDF ──────────────────────────────────────────────────────────────────────

def _block_overlaps_table(
    block_bbox: Sequence[float],
    table_bboxes: Sequence[Sequence[float]],
) -> bool:
    x0, y0, x1, y1 = block_bbox[:4]
    block_area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if block_area <= 0:
        return False
    for tx0, ty0, tx1, ty1 in table_bboxes:
        ix0, iy0 = max(x0, tx0), max(y0, ty0)
        ix1, iy1 = min(x1, tx1), min(y1, ty1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        overlap = (ix1 - ix0) * (iy1 - iy0)
        if overlap / block_area >= 0.5:
            return True
    return False


def parse_pdf(
    file_bytes: bytes,
    *,
    min_text_chars: int = _OCR_MIN_TEXT_CHARS,
) -> ParserOutput:
    """Extract text blocks and tables from a PDF, preserving page numbers."""
    min_text_chars = max(1, int(min_text_chars))
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ExtractionError("PyMuPDF is required to parse PDF documents") from exc

    blocks: list[ExtractedBlock] = []
    ocr_candidates: list[int] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise DocumentValidationError("Encrypted PDFs are not supported")
            metadata = {
                key: value
                for key, value in (document.metadata or {}).items()
                if value
            }
            page_count = document.page_count

            for page_number, page in enumerate(document, start=1):
                table_bboxes: list[tuple[float, float, float, float]] = []
                table_finder = None
                try:
                    table_finder = page.find_tables()
                except Exception as exc:  # pragma: no cover - malformed tables
                    warnings.append(f"Page {page_number}: table detection failed ({exc})")

                if table_finder is not None:
                    for table_index, table in enumerate(table_finder, start=1):
                        try:
                            rows = table.extract() or []
                            bx0, by0, bx1, by1 = table.bbox
                            table_bbox = (
                                float(bx0),
                                float(by0),
                                float(bx1),
                                float(by1),
                            )
                        except Exception as exc:  # pragma: no cover - defensive
                            warnings.append(
                                f"Page {page_number}: table {table_index} could not be read ({exc})"
                            )
                            continue
                        markdown = rows_to_markdown(rows)
                        if not markdown:
                            continue
                        table_bboxes.append(table_bbox)
                        headers = _markdown_header_cells(markdown.splitlines()[0])
                        blocks.append(
                            ExtractedBlock(
                                text=markdown,
                                block_type="table",
                                page_number=page_number,
                                table_headers=tuple(headers),
                                metadata={
                                    "table_index": table_index,
                                    "row_count": max(0, len(rows) - 1),
                                    "column_count": len(headers),
                                },
                            )
                        )

                page_has_images = bool(page.get_images(full=False))
                try:
                    raw_blocks = page.get_text("blocks", sort=True)
                except TypeError:  # older/newer PyMuPDF signature
                    raw_blocks = page.get_text("blocks")

                for raw_block in raw_blocks:
                    if len(raw_block) >= 7 and raw_block[6] != 0:
                        # 1 = image block; images are handled by the OCR stage.
                        continue
                    text = str(raw_block[4]).strip() if len(raw_block) > 4 else ""
                    if not text:
                        continue
                    if len(raw_block) >= 4 and _block_overlaps_table(
                        raw_block[:4], table_bboxes
                    ):
                        # The Markdown table block represents this text with
                        # structure intact; do not index the flattened copy.
                        continue
                    blocks.append(
                        ExtractedBlock(
                            text=text,
                            block_type="text",
                            page_number=page_number,
                            metadata={
                                "page_has_images": page_has_images,
                                "block_number": raw_block[5] if len(raw_block) > 5 else None,
                            },
                        )
                    )

                extracted_characters = len(
                    "".join(page.get_text("text").split())
                )
                if page_has_images and extracted_characters < min_text_chars:
                    ocr_candidates.append(page_number)

            return ParserOutput(
                blocks=blocks,
                page_count=page_count,
                ocr_candidates=ocr_candidates,
                warnings=warnings,
                metadata=metadata,
            )
    except DocumentValidationError:
        raise
    except Exception as exc:
        raise ExtractionError(f"PDF parsing failed: {exc}") from exc


# ── DOCX ─────────────────────────────────────────────────────────────────────

def _docx_heading_level(paragraph: Any) -> int:
    style = getattr(paragraph, "style", None)
    name = (getattr(style, "name", "") or "").strip().lower()
    if name == "title":
        return 1
    match = re.match(r"heading\s*(\d+)", name)
    if match:
        return int(match.group(1))
    style_id = (getattr(style, "style_id", "") or "").strip().lower()
    match = re.match(r"heading(\d+)", style_id)
    if match:
        return int(match.group(1))
    return 0


def parse_docx(file_bytes: bytes) -> ParserOutput:
    """Extract DOCX paragraphs and tables while preserving heading hierarchy."""
    blocks: list[ExtractedBlock] = []
    try:
        import docx
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError:  # pragma: no cover - optional dependency path
        return _parse_docx_xml_fallback(file_bytes)

    try:
        document = docx.Document(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ExtractionError(f"DOCX parsing failed: {exc}") from exc

    tracker = _HeadingTracker()
    page_estimator = _PageEstimator()
    current_section_path: str | None = None
    current_section_title: str | None = None
    table_index = 0
    warnings: list[str] = []

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            level = _docx_heading_level(paragraph)
            if level:
                label, path = tracker.add(level, text)
                current_section_path = path
                current_section_title = label
            page_estimator.consume(text)
            blocks.append(
                ExtractedBlock(
                    text=text,
                    block_type="text",
                    page_number=page_estimator.page,
                    section_path=current_section_path,
                    section_title=current_section_title,
                    metadata={"heading_level": level} if level else {},
                )
            )
        elif isinstance(child, CT_Tbl):
            table = Table(child, document)
            table_index += 1
            rows = [
                [cell.text.replace("\n", " ").strip() for cell in row.cells]
                for row in table.rows
            ]
            markdown = rows_to_markdown(rows)
            if not markdown:
                continue
            headers = _markdown_header_cells(markdown.splitlines()[0])
            page_estimator.consume(markdown)
            blocks.append(
                ExtractedBlock(
                    text=markdown,
                    block_type="table",
                    page_number=page_estimator.page,
                    section_path=current_section_path,
                    section_title=current_section_title,
                    table_headers=tuple(headers),
                    metadata={
                        "table_index": table_index,
                        "row_count": max(0, len(rows) - 1),
                        "column_count": len(headers),
                    },
                )
            )

    if not blocks:
        warnings.append("DOCX contained no extractable text or tables")
    return ParserOutput(blocks=blocks, warnings=warnings)


def _parse_docx_xml_fallback(file_bytes: bytes) -> ParserOutput:
    """Best-effort extraction when python-docx is unavailable."""
    try:
        import xml.etree.ElementTree as ET
        import zipfile

        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            xml_content = archive.read("word/document.xml")
        tree = ET.fromstring(xml_content)
        paragraphs: list[str] = []
        for paragraph in tree.iter():
            if not paragraph.tag.endswith("}p"):
                continue
            text = "".join(
                node.text or ""
                for node in paragraph.iter()
                if node.tag.endswith("}t")
            ).strip()
            if text:
                paragraphs.append(text)
        combined = "\n\n".join(paragraphs).strip()
        if not combined:
            return ParserOutput(blocks=[], warnings=["DOCX contained no extractable text"])
        return ParserOutput(
            blocks=[ExtractedBlock(text=combined, page_number=1)],
            warnings=["python-docx unavailable; used XML fallback"],
        )
    except Exception as exc:
        return ParserOutput(blocks=[], warnings=[f"DOCX fallback failed: {exc}"])


# ── Spreadsheets ─────────────────────────────────────────────────────────────

def _format_dataframe(df: Any) -> tuple[list[str], list[list[str]]]:
    headers = _unique_headers(df.columns)
    rows = [
        [_escape_cell(value) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    return headers, rows


def _spreadsheet_summary(
    *,
    sheet_name: str,
    headers: Sequence[str],
    row_count: int,
    numeric_columns: Sequence[str],
    dtypes: dict[str, str] | None = None,
    delimiter: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sheet_name": sheet_name,
        "row_count": int(row_count),
        "column_count": len(headers),
        "column_names": list(headers),
        "numeric_columns": list(numeric_columns),
    }
    if dtypes:
        summary["dtypes"] = dtypes
    if delimiter:
        summary["delimiter"] = delimiter
    return summary


def _numeric_columns(df: Any) -> list[str]:
    try:
        import pandas as pd

        selected = df.select_dtypes(include=["number"])
        return [
            str(column)
            for column in selected.columns
            if not pd.api.types.is_bool_dtype(selected[column].dtype)
        ]
    except Exception:  # pragma: no cover - defensive for exotic dtypes
        return []


def parse_xlsx(file_bytes: bytes) -> ParserOutput:
    """Parse every worksheet into its own Markdown table block."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("pandas is required to parse spreadsheets") from exc

    try:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    except Exception as exc:
        raise ExtractionError(f"XLSX parsing failed: {exc}") from exc

    blocks: list[ExtractedBlock] = []
    sheet_names: list[str] = []
    summaries: dict[str, Any] = {}
    warnings: list[str] = []

    for raw_sheet_name, raw_df in sheets.items():
        sheet_name = str(raw_sheet_name)
        df = raw_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if df.empty:
            warnings.append(f"Sheet '{sheet_name}' is empty and was skipped")
            continue
        sheet_names.append(sheet_name)
        headers, rows = _format_dataframe(df)
        markdown = rows_to_markdown(rows, header=headers)
        numeric_columns = _numeric_columns(df)
        dtypes = {str(column): str(dtype) for column, dtype in df.dtypes.items()}
        summary = _spreadsheet_summary(
            sheet_name=sheet_name,
            headers=headers,
            row_count=len(df),
            numeric_columns=numeric_columns,
            dtypes=dtypes,
        )
        summaries[sheet_name] = summary
        blocks.append(
            ExtractedBlock(
                text=markdown,
                block_type="table",
                sheet_name=sheet_name,
                row_range=f"1-{len(df)}",
                table_headers=tuple(headers),
                metadata=summary,
            )
        )

    if not blocks:
        warnings.append("Workbook contained no non-empty sheets")
    metadata: dict[str, Any] = {"sheets": summaries}
    if len(summaries) == 1:
        metadata.update(next(iter(summaries.values())))
    return ParserOutput(
        blocks=blocks,
        sheet_names=sheet_names,
        warnings=warnings,
        metadata=metadata,
    )


def _read_csv_dataframe(file_bytes: bytes) -> tuple[Any, str, str]:
    import csv

    import pandas as pd

    def sniff_delimiter(encoding: str) -> str:
        try:
            sample = file_bytes[:64 * 1024].decode(encoding, errors="replace")
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except Exception:
            return ","

    last_error: Exception | None = None
    encodings = ["utf-8-sig"]
    if file_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.append("utf-16")
    encodings.extend(["cp1252", "latin-1"])
    for encoding in encodings:
        try:
            delimiter = sniff_delimiter(encoding)
            df = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                sep=delimiter,
                engine="python",
            )
            return df, encoding, delimiter
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            break
    raise ExtractionError(f"CSV parsing failed: {last_error}") from last_error


def parse_csv(file_bytes: bytes) -> ParserOutput:
    """Parse a CSV file into one structured Markdown table block."""
    df, encoding, delimiter = _read_csv_dataframe(file_bytes)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return ParserOutput(blocks=[], warnings=["CSV contained no data rows"])

    headers, rows = _format_dataframe(df)
    markdown = rows_to_markdown(rows, header=headers)
    numeric_columns = _numeric_columns(df)
    dtypes = {str(column): str(dtype) for column, dtype in df.dtypes.items()}
    summary = _spreadsheet_summary(
        sheet_name="",
        headers=headers,
        row_count=len(df),
        numeric_columns=numeric_columns,
        dtypes=dtypes,
        delimiter=delimiter or None,
    )
    summary["encoding"] = encoding
    block = ExtractedBlock(
        text=markdown,
        block_type="table",
        row_range=f"1-{len(df)}",
        table_headers=tuple(headers),
        metadata=summary,
    )
    metadata: dict[str, Any] = {"sheets": {"": summary}, **summary}
    return ParserOutput(
        blocks=[block],
        warnings=[],
        metadata=metadata,
    )


# ── Plain text and Markdown ──────────────────────────────────────────────────

def _decode_text_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return file_bytes.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return file_bytes.decode("latin-1", errors="replace")


def parse_text(file_bytes: bytes, *, file_name: str | None = None) -> ParserOutput:
    """Parse a plain-text file as a single page-1 block."""
    text = _decode_text_bytes(file_bytes).strip()
    if not text:
        return ParserOutput(blocks=[], warnings=["Text file was empty"])
    return ParserOutput(
        blocks=[ExtractedBlock(text=text, page_number=1, metadata={"file_name": file_name})],
        page_count=1,
    )


def parse_markdown(
    file_bytes: bytes,
    *,
    file_name: str | None = None,
) -> ParserOutput:
    """Parse Markdown into headings, fenced code, tables, and prose blocks."""
    text = _decode_text_bytes(file_bytes)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    tracker = _HeadingTracker()
    section_path: str | None = None
    section_title: str | None = None
    blocks: list[ExtractedBlock] = []
    prose: list[str] = []
    index = 0

    def flush_prose() -> None:
        nonlocal prose
        if not prose:
            return
        content = "\n".join(prose).strip()
        prose = []
        if content:
            blocks.append(
                ExtractedBlock(
                    text=content,
                    page_number=1,
                    section_path=section_path,
                    section_title=section_title,
                )
            )

    while index < len(lines):
        line = lines[index]

        heading = _HEADING_MD_RE.match(line)
        if heading:
            flush_prose()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            label, path = tracker.add(level, title)
            section_path = path
            section_title = label
            blocks.append(
                ExtractedBlock(
                    text=label,
                    page_number=1,
                    section_path=section_path,
                    section_title=section_title,
                    metadata={"heading_level": level},
                )
            )
            index += 1
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            flush_prose()
            fence_marker = fence.group(1)
            code_lines = [line]
            index += 1
            while index < len(lines):
                code_lines.append(lines[index])
                if lines[index].strip().startswith(fence_marker):
                    index += 1
                    break
                index += 1
            blocks.append(
                ExtractedBlock(
                    text="\n".join(code_lines),
                    block_type="code",
                    page_number=1,
                    section_path=section_path,
                    section_title=section_title,
                )
            )
            continue

        if _table_start(lines, index):
            flush_prose()
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines):
                candidate = lines[index]
                if not candidate.strip() or "|" not in candidate:
                    break
                table_lines.append(candidate)
                index += 1
            markdown_table = "\n".join(table_lines)
            blocks.append(
                ExtractedBlock(
                    text=markdown_table,
                    block_type="table",
                    page_number=1,
                    section_path=section_path,
                    section_title=section_title,
                    table_headers=tuple(_markdown_header_cells(table_lines[0])),
                )
            )
            continue

        if not line.strip():
            flush_prose()
        else:
            prose.append(line)
        index += 1

    flush_prose()
    if not blocks:
        return ParserOutput(blocks=[], warnings=["Markdown contained no content"])
    return ParserOutput(
        blocks=blocks,
        page_count=1,
        metadata={"file_name": file_name},
    )
