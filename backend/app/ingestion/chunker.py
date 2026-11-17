"""Structure-aware chunking with canonical metadata and stable hashes.

Chunk boundaries respect document structure:

* Small Markdown tables and fenced code blocks are never split.
* Large tables are split by rows and **every** resulting chunk starts with the
  original header row so spreadsheet chunks are never orphaned.
* Large code blocks are split on line boundaries and re-open/close the fence.
* Ordinary prose uses the existing recursive character splitter (character
  sizing preserved from the pre-desktop pipeline).

The output contract is :class:`~app.ingestion.types.DocumentChunkPayload`.
Chunk-level SHA-256 deduplication happens before indices are assigned, so
``chunk_index`` always matches the returned list position.
"""
from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Sequence

from app.ingestion.cleaner import compute_content_hash
from app.ingestion.types import DocumentChunkPayload, ExtractedBlock

__all__ = ["chunk_blocks"]

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_DELIMITER_RE = re.compile(
    r"^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)+\|?\s*$"
)
_CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://quantiva.local/chunk")


def _is_fence_start(line: str) -> bool:
    return bool(_FENCE_RE.match(line))


def _is_table_delimiter(line: str) -> bool:
    return bool(_TABLE_DELIMITER_RE.match(line))


def _is_table_start(lines: Sequence[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and _is_table_delimiter(lines[index + 1])
    )


# ── Generic prose splitting ───────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    effective_overlap = max(0, min(chunk_overlap, max(0, chunk_size - 1)))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=effective_overlap,
        separators=["\n\n", "\n", " ", ""],
        strip_whitespace=True,
    )
    return [piece for piece in splitter.split_text(text) if piece.strip()]


# ── Markdown table splitting ─────────────────────────────────────────────────

def _table_parts(text: str) -> tuple[list[str], str, str, list[str], list[str]] | None:
    """Return ``(preamble, header, divider, rows, tail)`` for a table block."""
    lines = text.splitlines()
    start = next(
        (index for index in range(len(lines)) if _is_table_start(lines, index)),
        None,
    )

    if start is None:
        # Some callers (or older extracts) produce pipe rows without a divider.
        start = next((index for index, line in enumerate(lines) if "|" in line), None)
        if start is None:
            return None
        header = lines[start].strip()
        if not header.startswith("|"):
            header = "| " + header.strip("|").strip() + " |"
        divider = "| " + " | ".join("---" for _ in _header_cells(header)) + " |"
        body_start = start + 1
    else:
        header = lines[start].strip()
        divider = lines[start + 1].strip()
        body_start = start + 2

    preamble = [line for line in lines[:start] if line.strip()]
    rows: list[str] = []
    index = body_start
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            break
        if "|" not in line:
            break
        rows.append(line.strip())
        index += 1
    tail = [line for line in lines[index:] if line.strip()]
    return preamble, header, divider, rows, tail


def _header_cells(header: str) -> list[str]:
    stripped = header.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")] if stripped else []


def _split_markdown_table(text: str, chunk_size: int) -> list[str]:
    """Split a large table by rows, repeating the Markdown header each time."""
    stripped = text.strip()
    if len(stripped) <= chunk_size:
        return [stripped]

    parts = _table_parts(stripped)
    if parts is None:
        return _split_text(stripped, chunk_size, 0)
    preamble, header, divider, rows, tail = parts

    prefix = ("\n".join(preamble) + "\n\n") if preamble else ""
    header_block = f"{header}\n{divider}"
    base = f"{prefix}{header_block}"
    if len(base) + 2 >= chunk_size:
        # A header that consumes the entire chunk budget cannot be repeated
        # alongside any row; fall back to ordinary splitting.
        return _split_text(stripped, chunk_size, 0)

    if not rows:
        return [stripped]

    chunks: list[str] = []
    current: list[str] = []
    current_length = len(base)

    def flush() -> None:
        nonlocal current, current_length
        if current:
            chunks.append(base + "\n" + "\n".join(current))
            current = []
            current_length = len(base)

    row_budget = max(1, chunk_size - len(base) - 1)
    for row in rows:
        row_length = len(row) + 1
        if current and current_length + row_length > chunk_size:
            flush()
        if len(row) > row_budget:
            # Extremely wide/long row: split it, but still prefix every piece
            # with the header so column meaning is preserved.
            flush()
            for offset in range(0, len(row), row_budget):
                piece = row[offset : offset + row_budget]
                chunks.append(base + "\n" + piece)
            continue
        current.append(row)
        current_length += row_length

    flush()

    if tail:
        tail_text = "\n".join(tail)
        if len(chunks[-1]) + len(tail_text) + 2 <= chunk_size:
            chunks[-1] = chunks[-1] + "\n\n" + tail_text
        else:
            chunks.append(tail_text)
    return chunks


# ── Code block splitting ─────────────────────────────────────────────────────

def _split_code_block(text: str, chunk_size: int) -> list[str]:
    """Split a large fenced code block while keeping fences balanced."""
    stripped = text.strip("\n")
    if len(stripped) <= chunk_size:
        return [stripped]

    lines = stripped.splitlines()
    opening_match = _FENCE_RE.match(lines[0]) if lines else None
    if opening_match is None:
        return _split_text(stripped, chunk_size, 0)

    fence = opening_match.group(1)
    opening = lines[0].strip()
    has_closing = (
        len(lines) > 1
        and lines[-1].strip().startswith(fence[0] * len(fence))
    )
    closing = lines[-1].strip() if has_closing else fence
    body = lines[1:-1] if has_closing else lines[1:]
    budget = chunk_size - len(opening) - len(closing) - 2
    if budget < 1:
        return _split_text(stripped, chunk_size, 0)

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    def flush() -> None:
        nonlocal current, current_length
        if not current:
            return
        chunks.append("\n".join([opening, *current, closing]))
        current = []
        current_length = 0

    for line in body:
        pieces = [line] if len(line) <= budget else [
            line[offset : offset + budget] for offset in range(0, len(line), budget)
        ]
        for piece in pieces:
            piece_length = len(piece) + 1
            if current and current_length + piece_length > budget:
                flush()
            current.append(piece)
            current_length += piece_length

    flush()
    return chunks or [stripped]


# ── Protected-segment detection ──────────────────────────────────────────────

def _protected_segments(text: str) -> list[tuple[str, str]]:
    """Identify fenced code and Markdown tables inside a prose block."""
    lines = text.splitlines()
    segments: list[tuple[str, str]] = []
    prose: list[str] = []
    index = 0

    def flush_prose() -> None:
        nonlocal prose
        if prose:
            value = "\n".join(prose).strip()
            if value:
                segments.append(("text", value))
            prose = []

    while index < len(lines):
        line = lines[index]

        if _is_fence_start(line):
            flush_prose()
            fence_start = line
            fence_match = _FENCE_RE.match(line)
            fence_token = fence_match.group(1) if fence_match else "```"
            code_lines = [line]
            index += 1
            while index < len(lines):
                code_lines.append(lines[index])
                if lines[index].strip().startswith(fence_token[0] * len(fence_token)):
                    index += 1
                    break
                index += 1
            if not code_lines[-1].strip().startswith(fence_token[0] * len(fence_token)):
                code_lines.append(fence_token)
            segments.append(("code", "\n".join([fence_start, *code_lines[1:]]).strip("\n")))
            continue

        if _is_table_start(lines, index):
            flush_prose()
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines):
                candidate = lines[index]
                if not candidate.strip() or "|" not in candidate:
                    break
                table_lines.append(candidate)
                index += 1
            segments.append(("table", "\n".join(table_lines)))
            continue

        prose.append(line)
        index += 1

    flush_prose()
    return segments


# ── Public API ───────────────────────────────────────────────────────────────

def _make_chunk_id(
    document_id: str,
    knowledge_base_id: str,
    chunk_index: int,
    content_hash: str,
) -> str:
    return str(
        uuid.uuid5(
            _CHUNK_NAMESPACE,
            f"{document_id}:{knowledge_base_id}:{chunk_index}:{content_hash}",
        )
    )


def _fragments_for_block(
    block: ExtractedBlock,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, str]]:
    """Return ``(fragment_kind, text)`` pairs for one extracted block."""
    if block.block_type == "table":
        return [
            ("table", fragment)
            for fragment in _split_markdown_table(block.text, chunk_size)
            if fragment.strip()
        ]

    fragments: list[tuple[str, str]] = []
    for kind, segment in _protected_segments(block.text):
        if kind == "table":
            pieces = _split_markdown_table(segment, chunk_size)
        elif kind == "code":
            pieces = _split_code_block(segment, chunk_size)
        else:
            pieces = _split_text(segment, chunk_size, chunk_overlap)
        fragments.extend((kind, piece) for piece in pieces if piece.strip())
    return fragments


def chunk_blocks(
    blocks: Iterable[ExtractedBlock],
    document_id: str,
    knowledge_base_id: str,
    source_file: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[DocumentChunkPayload]:
    """Turn extracted blocks into canonical, deduplicated chunk payloads."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if not document_id:
        raise ValueError("document_id is required")
    if not knowledge_base_id:
        raise ValueError("knowledge_base_id is required")

    candidates: list[dict[str, object]] = []
    seen_hashes: set[str] = set()

    for block in blocks:
        text = block.text or ""
        if not text.strip():
            continue
        section_title = block.section_title
        if not section_title and block.section_path:
            section_title = block.section_path.split(">")[-1].strip()

        for kind, fragment in _fragments_for_block(block, chunk_size, chunk_overlap):
            content = fragment.strip()
            if not content:
                continue
            content_hash = compute_content_hash(content)
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            candidates.append(
                {
                    "content": content,
                    "content_hash": content_hash,
                    "page_number": block.page_number,
                    "section_title": section_title,
                    "sheet_name": block.sheet_name,
                    "is_table": kind == "table" or block.block_type == "table",
                }
            )

    payloads: list[DocumentChunkPayload] = []
    for chunk_index, candidate in enumerate(candidates):
        content = str(candidate["content"])
        content_hash = str(candidate["content_hash"])
        payloads.append(
            {
                "chunk_id": _make_chunk_id(
                    document_id,
                    knowledge_base_id,
                    chunk_index,
                    content_hash,
                ),
                "document_id": document_id,
                "knowledge_base_id": knowledge_base_id,
                "page_number": candidate["page_number"],  # type: ignore[typeddict-item]
                "chunk_index": chunk_index,
                "content": content,
                "source_file": source_file,
                "section_title": candidate["section_title"],  # type: ignore[typeddict-item]
                "sheet_name": candidate["sheet_name"],  # type: ignore[typeddict-item]
                "is_table": bool(candidate["is_table"]),
                "content_hash": content_hash,
                "char_count": len(content),
            }
        )
    return payloads
