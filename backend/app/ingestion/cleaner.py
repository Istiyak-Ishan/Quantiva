"""Text normalisation, SHA-256 hashing, and duplicate suppression.

The cleaner runs before chunking.  It is deliberately conservative:

* Markdown tables keep their pipes/dividers.
* Fenced code blocks keep their internal whitespace and indentation.
* Prose gets Unicode/whitespace normalisation but no semantic rewriting.

Two hashes are produced:

* **File hash** — SHA-256 over the raw upload bytes.  Used to reject duplicate
  documents before any parsing work happens.
* **Content hash** — SHA-256 over clean, whitespace-normalised block/chunk
  text.  Used to deduplicate identical chunks and to identify citations.
"""
from __future__ import annotations

import hashlib
import re
import threading
import unicodedata
from pathlib import Path
from typing import Iterable

from app.ingestion.types import DocumentChunkPayload, ExtractedBlock

__all__ = [
    "DocumentHashRegistry",
    "clean_block",
    "clean_blocks",
    "clean_text",
    "compute_content_hash",
    "compute_file_hash",
    "compute_file_hash_from_path",
    "deduplicate_chunks",
    "deduplicate_blocks",
]

_FILE_READ_CHUNK = 1024 * 1024
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_HASH_WHITESPACE_RE = re.compile(r"\s+")


# ── SHA-256 hashing ───────────────────────────────────────────────────────────

def compute_file_hash(data: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of raw file bytes."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("compute_file_hash expects bytes")
    digest = hashlib.sha256()
    digest.update(bytes(data))
    return digest.hexdigest()


def compute_file_hash_from_path(path: str | Path) -> str:
    """Hash a file on disk without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(_FILE_READ_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def compute_content_hash(text: str) -> str:
    """Return a stable SHA-256 fingerprint for block/chunk text.

    Whitespace is normalised (but letter case is preserved) so a table that
    differs only by incidental line wrapping is still recognised as a
    duplicate.
    """
    if not text:
        return hashlib.sha256(b"").hexdigest()
    normalized = _HASH_WHITESPACE_RE.sub(" ", clean_text(text)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_control_characters(text: str) -> str:
    # Keep tab and newline; remove NULs, zero-width joiners, BOMs, etc.
    return "".join(
        character
        for character in text
        if character in "\n\t"
        or unicodedata.category(character)[0] != "C"
    )


def clean_text(text: str) -> str:
    """Normalise prose while preserving Markdown structure.

    * Unicode NFKC (collapses compatibility variants)
    * CRLF/CR → LF
    * non-breaking/zero-width spaces normalised
    * trailing whitespace removed line-by-line
    * runs of 3+ blank lines collapsed to two
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _normalise_newlines(text)
    text = _strip_control_characters(text)
    text = (
        text.replace("\u00a0", " ")
        .replace("\u2007", " ")
        .replace("\u202f", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )
    text = text.replace("\t", "    ")
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _clean_code(text: str) -> str:
    """Only normalise line endings in code so indentation survives."""
    text = _normalise_newlines(text)
    text = _strip_control_characters(text)
    return text.strip("\n")


def clean_block(block: ExtractedBlock) -> ExtractedBlock:
    """Clean one extracted block in place and attach its content hash."""
    if block.block_type == "code":
        block.text = _clean_code(block.text)
    else:
        block.text = clean_text(block.text)
    block.content_hash = compute_content_hash(block.text) if block.text else ""
    return block


# ── Block-level deduplication ────────────────────────────────────────────────

def _normalise_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _table_cell_text(markdown: str) -> str:
    """Normalise a Markdown table to cell content for duplicate checks."""
    cells: list[str] = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line or set(line) <= set("|-: "):
            continue
        cells.extend(cell.strip() for cell in line.strip("|").split("|"))
    return _normalise_for_compare(" ".join(cell for cell in cells if cell))


def deduplicate_blocks(blocks: Iterable[ExtractedBlock]) -> list[ExtractedBlock]:
    """Drop exact duplicates and plain-text copies of extracted tables.

    PDF extraction frequently emits both a table-as-Markdown block and the
    same content as unstructured text.  Keeping both wastes embeddings and
    makes keyword search return the same passage twice.
    """
    block_list = list(blocks)
    table_texts = [
        cell_text
        for block in block_list
        if block.block_type == "table"
        for cell_text in [_table_cell_text(block.text)]
        if cell_text
    ]

    kept: list[ExtractedBlock] = []
    seen: set[str] = set()
    for block in block_list:
        normalized = _normalise_for_compare(block.text)
        if not normalized or normalized in seen:
            continue
        if block.block_type == "text" and any(
            len(table_text) >= 20
            and (normalized in table_text or table_text in normalized)
            for table_text in table_texts
        ):
            # The structured table block already represents this content.
            continue
        seen.add(normalized)
        kept.append(block)
    return kept


def clean_blocks(blocks: Iterable[ExtractedBlock]) -> list[ExtractedBlock]:
    """Clean, hash, and deduplicate a parser's output in one pass."""
    cleaned = [clean_block(block) for block in blocks]
    cleaned = [block for block in cleaned if block.text]
    return deduplicate_blocks(cleaned)


# ── Chunk-level deduplication ────────────────────────────────────────────────

def deduplicate_chunks(
    chunks: Iterable[DocumentChunkPayload],
) -> list[DocumentChunkPayload]:
    """Remove chunks whose normalised SHA-256 content hash was already seen.

    The first occurrence wins; ``chunk_index`` is left untouched here because
    ``chunker.chunk_blocks`` re-numbers after deduplication.  Callers that use
    this helper independently should re-index their payloads.
    """
    kept: list[DocumentChunkPayload] = []
    seen: set[str] = set()
    for chunk in chunks:
        content_hash = chunk.get("content_hash") or compute_content_hash(
            chunk.get("content", "")
        )
        if content_hash in seen:
            continue
        seen.add(content_hash)
        kept.append(chunk)
    return kept


class DocumentHashRegistry:
    """Thread-safe in-memory SHA-256 → document-id registry.

    This is a small convenience adapter for the local job manager.  The
    production lookup can be backed by SQLite; the pipeline only requires a
    ``Callable[[str], str | None]``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hashes: dict[str, str] = {}

    def add(self, content_hash: str, document_id: str) -> None:
        with self._lock:
            self._hashes[content_hash] = document_id

    def find(self, content_hash: str) -> str | None:
        with self._lock:
            return self._hashes.get(content_hash)

    def discard(self, content_hash: str) -> None:
        with self._lock:
            self._hashes.pop(content_hash, None)

    def __call__(self, content_hash: str) -> str | None:
        return self.find(content_hash)
