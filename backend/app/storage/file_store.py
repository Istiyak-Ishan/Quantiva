"""Local filesystem document storage for Quantiva desktop.

Manages original uploaded documents and extracted/processed artifacts under
the configured QUANTIVA_DATA_DIR. All database references store relative paths
so the entire data directory can be moved (e.g. from C: to D:) without breaking
file references.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, BinaryIO
import uuid

from app.config import settings

logger = logging.getLogger(__name__)

# Reserved device names in Windows (case-insensitive, even with extensions)
WINDOWS_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    "CLOCK$",
})

ILLEGAL_CHARS_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
DEFAULT_CHUNK_SIZE = 64 * 1024  # 64 KB


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize a filename for Windows filesystems and prevent path traversal.

    1. Strips directory navigation characters (/, \\, ..).
    2. Replaces Windows illegal characters (<>:"/\\|?* and control characters) with '_'.
    3. Handles Windows reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9).
    4. Trims leading and trailing spaces and dots (invalid under Windows).
    5. Bounds filename length to prevent exceeding Windows MAX_PATH limits.
    """
    if not filename or not isinstance(filename, str):
        return "document"

    # Normalize backslashes to forward slashes to extract only the leaf name
    leaf = filename.strip().replace("\\", "/").rstrip("/")
    if "/" in leaf:
        leaf = leaf.split("/")[-1]

    # Replace illegal characters
    cleaned = ILLEGAL_CHARS_PATTERN.sub("_", leaf)

    # Trim leading and trailing spaces and dots
    cleaned = cleaned.strip(". ")

    if not cleaned:
        return "document"

    # Separate stem and extension
    dot_index = cleaned.rfind(".")
    if dot_index > 0:
        stem = cleaned[:dot_index].rstrip(". ")
        suffix = cleaned[dot_index:]
    else:
        stem = cleaned
        suffix = ""

    if not stem:
        stem = "document"

    # Check for Windows reserved names.
    # Note: on Windows, reserved names apply to the base stem (e.g. CON.txt or aux.tar.gz)
    stem_base = stem.split(".")[0]
    if stem.upper() in WINDOWS_RESERVED_NAMES or stem_base.upper() in WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"

    # Enforce max length preserving suffix
    if len(stem) + len(suffix) > max_length:
        allowed_stem = max(1, max_length - len(suffix))
        stem = stem[:allowed_stem].rstrip(". ")

    result = f"{stem}{suffix}".strip(". ")
    return result if result else "document"


@dataclass(frozen=True, slots=True)
class StoredFile:
    """Metadata describing a file saved by LocalFileStore."""

    knowledge_base_id: str
    document_id: str
    filename: str
    relative_path: str
    absolute_path: Path
    file_size_bytes: int
    sha256_hash: str


class LocalFileStore:
    """Local filesystem storage for documents and artifacts.

    Persists original files under:
        QUANTIVA_DATA_DIR / "documents" / "original" / {kb_id} / {doc_id}_{safe_filename}

    Persists processed artifacts under:
        QUANTIVA_DATA_DIR / "documents" / "processed" / {kb_id} / {doc_id}.json

    Stores relative paths in database records so data directory migration
    maintains complete integrity.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        if data_dir is not None:
            self._data_dir = Path(data_dir).expanduser().resolve()
        else:
            self._data_dir = Path(settings.QUANTIVA_DATA_DIR).expanduser().resolve()

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def original_dir(self) -> Path:
        return self._data_dir / "documents" / "original"

    @property
    def processed_dir(self) -> Path:
        return self._data_dir / "documents" / "processed"

    # ── Path Helpers ──────────────────────────────────────────────────────────

    def get_original_relative_path(
        self,
        knowledge_base_id: str,
        document_id: str,
        filename: str,
    ) -> str:
        """Return canonical relative path using forward slashes for database storage."""
        safe_kb_id = sanitize_filename(knowledge_base_id)
        safe_doc_id = sanitize_filename(document_id)
        safe_name = sanitize_filename(filename)
        return f"documents/original/{safe_kb_id}/{safe_doc_id}_{safe_name}"

    def get_processed_relative_path(
        self,
        knowledge_base_id: str,
        document_id: str,
        ext: str = "json",
    ) -> str:
        """Return canonical relative path for processed artifacts."""
        safe_kb_id = sanitize_filename(knowledge_base_id)
        safe_doc_id = sanitize_filename(document_id)
        clean_ext = ext.lstrip(".") if ext else "json"
        safe_ext = sanitize_filename(clean_ext)
        return f"documents/processed/{safe_kb_id}/{safe_doc_id}.{safe_ext}"

    def resolve_relative_path(
        self,
        relative_path: str | Path,
        base_dir: Path | str | None = None,
    ) -> Path:
        """Resolve a relative path against the current or relocated data directory.

        Guards strictly against directory traversal: if the resolved path escapes
        the target root, raises ValueError.
        """
        root = Path(base_dir or self._data_dir).expanduser().resolve()
        raw_str = str(relative_path).strip()
        rel_obj = Path(raw_str)

        if rel_obj.is_absolute() or rel_obj.drive or raw_str.startswith(("/", "\\")):
            raise ValueError(
                f"Path traversal detected: absolute path '{relative_path}' is not permitted"
            )

        resolved = (root / rel_obj).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Path traversal detected: '{relative_path}' escapes base directory '{root}'"
            ) from exc

        return resolved

    # ── Upload & Write Operations ─────────────────────────────────────────────

    def save_original(
        self,
        knowledge_base_id: str,
        document_id: str,
        filename: str,
        content: bytes | BinaryIO,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> StoredFile:
        """Stream write original document in chunks and compute SHA-256 in a single pass."""
        safe_filename = sanitize_filename(filename)
        rel_path = self.get_original_relative_path(knowledge_base_id, document_id, filename)
        dest_path = self.resolve_relative_path(rel_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = dest_path.parent / f".tmp_{document_id}_{uuid.uuid4().hex}"
        hasher = hashlib.sha256()
        total_bytes = 0

        try:
            with open(temp_path, "wb") as out_file:
                if isinstance(content, (bytes, bytearray, memoryview)):
                    for offset in range(0, len(content), chunk_size):
                        chunk = content[offset : offset + chunk_size]
                        hasher.update(chunk)
                        out_file.write(chunk)
                        total_bytes += len(chunk)
                else:
                    while True:
                        chunk = content.read(chunk_size)
                        if not chunk:
                            break
                        hasher.update(chunk)
                        out_file.write(chunk)
                        total_bytes += len(chunk)

            os.replace(temp_path, dest_path)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

        return StoredFile(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            filename=safe_filename,
            relative_path=rel_path,
            absolute_path=dest_path,
            file_size_bytes=total_bytes,
            sha256_hash=hasher.hexdigest(),
        )

    async def save_original_stream(
        self,
        knowledge_base_id: str,
        document_id: str,
        filename: str,
        stream: AsyncIterator[bytes],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> StoredFile:
        """Stream write original document from an async iterator, hashing in a single pass."""
        safe_filename = sanitize_filename(filename)
        rel_path = self.get_original_relative_path(knowledge_base_id, document_id, filename)
        dest_path = self.resolve_relative_path(rel_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = dest_path.parent / f".tmp_{document_id}_{uuid.uuid4().hex}"
        hasher = hashlib.sha256()
        total_bytes = 0

        try:
            with open(temp_path, "wb") as out_file:
                async for chunk in stream:
                    if not chunk:
                        continue
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8")
                    hasher.update(chunk)
                    out_file.write(chunk)
                    total_bytes += len(chunk)

            os.replace(temp_path, dest_path)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

        return StoredFile(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            filename=safe_filename,
            relative_path=rel_path,
            absolute_path=dest_path,
            file_size_bytes=total_bytes,
            sha256_hash=hasher.hexdigest(),
        )

    def save_processed(
        self,
        knowledge_base_id: str,
        document_id: str,
        content: str | bytes | dict[str, Any] | list[Any],
        ext: str = "json",
    ) -> StoredFile:
        """Store extracted/processed markdown or JSON artifact."""
        clean_ext = ext.lstrip(".") if ext else "json"
        rel_path = self.get_processed_relative_path(knowledge_base_id, document_id, clean_ext)
        dest_path = self.resolve_relative_path(rel_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, (dict, list)):
            payload = json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8")
        elif isinstance(content, str):
            payload = content.encode("utf-8")
        elif isinstance(content, (bytes, bytearray)):
            payload = bytes(content)
        else:
            payload = str(content).encode("utf-8")

        temp_path = dest_path.parent / f".tmp_{document_id}_{uuid.uuid4().hex}"
        hasher = hashlib.sha256(payload)

        try:
            with open(temp_path, "wb") as out_file:
                out_file.write(payload)
            os.replace(temp_path, dest_path)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise

        return StoredFile(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            filename=f"{document_id}.{clean_ext}",
            relative_path=rel_path,
            absolute_path=dest_path,
            file_size_bytes=len(payload),
            sha256_hash=hasher.hexdigest(),
        )

    async def save_processed_async(
        self,
        knowledge_base_id: str,
        document_id: str,
        content: str | bytes | dict[str, Any] | list[Any],
        ext: str = "json",
    ) -> StoredFile:
        """Async convenience wrapper around save_processed."""
        return await asyncio.to_thread(
            self.save_processed,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            content=content,
            ext=ext,
        )

    # ── Read Operations ───────────────────────────────────────────────────────

    def read_original(
        self,
        knowledge_base_id: str,
        document_id: str,
        filename: str | None = None,
    ) -> bytes:
        """Read original file bytes by knowledge base and document ID."""
        if filename:
            rel_path = self.get_original_relative_path(knowledge_base_id, document_id, filename)
            target = self.resolve_relative_path(rel_path)
            if target.is_file():
                return target.read_bytes()

        # Fallback: search knowledge base directory for document_id prefix
        kb_dir = self.original_dir / sanitize_filename(knowledge_base_id)
        if kb_dir.is_dir():
            safe_doc_id = sanitize_filename(document_id)
            for match in kb_dir.glob(f"{safe_doc_id}_*"):
                if match.is_file():
                    return match.read_bytes()

        raise FileNotFoundError(
            f"Original document file not found for doc '{document_id}' in KB '{knowledge_base_id}'"
        )

    def read_processed(
        self,
        knowledge_base_id: str,
        document_id: str,
        ext: str = "json",
    ) -> Any:
        """Read and parse processed artifact (JSON or text)."""
        rel_path = self.get_processed_relative_path(knowledge_base_id, document_id, ext)
        target = self.resolve_relative_path(rel_path)
        if not target.is_file():
            raise FileNotFoundError(f"Processed artifact not found: {target}")

        raw = target.read_bytes()
        clean_ext = ext.lstrip(".").lower()
        if clean_ext == "json":
            return json.loads(raw.decode("utf-8"))
        return raw.decode("utf-8")

    def read_by_relative_path(
        self,
        relative_path: str | Path,
        base_dir: Path | str | None = None,
    ) -> bytes:
        """Read raw bytes of any managed file by relative path."""
        target = self.resolve_relative_path(relative_path, base_dir=base_dir)
        if not target.is_file():
            raise FileNotFoundError(f"File not found at: {target}")
        return target.read_bytes()

    # ── Deletion Operations ───────────────────────────────────────────────────

    def delete_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        original_rel_path: str | None = None,
        processed_rel_path: str | None = None,
    ) -> bool:
        """Delete both original and processed files cleanly for a document."""
        deleted_any = False

        # 1. Clean original file
        if original_rel_path:
            try:
                target = self.resolve_relative_path(original_rel_path)
                if target.is_file():
                    target.unlink()
                    deleted_any = True
            except (ValueError, OSError) as exc:
                logger.warning("Failed deleting original path %s: %s", original_rel_path, exc)
        else:
            orig_kb_dir = self.original_dir / sanitize_filename(knowledge_base_id)
            if orig_kb_dir.is_dir():
                safe_doc_id = sanitize_filename(document_id)
                for f in orig_kb_dir.glob(f"{safe_doc_id}_*"):
                    try:
                        if f.is_file():
                            f.unlink()
                            deleted_any = True
                    except OSError as exc:
                        logger.warning("Failed unlinking original file %s: %s", f, exc)

        # 2. Clean processed file
        if processed_rel_path:
            try:
                target = self.resolve_relative_path(processed_rel_path)
                if target.is_file():
                    target.unlink()
                    deleted_any = True
            except (ValueError, OSError) as exc:
                logger.warning("Failed deleting processed path %s: %s", processed_rel_path, exc)
        else:
            proc_kb_dir = self.processed_dir / sanitize_filename(knowledge_base_id)
            if proc_kb_dir.is_dir():
                safe_doc_id = sanitize_filename(document_id)
                for f in proc_kb_dir.glob(f"{safe_doc_id}.*"):
                    try:
                        if f.is_file():
                            f.unlink()
                            deleted_any = True
                    except OSError as exc:
                        logger.warning("Failed unlinking processed file %s: %s", f, exc)

        return deleted_any

    def delete_knowledge_base(self, knowledge_base_id: str) -> bool:
        """Recursively delete all original and processed files for a knowledge base."""
        safe_kb_id = sanitize_filename(knowledge_base_id)
        orig_kb_dir = self.original_dir / safe_kb_id
        proc_kb_dir = self.processed_dir / safe_kb_id

        if orig_kb_dir.exists():
            shutil.rmtree(orig_kb_dir, ignore_errors=True)

        if proc_kb_dir.exists():
            shutil.rmtree(proc_kb_dir, ignore_errors=True)

        return True


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "LocalFileStore",
    "StoredFile",
    "WINDOWS_RESERVED_NAMES",
    "sanitize_filename",
]
