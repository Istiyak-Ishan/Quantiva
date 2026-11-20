"""
backend/app/core/kb_archive.py
────────────────────────────────────────────────────────────────────────────────
Portable Knowledge Base export / import for Quantiva Desktop (.qkb.zip).

A ``.qkb.zip`` archive is self-contained: it carries the Knowledge Base
metadata, the embedding/index configuration specs, every Document row, every
chunk row (so text never has to be parsed or extracted again), the original
uploaded files, and the pre-computed dense vectors.

Archive layout::

    kb_metadata.json     format/version, KB metadata, embedding + index specs,
                         counts, and SHA-256 checksums for every other member
    documents.json       all Document records (portable, machine-neutral form)
    chunks.jsonl         one JSON object per DocumentChunk record
    vectors.jsonl        one JSON object per dense point: {"chunk_id", "vector"}
    original_files/      original uploaded PDF/DOCX/XLSX/... payloads
    processed_files/     optional processed/extracted artifacts

Import rules (architectural invariants):
- Fresh UUIDs are allocated for the Knowledge Base, every Document and every
  DocumentChunk, so the same archive can be imported any number of times on
  any machine without primary-key collisions.
- The target machine's embedding configuration wins.  The archived spec is
  matched by id, then by (provider, model, dimensions), and only cloned when
  no local equivalent exists.  Cloud credentials never travel in an archive.
- Dense vectors are restored directly into ``kb_{new_id}`` only when the
  target index space (provider, model, dimensions) is identical to the one
  the vectors were produced with.  Otherwise the archive still imports with
  ``IndexMetadata.needs_reindex = True``.
- Sparse FTS5 rows are recreated from the restored chunk rows.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Callable, Coroutine, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import ensure_data_directories, settings
from app.core.knowledge_bases import KnowledgeBaseService, get_knowledge_base_service
from app.ingestion.types import DocumentChunkPayload
from app.storage.file_store import LocalFileStore, sanitize_filename
from app.storage.models import (
    Document,
    DocumentChunk,
    EmbeddingConfiguration,
    IndexMetadata,
    KnowledgeBase,
    RetrievalConfiguration,
    utc_now,
)
from app.storage.sparse_store import initialize_fts5_schema
from app.storage.vector_store import collection_name_for_knowledge_base

logger = logging.getLogger(__name__)

# ── Archive Contract ──────────────────────────────────────────────────────────

QKB_FORMAT = "quantiva-knowledge-base"
QKB_FORMAT_VERSION = 1

KB_METADATA_FILE = "kb_metadata.json"
DOCUMENTS_FILE = "documents.json"
CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.jsonl"
ORIGINAL_FILES_DIR = "original_files"
PROCESSED_FILES_DIR = "processed_files"

EXPORT_DIRNAME = "exports"
ARCHIVE_SUFFIX = ".qkb.zip"

#: Chunk/vector batch size used while rebuilding the dense index.
VECTOR_RESTORE_BATCH_SIZE = 128

_T = TypeVar("_T")


# ── Exceptions ────────────────────────────────────────────────────────────────

class KBArchiveError(Exception):
    """Base class for portable Knowledge Base archive failures."""


class KBArchiveValidationError(KBArchiveError, ValueError):
    """Raised when an archive is missing, corrupted, tampered with, or unsupported."""


# ── Small helpers ─────────────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_bytes(payload: Any, *, indent: int | None = 2) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        default=_json_default,
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Any]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
        for row in rows
    ).encode("utf-8")


def _batched(rows: Sequence[_T], size: int) -> Iterator[Sequence[_T]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _run_sync(factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """Execute an async implementation from a synchronous public API.

    When called from a synchronous desktop/Tauri command the coroutine simply
    runs on a fresh loop.  When a loop is already active (FastAPI request,
    pytest-asyncio) it is executed on a helper thread so callers never receive
    "event loop is already running" errors.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qkb-archive") as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


def _safe_relative(root: Path, relative: str | Path) -> Path:
    """Resolve *relative* inside *root*, rejecting zip-slip style escapes."""
    base = Path(root).expanduser().resolve()
    candidate = (base / str(relative)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise KBArchiveValidationError(
            f"Unsafe archive entry path: '{relative}' escapes the archive root"
        ) from exc
    return candidate


def _unique_name(session: Session, model: Any, base: str, max_length: int = 255) -> str:
    """Return a name that is unique for *model* (Embedding/Retrieval config)."""
    cleaned = (base or "Imported").strip() or "Imported"
    candidate = cleaned[:max_length]
    counter = 2
    while session.scalar(select(model.id).where(model.name == candidate)) is not None:
        suffix = f" ({counter})"
        candidate = f"{cleaned[: max_length - len(suffix)]}{suffix}"
        counter += 1
    return candidate


# ── Export snapshot ───────────────────────────────────────────────────────────

def _snapshot_knowledge_base(
    session_factory: sessionmaker[Session],
    knowledge_base_id: str,
) -> dict[str, Any]:
    """Read a machine-neutral, fully detached snapshot of one Knowledge Base."""
    with session_factory() as session:
        kb = session.get(KnowledgeBase, knowledge_base_id)
        if kb is None:
            raise KBArchiveError(f"KnowledgeBase not found: {knowledge_base_id}")

        index_meta = session.scalar(
            select(IndexMetadata).where(
                IndexMetadata.knowledge_base_id == knowledge_base_id
            )
        )
        embedding_config = (
            session.get(EmbeddingConfiguration, kb.embedding_config_id)
            if kb.embedding_config_id
            else None
        )
        retrieval_config = (
            session.get(RetrievalConfiguration, kb.retrieval_config_id)
            if kb.retrieval_config_id
            else None
        )

        documents = session.scalars(
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
            .order_by(Document.created_at.asc(), Document.id.asc())
        ).all()
        chunks = session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.knowledge_base_id == knowledge_base_id)
            .order_by(DocumentChunk.document_id.asc(), DocumentChunk.chunk_index.asc())
        ).all()

        snapshot: dict[str, Any] = {
            "knowledge_base": {
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "embedding_config_id": kb.embedding_config_id,
                "retrieval_config_id": kb.retrieval_config_id,
                "document_count": int(kb.document_count or 0),
                "chunk_count": int(kb.chunk_count or 0),
                "status": kb.status,
                "created_at": _iso(kb.created_at),
                "updated_at": _iso(kb.updated_at),
            },
            "embedding_configuration": (
                {
                    "id": embedding_config.id,
                    "name": embedding_config.name,
                    "provider_type": embedding_config.provider_type,
                    "model_name": embedding_config.model_name,
                    "base_url": embedding_config.base_url,
                    "dimensions": int(embedding_config.dimensions),
                    "batch_size": int(embedding_config.batch_size or 32),
                    "is_default": bool(embedding_config.is_default),
                    # credential_key_id / has_api_key are deliberately omitted:
                    # secrets live in the OS credential vault and never travel.
                }
                if embedding_config is not None
                else None
            ),
            "retrieval_configuration": (
                {
                    "id": retrieval_config.id,
                    "name": retrieval_config.name,
                    "search_mode": retrieval_config.search_mode,
                    "dense_top_k": int(retrieval_config.dense_top_k),
                    "sparse_top_k": int(retrieval_config.sparse_top_k),
                    "rrf_k": int(retrieval_config.rrf_k),
                    "dense_weight": float(retrieval_config.dense_weight),
                    "sparse_weight": float(retrieval_config.sparse_weight),
                    "enable_reranking": bool(retrieval_config.enable_reranking),
                    "reranker_provider": retrieval_config.reranker_provider,
                    "reranker_model": retrieval_config.reranker_model,
                    "final_top_k": int(retrieval_config.final_top_k),
                    "enable_agentic_mode": bool(retrieval_config.enable_agentic_mode),
                    "max_agent_iterations": int(retrieval_config.max_agent_iterations),
                    "is_default": bool(retrieval_config.is_default),
                }
                if retrieval_config is not None
                else None
            ),
            "index_metadata": (
                {
                    "id": index_meta.id,
                    "vector_collection_name": index_meta.vector_collection_name,
                    "embedding_provider": index_meta.embedding_provider,
                    "embedding_model_name": index_meta.embedding_model_name,
                    "embedding_dimensions": int(index_meta.embedding_dimensions),
                    "vector_count": int(index_meta.vector_count or 0),
                    "sparse_count": int(index_meta.sparse_count or 0),
                    "needs_reindex": bool(index_meta.needs_reindex),
                    "last_indexed_at": _iso(index_meta.last_indexed_at),
                }
                if index_meta is not None
                else None
            ),
            "documents": [
                {
                    "id": document.id,
                    "filename": document.filename,
                    "original_rel_path": document.original_rel_path,
                    "processed_rel_path": document.processed_rel_path,
                    "file_type": document.file_type,
                    "file_size_bytes": int(document.file_size_bytes or 0),
                    "sha256_hash": document.sha256_hash,
                    "page_count": document.page_count,
                    "chunk_count": int(document.chunk_count or 0),
                    "status": document.status,
                    "error_message": document.error_message,
                    "ocr_applied": bool(document.ocr_applied),
                    "metadata_json": dict(document.metadata_json or {}),
                    "created_at": _iso(document.created_at),
                    "updated_at": _iso(document.updated_at),
                }
                for document in documents
            ],
            "chunks": [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": int(chunk.chunk_index),
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "sheet_name": chunk.sheet_name,
                    "is_table": bool(chunk.is_table),
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "char_count": int(chunk.char_count or 0),
                    "metadata_json": dict(chunk.metadata_json or {}),
                    "created_at": _iso(chunk.created_at),
                }
                for chunk in chunks
            ],
        }
        return snapshot


def _archive_filename(kb_name: str, knowledge_base_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_filename(kb_name, max_length=80).replace(" ", "_")
    if not safe_name or safe_name == "document":
        safe_name = "knowledge_base"
    return f"{safe_name}_{knowledge_base_id[:8]}_{timestamp}{ARCHIVE_SUFFIX}"


async def _collect_dense_vectors(
    vector_store: Any,
    collection_name: str,
) -> list[dict[str, Any]]:
    scroll = getattr(vector_store, "scroll_vectors", None)
    if scroll is None:
        logger.warning(
            "Vector store %s does not support scroll_vectors; exporting without dense vectors.",
            type(vector_store).__name__,
        )
        return []
    records = await scroll(collection_name)
    return list(records or [])


# ── Public Export API ─────────────────────────────────────────────────────────

def export_knowledge_base(
    kb_id: str,
    *,
    service: KnowledgeBaseService | None = None,
    destination_dir: Path | str | None = None,
) -> Path:
    """Package a single Knowledge Base into a self-contained ``.qkb.zip``.

    Args:
        kb_id: Identifier of the Knowledge Base to export.
        service: Optional pre-built :class:`KnowledgeBaseService` (used by tests
            and dependency injection); defaults to the process singleton.
        destination_dir: Optional override for the destination folder.  The
            default is ``QUANTIVA_DATA_DIR / "exports"``.

    Returns:
        Absolute path to the generated archive.
    """
    return _run_sync(
        lambda: export_knowledge_base_async(
            kb_id,
            service=service,
            destination_dir=destination_dir,
        )
    )


async def export_knowledge_base_async(
    kb_id: str,
    *,
    service: KnowledgeBaseService | None = None,
    destination_dir: Path | str | None = None,
) -> Path:
    """Async implementation backing :func:`export_knowledge_base`."""
    kb_id_clean = str(kb_id).strip()
    if not kb_id_clean:
        raise ValueError("kb_id is required")

    svc = service or get_knowledge_base_service()
    snapshot = await asyncio.to_thread(
        _snapshot_knowledge_base, svc.session_factory, kb_id_clean
    )

    index_spec = snapshot["index_metadata"] or {}
    collection_name = (
        index_spec.get("vector_collection_name")
        or collection_name_for_knowledge_base(kb_id_clean)
    )
    exported_vectors = await _collect_dense_vectors(svc.vector_store, collection_name)

    chunk_ids = {str(chunk["id"]) for chunk in snapshot["chunks"]}
    vector_rows: list[dict[str, Any]] = []
    skipped_vectors = 0
    for record in exported_vectors:
        chunk_id = record.get("chunk_id")
        vector = record.get("vector") or []
        if not chunk_id or chunk_id not in chunk_ids or not vector:
            skipped_vectors += 1
            continue
        vector_rows.append(
            {
                "chunk_id": str(chunk_id),
                "point_id": record.get("point_id"),
                "document_id": (record.get("payload") or {}).get("document_id"),
                "vector": [float(value) for value in vector],
            }
        )

    data_dir = Path(getattr(svc.file_store, "data_dir", settings.QUANTIVA_DATA_DIR))
    if destination_dir is not None:
        destination = Path(destination_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
    else:
        destination = ensure_data_directories(data_dir).exports

    kb_name = str(snapshot["knowledge_base"].get("name") or "knowledge_base")
    archive_path = destination / _archive_filename(kb_name, kb_id_clean)
    if archive_path.exists():  # same-second export collision
        archive_path = destination / (
            f"{archive_path.stem}_{uuid.uuid4().hex[:8]}{ARCHIVE_SUFFIX}"
        )

    temp_archive = destination / f".tmp_{uuid.uuid4().hex}{ARCHIVE_SUFFIX}"
    checksums: dict[str, str] = {}
    missing_files: list[str] = []

    try:
        with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Original uploaded files, namespaced per document id (no collisions)
            for document in snapshot["documents"]:
                relative = document.get("original_rel_path")
                if not relative:
                    continue
                try:
                    source = svc.file_store.resolve_relative_path(relative)
                except (ValueError, OSError):
                    missing_files.append(str(relative))
                    continue
                if not source.is_file():
                    missing_files.append(str(relative))
                    continue
                arcname = (
                    f"{ORIGINAL_FILES_DIR}/{document['id']}/"
                    f"{sanitize_filename(str(document['filename']))}"
                )
                data = await asyncio.to_thread(source.read_bytes)
                zf.writestr(arcname, data)
                checksums[arcname] = _sha256(data)
                document["original_archive_path"] = arcname

            # 2. Optional processed artifacts (best effort)
            for document in snapshot["documents"]:
                relative = document.get("processed_rel_path")
                if not relative:
                    continue
                try:
                    source = svc.file_store.resolve_relative_path(relative)
                except (ValueError, OSError):
                    continue
                if not source.is_file():
                    continue
                arcname = (
                    f"{PROCESSED_FILES_DIR}/{document['id']}/"
                    f"{source.name}"
                )
                data = await asyncio.to_thread(source.read_bytes)
                zf.writestr(arcname, data)
                checksums[arcname] = _sha256(data)
                document["processed_archive_path"] = arcname

            # 3. Relational records (chunks carry the text; no re-extraction needed)
            documents_payload = _json_bytes(snapshot["documents"])
            zf.writestr(DOCUMENTS_FILE, documents_payload)
            checksums[DOCUMENTS_FILE] = _sha256(documents_payload)

            chunks_payload = _jsonl_bytes(snapshot["chunks"])
            zf.writestr(CHUNKS_FILE, chunks_payload)
            checksums[CHUNKS_FILE] = _sha256(chunks_payload)

            vectors_payload = _jsonl_bytes(vector_rows)
            zf.writestr(VECTORS_FILE, vectors_payload)
            checksums[VECTORS_FILE] = _sha256(vectors_payload)

            # 4. Manifest written last so it can describe every other member
            metadata = {
                "format": QKB_FORMAT,
                "format_version": QKB_FORMAT_VERSION,
                "app_version": settings.APP_VERSION,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "knowledge_base": snapshot["knowledge_base"],
                "embedding_configuration": snapshot["embedding_configuration"],
                "retrieval_configuration": snapshot["retrieval_configuration"],
                "index_metadata": snapshot["index_metadata"],
                "counts": {
                    "documents": len(snapshot["documents"]),
                    "chunks": len(snapshot["chunks"]),
                    "vectors": len(vector_rows),
                    "skipped_vectors": skipped_vectors,
                },
                "missing_files": missing_files,
                "checksums": checksums,
            }
            zf.writestr(KB_METADATA_FILE, _json_bytes(metadata))

        await asyncio.to_thread(os.replace, temp_archive, archive_path)
    except BaseException:
        try:
            temp_archive.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    logger.info(
        "Exported knowledge base %s to %s (documents=%d, chunks=%d, vectors=%d)",
        kb_id_clean,
        archive_path,
        len(snapshot["documents"]),
        len(snapshot["chunks"]),
        len(vector_rows),
    )
    return archive_path


# ── Archive reading / validation ──────────────────────────────────────────────

def _read_json_member(zf: zipfile.ZipFile, name: str) -> Any:
    try:
        raw = zf.read(name)
    except KeyError as exc:
        raise KBArchiveValidationError(
            f"Invalid .qkb archive: required member '{name}' is missing"
        ) from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KBArchiveValidationError(
            f"Corrupted .qkb archive: failed parsing '{name}'"
        ) from exc


def _read_jsonl_member(zf: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    if name not in zf.namelist():
        return []
    try:
        raw = zf.read(name).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        raise KBArchiveValidationError(
            f"Corrupted .qkb archive: cannot decode '{name}'"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise KBArchiveValidationError(
                f"Corrupted .qkb archive: invalid JSON on {name}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise KBArchiveValidationError(
                f"Corrupted .qkb archive: {name}:{line_number} is not an object"
            )
        rows.append(value)
    return rows


def _validate_member_paths(zf: zipfile.ZipFile) -> None:
    for member in zf.infolist():
        normalized = member.filename.replace("\\", "/")
        parts = Path(normalized).parts
        if (
            normalized.startswith(("/", "../"))
            or ".." in parts
            or (parts and ":" in parts[0])
        ):
            raise KBArchiveValidationError(
                f"Unsafe archive member path: '{member.filename}'"
            )


def _verify_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise KBArchiveValidationError("Invalid .qkb archive: kb_metadata.json is not an object")
    if metadata.get("format") != QKB_FORMAT:
        raise KBArchiveValidationError(
            "Invalid .qkb archive: unexpected format "
            f"'{metadata.get('format')}' (expected '{QKB_FORMAT}')"
        )
    try:
        version = int(metadata.get("format_version") or 0)
    except (TypeError, ValueError) as exc:
        raise KBArchiveValidationError(
            "Invalid .qkb archive: format_version is not an integer"
        ) from exc
    if version <= 0 or version > QKB_FORMAT_VERSION:
        raise KBArchiveValidationError(
            f"Unsupported .qkb format_version {version} "
            f"(this build supports up to {QKB_FORMAT_VERSION})"
        )
    return metadata


def _verify_checksums(zf: zipfile.ZipFile, metadata: dict[str, Any]) -> None:
    checksums = metadata.get("checksums")
    if not checksums:
        return
    if not isinstance(checksums, dict):
        raise KBArchiveValidationError(
            "Invalid .qkb archive: checksums must be an object"
        )
    names = set(zf.namelist())
    for arcname, expected in checksums.items():
        if arcname not in names:
            raise KBArchiveValidationError(
                f"Corrupted .qkb archive: expected member '{arcname}' is missing"
            )
        actual = _sha256(zf.read(arcname))
        if actual != expected:
            raise KBArchiveValidationError(
                f"Archive integrity verification failed for '{arcname}' "
                f"(expected {expected}, got {actual})"
            )


def _extract_archive(zf: zipfile.ZipFile, destination: Path) -> None:
    for member in zf.infolist():
        if member.is_dir():
            continue
        target = _safe_relative(destination, member.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member, "r") as source, open(target, "wb") as sink:
            shutil.copyfileobj(source, sink)


# ── Import resolution helpers ─────────────────────────────────────────────────

def _resolve_kb_name(
    session: Session,
    requested_name: str | None,
    original_name: str,
) -> str:
    if requested_name is not None:
        candidate = str(requested_name).strip()
        if not candidate:
            raise ValueError("Imported knowledge base name cannot be empty")
        existing = session.scalar(
            select(KnowledgeBase.id).where(KnowledgeBase.name == candidate)
        )
        if existing is not None:
            raise ValueError(
                f"KnowledgeBase with name '{candidate}' already exists."
            )
        return candidate

    base = f"{str(original_name).strip() or 'Imported Knowledge Base'} (Imported)"
    candidate = base[:255]
    counter = 2
    while session.scalar(
        select(KnowledgeBase.id).where(KnowledgeBase.name == candidate)
    ) is not None:
        suffix = f" {counter}"
        candidate = f"{base[: 255 - len(suffix)]}{suffix}"
        counter += 1
    return candidate


def _resolve_embedding_configuration(
    session: Session,
    spec: dict[str, Any] | None,
) -> EmbeddingConfiguration | None:
    """Match the archived embedding spec against the target machine.

    Order: exact id + identical model space -> exact provider/model/dimensions
    -> clone the archived spec (without credentials) -> local default.
    """
    if not spec:
        return (
            session.scalar(
                select(EmbeddingConfiguration).where(
                    EmbeddingConfiguration.is_default == True
                )
            )
            or session.scalar(select(EmbeddingConfiguration))
        )

    provider = str(spec.get("provider_type") or "").strip()
    model_name = str(spec.get("model_name") or "").strip()
    try:
        dimensions = int(spec.get("dimensions") or 0)
    except (TypeError, ValueError):
        dimensions = 0

    spec_id = spec.get("id")
    if spec_id:
        by_id = session.get(EmbeddingConfiguration, str(spec_id))
        if (
            by_id is not None
            and by_id.provider_type == provider
            and by_id.model_name == model_name
            and int(by_id.dimensions) == dimensions
        ):
            return by_id

    if provider and model_name and dimensions > 0:
        exact = session.scalar(
            select(EmbeddingConfiguration).where(
                EmbeddingConfiguration.provider_type == provider,
                EmbeddingConfiguration.model_name == model_name,
                EmbeddingConfiguration.dimensions == dimensions,
            )
        )
        if exact is not None:
            return exact

    # Clone the archived spec when the target machine has no equivalent.
    if not provider or not model_name or dimensions <= 0:
        return (
            session.scalar(
                select(EmbeddingConfiguration).where(
                    EmbeddingConfiguration.is_default == True
                )
            )
            or session.scalar(select(EmbeddingConfiguration))
        )

    cloned = EmbeddingConfiguration(
        id=str(uuid.uuid4()),
        name=_unique_name(
            session,
            EmbeddingConfiguration,
            str(spec.get("name") or f"{model_name} (Imported)"),
        ),
        provider_type=provider,
        model_name=model_name,
        base_url=spec.get("base_url"),
        credential_key_id=None,
        has_api_key=False,
        dimensions=dimensions,
        batch_size=int(spec.get("batch_size") or 32),
        is_default=False,
    )
    session.add(cloned)
    session.flush()
    logger.info(
        "Created local embedding configuration '%s' for imported knowledge base.",
        cloned.name,
    )
    return cloned


def _resolve_retrieval_configuration(
    session: Session,
    spec: dict[str, Any] | None,
) -> RetrievalConfiguration | None:
    if not spec:
        return (
            session.scalar(
                select(RetrievalConfiguration).where(
                    RetrievalConfiguration.is_default == True
                )
            )
            or session.scalar(select(RetrievalConfiguration))
        )

    spec_id = spec.get("id")
    if spec_id:
        by_id = session.get(RetrievalConfiguration, str(spec_id))
        if by_id is not None:
            return by_id

    name = str(spec.get("name") or "").strip()
    if name:
        by_name = session.scalar(
            select(RetrievalConfiguration).where(
                RetrievalConfiguration.name == name
            )
        )
        if by_name is not None:
            return by_name

    cloned = RetrievalConfiguration(
        id=str(uuid.uuid4()),
        name=_unique_name(
            session,
            RetrievalConfiguration,
            name or "Imported Retrieval Configuration",
        ),
        search_mode=str(spec.get("search_mode") or "hybrid"),
        dense_top_k=int(spec.get("dense_top_k") or 20),
        sparse_top_k=int(spec.get("sparse_top_k") or 20),
        rrf_k=int(spec.get("rrf_k") or 60),
        dense_weight=float(spec.get("dense_weight") or 1.0),
        sparse_weight=float(spec.get("sparse_weight") or 1.0),
        enable_reranking=bool(spec.get("enable_reranking", True)),
        reranker_provider=spec.get("reranker_provider"),
        reranker_model=spec.get("reranker_model"),
        final_top_k=int(spec.get("final_top_k") or 8),
        enable_agentic_mode=bool(spec.get("enable_agentic_mode", True)),
        max_agent_iterations=int(spec.get("max_agent_iterations") or 5),
        is_default=False,
    )
    session.add(cloned)
    session.flush()
    return cloned


def _target_embedding_space(
    embedding_config: EmbeddingConfiguration | None,
    index_spec: dict[str, Any],
) -> tuple[int, str, str]:
    if embedding_config is not None:
        return (
            int(embedding_config.dimensions),
            str(embedding_config.provider_type),
            str(embedding_config.model_name),
        )
    return (
        int(index_spec.get("embedding_dimensions") or 0),
        str(index_spec.get("embedding_provider") or ""),
        str(index_spec.get("embedding_model_name") or ""),
    )


def _restored_document_status(archived: Any, chunk_count: int) -> str:
    value = str(archived or "").strip().upper()
    if value in {"READY", "FAILED", "ERROR", "COMPLETED"}:
        return value
    # A queued/processing source document has no worker on the target machine,
    # so derive a truthful terminal status from the chunks that came with it.
    return "READY" if chunk_count > 0 else "PENDING"


def _materialize_document_files(
    file_store: LocalFileStore,
    extracted_dir: Path,
    knowledge_base_id: str,
    documents: Sequence[dict[str, Any]],
    document_id_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Restore original/processed payloads into LocalFileStore.

    Returns per-document storage facts in the same order as *documents*.
    """
    materialized: list[dict[str, Any]] = []
    for document in documents:
        old_id = str(document["id"])
        new_id = document_id_map[old_id]
        filename = str(document.get("filename") or "document.bin")
        archived_hash = str(document.get("sha256_hash") or "")

        stored_original = None
        archived_original = document.get("original_archive_path")
        if archived_original:
            source = _safe_relative(extracted_dir, str(archived_original))
            if source.is_file():
                stored_original = file_store.save_original(
                    knowledge_base_id=knowledge_base_id,
                    document_id=new_id,
                    filename=filename,
                    content=source.read_bytes(),
                )
                if archived_hash and stored_original.sha256_hash != archived_hash:
                    logger.warning(
                        "Imported file hash mismatch for document %s (%s != %s)",
                        old_id,
                        stored_original.sha256_hash,
                        archived_hash,
                    )

        processed_rel_path = None
        archived_processed = document.get("processed_archive_path")
        if archived_processed:
            source = _safe_relative(extracted_dir, str(archived_processed))
            if source.is_file():
                stored_processed = file_store.save_processed(
                    knowledge_base_id=knowledge_base_id,
                    document_id=new_id,
                    content=source.read_bytes(),
                    ext=source.suffix.lstrip(".") or "json",
                )
                processed_rel_path = stored_processed.relative_path

        materialized.append(
            {
                "original_rel_path": (
                    stored_original.relative_path
                    if stored_original is not None
                    else file_store.get_original_relative_path(
                        knowledge_base_id, new_id, filename
                    )
                ),
                "processed_rel_path": processed_rel_path,
                "sha256_hash": (
                    stored_original.sha256_hash
                    if stored_original is not None
                    else archived_hash
                ),
                "file_size_bytes": (
                    stored_original.file_size_bytes
                    if stored_original is not None
                    else int(document.get("file_size_bytes") or 0)
                ),
            }
        )
    return materialized


_SPARSE_BACKFILL_SQL = """
INSERT INTO document_chunks_fts(
    chunk_id, document_id, knowledge_base_id,
    source_file, section_title, content
)
SELECT
    chunk.id,
    chunk.document_id,
    chunk.knowledge_base_id,
    document.filename,
    COALESCE(chunk.section_title, ''),
    chunk.content
FROM document_chunks AS chunk
JOIN documents AS document ON document.id = chunk.document_id
WHERE chunk.knowledge_base_id = :kb_id
  AND NOT EXISTS (
      SELECT 1
      FROM document_chunks_fts AS fts
      WHERE fts.chunk_id = chunk.id
  );
"""

_SPARSE_COUNT_SQL = (
    "SELECT COUNT(*) FROM document_chunks_fts WHERE knowledge_base_id = :kb_id"
)


def _synchronize_sparse_index(bind: Any, knowledge_base_id: str, fallback: int) -> int:
    """Ensure FTS5 rows exist for every restored chunk and return the count."""
    if not isinstance(bind, Engine):
        return fallback
    try:
        with bind.begin() as connection:
            connection.execute(text(_SPARSE_BACKFILL_SQL), {"kb_id": knowledge_base_id})
            count = connection.execute(
                text(_SPARSE_COUNT_SQL), {"kb_id": knowledge_base_id}
            ).scalar()
        return int(count or 0)
    except Exception as exc:  # pragma: no cover - defensive for non-FTS binds
        logger.warning(
            "Sparse index synchronization failed for imported KB %s: %s",
            knowledge_base_id,
            exc,
        )
        return fallback


def _persist_imported_knowledge_base(
    session_factory: sessionmaker[Session],
    file_store: LocalFileStore,
    extracted_dir: Path,
    metadata: dict[str, Any],
    documents: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    new_name: str | None,
) -> dict[str, Any]:
    """Insert fresh-UUID KB/Document/DocumentChunk rows and restore files."""
    kb_spec = dict(metadata.get("knowledge_base") or {})
    embedding_spec = metadata.get("embedding_configuration")
    retrieval_spec = metadata.get("retrieval_configuration")
    index_spec = dict(metadata.get("index_metadata") or {})

    new_kb_id = str(uuid.uuid4())
    document_id_map = {str(document["id"]): str(uuid.uuid4()) for document in documents}
    chunk_id_map = {str(chunk["id"]): str(uuid.uuid4()) for chunk in chunks}

    with session_factory() as session:
        bind = session.get_bind()
    if isinstance(bind, Engine):
        try:
            initialize_fts5_schema(bind)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("FTS5 schema initialization notice: %s", exc)

    # A brand new UUID should never collide, but never trust leftover state.
    file_store.delete_knowledge_base(new_kb_id)

    try:
        materialized = _materialize_document_files(
            file_store,
            extracted_dir,
            new_kb_id,
            documents,
            document_id_map,
        )

        chunks_by_document: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            chunks_by_document.setdefault(str(chunk["document_id"]), []).append(chunk)

        with session_factory() as session:
            kb_name = _resolve_kb_name(
                session,
                new_name,
                str(kb_spec.get("name") or "Imported Knowledge Base"),
            )
            embedding_config = _resolve_embedding_configuration(
                session, embedding_spec
            )
            retrieval_config = _resolve_retrieval_configuration(
                session, retrieval_spec
            )
            target_dims, target_provider, target_model = _target_embedding_space(
                embedding_config, index_spec
            )

            knowledge_base = KnowledgeBase(
                id=new_kb_id,
                name=kb_name,
                description=kb_spec.get("description"),
                embedding_config_id=(
                    embedding_config.id if embedding_config is not None else None
                ),
                retrieval_config_id=(
                    retrieval_config.id if retrieval_config is not None else None
                ),
                document_count=len(documents),
                chunk_count=len(chunks),
                status=str(kb_spec.get("status") or "ready"),
            )
            created_at = _parse_datetime(kb_spec.get("created_at"))
            updated_at = _parse_datetime(kb_spec.get("updated_at"))
            if created_at is not None:
                knowledge_base.created_at = created_at
            if updated_at is not None:
                knowledge_base.updated_at = updated_at
            session.add(knowledge_base)
            session.flush()

            index_metadata = IndexMetadata(
                id=str(uuid.uuid4()),
                knowledge_base_id=new_kb_id,
                vector_collection_name=collection_name_for_knowledge_base(new_kb_id),
                embedding_provider=target_provider,
                embedding_model_name=target_model,
                embedding_dimensions=target_dims,
                vector_count=0,
                sparse_count=0,
                needs_reindex=True,
                last_indexed_at=None,
            )
            session.add(index_metadata)

            for document, stored in zip(documents, materialized):
                old_id = str(document["id"])
                chunk_count = len(chunks_by_document.get(old_id, ()))
                document_row = Document(
                    id=document_id_map[old_id],
                    knowledge_base_id=new_kb_id,
                    filename=str(document.get("filename") or "document.bin"),
                    original_rel_path=stored["original_rel_path"],
                    processed_rel_path=stored["processed_rel_path"],
                    file_type=str(
                        document.get("file_type")
                        or Path(str(document.get("filename") or "")).suffix.lstrip(".")
                        or "bin"
                    ),
                    file_size_bytes=int(stored["file_size_bytes"]),
                    sha256_hash=str(stored["sha256_hash"]),
                    page_count=document.get("page_count"),
                    chunk_count=chunk_count,
                    status=_restored_document_status(
                        document.get("status"), chunk_count
                    ),
                    error_message=document.get("error_message"),
                    ocr_applied=bool(document.get("ocr_applied", False)),
                    metadata_json=dict(document.get("metadata_json") or {}),
                )
                doc_created = _parse_datetime(document.get("created_at"))
                doc_updated = _parse_datetime(document.get("updated_at"))
                if doc_created is not None:
                    document_row.created_at = doc_created
                if doc_updated is not None:
                    document_row.updated_at = doc_updated
                session.add(document_row)

            # Flush documents before chunks so FTS5 triggers can resolve
            # ``source_file`` from the documents table.
            session.flush()

            for chunk in chunks:
                content = str(chunk.get("content") or "")
                chunk_row = DocumentChunk(
                    id=chunk_id_map[str(chunk["id"])],
                    document_id=document_id_map[str(chunk["document_id"])],
                    knowledge_base_id=new_kb_id,
                    chunk_index=int(chunk.get("chunk_index") or 0),
                    page_number=chunk.get("page_number"),
                    section_title=chunk.get("section_title"),
                    sheet_name=chunk.get("sheet_name"),
                    is_table=bool(chunk.get("is_table", False)),
                    content=content,
                    content_hash=str(chunk.get("content_hash") or ""),
                    char_count=int(chunk.get("char_count") or len(content)),
                    metadata_json=dict(chunk.get("metadata_json") or {}),
                )
                chunk_created = _parse_datetime(chunk.get("created_at"))
                if chunk_created is not None:
                    chunk_row.created_at = chunk_created
                session.add(chunk_row)

            session.commit()
    except BaseException:
        # Never leave orphaned documents behind when the relational insert fails.
        file_store.delete_knowledge_base(new_kb_id)
        raise

    sparse_count = _synchronize_sparse_index(bind, new_kb_id, len(chunks))

    archive_dims = int(index_spec.get("embedding_dimensions") or 0)
    archive_provider = str(index_spec.get("embedding_provider") or "")
    archive_model = str(index_spec.get("embedding_model_name") or "")
    vectors_compatible = (
        target_dims > 0
        and archive_dims == target_dims
        and archive_provider == target_provider
        and archive_model == target_model
    )

    return {
        "knowledge_base_id": new_kb_id,
        "document_id_map": document_id_map,
        "chunk_id_map": chunk_id_map,
        "document_filenames": {
            str(document["id"]): str(document.get("filename") or "")
            for document in documents
        },
        "embedding_dimensions": target_dims,
        "embedding_provider": target_provider,
        "embedding_model_name": target_model,
        "sparse_count": sparse_count,
        "chunk_count": len(chunks),
        "vectors_compatible": vectors_compatible,
        "archive_dimensions": archive_dims,
    }


def _build_chunk_payloads(
    chunks: Sequence[dict[str, Any]],
    knowledge_base_id: str,
    document_id_map: dict[str, str],
    chunk_id_map: dict[str, str],
    document_filenames: dict[str, str],
) -> list[DocumentChunkPayload]:
    payloads: list[DocumentChunkPayload] = []
    for chunk in chunks:
        old_doc_id = str(chunk["document_id"])
        content = str(chunk.get("content") or "")
        payloads.append(
            {
                "chunk_id": chunk_id_map[str(chunk["id"])],
                "document_id": document_id_map[old_doc_id],
                "knowledge_base_id": knowledge_base_id,
                "page_number": chunk.get("page_number"),
                "chunk_index": int(chunk.get("chunk_index") or 0),
                "content": content,
                "source_file": document_filenames.get(old_doc_id, ""),
                "section_title": chunk.get("section_title"),
                "sheet_name": chunk.get("sheet_name"),
                "is_table": bool(chunk.get("is_table", False)),
                "content_hash": str(chunk.get("content_hash") or ""),
                "char_count": int(chunk.get("char_count") or len(content)),
            }
        )
    return payloads


# ── Public Import API ─────────────────────────────────────────────────────────

def import_knowledge_base(
    archive_path: Path,
    new_name: str | None = None,
    *,
    service: KnowledgeBaseService | None = None,
) -> KnowledgeBase:
    """Import a ``.qkb.zip`` archive as a brand new Knowledge Base.

    Every imported Knowledge Base, Document, and DocumentChunk receives a fresh
    UUID, so importing the same archive repeatedly never collides.
    """
    return _run_sync(
        lambda: import_knowledge_base_async(
            archive_path,
            new_name,
            service=service,
        )
    )


async def import_knowledge_base_async(
    archive_path: Path,
    new_name: str | None = None,
    *,
    service: KnowledgeBaseService | None = None,
) -> KnowledgeBase:
    """Async implementation backing :func:`import_knowledge_base`."""
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Knowledge base archive not found at: {archive}")
    if not zipfile.is_zipfile(archive):
        raise KBArchiveValidationError(
            f"Invalid .qkb archive: '{archive}' is not a valid zip file"
        )

    svc = service or get_knowledge_base_service()

    with zipfile.ZipFile(archive, "r") as zf:
        _validate_member_paths(zf)
        metadata = _verify_metadata(_read_json_member(zf, KB_METADATA_FILE))
        _verify_checksums(zf, metadata)

        documents = _read_json_member(zf, DOCUMENTS_FILE)
        if not isinstance(documents, list):
            raise KBArchiveValidationError(
                "Invalid .qkb archive: documents.json must be a list"
            )
        chunks = _read_jsonl_member(zf, CHUNKS_FILE)
        vectors = _read_jsonl_member(zf, VECTORS_FILE)

        staging_dir = Path(tempfile.mkdtemp(prefix="quantiva_qkb_import_"))
        try:
            await asyncio.to_thread(_extract_archive, zf, staging_dir)

            for document in documents:
                if not document.get("id"):
                    raise KBArchiveValidationError(
                        "Invalid .qkb archive: document record without id"
                    )
            document_ids = {str(document["id"]) for document in documents}
            for chunk in chunks:
                if not chunk.get("id") or not chunk.get("document_id"):
                    raise KBArchiveValidationError(
                        "Invalid .qkb archive: chunk record without id/document_id"
                    )
                if str(chunk["document_id"]) not in document_ids:
                    raise KBArchiveValidationError(
                        "Invalid .qkb archive: chunk "
                        f"'{chunk['id']}' references unknown document "
                        f"'{chunk['document_id']}'"
                    )

            persisted = await asyncio.to_thread(
                _persist_imported_knowledge_base,
                svc.session_factory,
                svc.file_store,
                staging_dir,
                metadata,
                documents,
                chunks,
                new_name,
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    knowledge_base_id = str(persisted["knowledge_base_id"])
    dimensions = int(persisted["embedding_dimensions"])
    collection_name = collection_name_for_knowledge_base(knowledge_base_id)

    vector_map: dict[str, list[float]] = {}
    for row in vectors:
        chunk_id = row.get("chunk_id")
        values = row.get("vector")
        if not chunk_id or not isinstance(values, list) or not values:
            continue
        try:
            vector_map[str(chunk_id)] = [float(value) for value in values]
        except (TypeError, ValueError):
            continue

    vectors_compatible = bool(persisted["vectors_compatible"])
    vector_rows: list[tuple[DocumentChunkPayload, list[float]]] = []
    chunk_id_map: dict[str, str] = persisted["chunk_id_map"]
    document_id_map: dict[str, str] = persisted["document_id_map"]
    document_filenames: dict[str, str] = persisted["document_filenames"]

    payloads = _build_chunk_payloads(
        chunks,
        knowledge_base_id,
        document_id_map,
        chunk_id_map,
        document_filenames,
    )
    for chunk, payload in zip(chunks, payloads):
        vector = vector_map.get(str(chunk["id"]))
        if vector is None:
            continue
        if dimensions <= 0 or len(vector) != dimensions:
            vectors_compatible = False
            logger.info(
                "Dense vector for chunk %s has %d dimensions but target index "
                "expects %d; import will require a reindex.",
                chunk["id"],
                len(vector),
                dimensions,
            )
            vector_rows = []
            break
        vector_rows.append((payload, vector))

    needs_reindex = True
    restored_vectors = 0
    try:
        await svc.vector_store.ensure_collection(
            collection_name,
            dimensions,
            index_metadata={
                "vector_collection_name": collection_name,
                "embedding_dimensions": dimensions,
            },
        )

        if vectors_compatible and vector_rows:
            for batch in _batched(vector_rows, VECTOR_RESTORE_BATCH_SIZE):
                await svc.vector_store.upsert_chunks(
                    collection_name=collection_name,
                    chunks=[payload for payload, _ in batch],
                    embeddings=[vector for _, vector in batch],
                    metadata={"embedding_dimensions": dimensions},
                )
            collection_info = await svc.vector_store.get_collection_info(
                collection_name
            )
            restored_vectors = (
                int(collection_info.point_count or 0)
                if collection_info is not None
                else 0
            )
            needs_reindex = restored_vectors < int(persisted["chunk_count"])
        else:
            needs_reindex = int(persisted["chunk_count"]) > 0
    except Exception as exc:
        logger.error(
            "Dense vector restore failed for imported KB %s: %s",
            knowledge_base_id,
            exc,
        )
        # Never leave a half-written dense index: recreate it empty and let the
        # reindex flag drive recovery.
        try:
            await svc.vector_store.delete_collection(collection_name)
            await svc.vector_store.ensure_collection(collection_name, dimensions)
        except Exception as cleanup_exc:  # pragma: no cover - defensive
            logger.warning(
                "Could not reset vector collection %s: %s",
                collection_name,
                cleanup_exc,
            )
        needs_reindex = True
        restored_vectors = 0

    try:
        with svc.session_factory() as session:
            index_metadata = session.scalar(
                select(IndexMetadata).where(
                    IndexMetadata.knowledge_base_id == knowledge_base_id
                )
            )
            if index_metadata is not None:
                index_metadata.vector_count = restored_vectors
                index_metadata.sparse_count = int(persisted["sparse_count"])
                index_metadata.needs_reindex = needs_reindex
                index_metadata.last_indexed_at = (
                    utc_now()
                    if not needs_reindex and restored_vectors > 0
                    else None
                )
            session.commit()

            knowledge_base = session.get(KnowledgeBase, knowledge_base_id)
            if knowledge_base is None:  # pragma: no cover - defensive
                raise KBArchiveError(
                    f"Imported KnowledgeBase disappeared: {knowledge_base_id}"
                )
            session.refresh(knowledge_base)
            session.expunge(knowledge_base)
    except BaseException:
        try:
            await svc.delete_knowledge_base(knowledge_base_id)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.exception(
                "Failed cleaning up partially imported KB %s", knowledge_base_id
            )
        raise

    logger.info(
        "Imported knowledge base '%s' (%s) from %s "
        "(documents=%d, chunks=%d, vectors=%d, needs_reindex=%s)",
        knowledge_base.name,
        knowledge_base_id,
        archive.name,
        len(documents),
        len(chunks),
        restored_vectors,
        needs_reindex,
    )
    return knowledge_base


__all__ = [
    "ARCHIVE_SUFFIX",
    "CHUNKS_FILE",
    "DOCUMENTS_FILE",
    "EXPORT_DIRNAME",
    "KB_METADATA_FILE",
    "QKB_FORMAT",
    "QKB_FORMAT_VERSION",
    "VECTORS_FILE",
    "KBArchiveError",
    "KBArchiveValidationError",
    "export_knowledge_base",
    "export_knowledge_base_async",
    "import_knowledge_base",
    "import_knowledge_base_async",
]
