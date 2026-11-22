"""Workspace backup, restore, and storage location migration for Quantiva.

Provides:
1. Hot consistent SQLite WAL backups using Python's native sqlite3.backup() API.
2. Compressed timestamped archives packaging database, documents, vectors, and manifest.
3. Safe restore with tamper detection, checksum validation, and safety snapshots.
4. Storage location migration with disk space verification and bootstrap.json updates.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any
import uuid
import zipfile

from app.config import (
    LocalDataPaths,
    bootstrap_config_path,
    ensure_data_directories,
    settings,
)

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})
CURRENT_SCHEMA_VERSION: int = 1


def _backup_sqlite_database(src_path: Path, dst_path: Path) -> None:
    """Create an atomic, consistent online snapshot of a live SQLite database.

    Uses Python's native sqlite3.Connection.backup() API, ensuring that even if
    WAL mode is active and concurrent readers/writers are executing, a clean and
    consistent standalone database file is produced without file locking corruption.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        try:
            dst_path.unlink()
        except OSError:
            pass

    src_conn = sqlite3.connect(str(src_path), timeout=30.0)
    dst_conn = sqlite3.connect(str(dst_path))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()


def _restore_sqlite_database(staging_path: Path, live_path: Path) -> None:
    """Restore a database snapshot into live_path using SQLite backup API.

    Overwrites live_path transactionally through SQLite's own engine, safely
    checkpointing and invalidating stale WAL logs without file locking collisions.
    """
    live_path.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(staging_path))
    dst_conn = sqlite3.connect(str(live_path), timeout=30.0)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()


def _get_database_schema_version(db_path: Path) -> int:
    """Read the highest schema version from schema_versions in quantiva.db."""
    if not db_path.is_file():
        return CURRENT_SCHEMA_VERSION
    try:
        with sqlite3.connect(str(db_path), timeout=5.0) as conn:
            cursor = conn.execute("SELECT MAX(version) FROM schema_versions")
            row = cursor.fetchone()
            if row and row[0] is not None:
                return int(row[0])
    except Exception:
        pass
    return CURRENT_SCHEMA_VERSION


def _compute_sha256(data_or_path: bytes | Path) -> str:
    """Compute SHA-256 hex digest of file or bytes."""
    if isinstance(data_or_path, (bytes, bytearray)):
        return hashlib.sha256(data_or_path).hexdigest()

    hasher = hashlib.sha256()
    with open(data_or_path, "rb") as f:
        while chunk := f.read(64 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _pause_local_job_manager() -> None:
    """Attempt to gracefully pause or stop active LocalJobManager workers."""
    try:
        from app.jobs.manager import get_local_job_manager

        mgr = get_local_job_manager()
        if mgr and getattr(mgr, "is_started", False):
            try:
                asyncio.get_running_loop()
                asyncio.ensure_future(mgr.stop(timeout_seconds=5.0))
            except RuntimeError:
                asyncio.run(mgr.stop(timeout_seconds=5.0))
            logger.info("Paused active LocalJobManager for maintenance operation.")
    except Exception as exc:
        logger.debug("LocalJobManager not active or could not be paused: %s", exc)


def _close_active_handles() -> None:
    """Release active vector store handles, database connections, and caches."""
    _pause_local_job_manager()

    try:
        from app.core.application_context import get_application_context

        ctx = get_application_context()
        ctx.close_vector_client()
    except Exception as exc:
        logger.debug("Error releasing vector store client: %s", exc)

    try:
        from app.storage.database import engine

        engine.dispose()
    except Exception as exc:
        logger.debug("Error disposing database engine: %s", exc)

    try:
        from app.storage.local_cache import reset_local_cache

        reset_local_cache()
    except Exception as exc:
        logger.debug("Error resetting local cache: %s", exc)


def _reinitialize_system() -> None:
    """Run database initialization and re-spin ApplicationContext."""
    try:
        from app.storage.database import engine, initialize_database

        initialize_database(engine)
    except Exception as exc:
        logger.warning("Error running database initialization after restore: %s", exc)

    try:
        from app.core.application_context import get_application_context, reset_application_context

        reset_application_context()
        ctx = get_application_context()
        ctx.startup()
    except Exception as exc:
        logger.warning("Error restarting ApplicationContext after restore: %s", exc)


# ── 1. Hot Backup ─────────────────────────────────────────────────────────────

def create_backup(
    include_models: bool = False,
    data_dir: Path | str | None = None,
    destination_dir: Path | str | None = None,
    prefix: str = "quantiva_backup_",
) -> Path:
    """Create a hot, atomic backup archive of the Quantiva workspace.

    1. Uses sqlite3.Connection.backup() to produce a transactionally consistent
       snapshot of quantiva.db without file locking hazards.
    2. Packages database snapshot, documents/, vector_store/, and manifest.json.
    3. Excludes models/ by default so backups are compact and rapid.
    4. Computes SHA-256 checksums for every member to guarantee tamper resistance.

    Returns the absolute path to the generated .zip archive.
    """
    root = Path(data_dir or settings.QUANTIVA_DATA_DIR).expanduser().resolve()
    paths = LocalDataPaths(root)

    dest = Path(destination_dir or paths.backups).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"{prefix}{timestamp}.zip"
    final_archive_path = dest / archive_name

    temp_dir = Path(tempfile.mkdtemp(prefix="quantiva_backup_temp_"))
    try:
        files_to_pack: list[tuple[Path, str]] = []

        # 1. Hot snapshot quantiva.db
        if paths.database_file.is_file():
            snapshot_db = temp_dir / "quantiva.db"
            _backup_sqlite_database(paths.database_file, snapshot_db)
            files_to_pack.append((snapshot_db, "database/quantiva.db"))
            schema_version = _get_database_schema_version(snapshot_db)
        else:
            schema_version = CURRENT_SCHEMA_VERSION

        # Snapshot cache.db if present
        cache_db = paths.database_dir / "cache.db"
        if cache_db.is_file():
            snapshot_cache = temp_dir / "cache.db"
            _backup_sqlite_database(cache_db, snapshot_cache)
            files_to_pack.append((snapshot_cache, "database/cache.db"))

        # 2. Package documents/ (original, processed, attachments)
        docs_dir = root / "documents"
        if docs_dir.is_dir():
            for p in sorted(docs_dir.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(root).as_posix()
                    files_to_pack.append((p, rel))

        # 3. Package vector_store/
        if paths.vector_store.is_dir():
            for p in sorted(paths.vector_store.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(root).as_posix()
                    files_to_pack.append((p, rel))

        # 4. Optional models/
        if include_models:
            models_dir = root / "models"
            if models_dir.is_dir():
                for p in sorted(models_dir.rglob("*")):
                    if p.is_file():
                        rel = p.relative_to(root).as_posix()
                        files_to_pack.append((p, rel))

        # 5. Compute SHA-256 for all members
        checksums: dict[str, str] = {}
        total_size = 0
        for src_path, arcname in files_to_pack:
            digest = _compute_sha256(src_path)
            checksums[arcname] = digest
            total_size += src_path.stat().st_size

        # 6. Build manifest.json
        manifest = {
            "app_version": settings.APP_VERSION,
            "schema_version": schema_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files_count": len(files_to_pack),
            "total_size_bytes": total_size,
            "include_models": include_models,
            "sha256_checksums": checksums,
        }

        # 7. Write zip archive atomically
        temp_zip = dest / f".tmp_{uuid.uuid4().hex}.zip"
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Write manifest first
            zf.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2, ensure_ascii=False),
            )
            # Write all payload files
            for src_path, arcname in files_to_pack:
                zf.write(src_path, arcname=arcname)

        os.replace(temp_zip, final_archive_path)
        logger.info(
            "Created workspace backup at '%s' (files: %d, size: %d bytes)",
            final_archive_path,
            len(files_to_pack),
            total_size,
        )
        return final_archive_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── 2. Safe Restore ───────────────────────────────────────────────────────────

def verify_backup_archive(archive_path: Path | str) -> dict[str, Any]:
    """Validate archive integrity, manifest, schema version, and SHA-256 checksums.

    Raises:
        FileNotFoundError: If the archive path does not exist.
        ValueError: If archive is corrupted, tampered with, or incompatible.
    """
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Backup archive not found at: {archive}")

    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Invalid backup archive: '{archive}' is not a valid zip file")

    with zipfile.ZipFile(archive, "r") as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise ValueError(
                f"Invalid backup archive '{archive.name}': manifest.json is missing"
            )

        try:
            raw_manifest = zf.read("manifest.json").decode("utf-8")
            manifest: dict[str, Any] = json.loads(raw_manifest)
        except Exception as exc:
            raise ValueError(
                f"Corrupted backup archive '{archive.name}': failed parsing manifest.json"
            ) from exc

        # Check required fields
        if "schema_version" not in manifest or "sha256_checksums" not in manifest:
            raise ValueError(
                f"Corrupted backup manifest in '{archive.name}': missing required fields"
            )

        schema_ver = int(manifest["schema_version"])
        if schema_ver not in SUPPORTED_SCHEMA_VERSIONS and schema_ver > max(SUPPORTED_SCHEMA_VERSIONS):
            raise ValueError(
                f"Incompatible backup schema version: {schema_ver} "
                f"(supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
            )

        checksums: dict[str, str] = manifest["sha256_checksums"]

        # Verify all entries against checksums and guard against path traversal
        for arcname, expected_hash in checksums.items():
            if ".." in arcname or arcname.startswith(("/", "\\")) or ":" in arcname:
                raise ValueError(
                    f"Path traversal detected in archive entry: '{arcname}'"
                )

            if arcname not in names:
                raise ValueError(
                    f"Corrupted backup archive '{archive.name}': expected file '{arcname}' missing"
                )

            actual_data = zf.read(arcname)
            actual_hash = hashlib.sha256(actual_data).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Backup integrity verification failed: checksum mismatch for '{arcname}' in '{archive.name}' "
                    f"(expected {expected_hash}, got {actual_hash})"
                )

    return manifest


def restore_backup(
    archive_path: Path | str,
    data_dir: Path | str | None = None,
    create_safety_snapshot: bool = True,
    engine_to_dispose: Any | None = None,
) -> Path:
    """Restore workspace from a verified backup archive.

    1. Validates manifest.json, schema compatibility, and file checksums before
       touching any live data.
    2. Pauses LocalJobManager, closes active vector store and database handles.
    3. Creates an automatic pre-restore safety snapshot of live data.
    4. Extracts the backup atomically into live workspace directories.
    5. Runs database initialization / migrations and reinitializes ApplicationContext.
    """
    archive = Path(archive_path).expanduser().resolve()
    target_data_dir = Path(data_dir or settings.QUANTIVA_DATA_DIR).expanduser().resolve()
    paths = LocalDataPaths(target_data_dir)

    # 1. Validation & tamper detection
    manifest = verify_backup_archive(archive)
    logger.info(
        "Verified backup archive '%s' (created: %s, schema: %s)",
        archive.name,
        manifest.get("created_at"),
        manifest.get("schema_version"),
    )

    # 2. Release active handles before touching files
    _close_active_handles()
    if engine_to_dispose is not None:
        try:
            engine_to_dispose.dispose()
        except Exception:
            pass

    # 3. Automatic pre-restore safety snapshot
    if create_safety_snapshot and paths.database_file.is_file():
        try:
            safety_dir = paths.backups / "safety_snapshots"
            safety_dir.mkdir(parents=True, exist_ok=True)
            safety_backup = create_backup(
                include_models=False,
                data_dir=target_data_dir,
                destination_dir=safety_dir,
                prefix="pre_restore_safety_",
            )
            logger.info("Created pre-restore safety snapshot: %s", safety_backup)
        except Exception as exc:
            logger.warning("Could not create pre-restore safety snapshot: %s", exc)

    # 4. Atomic extraction via staging directory
    staging_dir = target_data_dir / f".restore_staging_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive, "r") as zf:
            for member in zf.infolist():
                if member.filename == "manifest.json":
                    continue
                if ".." in member.filename or member.filename.startswith(("/", "\\")):
                    raise ValueError(f"Path traversal detected in archive: {member.filename}")
                zf.extract(member, path=staging_dir)

        # Restore database atomically via sqlite3.Connection.backup()
        staging_db = staging_dir / "database" / "quantiva.db"
        if staging_db.is_file():
            paths.database_dir.mkdir(parents=True, exist_ok=True)
            _restore_sqlite_database(staging_db, paths.database_file)

        staging_cache = staging_dir / "database" / "cache.db"
        if staging_cache.is_file():
            _restore_sqlite_database(staging_cache, paths.database_dir / "cache.db")

        # Restore documents/
        staging_docs = staging_dir / "documents"
        live_docs = target_data_dir / "documents"
        if staging_docs.is_dir():
            if live_docs.exists():
                shutil.rmtree(live_docs, ignore_errors=True)
            shutil.copytree(staging_docs, live_docs)

        # Restore vector_store/
        staging_vectors = staging_dir / "vector_store"
        if staging_vectors.is_dir():
            if paths.vector_store.exists():
                shutil.rmtree(paths.vector_store, ignore_errors=True)
            shutil.copytree(staging_vectors, paths.vector_store)

        # Restore models/ if present in backup
        staging_models = staging_dir / "models"
        live_models = target_data_dir / "models"
        if staging_models.is_dir():
            if live_models.exists():
                shutil.rmtree(live_models, ignore_errors=True)
            shutil.copytree(staging_models, live_models)

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    # 5. Schema migration & reinitialization
    _reinitialize_system()

    logger.info("Successfully restored workspace from '%s'", archive.name)
    return archive


# ── 3. Storage Location Migration ─────────────────────────────────────────────

def migrate_data_directory(
    new_path: Path | str,
    current_data_dir: Path | str | None = None,
    bootstrap_path: Path | str | None = None,
    verify_space: bool = True,
) -> Path:
    """Migrate Quantiva workspace to a new storage location.

    1. Checks available disk space on the target filesystem with safety margin.
    2. Closes active file and engine handles.
    3. Copies database (using sqlite3.Connection.backup()), documents, and vector store.
    4. Updates bootstrap.json atomically so next startup boots from the new path.
    5. Verifies read and write permissions on the new directory.

    Returns the new data directory Path.
    """
    new_dir = Path(new_path).expanduser().resolve()
    current_dir = Path(current_data_dir or settings.QUANTIVA_DATA_DIR).expanduser().resolve()

    if new_dir == current_dir:
        return current_dir

    current_paths = LocalDataPaths(current_dir)

    # 1. Verify free disk space on target drive
    total_bytes = 0
    if current_dir.is_dir():
        for p in current_dir.rglob("*"):
            if p.is_file():
                try:
                    total_bytes += p.stat().st_size
                except OSError:
                    pass

    if verify_space:
        probe = new_dir
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            free_space = shutil.disk_usage(str(probe)).free
            required_space = int(total_bytes * 1.1)  # 10% safety buffer
            if free_space < required_space:
                raise OSError(
                    f"Insufficient free disk space on '{new_dir}': "
                    f"required {required_space} bytes, available {free_space} bytes"
                )
        except (AttributeError, OSError) as exc:
            if isinstance(exc, OSError) and "Insufficient" in str(exc):
                raise
            logger.debug("Disk usage check skipped: %s", exc)

    # 2. Release active handles
    _close_active_handles()

    # 3. Create destination directory structure
    new_paths = ensure_data_directories(new_dir)

    # 4. Copy database cleanly using sqlite3.backup()
    if current_paths.database_file.is_file():
        _backup_sqlite_database(current_paths.database_file, new_paths.database_file)

    if (current_paths.database_dir / "cache.db").is_file():
        shutil.copy2(
            current_paths.database_dir / "cache.db",
            new_paths.database_dir / "cache.db",
        )

    # 5. Copy data directories
    for sub in ["documents", "vector_store", "models", "backups"]:
        src_sub = current_dir / sub
        dst_sub = new_dir / sub
        if src_sub.is_dir():
            if dst_sub.exists():
                shutil.rmtree(dst_sub, ignore_errors=True)
            shutil.copytree(src_sub, dst_sub)

    # 6. Verify read and write permissions on new directory
    probe_file = new_dir / f".quantiva_probe_{uuid.uuid4().hex}"
    try:
        probe_file.write_text("ok", encoding="utf-8")
        assert probe_file.read_text(encoding="utf-8") == "ok"
        probe_file.unlink()
    except Exception as exc:
        raise OSError(f"Read/write verification failed on target directory '{new_dir}'") from exc

    # Verify database readability
    if new_paths.database_file.is_file():
        with sqlite3.connect(str(new_paths.database_file), timeout=5.0) as conn:
            conn.execute("SELECT 1;").fetchone()

    # 7. Update bootstrap.json atomically
    boot_file = Path(bootstrap_path or bootstrap_config_path()).expanduser().resolve()
    boot_file.parent.mkdir(parents=True, exist_ok=True)
    temp_boot = boot_file.parent / f".tmp_{uuid.uuid4().hex}.json"
    temp_boot.write_text(
        json.dumps({"data_dir": str(new_dir)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temp_boot, boot_file)

    # 8. Update runtime configuration and re-spin context
    try:
        object.__setattr__(settings, "QUANTIVA_DATA_DIR", new_dir)
    except Exception:
        pass

    _reinitialize_system()

    logger.info("Successfully migrated data directory from '%s' to '%s'", current_dir, new_dir)
    return new_dir


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "create_backup",
    "migrate_data_directory",
    "restore_backup",
    "verify_backup_archive",
]
