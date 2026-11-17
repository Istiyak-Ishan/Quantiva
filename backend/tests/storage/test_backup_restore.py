"""Tests for Quantiva workspace backup, restore, and storage migration.

Covers:
- Hot consistent SQLite WAL backups during concurrent readers and writers
- Packaging database snapshot, documents/, vector_store/, and manifest.json
- Model exclusion by default
- Tamper detection on invalid archives (checksum mismatch, missing manifest, schema incompatibility, zip slip)
- Full workspace state restoration preserving database rows, vectors, and files
- Storage location migration with disk space validation and bootstrap.json updates
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import zipfile

import pytest
from sqlalchemy import select

from app.config import LocalDataPaths, ensure_data_directories
from app.storage.backup import (
    CURRENT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    create_backup,
    migrate_data_directory,
    restore_backup,
    verify_backup_archive,
)
from app.storage.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from app.storage.file_store import LocalFileStore
from app.storage.models import ApplicationProfile, Document, KnowledgeBase


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def workspace_env(tmp_path: Path):
    """Sets up an isolated, initialized workspace for testing."""
    data_dir = tmp_path / "quantiva_workspace"
    paths = ensure_data_directories(data_dir)

    engine = create_sqlite_engine(data_dir)
    initialize_database(engine)
    session_factory = create_session_factory(engine)
    file_store = LocalFileStore(data_dir=data_dir)

    # Bootstrap file
    bootstrap_file = tmp_path / "bootstrap.json"
    bootstrap_file.write_text(
        json.dumps({"data_dir": str(data_dir)}),
        encoding="utf-8",
    )

    try:
        yield {
            "root": data_dir,
            "paths": paths,
            "engine": engine,
            "factory": session_factory,
            "file_store": file_store,
            "bootstrap_file": bootstrap_file,
        }
    finally:
        engine.dispose()


# ── 1. Hot Backup During Concurrent Reads/Writes ──────────────────────────────

def test_hot_backup_during_concurrent_writes(workspace_env):
    """Verify create_backup runs cleanly while other threads write to SQLite WAL."""
    engine = workspace_env["engine"]
    data_dir = workspace_env["root"]
    db_path = workspace_env["paths"].database_file

    stop_event = threading.Event()
    write_count = [0]
    errors: list[Exception] = []

    def writer_task():
        try:
            # Use raw SQLite connection to perform rapid inserts
            conn = sqlite3.connect(str(db_path), timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            while not stop_event.is_set():
                conn.execute(
                    "INSERT INTO application_profiles (profile_id, display_name, workspace_name, workspace_preferences_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, '{}', datetime('now'), datetime('now'))",
                    (f"prof_{write_count[0]}", f"User {write_count[0]}", "WS"),
                )
                conn.commit()
                write_count[0] += 1
                time.sleep(0.005)
            conn.close()
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer_future = executor.submit(writer_task)
        time.sleep(0.02)  # Allow writes to begin

        # Execute hot backup while writer is actively inserting
        archive_path = create_backup(
            include_models=False,
            data_dir=data_dir,
        )

        stop_event.set()
        writer_future.result()

    assert len(errors) == 0
    assert write_count[0] > 0
    assert archive_path.is_file()

    # Validate the snapshot database extracted from the backup
    with zipfile.ZipFile(archive_path, "r") as zf:
        db_bytes = zf.read("database/quantiva.db")

    test_db = data_dir / "temp_extracted_check.db"
    test_db.write_bytes(db_bytes)
    conn = sqlite3.connect(str(test_db))
    try:
        integrity = conn.execute("PRAGMA integrity_check;").fetchone()
        assert integrity[0] == "ok"
        count = conn.execute("SELECT COUNT(*) FROM application_profiles;").fetchone()[0]
        assert count > 0
    finally:
        conn.close()
        test_db.unlink(missing_ok=True)


# ── 2. Models Cache Exclusion ─────────────────────────────────────────────────

def test_backup_excludes_models_by_default(workspace_env):
    """Backups must exclude the heavy models/ directory unless explicitly requested."""
    data_dir = workspace_env["root"]
    models_dir = data_dir / "models" / "embeddings"
    models_dir.mkdir(parents=True, exist_ok=True)
    heavy_weights = models_dir / "weights.bin"
    heavy_weights.write_bytes(b"Simulated 2GB Model Weights " * 100)

    # 1. Default backup -> models excluded
    default_archive = create_backup(include_models=False, data_dir=data_dir)
    with zipfile.ZipFile(default_archive, "r") as zf:
        names = zf.namelist()
        assert not any(n.startswith("models/") for n in names)
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["include_models"] is False

    # 2. Explicit include_models=True -> models included
    full_archive = create_backup(include_models=True, data_dir=data_dir)
    with zipfile.ZipFile(full_archive, "r") as zf:
        names = zf.namelist()
        assert any(n.startswith("models/") for n in names)
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        assert manifest["include_models"] is True


# ── 3. Tamper Detection & Archive Validation ──────────────────────────────────

def test_tamper_detection_checksum_mismatch(workspace_env, tmp_path: Path):
    """Any modified byte inside the archive must fail SHA-256 verification."""
    data_dir = workspace_env["root"]
    archive_path = create_backup(data_dir=data_dir)

    # Tamper with the zip: modify a byte in database/quantiva.db
    tampered_zip_path = tmp_path / "tampered_backup.zip"
    with zipfile.ZipFile(archive_path, "r") as src_zf, zipfile.ZipFile(tampered_zip_path, "w") as dst_zf:
        for item in src_zf.infolist():
            data = src_zf.read(item.filename)
            if item.filename == "database/quantiva.db":
                # Flip bytes
                data = b"TAMPERED_BYTES" + data[14:]
            dst_zf.writestr(item, data)

    # verify_backup_archive must reject it
    with pytest.raises(ValueError, match="Backup integrity verification failed.*checksum mismatch"):
        verify_backup_archive(tampered_zip_path)

    # restore_backup must also reject it before touching live data
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore_backup(tampered_zip_path, data_dir=data_dir)


def test_tamper_detection_missing_manifest(workspace_env, tmp_path: Path):
    """Archives missing manifest.json must be rejected."""
    data_dir = workspace_env["root"]
    archive_path = create_backup(data_dir=data_dir)

    no_manifest_zip = tmp_path / "no_manifest.zip"
    with zipfile.ZipFile(archive_path, "r") as src_zf, zipfile.ZipFile(no_manifest_zip, "w") as dst_zf:
        for item in src_zf.infolist():
            if item.filename != "manifest.json":
                dst_zf.writestr(item, src_zf.read(item.filename))

    with pytest.raises(ValueError, match="manifest.json is missing"):
        verify_backup_archive(no_manifest_zip)


def test_tamper_detection_incompatible_schema_version(workspace_env, tmp_path: Path):
    """Archives specifying an unsupported schema version must be rejected."""
    data_dir = workspace_env["root"]
    archive_path = create_backup(data_dir=data_dir)

    incompat_zip = tmp_path / "incompat_schema.zip"
    with zipfile.ZipFile(archive_path, "r") as src_zf, zipfile.ZipFile(incompat_zip, "w") as dst_zf:
        for item in src_zf.infolist():
            data = src_zf.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data.decode("utf-8"))
                manifest["schema_version"] = 999  # Future incompatible schema
                data = json.dumps(manifest).encode("utf-8")
            dst_zf.writestr(item, data)

    with pytest.raises(ValueError, match="Incompatible backup schema version: 999"):
        verify_backup_archive(incompat_zip)


def test_tamper_detection_path_traversal(workspace_env, tmp_path: Path):
    """Archives containing malicious relative or absolute paths must be rejected."""
    traversal_zip = tmp_path / "traversal_attack.zip"
    manifest = {
        "app_version": "1.0.0",
        "schema_version": 1,
        "created_at": "2026-11-17T00:00:00Z",
        "sha256_checksums": {
            "../../escape.txt": hashlib.sha256(b"evil").hexdigest(),
        },
    }
    with zipfile.ZipFile(traversal_zip, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("../../escape.txt", b"evil")

    with pytest.raises(ValueError, match="Path traversal detected"):
        verify_backup_archive(traversal_zip)


# ── 4. Full Workspace State Restoration ───────────────────────────────────────

def test_full_workspace_state_restoration(workspace_env):
    """Verify restoring a backup fully recovers SQLite rows, original documents, and vectors."""
    data_dir = workspace_env["root"]
    factory = workspace_env["factory"]
    file_store = workspace_env["file_store"]

    # 1. Populate workspace state:
    # A. SQLite database records
    with factory() as session:
        kb = KnowledgeBase(id="kb_q4", name="Q4 Financials", description="Earnings reports")
        doc = Document(
            id="doc_q4_1",
            knowledge_base_id="kb_q4",
            filename="report.pdf",
            original_rel_path="documents/original/kb_q4/doc_q4_1_report.pdf",
            processed_rel_path="documents/processed/kb_q4/doc_q4_1.json",
            file_type="pdf",
            file_size_bytes=1024,
            sha256_hash="hash12345",
        )
        profile = ApplicationProfile(
            profile_id="user_admin",
            display_name="Admin User",
            workspace_name="Corporate Intelligence",
        )
        session.add_all([kb, doc, profile])
        session.commit()

    # B. Documents (original and processed)
    doc_bytes = b"%PDF-1.7 Original Document Contents for Q4"
    orig_file = file_store.save_original(
        knowledge_base_id="kb_q4",
        document_id="doc_q4_1",
        filename="report.pdf",
        content=doc_bytes,
    )
    proc_json = {"summary": "Q4 Revenue grew 45%", "revenue_mil": 450}
    proc_file = file_store.save_processed(
        knowledge_base_id="kb_q4",
        document_id="doc_q4_1",
        content=proc_json,
    )

    # C. Vector Store data
    vector_dir = data_dir / "vector_store" / "collections" / "quantiva_v2"
    vector_dir.mkdir(parents=True, exist_ok=True)
    vector_data_file = vector_dir / "storage.sqlite"
    vector_data_file.write_bytes(b"EMBEDDED_VECTOR_POINTS_DENSE_ARRAY")

    # 2. Create the backup
    archive_path = create_backup(include_models=False, data_dir=data_dir)
    assert archive_path.is_file()

    # 3. Simulate disaster / state loss:
    # Delete database records
    with factory() as session:
        session.query(Document).delete()
        session.query(KnowledgeBase).delete()
        session.query(ApplicationProfile).delete()
        session.commit()

        # Confirm live DB is empty
        assert session.scalar(select(Document)) is None
        assert session.scalar(select(KnowledgeBase)) is None

    # Delete physical files
    file_store.delete_document(
        knowledge_base_id="kb_q4",
        document_id="doc_q4_1",
        original_rel_path=orig_file.relative_path,
        processed_rel_path=proc_file.relative_path,
    )
    shutil.rmtree(data_dir / "vector_store", ignore_errors=True)

    assert not orig_file.absolute_path.exists()
    assert not proc_file.absolute_path.exists()
    assert not vector_data_file.exists()

    # 4. Perform Safe Restore
    restored_archive = restore_backup(
        archive_path=archive_path,
        data_dir=data_dir,
        create_safety_snapshot=True,
        engine_to_dispose=workspace_env["engine"],
    )
    assert restored_archive == archive_path

    # Create fresh factory from restored database
    restored_engine = create_sqlite_engine(data_dir)
    restored_factory = create_session_factory(restored_engine)

    # 5. Verify Complete State Recovery:
    # A. SQLite database records restored
    with restored_factory() as session:
        restored_kb = session.scalar(select(KnowledgeBase).where(KnowledgeBase.id == "kb_q4"))
        assert restored_kb is not None
        assert restored_kb.name == "Q4 Financials"

        restored_doc = session.scalar(select(Document).where(Document.id == "doc_q4_1"))
        assert restored_doc is not None
        assert restored_doc.filename == "report.pdf"

        restored_profile = session.scalar(select(ApplicationProfile).where(ApplicationProfile.profile_id == "user_admin"))
        assert restored_profile is not None
        assert restored_profile.display_name == "Admin User"

    # B. Documents restored
    assert orig_file.absolute_path.is_file()
    assert orig_file.absolute_path.read_bytes() == doc_bytes
    assert proc_file.absolute_path.is_file()
    assert file_store.read_processed("kb_q4", "doc_q4_1") == proc_json

    # C. Vector Store data restored
    assert vector_data_file.is_file()
    assert vector_data_file.read_bytes() == b"EMBEDDED_VECTOR_POINTS_DENSE_ARRAY"

    # D. Pre-restore safety snapshot was created
    safety_snapshots = list((data_dir / "backups" / "safety_snapshots").glob("pre_restore_safety_*.zip"))
    assert len(safety_snapshots) > 0


# ── 5. Data Directory Migration ───────────────────────────────────────────────

def test_migrate_data_directory_success(workspace_env, tmp_path: Path):
    """Verify migrate_data_directory moves all workspace data and updates bootstrap.json."""
    src_dir = workspace_env["root"]
    bootstrap_file = workspace_env["bootstrap_file"]
    file_store = workspace_env["file_store"]
    factory = workspace_env["factory"]

    # Setup some state in source
    with factory() as session:
        kb = KnowledgeBase(id="kb_migrate", name="Migration KB")
        session.add(kb)
        session.commit()

    file_store.save_original(
        knowledge_base_id="kb_migrate",
        document_id="doc_mig",
        filename="data.txt",
        content=b"Migration Payload",
    )

    # Destination directory (e.g. D:\QuantivaData)
    target_dir = tmp_path / "new_disk_drive" / "QuantivaData"

    # Run migration
    migrated_path = migrate_data_directory(
        new_path=target_dir,
        current_data_dir=src_dir,
        bootstrap_path=bootstrap_file,
        verify_space=True,
    )

    assert migrated_path == target_dir
    assert target_dir.is_dir()

    # 1. Verify bootstrap.json updated
    bootstrap_data = json.loads(bootstrap_file.read_text(encoding="utf-8"))
    assert bootstrap_data["data_dir"] == str(target_dir)

    # 2. Verify files and database in target
    target_db = target_dir / "database" / "quantiva.db"
    assert target_db.is_file()

    with sqlite3.connect(str(target_db)) as conn:
        row = conn.execute("SELECT name FROM knowledge_bases WHERE id = 'kb_migrate'").fetchone()
        assert row is not None
        assert row[0] == "Migration KB"

    target_doc = target_dir / "documents" / "original" / "kb_migrate" / "doc_mig_data.txt"
    assert target_doc.is_file()
    assert target_doc.read_bytes() == b"Migration Payload"


def test_migrate_data_directory_insufficient_space(workspace_env, monkeypatch, tmp_path: Path):
    """Migration must fail closed if target drive does not have sufficient space."""
    src_dir = workspace_env["root"]
    target_dir = tmp_path / "tiny_drive" / "Quantiva"

    # Create dummy 10MB file in src
    dummy = src_dir / "documents" / "original" / "heavy.bin"
    dummy.parent.mkdir(parents=True, exist_ok=True)
    dummy.write_bytes(b"x" * (1024 * 1024))

    # Mock shutil.disk_usage to return only 100 bytes free
    class MockUsage:
        free = 100
        total = 1000
        used = 900

    monkeypatch.setattr(shutil, "disk_usage", lambda _: MockUsage())

    with pytest.raises(OSError, match="Insufficient free disk space"):
        migrate_data_directory(
            new_path=target_dir,
            current_data_dir=src_dir,
            verify_space=True,
        )
