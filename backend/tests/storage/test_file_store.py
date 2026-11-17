"""Tests for LocalFileStore and LocalCache storage components.

Covers:
- Upload streaming and single-pass SHA-256 computation
- Windows reserved filename sanitization (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
- Path traversal prevention
- Relative path storage and resolution across relocated data directories
- Original and processed artifact persistence and deletion cleanup
- Thread-safe TTL LRU LocalCache with optional SQLite persistence
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import pytest

from app.storage.file_store import (
    LocalFileStore,
    StoredFile,
    WINDOWS_RESERVED_NAMES,
    sanitize_filename,
)
from app.storage.local_cache import LocalCache, get_local_cache, reset_local_cache


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "quantiva_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def file_store(temp_data_dir: Path) -> LocalFileStore:
    return LocalFileStore(data_dir=temp_data_dir)


# ── 1. Windows Filename Sanitization & Reserved Names ─────────────────────────

def test_sanitize_windows_reserved_names():
    """Verify all reserved device names are safely prefixed."""
    for reserved in ["CON", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "LPT9"]:
        # Exact reserved name
        assert sanitize_filename(reserved) == f"_{reserved}"
        # Lowercase
        assert sanitize_filename(reserved.lower()) == f"_{reserved.lower()}"
        # With extension
        assert sanitize_filename(f"{reserved}.txt") == f"_{reserved}.txt"
        assert sanitize_filename(f"{reserved.lower()}.pdf") == f"_{reserved.lower()}.pdf"
        # Compound extension
        assert sanitize_filename(f"{reserved}.tar.gz") == f"_{reserved}.tar.gz"


def test_sanitize_windows_illegal_characters():
    """Illegal Windows characters <>:"/\\|?* must be replaced with underscores."""
    raw = 'report<2024>:Q1"draft"|analysis?*final.pdf'
    sanitized = sanitize_filename(raw)
    for char in '<>:"/\\|?*':
        assert char not in sanitized
    assert sanitized == "report_2024__Q1_draft__analysis__final.pdf"


def test_sanitize_trailing_dots_and_spaces():
    """Windows files cannot end in dots or spaces."""
    assert sanitize_filename("filename...   ") == "filename"
    assert sanitize_filename("my_document.pdf... ") == "my_document.pdf"
    assert sanitize_filename("spaces at end   ") == "spaces at end"


def test_sanitize_empty_and_dot_names():
    """Empty or dot-only inputs must resolve to a safe default name."""
    assert sanitize_filename("") == "document"
    assert sanitize_filename("   ") == "document"
    assert sanitize_filename(".") == "document"
    assert sanitize_filename("..") == "document"
    assert sanitize_filename("...") == "document"


def test_sanitize_bounds_filename_length():
    """Excessive length filenames must be safely truncated preserving suffix."""
    long_stem = "a" * 300
    sanitized = sanitize_filename(f"{long_stem}.pdf", max_length=100)
    assert len(sanitized) <= 100
    assert sanitized.endswith(".pdf")
    assert sanitized.startswith("aaaa")


# ── 2. Path Traversal Blocking ────────────────────────────────────────────────

def test_sanitize_filename_strips_directory_components():
    """sanitize_filename must extract leaf names and eliminate path traversal."""
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\calc.exe") == "calc.exe"
    assert sanitize_filename("/usr/local/bin/secret.pdf") == "secret.pdf"
    assert sanitize_filename("C:\\Users\\admin\\document.docx") == "document.docx"


def test_resolve_relative_path_blocks_traversal(file_store: LocalFileStore, temp_data_dir: Path):
    """resolve_relative_path must raise ValueError when paths attempt to escape."""
    # Parent directory escape
    with pytest.raises(ValueError, match="escapes base directory|not permitted"):
        file_store.resolve_relative_path("../outside.txt")

    with pytest.raises(ValueError, match="escapes base directory|not permitted"):
        file_store.resolve_relative_path("documents/../../secret.txt")

    # Absolute paths must be rejected
    with pytest.raises(ValueError, match="absolute path"):
        file_store.resolve_relative_path("/etc/passwd")

    with pytest.raises(ValueError, match="absolute path"):
        file_store.resolve_relative_path("C:\\Windows\\System32\\calc.exe")


# ── 3. Streaming Upload & Single-Pass SHA-256 Hashing ─────────────────────────

def test_save_original_sync_bytes(file_store: LocalFileStore, temp_data_dir: Path):
    """Test saving bytes synchronously with chunked streaming and SHA-256 computation."""
    content = b"Quantiva Local Storage Test Content " * 1000
    expected_hash = hashlib.sha256(content).hexdigest()
    expected_size = len(content)

    result = file_store.save_original(
        knowledge_base_id="kb_finance",
        document_id="doc_101",
        filename="q4_financials:final.pdf",
        content=content,
        chunk_size=512,
    )

    assert isinstance(result, StoredFile)
    assert result.knowledge_base_id == "kb_finance"
    assert result.document_id == "doc_101"
    assert result.filename == "q4_financials_final.pdf"
    assert result.sha256_hash == expected_hash
    assert result.file_size_bytes == expected_size
    assert result.relative_path == "documents/original/kb_finance/doc_101_q4_financials_final.pdf"

    # Verify physical file on disk
    expected_disk_path = (
        temp_data_dir / "documents" / "original" / "kb_finance" / "doc_101_q4_financials_final.pdf"
    )
    assert result.absolute_path == expected_disk_path
    assert expected_disk_path.is_file()
    assert expected_disk_path.read_bytes() == content


def test_save_original_sync_binary_io(file_store: LocalFileStore):
    """Test saving from a BinaryIO stream in chunks."""
    content = b"Binary stream content for testing " * 500
    stream = io.BytesIO(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    result = file_store.save_original(
        knowledge_base_id="kb_ops",
        document_id="doc_202",
        filename="CON.txt",  # Windows reserved name
        content=stream,
        chunk_size=256,
    )

    assert result.filename == "_CON.txt"
    assert result.sha256_hash == expected_hash
    assert result.file_size_bytes == len(content)
    assert result.absolute_path.is_file()
    assert result.absolute_path.read_bytes() == content


@pytest.mark.asyncio
async def test_save_original_async_stream(file_store: LocalFileStore):
    """Test saving from an async generator stream in chunks."""
    chunks = [f"Async chunk {i} ".encode("utf-8") * 50 for i in range(20)]
    full_content = b"".join(chunks)
    expected_hash = hashlib.sha256(full_content).hexdigest()

    async def async_chunk_generator():
        for ch in chunks:
            await asyncio.sleep(0.001)
            yield ch

    result = await file_store.save_original_stream(
        knowledge_base_id="kb_tech",
        document_id="doc_303",
        filename="specs.md",
        stream=async_chunk_generator(),
    )

    assert result.sha256_hash == expected_hash
    assert result.file_size_bytes == len(full_content)
    assert result.absolute_path.read_bytes() == full_content


def test_streaming_failure_leaves_no_orphan_files(file_store: LocalFileStore):
    """If an exception occurs during streaming, temporary files must be cleaned up."""
    class FailingStream:
        def __init__(self):
            self.count = 0

        def read(self, size):
            self.count += 1
            if self.count > 2:
                raise IOError("Simulated network/disk disconnect")
            return b"good bytes " * 10

    target_dir = file_store.original_dir / "kb_fail"

    with pytest.raises(IOError, match="Simulated network/disk disconnect"):
        file_store.save_original(
            knowledge_base_id="kb_fail",
            document_id="doc_failed",
            filename="crash.pdf",
            content=FailingStream(),
            chunk_size=10,
        )

    # Destination file must not exist
    dest = target_dir / "doc_failed_crash.pdf"
    assert not dest.exists()

    # No leftover .tmp files
    if target_dir.exists():
        tmp_files = list(target_dir.glob(".tmp_*"))
        assert len(tmp_files) == 0


# ── 4. Relative Path Resolution Across Relocated Directories ──────────────────

def test_relative_path_portability_across_drives(tmp_path: Path):
    """Verify that stored relative paths resolve seamlessly when data dir is relocated."""
    # Machine / Drive A (e.g. C:\Users\AppData\Quantiva)
    drive_a = tmp_path / "DriveC" / "Quantiva"
    store_a = LocalFileStore(data_dir=drive_a)

    content = b"Mission critical document content."
    saved = store_a.save_original(
        knowledge_base_id="kb_contracts",
        document_id="doc_alpha",
        filename="master_agreement.pdf",
        content=content,
    )

    # Database records ONLY the relative path
    recorded_rel_path = saved.relative_path
    assert recorded_rel_path == "documents/original/kb_contracts/doc_alpha_master_agreement.pdf"

    # User moves their data directory to Drive B (e.g. D:\QuantivaData)
    drive_b = tmp_path / "DriveD" / "QuantivaData"
    shutil.copytree(drive_a, drive_b)

    # Instantiate file store on Drive B
    store_b = LocalFileStore(data_dir=drive_b)

    # Resolve the stored relative path against Drive B
    resolved_path_b = store_b.resolve_relative_path(recorded_rel_path)

    assert resolved_path_b == drive_b / "documents" / "original" / "kb_contracts" / "doc_alpha_master_agreement.pdf"
    assert resolved_path_b.is_file()
    assert resolved_path_b.read_bytes() == content

    # Read helper by relative path
    assert store_b.read_by_relative_path(recorded_rel_path) == content


# ── 5. Processed Artifact Persistence & Retrieval ─────────────────────────────

def test_save_and_read_processed_json_artifact(file_store: LocalFileStore):
    """Processed parsed artifacts are saved as JSON and read back as structured data."""
    artifact_payload = {
        "document_id": "doc_parsed_1",
        "chunks": [
            {"index": 0, "text": "Introduction chunk", "page": 1},
            {"index": 1, "text": "Financial summary chunk", "page": 2},
        ],
        "tables": [{"rows": [["Year", "Revenue"], ["2024", "$10M"]]}],
    }

    stored = file_store.save_processed(
        knowledge_base_id="kb_docs",
        document_id="doc_parsed_1",
        content=artifact_payload,
        ext="json",
    )

    assert stored.relative_path == "documents/processed/kb_docs/doc_parsed_1.json"
    assert stored.absolute_path.is_file()

    # Read back and verify deserialization
    retrieved = file_store.read_processed("kb_docs", "doc_parsed_1", ext="json")
    assert retrieved == artifact_payload


def test_save_and_read_processed_markdown(file_store: LocalFileStore):
    """Processed artifacts can also be markdown text."""
    md_content = "# Executive Summary\n\n- Key Metric 1: 99.9%\n- Key Metric 2: $42M"

    stored = file_store.save_processed(
        knowledge_base_id="kb_docs",
        document_id="doc_md_1",
        content=md_content,
        ext="md",
    )

    assert stored.relative_path == "documents/processed/kb_docs/doc_md_1.md"
    retrieved = file_store.read_processed("kb_docs", "doc_md_1", ext="md")
    assert retrieved == md_content


# ── 6. Deletion Cleanup ───────────────────────────────────────────────────────

def test_delete_document_removes_both_original_and_processed(file_store: LocalFileStore):
    """Deleting a document cleanly purges both original and processed files."""
    # 1. Create document 1
    orig_1 = file_store.save_original(
        knowledge_base_id="kb_clean",
        document_id="doc_1",
        filename="invoice.pdf",
        content=b"Invoice 1",
    )
    proc_1 = file_store.save_processed(
        knowledge_base_id="kb_clean",
        document_id="doc_1",
        content={"parsed": "invoice 1"},
    )

    # 2. Create document 2 in same KB
    orig_2 = file_store.save_original(
        knowledge_base_id="kb_clean",
        document_id="doc_2",
        filename="receipt.pdf",
        content=b"Receipt 2",
    )
    proc_2 = file_store.save_processed(
        knowledge_base_id="kb_clean",
        document_id="doc_2",
        content={"parsed": "receipt 2"},
    )

    assert orig_1.absolute_path.is_file()
    assert proc_1.absolute_path.is_file()
    assert orig_2.absolute_path.is_file()
    assert proc_2.absolute_path.is_file()

    # Delete document 1
    deleted = file_store.delete_document(
        knowledge_base_id="kb_clean",
        document_id="doc_1",
        original_rel_path=orig_1.relative_path,
        processed_rel_path=proc_1.relative_path,
    )
    assert deleted is True

    # Document 1 files are gone
    assert not orig_1.absolute_path.exists()
    assert not proc_1.absolute_path.exists()

    # Document 2 files remain intact!
    assert orig_2.absolute_path.is_file()
    assert proc_2.absolute_path.is_file()


def test_delete_knowledge_base_purges_entire_folder_hierarchy(file_store: LocalFileStore):
    """Deleting a knowledge base recursively removes all its documents."""
    # Setup files in target KB
    for i in range(3):
        file_store.save_original(
            knowledge_base_id="kb_to_delete",
            document_id=f"doc_{i}",
            filename=f"file_{i}.txt",
            content=f"Content {i}".encode(),
        )
        file_store.save_processed(
            knowledge_base_id="kb_to_delete",
            document_id=f"doc_{i}",
            content={"id": i},
        )

    # Setup files in another KB
    other_orig = file_store.save_original(
        knowledge_base_id="kb_survivor",
        document_id="doc_safe",
        filename="safe.txt",
        content=b"Survivor content",
    )

    # Delete target KB
    file_store.delete_knowledge_base("kb_to_delete")

    # Verify target folders are completely gone
    assert not (file_store.original_dir / "kb_to_delete").exists()
    assert not (file_store.processed_dir / "kb_to_delete").exists()

    # Verify survivor is untouched
    assert other_orig.absolute_path.is_file()


# ── 7. LocalCache (In-Memory LRU, TTL, and SQLite Persistence) ────────────────

def test_local_cache_lru_and_capacity():
    """Verify in-memory LRU eviction when capacity is exceeded."""
    cache = LocalCache(max_size=3, default_ttl=3600, persist_to_sqlite=False)
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    cache.set("k3", "v3")

    assert len(cache) == 3
    assert cache.get("k1") == "v1"  # k1 accessed, making k2 least recently used

    # Adding k4 must evict k2
    cache.set("k4", "v4")
    assert cache.get("k2") is None
    assert cache.get("k1") == "v1"
    assert cache.get("k3") == "v3"
    assert cache.get("k4") == "v4"


def test_local_cache_ttl_expiration():
    """Verify items expire after their TTL and are pruned."""
    cache = LocalCache(max_size=10, default_ttl=0.05, persist_to_sqlite=False)
    cache.set("fast_expire", "val", ttl=0.02)
    cache.set("longer_expire", "val2", ttl=1.0)

    assert cache.get("fast_expire") == "val"
    time.sleep(0.04)

    # fast_expire has expired
    assert cache.get("fast_expire") is None
    # longer_expire still active
    assert cache.get("longer_expire") == "val2"

    # Cleanup expired
    pruned = cache.cleanup_expired()
    assert pruned >= 0


def test_local_cache_sqlite_persistence_across_instances(tmp_path: Path):
    """Verify SQLite persistence reloads items into fresh cache instances."""
    sqlite_db = tmp_path / "cache.db"

    # Instance 1: write items
    cache_1 = LocalCache(sqlite_path=sqlite_db, persist_to_sqlite=True, max_size=100)
    cache_1.set("app_setting", {"theme": "dark", "fontSize": 14})
    cache_1.set_embedding("query text", "e5-small", 384, [0.123, -0.456, 0.789])
    cache_1.set_query_cache("kb_1", "What is EBITDA?", {"answer": "Earnings...", "citations": []})

    # Instance 2: completely new in-memory instance pointing to same SQLite database
    cache_2 = LocalCache(sqlite_path=sqlite_db, persist_to_sqlite=True, max_size=100)

    # Items must reload from SQLite
    assert cache_2.get("app_setting") == {"theme": "dark", "fontSize": 14}
    assert cache_2.get_embedding("query text", "e5-small", 384) == [0.123, -0.456, 0.789]
    query_hit = cache_2.get_query_cache("kb_1", "What is EBITDA?")
    assert query_hit is not None
    assert query_hit["answer"] == "Earnings..."


def test_local_cache_thread_safety():
    """Verify concurrent reads and writes across multiple threads without errors."""
    cache = LocalCache(max_size=500, default_ttl=60, persist_to_sqlite=False)

    def worker(worker_id: int):
        for i in range(100):
            key = f"thread_key_{worker_id}_{i % 20}"
            cache.set(key, i)
            val = cache.get(key)
            assert val is not None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in futures:
            f.result()

    assert len(cache) > 0


def test_local_cache_ping_and_async_methods():
    """Verify health check / readiness probes and async methods."""
    cache = LocalCache(max_size=10, persist_to_sqlite=False)
    assert cache.ping() is True

    async def test_async():
        assert await cache.aping() is True
        await cache.aset("async_key", "async_val")
        assert await cache.aget("async_key") == "async_val"
        assert await cache.adelete("async_key") is True
        assert await cache.aget("async_key") is None

    asyncio.run(test_async())


def test_global_cache_singleton_and_reset(temp_data_dir: Path):
    """Verify get_local_cache singleton behavior and reset."""
    reset_local_cache()
    c1 = get_local_cache(data_dir=temp_data_dir)
    c2 = get_local_cache()
    assert c1 is c2

    c1.set("singleton_test", "yes")
    assert c2.get("singleton_test") == "yes"

    reset_local_cache()
    c3 = get_local_cache()
    assert c3 is not c1
    reset_local_cache()
