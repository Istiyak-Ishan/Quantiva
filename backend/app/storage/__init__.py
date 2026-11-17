"""Provider-agnostic persistence contracts for Quantiva.

Concrete adapters (embedded Qdrant/LanceDB, SQLite FTS5, etc.) live outside
this package and are injected into the retrieval pipeline.
"""
from app.storage.backup import (
    CURRENT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    create_backup,
    migrate_data_directory,
    restore_backup,
    verify_backup_archive,
)
from app.storage.file_store import (
    DEFAULT_CHUNK_SIZE,
    LocalFileStore,
    StoredFile,
    WINDOWS_RESERVED_NAMES,
    sanitize_filename,
)
from app.storage.interfaces import (
    CollectionInfo,
    SearchHit,
    SparseIndexRepository,
    SparseSearchResult,
    VectorSearchResult,
    VectorStoreRepository,
)
from app.storage.local_cache import (
    LocalCache,
    get_local_cache,
    reset_local_cache,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "CollectionInfo",
    "DEFAULT_CHUNK_SIZE",
    "LocalCache",
    "LocalFileStore",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SearchHit",
    "SparseIndexRepository",
    "SparseSearchResult",
    "StoredFile",
    "VectorSearchResult",
    "VectorStoreRepository",
    "WINDOWS_RESERVED_NAMES",
    "create_backup",
    "get_local_cache",
    "migrate_data_directory",
    "reset_local_cache",
    "restore_backup",
    "sanitize_filename",
    "verify_backup_archive",
]
