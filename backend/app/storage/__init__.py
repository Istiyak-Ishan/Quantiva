"""Provider-agnostic persistence contracts for Quantiva.

Concrete adapters (embedded Qdrant/LanceDB, SQLite FTS5, etc.) live outside
this package and are injected into the retrieval pipeline.
"""
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
    "CollectionInfo",
    "DEFAULT_CHUNK_SIZE",
    "LocalCache",
    "LocalFileStore",
    "SearchHit",
    "SparseIndexRepository",
    "SparseSearchResult",
    "StoredFile",
    "VectorSearchResult",
    "VectorStoreRepository",
    "WINDOWS_RESERVED_NAMES",
    "get_local_cache",
    "reset_local_cache",
    "sanitize_filename",
]
