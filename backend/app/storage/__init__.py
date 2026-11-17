"""Provider-agnostic persistence contracts for Quantiva.

Concrete adapters (embedded Qdrant/LanceDB, SQLite FTS5, etc.) live outside
this package and are injected into the retrieval pipeline.
"""
from app.storage.interfaces import (
    CollectionInfo,
    SearchHit,
    SparseIndexRepository,
    SparseSearchResult,
    VectorSearchResult,
    VectorStoreRepository,
)

__all__ = [
    "CollectionInfo",
    "SearchHit",
    "SparseIndexRepository",
    "SparseSearchResult",
    "VectorSearchResult",
    "VectorStoreRepository",
]
