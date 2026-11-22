"""Thread-safe TTL LRU in-memory cache with optional SQLite persistence.

Replaces Redis caching dependencies for Quantiva desktop environments,
providing zero-infrastructure caching for LLM queries, embeddings, and hot data.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expires_at: float | None


class LocalCache:
    """Thread-safe LRU cache with per-item TTL and optional SQLite persistence."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float | None = 3600.0,
        sqlite_path: Path | str | None = None,
        persist_to_sqlite: bool = False,
    ) -> None:
        self._max_size = max(1, max_size)
        self._default_ttl = default_ttl
        self._persist_to_sqlite = persist_to_sqlite or (sqlite_path is not None)

        if sqlite_path is not None:
            self._sqlite_path: Path | None = Path(sqlite_path).expanduser().resolve()
        elif self._persist_to_sqlite:
            db_dir = Path(settings.QUANTIVA_DATA_DIR).expanduser().resolve() / "database"
            self._sqlite_path = db_dir / "cache.db"
        else:
            self._sqlite_path = None

        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        if self._persist_to_sqlite and self._sqlite_path is not None:
            self._init_sqlite()

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def default_ttl(self) -> float | None:
        return self._default_ttl

    @property
    def sqlite_path(self) -> Path | None:
        return self._sqlite_path

    # ── SQLite Persistence ────────────────────────────────────────────────────

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        if self._sqlite_path is None:
            raise RuntimeError("SQLite path is not configured for this cache")
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self._sqlite_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_sqlite(self) -> None:
        with self._lock:
            try:
                with self._get_sqlite_connection() as conn:
                    conn.execute("""
                    CREATE TABLE IF NOT EXISTS local_cache (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        expires_at REAL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    """)
                    conn.execute("""
                    CREATE INDEX IF NOT EXISTS ix_local_cache_expires_at
                    ON local_cache(expires_at);
                    """)
                    conn.commit()
            except Exception as exc:
                logger.warning("Failed initializing SQLite cache persistence: %s", exc)

    def _sqlite_get(self, key: str) -> tuple[Any, float | None] | None:
        if not self._persist_to_sqlite or self._sqlite_path is None:
            return None
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.execute(
                    "SELECT value, expires_at FROM local_cache WHERE key = ?",
                    (key,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                val_json, expires_at = row
                return json.loads(val_json), expires_at
        except Exception as exc:
            logger.debug("SQLite cache read error for '%s': %s", key, exc)
            return None

    def _sqlite_upsert(self, key: str, value: Any, expires_at: float | None) -> None:
        if not self._persist_to_sqlite or self._sqlite_path is None:
            return
        try:
            val_json = json.dumps(value, ensure_ascii=False)
            now = time.time()
            with self._get_sqlite_connection() as conn:
                conn.execute("""
                INSERT INTO local_cache(key, value, expires_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at;
                """, (key, val_json, expires_at, now, now))
                conn.commit()
        except Exception as exc:
            logger.debug("SQLite cache write error for '%s': %s", key, exc)

    def _sqlite_delete(self, key: str) -> bool:
        if not self._persist_to_sqlite or self._sqlite_path is None:
            return False
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.execute("DELETE FROM local_cache WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as exc:
            logger.debug("SQLite cache delete error for '%s': %s", key, exc)
            return False

    def _sqlite_clear(self) -> None:
        if not self._persist_to_sqlite or self._sqlite_path is None:
            return
        try:
            with self._get_sqlite_connection() as conn:
                conn.execute("DELETE FROM local_cache;")
                conn.commit()
        except Exception as exc:
            logger.debug("SQLite cache clear error: %s", exc)

    def _sqlite_cleanup_expired(self, now: float) -> int:
        if not self._persist_to_sqlite or self._sqlite_path is None:
            return 0
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM local_cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                )
                conn.commit()
                return cursor.rowcount
        except Exception as exc:
            logger.debug("SQLite cache purge error: %s", exc)
            return 0

    # ── Core Cache Operations ─────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Get an item from cache, respecting TTL and updating LRU order."""
        with self._lock:
            # 1. Check in-memory LRU
            if key in self._cache:
                entry = self._cache[key]
                if entry.expires_at is not None and time.time() > entry.expires_at:
                    del self._cache[key]
                    self._sqlite_delete(key)
                    return default
                self._cache.move_to_end(key)
                return entry.value

            # 2. Check SQLite persistence fallback
            if self._persist_to_sqlite:
                res = self._sqlite_get(key)
                if res is not None:
                    val, expires_at = res
                    if expires_at is not None and time.time() > expires_at:
                        self._sqlite_delete(key)
                        return default
                    # Populate in-memory LRU
                    if len(self._cache) >= self._max_size:
                        self._cache.popitem(last=False)
                    self._cache[key] = _CacheEntry(value=val, expires_at=expires_at)
                    return val

            return default

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store an item in cache with LRU eviction and optional TTL expiration."""
        with self._lock:
            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = (time.time() + effective_ttl) if (effective_ttl is not None and effective_ttl > 0) else None

            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = _CacheEntry(value=value, expires_at=expires_at)

            if self._persist_to_sqlite:
                self._sqlite_upsert(key, value, expires_at)

    def delete(self, key: str) -> bool:
        """Delete an item from both in-memory cache and SQLite storage."""
        with self._lock:
            in_mem = self._cache.pop(key, None) is not None
            in_sqlite = self._sqlite_delete(key)
            return in_mem or in_sqlite

    def clear(self) -> None:
        """Clear all items from cache and persistent backing store."""
        with self._lock:
            self._cache.clear()
            self._sqlite_clear()

    def cleanup_expired(self) -> int:
        """Purge all expired entries across memory and SQLite."""
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._cache.items()
                if v.expires_at is not None and now > v.expires_at
            ]
            for k in expired_keys:
                del self._cache[k]

            pruned = len(expired_keys)
            pruned += self._sqlite_cleanup_expired(now)
            return pruned

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        with self._lock:
            now = time.time()
            return sum(
                1 for v in self._cache.values()
                if v.expires_at is None or v.expires_at >= now
            )

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None and key not in self:
            raise KeyError(key)
        return val

    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)

    # ── Async Support ─────────────────────────────────────────────────────────

    async def aget(self, key: str, default: Any = None) -> Any:
        return self.get(key, default)

    async def aset(self, key: str, value: Any, ttl: float | None = None) -> None:
        self.set(key, value, ttl=ttl)

    async def adelete(self, key: str) -> bool:
        return self.delete(key)

    async def aclear(self) -> None:
        self.clear()

    def ping(self) -> bool:
        """Health check / readiness probe for cache layer."""
        return True

    async def aping(self) -> bool:
        return True

    # ── High-Level Embedding and Query Helpers ────────────────────────────────

    @staticmethod
    def make_embedding_key(text: str, model_name: str, dimensions: int) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"emb:{model_name}:{dimensions}:{text_hash}"

    def get_embedding(
        self,
        text: str,
        model_name: str,
        dimensions: int,
    ) -> list[float] | None:
        """Retrieve a cached embedding vector."""
        key = self.make_embedding_key(text, model_name, dimensions)
        return self.get(key)

    def set_embedding(
        self,
        text: str,
        model_name: str,
        dimensions: int,
        embedding: list[float],
        ttl: float | None = None,
    ) -> None:
        """Cache an embedding vector."""
        key = self.make_embedding_key(text, model_name, dimensions)
        self.set(key, embedding, ttl=ttl)

    @staticmethod
    def make_query_key(
        knowledge_base_id: str,
        query: str,
        context_hash: str = "",
    ) -> str:
        raw = f"{knowledge_base_id}:{query}:{context_hash}"
        q_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"query:{knowledge_base_id}:{q_hash}"

    def get_query_cache(
        self,
        knowledge_base_id: str,
        query: str,
        context_hash: str = "",
    ) -> Any | None:
        """Retrieve a cached query result."""
        key = self.make_query_key(knowledge_base_id, query, context_hash)
        return self.get(key)

    def set_query_cache(
        self,
        knowledge_base_id: str,
        query: str,
        result: Any,
        context_hash: str = "",
        ttl: float | None = None,
    ) -> None:
        """Cache a query result."""
        key = self.make_query_key(knowledge_base_id, query, context_hash)
        self.set(key, result, ttl=ttl)


_global_cache_lock = threading.RLock()
_global_cache: LocalCache | None = None


def get_local_cache(
    data_dir: Path | str | None = None,
    persist_to_sqlite: bool = True,
) -> LocalCache:
    """Return the global LocalCache singleton."""
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                sqlite_path = None
                if persist_to_sqlite:
                    root = Path(data_dir or settings.QUANTIVA_DATA_DIR).expanduser().resolve()
                    sqlite_path = root / "database" / "cache.db"
                _global_cache = LocalCache(
                    sqlite_path=sqlite_path,
                    persist_to_sqlite=persist_to_sqlite,
                )
    return _global_cache


def reset_local_cache() -> None:
    """Reset the global LocalCache singleton (for test isolation)."""
    global _global_cache
    with _global_cache_lock:
        if _global_cache is not None:
            _global_cache.clear()
            _global_cache = None


__all__ = [
    "LocalCache",
    "get_local_cache",
    "reset_local_cache",
]
