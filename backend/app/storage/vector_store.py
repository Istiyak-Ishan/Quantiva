"""Embedded Qdrant vector repository with strict process-wide lifecycle safety."""
from __future__ import annotations

import asyncio
import math
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import ensure_data_directories, settings
from app.ingestion.types import DocumentChunkPayload
from app.providers.interfaces.exceptions import EmbeddingDimensionMismatchError
from app.storage.interfaces import CollectionInfo, SearchHit, VectorStoreRepository

_T = TypeVar("_T")


def collection_name_for_knowledge_base(knowledge_base_id: str) -> str:
    """Return the isolated embedded collection name for one knowledge base."""
    normalized = str(knowledge_base_id).strip()
    if not normalized:
        raise ValueError("knowledge_base_id is required")
    return f"kb_{normalized}"


def _point_id(chunk_id: str) -> str:
    """Return a Qdrant-compatible deterministic UUID for any chunk identifier."""
    try:
        return str(uuid.UUID(str(chunk_id)))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"quantiva:chunk:{chunk_id}"))


def _vector_dimensions(vector_config: Any) -> int:
    if isinstance(vector_config, qdrant_models.VectorParams):
        return int(vector_config.size)
    if isinstance(vector_config, Mapping):
        if len(vector_config) != 1:
            raise ValueError("Named multi-vector collections are not supported")
        params = next(iter(vector_config.values()))
        if not isinstance(params, qdrant_models.VectorParams):
            raise ValueError("Unsupported Qdrant vector configuration")
        return int(params.size)
    size = getattr(vector_config, "size", None)
    if size is None:
        raise ValueError("Unable to determine Qdrant collection dimensions")
    return int(size)


class VectorStoreManager:
    """Process-wide owner of the only embedded :class:`QdrantClient`.

    Embedded Qdrant takes an exclusive directory lock.  Constructing this
    manager repeatedly returns the same object; attempting to point an active
    manager at another directory raises instead of opening a second client.
    After :meth:`close`, the singleton may be started again (including in tests)
    and persisted vectors remain on disk.
    """

    _instance: VectorStoreManager | None = None
    _instance_lock = threading.RLock()
    _initialized: bool

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        vector_store_dir: Path | str | None = None,
    ) -> None:
        explicit_path = data_dir is not None or vector_store_dir is not None
        # Asking for the singleton without a path must return the active owner,
        # not reinterpret the request as an attempt to switch directories.
        if self._initialized and self._client is not None and not explicit_path:
            return
        requested = self._resolve_store_dir(data_dir, vector_store_dir)
        with self._instance_lock:
            if not self._initialized:
                self._client: QdrantClient | None = None
                self._store_dir = requested
                self._operation_lock = threading.RLock()
                self._initialized = True
            elif self._client is not None and requested != self._store_dir:
                raise RuntimeError(
                    "The embedded Qdrant singleton is already active at "
                    f"'{self._store_dir}'. Close it before selecting '{requested}'."
                )
            elif self._client is None:
                self._store_dir = requested

    @staticmethod
    def _resolve_store_dir(
        data_dir: Path | str | None,
        vector_store_dir: Path | str | None,
    ) -> Path:
        if vector_store_dir is not None:
            path = Path(vector_store_dir).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        paths = ensure_data_directories(data_dir or settings.QUANTIVA_DATA_DIR)
        return paths.vector_store

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    @property
    def is_started(self) -> bool:
        return self._client is not None

    @property
    def client(self) -> QdrantClient:
        client = self._client
        if client is None:
            raise RuntimeError(
                "VectorStoreManager is not started. Initialize it during FastAPI lifespan startup."
            )
        return client

    def start(self) -> QdrantClient:
        with self._instance_lock:
            if self._client is None:
                self._store_dir.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(self._store_dir))
            return self._client

    def close(self) -> None:
        with self._instance_lock:
            client = self._client
            self._client = None
            if client is not None:
                client.close()

    def run_locked(self, operation: Callable[[QdrantClient], _T]) -> _T:
        """Run one client operation while excluding all other local accesses."""
        with self._operation_lock:
            return operation(self.client)

    def __enter__(self) -> VectorStoreManager:
        self.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


class LocalVectorStore(VectorStoreRepository):
    """Async repository adapter over the singleton embedded Qdrant client."""

    def __init__(self, manager: VectorStoreManager | None = None) -> None:
        self.manager = manager or VectorStoreManager()
        self._async_lock = asyncio.Lock()

    async def _run(self, operation: Callable[[QdrantClient], _T]) -> _T:
        async with self._async_lock:
            return await asyncio.to_thread(self.manager.run_locked, operation)

    @staticmethod
    def collection_name(knowledge_base_id: str) -> str:
        return collection_name_for_knowledge_base(knowledge_base_id)

    @staticmethod
    def _validate_vector(vector: Sequence[float], expected: int) -> list[float]:
        values = [float(value) for value in vector]
        actual = len(values)
        if actual != expected:
            raise EmbeddingDimensionMismatchError(
                expected=expected,
                actual=actual,
                provider="local_qdrant",
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding vectors must contain only finite values")
        if not any(value != 0.0 for value in values):
            raise ValueError("Embedding vectors must not be all zero")
        return values

    @staticmethod
    def _index_dimensions(
        metadata: Mapping[str, Any] | None,
        index_metadata: Any | None,
    ) -> int | None:
        if index_metadata is not None:
            value = getattr(index_metadata, "embedding_dimensions", None)
            if value is None and isinstance(index_metadata, Mapping):
                value = index_metadata.get("embedding_dimensions")
            if value is not None:
                return int(value)
        if metadata is not None and metadata.get("embedding_dimensions") is not None:
            return int(metadata["embedding_dimensions"])
        return None

    @staticmethod
    def _validate_index_metadata(
        collection_name: str,
        collection_dimensions: int,
        index_metadata: Any | None,
    ) -> None:
        if index_metadata is None:
            return
        metadata_collection = getattr(
            index_metadata, "vector_collection_name", None
        ) or getattr(index_metadata, "collection_name", None)
        if isinstance(index_metadata, Mapping):
            metadata_collection = (
                index_metadata.get("vector_collection_name")
                or index_metadata.get("collection_name")
                or metadata_collection
            )
        if metadata_collection and str(metadata_collection) != collection_name:
            raise ValueError(
                "IndexMetadata collection does not match the requested collection"
            )
        expected = getattr(index_metadata, "embedding_dimensions", None)
        if isinstance(index_metadata, Mapping):
            expected = index_metadata.get("embedding_dimensions", expected)
        if expected is not None and int(expected) != collection_dimensions:
            raise EmbeddingDimensionMismatchError(
                expected=int(expected),
                actual=collection_dimensions,
                provider="local_qdrant",
            )

    @staticmethod
    def _collection_info_sync(
        client: QdrantClient,
        collection_name: str,
    ) -> CollectionInfo | None:
        if not client.collection_exists(collection_name):
            return None
        info = client.get_collection(collection_name)
        dimensions = _vector_dimensions(info.config.params.vectors)
        distance = getattr(info.config.params.vectors, "distance", "cosine")
        if hasattr(distance, "value"):
            distance = distance.value
        return CollectionInfo(
            name=collection_name,
            dimensions=dimensions,
            distance_metric=str(distance).lower(),
            point_count=int(info.points_count or 0),
        )

    async def ensure_collection(
        self,
        collection_name: str,
        embedding_dimensions: int,
        *,
        index_metadata: Any | None = None,
    ) -> CollectionInfo:
        if not collection_name.strip():
            raise ValueError("collection_name is required")
        if embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")

        def operation(client: QdrantClient) -> CollectionInfo:
            info = self._collection_info_sync(client, collection_name)
            if info is None:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=int(embedding_dimensions),
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                info = self._collection_info_sync(client, collection_name)
                if info is None:  # pragma: no cover - defensive engine guard
                    raise RuntimeError("Qdrant did not create the requested collection")
            if info.dimensions != int(embedding_dimensions):
                raise EmbeddingDimensionMismatchError(
                    expected=info.dimensions,
                    actual=int(embedding_dimensions),
                    provider="local_qdrant",
                )
            self._validate_index_metadata(
                collection_name,
                info.dimensions,
                index_metadata,
            )
            return info

        return await self._run(operation)

    async def ensure_knowledge_base_collection(
        self,
        knowledge_base_id: str,
        embedding_dimensions: int,
        *,
        index_metadata: Any | None = None,
    ) -> CollectionInfo:
        return await self.ensure_collection(
            self.collection_name(knowledge_base_id),
            embedding_dimensions,
            index_metadata=index_metadata,
        )

    async def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[DocumentChunkPayload],
        embeddings: Sequence[Sequence[float]],
        *,
        metadata: Mapping[str, Any] | None = None,
        index_metadata: Any | None = None,
    ) -> list[str]:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk requires exactly one embedding")
        if not chunks:
            return []

        knowledge_base_ids = {str(chunk["knowledge_base_id"]) for chunk in chunks}
        if len(knowledge_base_ids) != 1:
            raise ValueError("One upsert batch may contain only one knowledge base")
        knowledge_base_id = next(iter(knowledge_base_ids))
        expected_collection = self.collection_name(knowledge_base_id)
        is_shadow_collection = collection_name.startswith(
            f"{expected_collection}_reindex_"
        )
        if collection_name != expected_collection and not is_shadow_collection:
            raise ValueError(
                f"Knowledge base '{knowledge_base_id}' must use collection "
                f"'{expected_collection}' (or its temporary reindex shadow), "
                f"not '{collection_name}'"
            )

        declared_dimensions = self._index_dimensions(metadata, index_metadata)
        vector_dimensions = len(embeddings[0])
        if declared_dimensions is not None and declared_dimensions != vector_dimensions:
            raise EmbeddingDimensionMismatchError(
                expected=declared_dimensions,
                actual=vector_dimensions,
                provider="local_qdrant",
            )
        requested_dimensions = declared_dimensions or vector_dimensions
        await self.ensure_collection(
            collection_name,
            requested_dimensions,
            index_metadata=index_metadata,
        )

        vectors = [
            self._validate_vector(vector, requested_dimensions)
            for vector in embeddings
        ]
        common_payload = dict(metadata or {})
        common_payload.pop("embedding_dimensions", None)
        point_ids: list[str] = []
        points: list[qdrant_models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = str(chunk["chunk_id"])
            point_id = _point_id(chunk_id)
            point_ids.append(point_id)
            payload = {
                **common_payload,
                "chunk_id": chunk_id,
                "document_id": str(chunk["document_id"]),
                "knowledge_base_id": knowledge_base_id,
                "page_number": chunk.get("page_number"),
                "chunk_index": int(chunk["chunk_index"]),
                "source_file": str(chunk["source_file"]),
                "is_table": bool(chunk["is_table"]),
                "content": str(chunk["content"]),
            }
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        def operation(client: QdrantClient) -> None:
            # Re-check under the same operation lock immediately before write.
            info = self._collection_info_sync(client, collection_name)
            if info is None:
                raise RuntimeError("Vector collection disappeared before upsert")
            if info.dimensions != requested_dimensions:
                raise EmbeddingDimensionMismatchError(
                    expected=info.dimensions,
                    actual=requested_dimensions,
                    provider="local_qdrant",
                )
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

        await self._run(operation)
        return point_ids

    @staticmethod
    def _query_filter(filters: Mapping[str, Any] | None):
        if not filters:
            return None
        conditions: list[Any] = []
        document_id = filters.get("document_id")
        document_ids = filters.get("document_ids")
        if document_id is not None:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchValue(value=str(document_id)),
                )
            )
        elif document_ids:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchAny(
                        any=[str(value) for value in document_ids]
                    ),
                )
            )
        knowledge_base_id = filters.get("knowledge_base_id")
        if knowledge_base_id is not None:
            conditions.append(
                qdrant_models.FieldCondition(
                    key="knowledge_base_id",
                    match=qdrant_models.MatchValue(value=str(knowledge_base_id)),
                )
            )
        return qdrant_models.Filter(must=conditions) if conditions else None

    async def search_dense(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        *,
        index_metadata: Any | None = None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        def operation(client: QdrantClient) -> list[SearchHit]:
            info = self._collection_info_sync(client, collection_name)
            if info is None:
                return []
            self._validate_index_metadata(
                collection_name,
                info.dimensions,
                index_metadata,
            )
            filter_dimensions = self._index_dimensions(filters, None)
            if filter_dimensions is not None and filter_dimensions != info.dimensions:
                raise EmbeddingDimensionMismatchError(
                    expected=info.dimensions,
                    actual=filter_dimensions,
                    provider="local_qdrant",
                )
            vector = self._validate_vector(query_vector, info.dimensions)
            response = client.query_points(
                collection_name=collection_name,
                query=vector,
                query_filter=self._query_filter(filters),
                limit=top_k,
                with_payload=True,
            )
            hits: list[SearchHit] = []
            for point in response.points:
                payload = dict(point.payload or {})
                hits.append(
                    SearchHit(
                        chunk_id=str(payload.get("chunk_id") or point.id),
                        score=float(point.score),
                        content=str(payload.get("content") or ""),
                        metadata=payload,
                    )
                )
            return hits

        return await self._run(operation)

    async def delete_by_document_id(
        self,
        collection_name: str,
        document_id: str,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> None:
        if not document_id:
            raise ValueError("document_id is required")

        def operation(client: QdrantClient) -> None:
            if not client.collection_exists(collection_name):
                return
            conditions: list[Any] = [
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchValue(value=str(document_id)),
                )
            ]
            if filters and filters.get("knowledge_base_id") is not None:
                conditions.append(
                    qdrant_models.FieldCondition(
                        key="knowledge_base_id",
                        match=qdrant_models.MatchValue(
                            value=str(filters["knowledge_base_id"])
                        ),
                    )
                )
            client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(must=conditions)
                ),
                wait=True,
            )

        await self._run(operation)

    async def delete_collection(self, collection_name: str) -> None:
        def operation(client: QdrantClient) -> None:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name=collection_name)

        await self._run(operation)

    async def get_collection_info(
        self,
        collection_name: str,
    ) -> CollectionInfo | None:
        return await self._run(
            lambda client: self._collection_info_sync(client, collection_name)
        )

    async def scroll_vectors(
        self,
        collection_name: str,
        *,
        batch_size: int = 256,
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        """Stream every point of a collection out as plain Python records.

        Used by portable Knowledge Base export so pre-computed embeddings can
        travel with the archive and be restored without re-embedding.  Missing
        collections return an empty list rather than raising, matching the
        delete/no-op conventions of this repository.
        """
        if not collection_name.strip():
            raise ValueError("collection_name is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        def operation(client: QdrantClient) -> list[dict[str, Any]]:
            if not client.collection_exists(collection_name):
                return []
            records: list[dict[str, Any]] = []
            offset: Any = None
            while True:
                points, offset = client.scroll(
                    collection_name=collection_name,
                    limit=int(batch_size),
                    offset=offset,
                    with_payload=with_payload,
                    with_vectors=True,
                )
                for point in points:
                    raw_vector: Any = point.vector
                    if isinstance(raw_vector, Mapping):
                        raw_vector = next(iter(raw_vector.values()), None)
                    vec_floats: list[float] = []
                    if isinstance(raw_vector, list):
                        for value in raw_vector:
                            try:
                                vec_floats.append(float(value))
                            except (TypeError, ValueError):
                                pass
                    payload = dict(point.payload or {}) if with_payload else {}
                    records.append(
                        {
                            "point_id": str(point.id),
                            "chunk_id": (
                                str(payload["chunk_id"])
                                if payload.get("chunk_id") is not None
                                else None
                            ),
                            "payload": payload,
                            "vector": vec_floats,
                        }
                    )
                if offset is None:
                    break
            return records

        return await self._run(operation)


__all__ = [
    "LocalVectorStore",
    "VectorStoreManager",
    "collection_name_for_knowledge_base",
]
