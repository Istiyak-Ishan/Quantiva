import uuid
import math
from typing import List, Dict, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from app.config import settings

class VectorStore:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL, check_compatibility=False, timeout=10)
        self.vector_size = settings.EMBEDDING_DIMENSIONS
        self.collection_name = settings.embedding_collection

    async def create_collection(self):
        """
        Idempotent function to ensure the collection exists.
        It uses Cosine similarity for matching vectors.
        """
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=self.vector_size,
                    distance=rest.Distance.COSINE
                )
            )
        else:
            info = await self.client.get_collection(self.collection_name)
            if not isinstance(info.config.params.vectors, rest.VectorParams) or info.config.params.vectors.size != self.vector_size:
                raise ValueError("Embedding dimensions do not match collection; use a new versioned collection and reindex")

    async def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], doc_id: str, user_id: str) -> List[str]:
        """
        Inserts chunks and their vector embeddings into Qdrant.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk requires exactly one embedding")
        if not user_id:
            raise ValueError("Tenant user_id is required")
        for vector in embeddings:
            if len(vector) != self.vector_size or not all(math.isfinite(v) for v in vector) or not any(vector):
                raise ValueError("Invalid embedding vector")
        points = []
        qdrant_ids = []
        
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{doc_id}:{i}"))
            qdrant_ids.append(qdrant_id)
            
            # The 'payload' acts like columns in a relational DB. We can filter on these later!
            payload = {
                "doc_id": str(doc_id),
                "user_id": str(user_id),
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunk_index": i,
                "text": chunk["text"],
                "page": chunk["metadata"].get("page"),
                "chunk_type": chunk["metadata"].get("chunk_type")
            }
            
            points.append(
                rest.PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload=payload
                )
            )
            
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points
            , wait=True
        )
        return qdrant_ids

    async def delete_document(self, doc_id: str, *, user_id: str):
        """Deletes all chunks associated with a specific document."""
        if not user_id:
            raise ValueError("Tenant user_id is required")
        # A failed first ingestion may never have created this collection.
        # Missing vectors are already cleaned up; transport errors still fail.
        if not await self.client.collection_exists(self.collection_name):
            return
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(key="user_id", match=rest.MatchValue(value=str(user_id))),
                    rest.FieldCondition(
                        key="doc_id",
                        match=rest.MatchValue(value=str(doc_id))
                    )
                ]
            ), wait=True
        )

    async def search(self, query_embedding: List[float], doc_ids: List[str], limit: int = 20, *, user_id: str) -> List[Dict[str, Any]]:
        """
        Searches Qdrant for the most similar chunks based on cosine similarity.
        Input: the embedded query vector, a list of document IDs to filter by, and result limit.
        Output: A list of scored chunks from Qdrant.
        """
        if not user_id:
            raise ValueError("Tenant user_id is required")
        if not doc_ids:
            return []
        if len(query_embedding) != self.vector_size:
            raise ValueError("Query embedding dimension does not match collection")
        # Both filters are mandatory, even when callers validated SQL ownership.
        doc_filter = rest.Filter(
            must=[
                rest.FieldCondition(key="user_id", match=rest.MatchValue(value=str(user_id))),
                rest.FieldCondition(key="embedding_model", match=rest.MatchValue(value=settings.EMBEDDING_MODEL)),
                rest.FieldCondition(
                    key="doc_id",
                    match=rest.MatchAny(any=doc_ids)
                )
            ]
        )
        
        # Search Qdrant
        search_result = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=doc_filter,
            limit=limit,
            with_payload=True # Crucial: asks Qdrant to return the text and metadata!
        )
        
        # Format the output. 
        # score is Cosine Similarity (closer to 1.0 means highly relevant)
        results = []
        for point in search_result.points:
            results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload
            })
            
        return results

vector_store = VectorStore()
