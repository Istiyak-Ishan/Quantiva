import uuid
from typing import List, Dict, Any
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from app.config import settings

class VectorStore:
    def __init__(self):
        self.client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.collection_name = "quantiva_documents"
        self.vector_size = 768 # Specific to text-embedding-004

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

    async def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], doc_id: str, user_id: str) -> List[uuid.UUID]:
        """
        Inserts chunks and their vector embeddings into Qdrant.
        """
        points = []
        qdrant_ids = []
        
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            qdrant_id = str(uuid.uuid4())
            qdrant_ids.append(qdrant_id)
            
            # The 'payload' acts like columns in a relational DB. We can filter on these later!
            payload = {
                "doc_id": str(doc_id),
                "user_id": str(user_id),
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
        )
        return qdrant_ids

    async def delete_document(self, doc_id: str):
        """Deletes all chunks associated with a specific document."""
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id",
                        match=rest.MatchValue(value=str(doc_id))
                    )
                ]
            )
        )

    async def search(self, query_embedding: List[float], doc_ids: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Searches Qdrant for the most similar chunks based on cosine similarity.
        Input: the embedded query vector, a list of document IDs to filter by, and result limit.
        Output: A list of scored chunks from Qdrant.
        """
        # Create a filter to ONLY search within the specified documents
        doc_filter = rest.Filter(
            must=[
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
