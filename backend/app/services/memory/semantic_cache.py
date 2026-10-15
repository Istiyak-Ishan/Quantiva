import json
import time
import hashlib
import numpy as np
from typing import List, Optional
import redis.asyncio as redis

from app.config import settings
from app.schemas.agent import AgentAnswer
from app.services.retrieval.embedding_service import embedding_service

class SemanticCache:
    def __init__(self):
        self.client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.SIMILARITY_THRESHOLD = 0.92
        self.TTL_SECONDS = 3600  # 1 hour
        self.PREFIX = "semantic_cache:"

    def _generate_key(self, doc_ids: List[str]) -> str:
        """
        Creates a deterministic hash from the document IDs.
        This ensures queries are only matched against cache entries
        that were generated for the EXACT same set of documents.
        """
        if not doc_ids:
            return f"{self.PREFIX}global"
        sorted_docs = sorted(doc_ids)
        docs_string = ",".join(sorted_docs)
        doc_hash = hashlib.md5(docs_string.encode()).hexdigest()
        return f"{self.PREFIX}{doc_hash}"

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(v1)
        b = np.array(v2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def get(self, query: str, doc_ids: List[str]) -> Optional[AgentAnswer]:
        """
        Finds the semantically closest cached answer for the exact same doc_ids.
        """
        cache_key = self._generate_key(doc_ids)
        
        # 1. Fetch all cached entries for this document combination
        cached_entries_raw = await self.client.hgetall(cache_key)
        if not cached_entries_raw:
            return None
            
        # 2. Embed the incoming query
        query_embedding = embedding_service.embed_text(query)
        
        best_match = None
        best_score = -1.0
        
        # 3. Iterate through cached queries and compute similarity
        for cached_query, cached_data_str in cached_entries_raw.items():
            cached_data = json.loads(cached_data_str)
            cached_embedding = cached_data["embedding"]
            
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_match = cached_data
                
        # 4. Return if it meets our strict threshold
        if best_match and best_score >= self.SIMILARITY_THRESHOLD:
            # We found a match! Return the cached AgentAnswer
            answer_data = best_match["answer"]
            answer = AgentAnswer(**answer_data)
            answer.cache_hit = True
            return answer
            
        return None

    async def set(self, query: str, doc_ids: List[str], answer: AgentAnswer):
        """
        Embeds the query and stores the answer in the cache hash.
        """
        cache_key = self._generate_key(doc_ids)
        query_embedding = embedding_service.embed_text(query)
        
        # Ensure cache_hit is False when saving
        answer.cache_hit = False
        
        cache_payload = {
            "query": query,
            "embedding": query_embedding,
            "answer": answer.model_dump(),
            "timestamp": time.time()
        }
        
        # Store in Redis Hash Map
        await self.client.hset(cache_key, query, json.dumps(cache_payload))
        # Reset the TTL for this document combination's cache bucket
        await self.client.expire(cache_key, self.TTL_SECONDS)

    async def invalidate(self, doc_id: str):
        """
        In a real system, you would iterate through keys and invalidate
        any hash map where `doc_id` is part of the doc_ids hash.
        For simplicity, we might flush the cache namespace if needed.
        """
        pass

semantic_cache = SemanticCache()
