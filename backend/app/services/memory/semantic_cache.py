"""Tenant-, corpus-, model-, and conversation-scoped optional answer cache."""
import asyncio
import hashlib
import json
import logging
import math

import redis.asyncio as redis

from app.config import settings
from app.schemas.agent import AgentAnswer
from app.services.retrieval.embedding_service import embedding_service

logger = logging.getLogger(__name__)


class SemanticCache:
    TTL_SECONDS = 3600
    SIMILARITY_THRESHOLD = .92
    PREFIX = "semantic_cache:v2:"

    def __init__(self):
        self.client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)

    def _generate_key(self, doc_ids, *, user_id, corpus_version="", context_fingerprint=""):
        if not user_id:
            raise ValueError("Tenant user_id is required")
        identity = json.dumps([sorted(doc_ids), corpus_version, context_fingerprint, settings.GEMINI_CHAT_MODEL, settings.EMBEDDING_MODEL, settings.EMBEDDING_DIMENSIONS])
        return f"{self.PREFIX}{user_id}:{hashlib.sha256(identity.encode()).hexdigest()}"

    def _doc_index(self, doc_id, user_id):
        return f"{self.PREFIX}doc:{user_id}:{doc_id}"

    def _cosine_similarity(self, a, b):
        if len(a) != len(b):
            return -1.
        denom = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
        return sum(x*y for x, y in zip(a, b)) / denom if denom else -1.

    async def get(self, query, doc_ids, *, user_id, corpus_version="", context_fingerprint=""):
        if not doc_ids:
            return None
        key = self._generate_key(doc_ids, user_id=user_id, corpus_version=corpus_version, context_fingerprint=context_fingerprint)
        try:
            entries = await self.client.hgetall(key)
            if not entries:
                return None
            query_vector = await asyncio.to_thread(embedding_service.embed_text, query, task_type="SEMANTIC_SIMILARITY")
            for value in entries.values():
                try:
                    data = json.loads(value)
                    if self._cosine_similarity(query_vector, data["embedding"]) >= self.SIMILARITY_THRESHOLD:
                        answer = AgentAnswer.model_validate(data["answer"])
                        answer.cache_hit = True
                        answer.tokens_used = answer.input_tokens = answer.output_tokens = 0
                        answer.cost_usd = 0.0
                        return answer
                except (ValueError, KeyError, TypeError):
                    continue
        except Exception:
            # Cache availability is optional. A miss still runs the real model.
            logger.warning("Semantic cache read unavailable", exc_info=False)
        return None

    async def set(self, query, doc_ids, answer, *, user_id, corpus_version="", context_fingerprint=""):
        if not doc_ids:
            return
        key = self._generate_key(doc_ids, user_id=user_id, corpus_version=corpus_version, context_fingerprint=context_fingerprint)
        try:
            vector = await asyncio.to_thread(embedding_service.embed_text, query, task_type="SEMANTIC_SIMILARITY")
            await self.client.hset(key, query, json.dumps({"embedding": vector, "answer": answer.model_dump()}))
            await self.client.expire(key, self.TTL_SECONDS)
            for doc_id in doc_ids:
                index = self._doc_index(doc_id, user_id)
                await self.client.sadd(index, key)
                await self.client.expire(index, self.TTL_SECONDS)
        except Exception:
            logger.warning("Semantic cache write unavailable", exc_info=False)

    async def invalidate(self, doc_id: str, *, user_id: str):
        # Do not swallow deletion failures: callers retain a retryable document.
        index = self._doc_index(doc_id, user_id)
        keys = await self.client.smembers(index)
        if keys:
            await self.client.delete(*keys)
        await self.client.delete(index)


semantic_cache = SemanticCache()
