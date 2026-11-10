from typing import List, Dict, Any
import asyncio
import time
from app.core.metrics import RETRIEVAL_LATENCY
from app.services.retrieval.vector_store import vector_store
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.embedding_service import embedding_service

from app.services.retrieval.reranker import reranker

class HybridRetriever:
    def __init__(self, store=None):
        self.vector_store = store if store is not None else vector_store

    def _rrf_merge(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.
        RRF Score = 1 / (k + rank)
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map = {}
        
        for rank, res in enumerate(dense_results):
            chunk_id = res["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))
            chunk_map[chunk_id] = res["payload"]
            
        for rank, res in enumerate(sparse_results):
            chunk_payload = res["chunk"]["payload"]
            chunk_id = res["chunk"]["qdrant_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))
            # Preserve the richest payload (dense search has full chunk text, preview might be truncated)
            if chunk_id not in chunk_map or len(chunk_payload.get("text", "")) > len(chunk_map[chunk_id].get("text", "")):
                chunk_map[chunk_id] = chunk_payload

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for chunk_id, score in sorted_rrf:
            final_results.append({
                "id": chunk_id,
                "score": score,
                "payload": chunk_map[chunk_id]
            })
            
        return final_results

    async def search(self, query: str, doc_ids: List[str], db_chunks: List[Any], top_k: int = 8, *, user_id: str) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        try:
            return await self._search(query, doc_ids, db_chunks, top_k, user_id=user_id)
        finally:
            RETRIEVAL_LATENCY.observe(time.perf_counter() - started)

    async def _search(self, query: str, doc_ids: List[str], db_chunks: List[Any], top_k: int = 8, *, user_id: str) -> List[Dict[str, Any]]:
        """
        Orchestrates the Hybrid Search.
        dense(20) + sparse(20) -> RRF(30) -> rerank(8)
        """
        # A. Dense (Vector) Search (Top 20)
        if not doc_ids:
            return []
        query_embedding = await asyncio.to_thread(embedding_service.embed_text, query)
        dense_results = await self.vector_store.search(query_embedding, doc_ids, limit=20, user_id=user_id)
        
        # B. Sparse (BM25) Search (Top 20)
        doc_ids_set = set(str(d) for d in doc_ids) if doc_ids else None
        bm25_input = []
        for c in db_chunks:
            if doc_ids_set and str(getattr(c, "doc_id", "")) not in doc_ids_set:
                continue
            bm25_input.append({
                "text": c.text,
                "qdrant_id": str(getattr(c, "qdrant_id", "")),
                "payload": {
                    "text": getattr(c, "text", getattr(c, "text_preview", "")),
                    "page": getattr(c, "page_number", 1),
                    "doc_id": str(getattr(c, "doc_id", ""))
                }
            })
            
        bm25 = BM25Retriever()
        bm25.build_index(bm25_input)
        sparse_results = bm25.search(query, limit=20)
        
        # C. RRF Merge (Top 30 candidates for reranking)
        hybrid_results = self._rrf_merge(dense_results, sparse_results, k=60)
        candidates = hybrid_results[:30]
        
        # D. Rerank (Top 8 final results)
        reranked_results = await asyncio.to_thread(reranker.rerank, query, candidates, top_k=top_k)
        
        return reranked_results

hybrid_retriever = HybridRetriever()
