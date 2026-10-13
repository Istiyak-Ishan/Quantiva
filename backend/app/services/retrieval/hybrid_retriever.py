from typing import List, Dict, Any
from app.services.retrieval.vector_store import vector_store
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.embedding_service import embedding_service

from app.services.retrieval.reranker import reranker

class HybridRetriever:
    def _rrf_merge(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.
        RRF Score = 1 / (k + rank)
        """
        rrf_scores = {}
        chunk_map = {}
        
        for rank, res in enumerate(dense_results):
            chunk_id = res["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))
            chunk_map[chunk_id] = res["payload"]
            
        for rank, res in enumerate(sparse_results):
            chunk_payload = res["chunk"]["payload"]
            chunk_id = res["chunk"]["qdrant_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank + 1))
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

    async def search(self, query: str, doc_ids: List[str], db_chunks: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Orchestrates the Hybrid Search.
        dense(20) + sparse(20) -> RRF(30) -> rerank(8)
        """
        # A. Dense (Vector) Search (Top 20)
        query_embedding = embedding_service.embed_text(query)
        dense_results = await vector_store.search(query_embedding, doc_ids, limit=20)
        
        # B. Sparse (BM25) Search (Top 20)
        bm25_input = []
        for c in db_chunks:
            bm25_input.append({
                "text": c.text_preview, 
                "qdrant_id": str(c.qdrant_id),
                "payload": {
                    "text": c.text_preview,
                    "page": c.page_number,
                    "doc_id": str(c.doc_id)
                }
            })
            
        bm25 = BM25Retriever()
        bm25.build_index(bm25_input)
        sparse_results = bm25.search(query, limit=20)
        
        # C. RRF Merge (Top 30 candidates for reranking)
        hybrid_results = self._rrf_merge(dense_results, sparse_results, k=60)
        candidates = hybrid_results[:30]
        
        # D. Rerank (Top 8 final results)
        reranked_results = reranker.rerank(query, candidates, top_k=top_k)
        
        return reranked_results

hybrid_retriever = HybridRetriever()
