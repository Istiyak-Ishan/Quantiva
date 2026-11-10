import re
import asyncio
import numpy as np
from typing import List, Dict, Any
from app.services.retrieval.embedding_service import embedding_service

class HallucinationChecker:
    def __init__(self):
        self.similarity_threshold = 0.65

    def _split_into_sentences(self, text: str) -> List[str]:
        # Basic sentence splitting (using regex for simplicity in this example)
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def check(self, answer: str, source_chunks: List[str]) -> Dict[str, Any]:
        """
        Checks if every sentence in the answer is supported by the source chunks.
        """
        if not source_chunks or not answer:
            return {"score": 0.0, "flagged_claims": [], "is_hallucinated": True}

        sentences = self._split_into_sentences(answer)
        if not sentences:
            return {"score": 0.0, "flagged_claims": [], "is_hallucinated": True}

        # Embed all source chunks once
        chunk_embeddings, sentence_embeddings = await asyncio.gather(
            asyncio.to_thread(embedding_service.embed_batch, source_chunks, task_type="SEMANTIC_SIMILARITY"),
            asyncio.to_thread(embedding_service.embed_batch, sentences, task_type="SEMANTIC_SIMILARITY"),
        )
        
        supported_sentences = 0
        flagged_claims = []

        for idx, sentence_emb in enumerate(sentence_embeddings):
            is_supported = False
            best_score = 0.0
            
            for chunk_emb in chunk_embeddings:
                sim = self._cosine_similarity(sentence_emb, chunk_emb)
                if sim > best_score:
                    best_score = sim
                    
                if sim >= self.similarity_threshold:
                    is_supported = True
                    break
                    
            if is_supported:
                supported_sentences += 1
            else:
                flagged_claims.append({
                    "sentence": sentences[idx],
                    "confidence_score": best_score
                })

        score = supported_sentences / len(sentences)
        
        return {
            "score": score,
            "flagged_claims": flagged_claims,
            "is_hallucinated": score < 0.6
        }

hallucination_checker = HallucinationChecker()
