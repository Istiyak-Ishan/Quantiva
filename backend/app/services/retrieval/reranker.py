from typing import List, Dict, Any
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

class CrossEncoderReranker:
    def __init__(self):
        # We use a lightweight cross-encoder model trained on MS MARCO.
        # It's fast enough for CPU and highly accurate.
        if CrossEncoder:
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        else:
            self.model = None

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Reranks the retrieved chunks using a Cross-Encoder.
        Input: Query string and a list of chunks (each containing 'payload' with 'text').
        Pairs (Query, Chunk_Text) go into the model.
        Output: The chunks sorted by their Cross-Encoder score (logits), top_k returned.
        """
        if not chunks:
            return []
            
        if not self.model:
            # Fallback for testing environments where PyTorch isn't installed
            print("WARNING: CrossEncoder not loaded. Returning original chunks.")
            return chunks[:top_k]

        # The model expects a list of pairs: [[query, text1], [query, text2], ...]
        pairs = [[query, c["payload"]["text"]] for c in chunks]
        
        # The score is a logit (can be negative or positive, higher is better)
        scores = self.model.predict(pairs)
        
        # Attach the new scores to the chunks
        for i, score in enumerate(scores):
            chunks[i]["rerank_score"] = float(score)
            
        # Sort by the new rerank score descending
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return chunks[:top_k]

reranker = CrossEncoderReranker()
