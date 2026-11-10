import re
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

def tokenize(text: str) -> List[str]:
    """Tokenize string into alphanumeric terms in lowercase."""
    return re.findall(r'\w+', text.lower()) if text else []

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.doc_chunks = []

    def build_index(self, doc_chunks: List[Dict[str, Any]]):
        """
        Builds an in-memory BM25 index from a list of chunks.
        Tokenization uses regex word matching for robust keyword extraction.
        """
        self.doc_chunks = doc_chunks
        if not doc_chunks:
            self.bm25 = None
            return
            
        tokenized_corpus = [
            tokenize(chunk.get("text", "")) for chunk in self.doc_chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Scores the document chunks against the query keywords using BM25.
        Returns the top results sorted by score.
        """
        if not self.bm25:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
            
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine chunks with their BM25 scores
        scored_chunks = []
        for i, score in enumerate(doc_scores):
            if score > 0: # Only return chunks that actually have a keyword match
                scored_chunks.append({
                    "chunk": self.doc_chunks[i],
                    "score": score
                })
                
        # Sort descending by score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:limit]

bm25_retriever = BM25Retriever()
