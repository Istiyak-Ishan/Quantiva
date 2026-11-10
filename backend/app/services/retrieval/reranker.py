"""Lazy cross-encoder lifecycle: no model downloads during application imports."""
from threading import Lock


class CrossEncoderReranker:
    def __init__(self):
        self.model = None
        self._lock = Lock()

    def rerank(self, query, chunks, top_k=8):
        if not chunks:
            return []
        # Called via to_thread by the async retriever. Model load/inference are
        # serialized because the CPU model is shared across requests.
        with self._lock:
            if self.model is None:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
            scores = self.model.predict([[query, c["payload"]["text"]] for c in chunks])
        return sorted([{**chunk, "rerank_score": float(score)} for chunk, score in zip(chunks, scores)], key=lambda c: c["rerank_score"], reverse=True)[:top_k]


reranker = CrossEncoderReranker()
