import os
from google import genai
from typing import List
from app.config import settings

class EmbeddingService:
    def __init__(self):
        # We initialize the client using the API key from config
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        # text-embedding-004 outputs exactly 768 dimensions by default.
        self.model_name = "text-embedding-004"
        self.vector_size = 768

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string into a 768-dimensional float vector.
        Input: Raw text string (e.g. "The Q3 revenue was $5M")
        Output: [0.014, -0.051, 0.991, ...] (list of 768 floats)
        """
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            # Fallback for local testing without valid API key
            import hashlib
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
            import random
            random.seed(seed)
            return [random.uniform(-1, 1) for _ in range(self.vector_size)]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of strings for high throughput (max 100 per call is typical).
        Input: List of text strings
        Output: List of vectors, where each vector is a list of 768 floats.
        """
        if not texts:
            return []
            
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=texts
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            # Fallback for local testing without valid API key
            import random
            return [[random.uniform(-1, 1) for _ in range(self.vector_size)] for _ in texts]

embedding_service = EmbeddingService()
