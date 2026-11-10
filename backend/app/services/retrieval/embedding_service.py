"""Validated Gemini text embeddings. Provider failure cannot produce an index."""
import math

from google import genai
from google.genai import types

from app.config import settings
from app.services.llm.errors import EmbeddingError


class EmbeddingService:
    def __init__(self, client=None):
        self.client = client
        self.model_name = settings.EMBEDDING_MODEL
        self.vector_size = settings.EMBEDDING_DIMENSIONS

    def _validate(self, response, expected):
        vectors = [embedding.values for embedding in response.embeddings or []]
        if len(vectors) != expected:
            raise EmbeddingError()
        normalized = []
        for vector in vectors:
            if vector is None or len(vector) != self.vector_size or not all(math.isfinite(v) for v in vector):
                raise EmbeddingError()
            norm = math.sqrt(sum(v * v for v in vector))
            if not norm:
                raise EmbeddingError()
            normalized.append([v / norm for v in vector])
        return normalized

    def embed_batch(self, texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        if not texts:
            return []
        if self.model_name != "gemini-embedding-001":
            # Other embedding APIs have different task/batch semantics; do not
            # silently apply this adapter to an incompatible model.
            raise EmbeddingError()
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.vector_size)
        contents = [types.Content(parts=[types.Part.from_text(text=text)]) for text in texts]
        try:
            if self.client is not None:
                response = self.client.models.embed_content(model=self.model_name, contents=contents, config=config)
            else:
                with genai.Client(api_key=settings.GOOGLE_API_KEY, http_options=types.HttpOptions(timeout=settings.GOOGLE_TIMEOUT_MS)) as client:
                    response = client.models.embed_content(model=self.model_name, contents=contents, config=config)
            return self._validate(response, len(texts))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError() from exc

    def embed_text(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        return self.embed_batch([text], task_type=task_type)[0]


embedding_service = EmbeddingService()
