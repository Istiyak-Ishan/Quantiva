import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.llm.errors import EmbeddingError
from app.services.retrieval.embedding_service import EmbeddingService


def test_embeddings_are_normalized_and_task_dimension_configured():
    client = MagicMock()
    service = EmbeddingService(client=client)
    service.vector_size = 3
    client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[3., 4., 0.])])
    vector = service.embed_text("a real query")
    assert vector == [0.6, 0.8, 0.0]
    config = client.models.embed_content.call_args.kwargs["config"]
    assert config.task_type == "RETRIEVAL_QUERY"
    assert config.output_dimensionality == 3


def test_embedding_error_never_returns_random_vectors():
    client = MagicMock()
    service = EmbeddingService(client=client)
    client.models.embed_content.side_effect = RuntimeError("quota")
    for texts in (["one"], ["one", "two"]):
        with pytest.raises(EmbeddingError):
            service.embed_batch(texts)


@pytest.mark.parametrize("vectors", [[], [[1., 2.]], [[0., 0., 0.]], [[math.nan, 0., 1.]]])
def test_malformed_vectors_rejected(vectors):
    client = MagicMock()
    service = EmbeddingService(client=client)
    service.vector_size = 3
    client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in vectors])
    with pytest.raises(EmbeddingError):
        service.embed_text("query")
