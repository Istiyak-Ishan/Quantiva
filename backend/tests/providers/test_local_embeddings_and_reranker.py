"""
tests/providers/test_local_embeddings_and_reranker.py
──────────────────────────────────────────────────────────────────────────────
Offline unit tests for the in-process SentenceTransformers embedding provider,
the local CrossEncoder reranker, the NoOp reranker, and the remote JSON
reranker adapter.

No model weights are downloaded and torch / sentence-transformers are never
imported: every heavy component is replaced by a tiny deterministic fake.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import settings
from app.providers.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)
from app.providers.interfaces.exceptions import (
    EmbeddingDimensionMismatchError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from app.providers.interfaces.reranker import (
    RerankDocument,
    RerankerProvider,
)
from app.providers.local_models import detect_device, l2_normalize, to_float_list
from app.providers.rerankers.api import APIRerankerProvider
from app.providers.rerankers.cross_encoder import (
    LocalCrossEncoderReranker,
    NoOpReranker,
)

E5_MODEL = "intfloat/multilingual-e5-small"
PLAIN_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ═══════════════════════════════════════════════════════════════════════════════
# Fakes & helpers
# ═══════════════════════════════════════════════════════════════════════════════

class FakeSentenceTransformer:
    """Deterministic stand-in for ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, dimensions: int = 4, delay: float = 0.0) -> None:
        self._dimensions = dimensions
        self.delay = delay
        self.calls: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.thread_idents: list[int] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimensions

    def _vector(self, text: str) -> list[float]:
        # Deliberately non-unit length so normalisation is observable.
        spread = float(sum(ord(char) for char in text) % 11) + 1.0
        tail = [
            float(len(text) + index) + 1.0
            for index in range(self._dimensions - 1)
        ]
        return ([spread] + tail)[: self._dimensions]

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.thread_idents.append(threading.get_ident())
        self.calls.append(list(texts))
        self.kwargs.append(dict(kwargs))
        if self.delay:
            time.sleep(self.delay)
        return [self._vector(text) for text in texts]


class FakeCrossEncoder:
    """Deterministic stand-in for ``sentence_transformers.CrossEncoder``."""

    def __init__(self, scores: list[float] | None = None, delay: float = 0.0) -> None:
        self.scores = scores
        self.delay = delay
        self.pairs: list[list[tuple[str, str]]] = []
        self.thread_idents: list[int] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.thread_idents.append(threading.get_ident())
        self.pairs.append(list(pairs))
        if self.delay:
            time.sleep(self.delay)
        if self.scores is not None:
            return list(self.scores)
        # Default: longer overlap with the query scores higher.
        query = pairs[0][0].lower() if pairs else ""
        return [
            float(len(set(query.split()) & set(document.lower().split()))) + 0.1
            for _query, document in pairs
        ]


def recording_factory(model: Any) -> tuple[Callable[[], Any], list[int]]:
    loads: list[int] = []

    def _factory() -> Any:
        loads.append(1)
        return model

    return _factory, loads


def make_local_embedder(
    model_name: str = E5_MODEL,
    *,
    fake: FakeSentenceTransformer | None = None,
    **kwargs: Any,
) -> tuple[SentenceTransformersEmbeddingProvider, FakeSentenceTransformer, list[int]]:
    fake = fake or FakeSentenceTransformer()
    factory, loads = recording_factory(fake)
    provider = SentenceTransformersEmbeddingProvider(
        model_name,
        model_factory=factory,
        cache_dir=kwargs.pop("cache_dir", None),
        **kwargs,
    )
    return provider, fake, loads


@contextlib.asynccontextmanager
async def loop_ticker(interval: float = 0.005):
    """Increment a counter on the event loop so blocking is detectable."""
    state = {"ticks": 0}

    async def _ticker() -> None:
        while True:
            await asyncio.sleep(interval)
            state["ticks"] += 1

    task = asyncio.create_task(_ticker())
    try:
        yield state
    finally:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helper units
# ═══════════════════════════════════════════════════════════════════════════════

class TestLocalModelHelpers:
    def test_detect_device_honours_explicit_preference(self) -> None:
        assert detect_device("cpu") == "cpu"
        assert detect_device("cuda:1") == "cuda:1"
        assert detect_device("CUDA") == "cuda"
        assert detect_device(None) in {"cpu", "cuda", "mps"}
        assert detect_device("auto") in {"cpu", "cuda", "mps"}

    def test_l2_normalize(self) -> None:
        assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
        assert l2_normalize([1.0, 0.0]) == [1.0, 0.0]
        with pytest.raises(ValueError):
            l2_normalize([0.0, 0.0])
        with pytest.raises(ValueError):
            l2_normalize([float("nan"), 1.0])
        with pytest.raises(ValueError):
            l2_normalize([])

    def test_to_float_list_accepts_numpy_like_values(self) -> None:
        class ArrayLike:
            def tolist(self) -> list[list[float]]:
                return [[0.25, 0.5]]

        assert to_float_list(ArrayLike()) == [0.25, 0.5]
        assert to_float_list([1, 2, 3]) == [1.0, 2.0, 3.0]
        assert to_float_list(7) == [7.0]
        with pytest.raises(TypeError):
            to_float_list([[1.0], [2.0]])

    def test_provider_modules_do_not_import_torch_at_import_time(self) -> None:
        """The packaged backend must boot without touching PyTorch weights."""
        script = (
            "import sys;"
            "import app.providers.embeddings.sentence_transformers;"
            "import app.providers.rerankers.cross_encoder;"
            "print(int('torch' in sys.modules), int('sentence_transformers' in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "0 0", completed.stderr


# ═══════════════════════════════════════════════════════════════════════════════
# SentenceTransformers embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class TestSentenceTransformersEmbeddings:
    def test_identity_and_data_dir_binding(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "QUANTIVA_DATA_DIR", tmp_path)

        provider = SentenceTransformersEmbeddingProvider(E5_MODEL)
        assert provider.provider_name() == "sentence_transformers"
        assert provider.model_name() == E5_MODEL
        assert provider.cache_dir() == tmp_path / "models" / "embeddings"
        assert provider.cache_dir().is_dir()

        custom = SentenceTransformersEmbeddingProvider(E5_MODEL, cache_dir=tmp_path / "custom")
        assert custom.cache_dir() == tmp_path / "custom"

    def test_default_model_comes_from_settings(self, tmp_path: Path) -> None:
        provider = SentenceTransformersEmbeddingProvider(cache_dir=tmp_path)
        assert provider.model_name() == settings.LOCAL_EMBEDDING_MODEL

    @pytest.mark.asyncio
    async def test_model_loads_lazily_and_only_once(self, tmp_path: Path) -> None:
        provider, fake, loads = make_local_embedder(cache_dir=tmp_path)

        assert provider.is_loaded() is False
        assert provider.get_dimensions() == 0
        assert loads == []

        await provider.embed_documents(["alpha"])
        assert loads == [1]
        assert provider.is_loaded() is True
        assert provider.get_dimensions() == fake.get_sentence_embedding_dimension()

        await provider.embed_documents(["beta"])
        assert loads == [1]

    @pytest.mark.asyncio
    async def test_warmup_loads_model_and_reports_dimensions(self, tmp_path: Path) -> None:
        provider, _fake, loads = make_local_embedder(cache_dir=tmp_path)
        assert await provider.warmup() == 4
        assert loads == [1]
        assert await provider.warmup() == 4
        assert loads == [1]

    @pytest.mark.asyncio
    async def test_e5_prefixes_for_documents_and_queries(self, tmp_path: Path) -> None:
        provider, fake, _ = make_local_embedder(E5_MODEL, cache_dir=tmp_path)

        await provider.embed_documents(
            ["alpha", "passage: beta", "query: gamma"]
        )
        assert fake.calls[0] == ["passage: alpha", "passage: beta", "query: gamma"]

        await provider.embed_query("delta")
        assert fake.calls[1] == ["query: delta"]

        await provider.embed_query("query: already-prefixed")
        assert fake.calls[2] == ["query: already-prefixed"]

    @pytest.mark.asyncio
    async def test_non_e5_models_get_no_prefixes(self, tmp_path: Path) -> None:
        provider, fake, _ = make_local_embedder(PLAIN_MODEL, cache_dir=tmp_path)
        await provider.embed_documents(["alpha", "beta"])
        await provider.embed_query("gamma")
        assert fake.calls == [["alpha", "beta"], ["gamma"]]

    @pytest.mark.asyncio
    async def test_e5_prefixes_can_be_disabled(self, tmp_path: Path) -> None:
        provider, fake, _ = make_local_embedder(
            E5_MODEL, cache_dir=tmp_path, use_e5_prefixes=False
        )
        assert provider.uses_e5_prefixes() is False
        await provider.embed_documents(["alpha"])
        assert fake.calls == [["alpha"]]

    @pytest.mark.asyncio
    async def test_vectors_are_unit_length_python_floats(self, tmp_path: Path) -> None:
        provider, fake, _ = make_local_embedder(cache_dir=tmp_path)
        vectors = await provider.embed_documents(["hello", "world"])

        assert len(vectors) == 2
        for vector in vectors:
            assert isinstance(vector, list)
            assert all(type(value) is float for value in vector)
            assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=1e-9)

        # The model is asked to normalise; we also normalise defensively.
        assert fake.kwargs[0]["normalize_embeddings"] is True
        assert fake.kwargs[0]["convert_to_numpy"] is True

    @pytest.mark.asyncio
    async def test_batching_splits_encode_calls(self, tmp_path: Path) -> None:
        provider, fake, _ = make_local_embedder(cache_dir=tmp_path)
        vectors = await provider.embed_documents(
            ["one", "two", "three", "four", "five"], batch_size=2
        )

        assert [len(call) for call in fake.calls] == [2, 2, 1]
        assert [kwargs["batch_size"] for kwargs in fake.kwargs] == [2, 2, 2]
        assert len(vectors) == 5
        assert provider.get_dimensions() == 4

    @pytest.mark.asyncio
    async def test_dimensions_detected_from_model(self, tmp_path: Path) -> None:
        fake = FakeSentenceTransformer(dimensions=8)
        provider, _fake, _ = make_local_embedder(cache_dir=tmp_path, fake=fake)
        assert provider.get_dimensions() == 0
        vectors = await provider.embed_documents(["alpha"])
        assert provider.get_dimensions() == 8
        assert len(vectors[0]) == 8

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises_before_returning(self, tmp_path: Path) -> None:
        provider, _fake, _ = make_local_embedder(cache_dir=tmp_path, dimensions=3)
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            await provider.embed_documents(["alpha"])
        assert exc_info.value.expected == 3
        assert exc_info.value.actual == 4

    @pytest.mark.asyncio
    async def test_encoding_runs_off_the_event_loop(self, tmp_path: Path) -> None:
        fake = FakeSentenceTransformer(delay=0.15)
        provider, _fake, _ = make_local_embedder(cache_dir=tmp_path, fake=fake)

        async with loop_ticker() as state:
            await provider.embed_documents(["alpha"])

        assert fake.thread_idents[0] != threading.get_ident()
        # The loop kept running while the worker thread was encoding.
        assert state["ticks"] >= 3

    @pytest.mark.asyncio
    async def test_input_validation(self, tmp_path: Path) -> None:
        provider, _fake, _ = make_local_embedder(cache_dir=tmp_path)

        with pytest.raises(ValueError):
            await provider.embed_documents([])
        with pytest.raises(ValueError):
            await provider.embed_documents(["ok"], batch_size=0)
        with pytest.raises(ValueError):
            await provider.embed_documents(["ok", "   "])
        with pytest.raises(ValueError):
            await provider.embed_query("")
        with pytest.raises(ValueError):
            await provider.embed_query("   ")

    @pytest.mark.asyncio
    async def test_test_embedding_success(self, tmp_path: Path) -> None:
        provider, _fake, _ = make_local_embedder(cache_dir=tmp_path)
        result = await provider.test_embedding()
        assert result.success is True
        assert result.provider == "sentence_transformers"
        assert result.detected_dimensions == 4
        assert result.error_code is None

    @pytest.mark.asyncio
    async def test_test_embedding_reports_load_failure(self, tmp_path: Path) -> None:
        def _boom() -> Any:
            raise ProviderError(
                "model files are missing",
                provider="sentence_transformers",
                model=E5_MODEL,
            )

        provider = SentenceTransformersEmbeddingProvider(
            E5_MODEL, cache_dir=tmp_path, model_factory=_boom
        )
        result = await provider.test_embedding()
        assert result.success is False
        assert result.error_code == "ProviderError"
        assert result.detected_dimensions is None

    @pytest.mark.asyncio
    async def test_missing_sentence_transformers_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        provider = SentenceTransformersEmbeddingProvider(E5_MODEL, cache_dir=tmp_path)

        with pytest.raises(ProviderError) as exc_info:
            await provider.warmup()

        assert "sentence-transformers" in str(exc_info.value)
        assert exc_info.value.retriable is False


# ═══════════════════════════════════════════════════════════════════════════════
# Local cross-encoder reranker
# ═══════════════════════════════════════════════════════════════════════════════

def make_documents() -> list[RerankDocument]:
    return [
        RerankDocument(
            id="doc-a",
            content="Oranges are citrus fruits.",
            metadata={"page": 1},
            initial_score=0.9,
        ),
        RerankDocument(
            id="doc-b",
            content="Apples grow on trees in orchards.",
            metadata={"page": 2},
            initial_score=0.5,
        ),
        RerankDocument(
            id="doc-c",
            content="A guide to database migrations.",
            metadata={"page": 3},
            initial_score=0.2,
        ),
    ]


class TestLocalCrossEncoderReranker:
    def test_identity_and_cache_binding(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "QUANTIVA_DATA_DIR", tmp_path)
        reranker = LocalCrossEncoderReranker()
        assert reranker.provider_name() == "cross_encoder"
        assert reranker.model_name() == RERANK_MODEL
        assert reranker.cache_dir() == tmp_path / "models" / "rerankers"
        assert reranker.cache_dir().is_dir()

    @pytest.mark.asyncio
    async def test_lazy_load_and_top_k_ordering(self, tmp_path: Path) -> None:
        fake = FakeCrossEncoder(scores=[0.12, 0.87, 0.44])
        factory, loads = recording_factory(fake)
        reranker = LocalCrossEncoderReranker(
            cache_dir=tmp_path, model_factory=factory
        )

        assert reranker.is_loaded() is False
        assert loads == []

        results = await reranker.rerank("apple cultivation", make_documents(), top_k=2)

        assert loads == [1]
        assert [result.id for result in results] == ["doc-b", "doc-c"]
        assert [result.rerank_score for result in results] == [0.87, 0.44]
        assert [result.rank for result in results] == [0, 1]
        assert results[0].metadata == {"page": 2}
        assert results[0].initial_score == 0.5
        assert results[0].content == "Apples grow on trees in orchards."

        await reranker.rerank("apple cultivation", make_documents(), top_k=1)
        assert loads == [1]

    @pytest.mark.asyncio
    async def test_pairs_are_query_document_tuples(self, tmp_path: Path) -> None:
        fake = FakeCrossEncoder(scores=[0.1, 0.2, 0.3])
        reranker = LocalCrossEncoderReranker(
            cache_dir=tmp_path, model_factory=lambda: fake
        )
        await reranker.rerank("fruit", make_documents(), top_k=3)

        assert fake.pairs[0] == [
            ("fruit", "Oranges are citrus fruits."),
            ("fruit", "Apples grow on trees in orchards."),
            ("fruit", "A guide to database migrations."),
        ]

    @pytest.mark.asyncio
    async def test_rerank_runs_off_the_event_loop(self, tmp_path: Path) -> None:
        fake = FakeCrossEncoder(scores=[0.1, 0.9, 0.4], delay=0.15)
        reranker = LocalCrossEncoderReranker(
            cache_dir=tmp_path, model_factory=lambda: fake
        )

        async with loop_ticker() as state:
            results = await reranker.rerank("apples", make_documents(), top_k=2)

        assert results[0].id == "doc-b"
        assert fake.thread_idents[0] != threading.get_ident()
        assert state["ticks"] >= 3

    @pytest.mark.asyncio
    async def test_empty_documents_and_validation(self, tmp_path: Path) -> None:
        fake = FakeCrossEncoder(scores=[])
        reranker = LocalCrossEncoderReranker(
            cache_dir=tmp_path, model_factory=lambda: fake
        )

        assert await reranker.rerank("anything", [], top_k=5) == []
        assert fake.pairs == []  # no model work for an empty candidate set

        with pytest.raises(ValueError):
            await reranker.rerank("", make_documents())
        with pytest.raises(ValueError):
            await reranker.rerank("q", make_documents(), top_k=0)

    @pytest.mark.asyncio
    async def test_score_count_mismatch_is_normalised(self, tmp_path: Path) -> None:
        fake = FakeCrossEncoder(scores=[0.5])
        reranker = LocalCrossEncoderReranker(
            cache_dir=tmp_path, model_factory=lambda: fake
        )
        with pytest.raises(ProviderError) as exc_info:
            await reranker.rerank("apples", make_documents(), top_k=2)
        assert exc_info.value.retriable is True
        assert "3 candidates" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_multi_label_scores_are_flattened(self, tmp_path: Path) -> None:
        class MultiLabel:
            def predict(self, pairs, **_kwargs):
                return [[0.1], [0.9], [0.5]]

        reranker = LocalCrossEncoderReranker(
            cache_dir=tmp_path, model_factory=MultiLabel
        )
        results = await reranker.rerank("apples", make_documents(), top_k=3)
        assert [result.id for result in results] == ["doc-b", "doc-c", "doc-a"]

    @pytest.mark.asyncio
    async def test_test_reranker_probe(self, tmp_path: Path) -> None:
        fake = FakeCrossEncoder(scores=[0.9, 0.1])
        reranker = LocalCrossEncoderReranker(
            cache_dir=tmp_path, model_factory=lambda: fake
        )
        result = await reranker.test_reranker()
        assert result.success is True
        assert result.provider == "cross_encoder"
        assert result.model == RERANK_MODEL

    @pytest.mark.asyncio
    async def test_test_reranker_reports_failure(self, tmp_path: Path) -> None:
        def _boom() -> Any:
            raise ProviderError(
                "weights unavailable",
                provider="cross_encoder",
                model=RERANK_MODEL,
            )

        reranker = LocalCrossEncoderReranker(cache_dir=tmp_path, model_factory=_boom)
        result = await reranker.test_reranker()
        assert result.success is False
        assert result.error_code == "ProviderError"

    @pytest.mark.asyncio
    async def test_missing_sentence_transformers_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        reranker = LocalCrossEncoderReranker(cache_dir=tmp_path)
        with pytest.raises(ProviderError) as exc_info:
            await reranker.warmup()
        assert "sentence-transformers" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════════
# NoOp reranker
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoOpReranker:
    def test_identity(self) -> None:
        reranker = NoOpReranker()
        assert isinstance(reranker, RerankerProvider)
        assert reranker.provider_name() == "none"
        assert reranker.model_name() == "disabled"

    @pytest.mark.asyncio
    async def test_preserves_order_and_scores(self) -> None:
        reranker = NoOpReranker()
        documents = make_documents()

        results = await reranker.rerank("ignored", documents, top_k=2)

        assert [result.id for result in results] == ["doc-a", "doc-b"]
        assert [result.rank for result in results] == [0, 1]
        assert [result.rerank_score for result in results] == [0.9, 0.5]
        assert results[0].initial_score == 0.9
        assert results[0].metadata == {"page": 1}

    @pytest.mark.asyncio
    async def test_empty_and_invalid_top_k(self) -> None:
        reranker = NoOpReranker()
        assert await reranker.rerank("q", [], top_k=5) == []
        with pytest.raises(ValueError):
            await reranker.rerank("q", make_documents(), top_k=-1)

    @pytest.mark.asyncio
    async def test_probe_always_succeeds(self) -> None:
        result = await NoOpReranker().test_reranker()
        assert result.success is True
        assert result.provider == "none"


# ═══════════════════════════════════════════════════════════════════════════════
# Remote JSON reranker
# ═══════════════════════════════════════════════════════════════════════════════

class Recorder:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.handler = handler
        self.requests: list[tuple[str, str, dict[str, Any], dict[str, str]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content) if request.content else {}
        except json.JSONDecodeError:
            body = {}
        headers = {key.lower(): value for key, value in request.headers.items()}
        self.requests.append((request.method, request.url.path, body, headers))
        return self.handler(request)


def make_api_reranker(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> tuple[APIRerankerProvider, Recorder]:
    recorder = Recorder(handler)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(recorder),
        base_url="https://reranker.test",
    )
    provider = APIRerankerProvider(
        base_url="https://reranker.test/v1",
        model="rerank-test",
        api_key="rerank-key",
        client=client,
        **kwargs,
    )
    return provider, recorder


def json_handler(payload: Any, status_code: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return _handler


class TestAPIReranker:
    def test_identity_and_style_validation(self) -> None:
        provider = APIRerankerProvider(
            base_url="https://api.cohere.com/v1",
            model="rerank-v3.5",
            style="cohere",
        )
        assert provider.provider_name() == "cohere"
        assert provider.model_name() == "rerank-v3.5"
        assert provider.base_url() == "https://api.cohere.com/v1"

        with pytest.raises(ValueError):
            APIRerankerProvider(style="bedrock")

    def test_presets(self) -> None:
        cohere = APIRerankerProvider.for_cohere(api_key="c-key")
        assert cohere.provider_name() == "cohere"
        assert cohere.base_url() == "https://api.cohere.com/v1"

        jina = APIRerankerProvider.for_jina(api_key="j-key")
        assert jina.provider_name() == "jina"
        assert jina.base_url() == "https://api.jina.ai/v1"

        compatible = APIRerankerProvider.for_openai_compatible(
            "http://127.0.0.1:8080/v1", model="local-rerank"
        )
        assert compatible.provider_name() == "openai_compatible"
        assert compatible.base_url() == "http://127.0.0.1:8080/v1"

    @pytest.mark.asyncio
    async def test_cohere_payload_and_result_ordering(self) -> None:
        payload = {
            "results": [
                {"index": 2, "relevance_score": 0.33},
                {"index": 0, "relevance_score": 0.91},
                {"index": 1, "relevance_score": 0.55},
            ]
        }
        provider, recorder = make_api_reranker(
            json_handler(payload), style="cohere", provider_label="cohere"
        )

        results = await provider.rerank("apples", make_documents(), top_k=2)

        assert [result.id for result in results] == ["doc-a", "doc-b"]
        assert [result.rerank_score for result in results] == [0.91, 0.55]
        assert [result.rank for result in results] == [0, 1]
        assert results[0].initial_score == 0.9

        method, path, body, headers = recorder.requests[0]
        assert method == "POST"
        assert path == "/v1/rerank"
        assert body["model"] == "rerank-test"
        assert body["query"] == "apples"
        assert body["documents"] == [document.content for document in make_documents()]
        assert body["top_n"] == 2
        assert body["return_documents"] is False
        assert headers["authorization"] == "Bearer rerank-key"

    @pytest.mark.asyncio
    async def test_tei_style_top_level_list_and_texts_field(self) -> None:
        response = [
            {"index": 1, "score": 0.75},
            {"index": 0, "score": 0.25},
        ]
        provider, recorder = make_api_reranker(
            json_handler(response), document_field="texts"
        )

        results = await provider.rerank("apples", make_documents(), top_k=5)

        assert [result.id for result in results] == ["doc-b", "doc-a"]
        assert recorder.requests[0][2]["texts"] == [
            document.content for document in make_documents()
        ]

    @pytest.mark.asyncio
    async def test_document_text_fallback_when_index_missing(self) -> None:
        response = {
            "results": [
                {
                    "document": {
                        "text": "Apples grow on trees in orchards."
                    },
                    "relevance_score": 0.8,
                }
            ]
        }
        provider, _ = make_api_reranker(json_handler(response))
        results = await provider.rerank("apples", make_documents(), top_k=3)
        assert [result.id for result in results] == ["doc-b"]

    @pytest.mark.asyncio
    async def test_http_error_normalisation(self) -> None:
        cases = [
            (401, {"message": "invalid api key"}, ProviderAuthenticationError),
            (429, {"message": "rate limit"}, ProviderRateLimitError),
            (500, {"message": "overloaded"}, ProviderError),
        ]
        for status, body, expected in cases:
            provider, _ = make_api_reranker(json_handler(body, status_code=status))
            with pytest.raises(expected):
                await provider.rerank("q", make_documents(), top_k=2)

    @pytest.mark.asyncio
    async def test_connection_error_is_typed(self) -> None:
        def _refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        provider, _ = make_api_reranker(
            _refuse, provider_label="lm_studio", is_local=True
        )
        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.rerank("q", make_documents(), top_k=2)

        assert "lm_studio" in str(exc_info.value)
        assert exc_info.value.retriable is True

    @pytest.mark.asyncio
    async def test_requires_api_key(self) -> None:
        recorder = Recorder(json_handler({"results": []}))
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(recorder),
            base_url="https://api.cohere.com/v1",
        )
        provider = APIRerankerProvider.for_cohere(api_key="", client=client)

        with pytest.raises(ProviderAuthenticationError):
            await provider.rerank("q", make_documents(), top_k=2)
        assert recorder.requests == []

    @pytest.mark.asyncio
    async def test_input_validation(self) -> None:
        provider, recorder = make_api_reranker(json_handler({"results": []}))

        assert await provider.rerank("q", [], top_k=3) == []
        with pytest.raises(ValueError):
            await provider.rerank("  ", make_documents(), top_k=3)
        with pytest.raises(ValueError):
            await provider.rerank("q", make_documents(), top_k=0)
        assert recorder.requests == []

    @pytest.mark.asyncio
    async def test_malformed_response_is_normalised(self) -> None:
        provider, _ = make_api_reranker(json_handler({"results": [{"score": 0.4}]}))
        with pytest.raises(ProviderError) as exc_info:
            await provider.rerank("q", make_documents(), top_k=2)
        assert exc_info.value.retriable is True

    @pytest.mark.asyncio
    async def test_probe_success_and_failure(self) -> None:
        provider, _ = make_api_reranker(
            json_handler({"results": [{"index": 0, "relevance_score": 0.9}]})
        )
        result = await provider.test_reranker()
        assert result.success is True
        assert result.provider == "openai_compatible"

        def _refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        failing, _ = make_api_reranker(_refuse)
        failed = await failing.test_reranker()
        assert failed.success is False
        assert failed.error_code == "ProviderConnectionError"
