"""
app/providers/embeddings/sentence_transformers.py
──────────────────────────────────────────────────────────────────────────────
In-process SentenceTransformers embedding provider.

Design constraints for the packaged desktop application:

* **No weights at import time.** Neither :mod:`sentence_transformers` nor
  :mod:`torch` is imported until the model is actually needed (first embedding
  call or explicit :meth:`warmup`), so application startup stays fast and the
  backend can boot before any model has been downloaded.
* **Data-directory binding.** Weights are downloaded into
  ``<QUANTIVA_DATA_DIR>/models/embeddings`` via ``cache_folder``; a fully
  downloaded model keeps working with Hugging Face offline mode enabled.
* **Event-loop safety.** CPU/GPU encoding runs in ``asyncio.to_thread``; the
  FastAPI event loop is never blocked during document indexing.

E5-family models (``intfloat/multilingual-e5-*``) require task prefixes, which
are applied automatically: ``"passage: "`` for documents and ``"query: "`` for
queries.

Conforms to :class:`~app.providers.interfaces.embedding.EmbeddingProvider`.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import settings
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.exceptions import ProviderError
from app.providers.interfaces.types import ConnectionTestResult
from app.providers.local_models import (
    EMBEDDINGS_SUBDIR,
    configure_offline_mode,
    detect_device,
    l2_normalize,
    model_cache_dir,
    to_float_matrix,
)

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "
_EXISTING_PREFIXES = ("passage:", "query:")

_UNAVAILABLE_HINT = (
    "Local embeddings require the 'sentence-transformers' package. "
    "Install it with 'pip install sentence-transformers' and restart Quantiva."
)
_LOAD_FAILED_HINT = (
    "The local embedding model could not be loaded. Check that the model name "
    "is correct and that Quantiva can reach it, or download it ahead of time "
    "in Settings > Models."
)


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Local, offline-capable embedding provider backed by SentenceTransformers."""

    def __init__(
        self,
        model: str | None = None,
        *,
        dimensions: int | None = None,
        device: str | None = None,
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        local_files_only: bool | None = None,
        offline: bool | None = None,
        trust_remote_code: bool = False,
        show_progress_bar: bool = False,
        max_seq_length: int | None = None,
        use_e5_prefixes: bool | None = None,
        model_instance: Any | None = None,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._model_name = (
            model
            or getattr(settings, "LOCAL_EMBEDDING_MODEL", None)
            or DEFAULT_EMBEDDING_MODEL
        )
        self._dimensions = int(dimensions) if dimensions else 0
        self._device_preference = (
            device
            if device is not None
            else getattr(settings, "MODEL_DEVICE", "auto")
        )
        self._cache_dir = model_cache_dir(EMBEDDINGS_SUBDIR, cache_dir)
        self._batch_size = int(batch_size)
        self._normalize_embeddings = bool(normalize_embeddings)
        self._local_files_only = (
            bool(local_files_only) if local_files_only is not None else False
        )
        self._offline = (
            bool(offline)
            if offline is not None
            else bool(getattr(settings, "HF_OFFLINE", False))
        )
        self._trust_remote_code = bool(trust_remote_code)
        self._show_progress_bar = bool(show_progress_bar)
        self._max_seq_length = max_seq_length
        self._use_e5_prefixes = use_e5_prefixes

        # Injection seams for tests / pre-warmed models; never required in prod.
        self._model = model_instance
        self._model_factory = model_factory

        self._load_lock = threading.Lock()
        self._device = ""

    # ── Identity & dimensions ─────────────────────────────────────────────────

    def provider_name(self) -> str:
        return "sentence_transformers"

    def model_name(self) -> str:
        return self._model_name

    def cache_dir(self) -> Path:
        """Directory that holds this model's downloaded weights."""
        return self._cache_dir

    def device(self) -> str:
        """Resolved device (``""`` until the model is loaded)."""
        return self._device

    def is_loaded(self) -> bool:
        """True once the model object exists in memory."""
        return self._model is not None

    def get_dimensions(self) -> int:
        """Return the vector length, or ``0`` until the model is loaded.

        Dimensions are discovered from the loaded model (or the first encoded
        batch).  Call :meth:`warmup` to resolve them without embedding text.
        """
        return self._dimensions

    # ── Model lifecycle (lazy, thread-safe) ───────────────────────────────────

    def _ensure_model_sync(self) -> Any:
        """Load the model exactly once, serialised across worker threads."""
        cached = self._model
        if cached is not None:
            return cached

        with self._load_lock:
            if self._model is not None:  # another thread loaded it first
                return self._model

            model = (
                self._model_factory()
                if self._model_factory is not None
                else self._create_model()
            )
            self._model = model
            self._detect_dimensions(model)
            return model

    def _create_model(self) -> Any:
        """Heavy path: import torch-backed packages and load weights from disk."""
        configure_offline_mode(self._offline)

        try:
            from sentence_transformers import (  # heavy, lazy by design
                SentenceTransformer,
            )
        except Exception as exc:
            raise self._error(_UNAVAILABLE_HINT, retriable=False) from exc

        self._device = detect_device(self._device_preference)
        kwargs: dict[str, Any] = {
            "device": self._device,
            "cache_folder": str(self._cache_dir),
            "trust_remote_code": self._trust_remote_code,
        }
        if self._local_files_only or self._offline:
            kwargs["local_files_only"] = True

        try:
            model = SentenceTransformer(self._model_name, **kwargs)
        except TypeError:
            # Older sentence-transformers releases lack ``local_files_only``.
            kwargs.pop("local_files_only", None)
            try:
                model = SentenceTransformer(self._model_name, **kwargs)
            except Exception as exc:
                # Normalise every load failure into a typed provider error.
                raise self._error(_LOAD_FAILED_HINT, retriable=False) from exc
        except Exception as exc:
            raise self._error(_LOAD_FAILED_HINT, retriable=False) from exc

        if self._max_seq_length and hasattr(model, "max_seq_length"):
            model.max_seq_length = int(self._max_seq_length)
        return model

    def _detect_dimensions(self, model: Any) -> None:
        """Populate ``get_dimensions()`` from the model without encoding text."""
        if self._dimensions:
            return
        getter = getattr(model, "get_sentence_embedding_dimension", None)
        if not callable(getter):
            return
        try:
            detected = getter()
        except Exception:  # noqa: BLE001 - detection is best-effort
            return
        if detected:
            self._dimensions = int(detected)

    async def warmup(self) -> int:
        """Load the model (downloading if necessary) and return its dimensions.

        Intended for an explicit "download / prepare model" action in Model
        Studio.  Safe to call multiple times; subsequent calls are no-ops.
        """
        await asyncio.to_thread(self._ensure_model_sync)
        return self._dimensions

    # ── E5 prefix conventions ─────────────────────────────────────────────────

    def uses_e5_prefixes(self) -> bool:
        """Whether E5 ``passage:``/``query:`` prefixes are applied."""
        if self._use_e5_prefixes is not None:
            return self._use_e5_prefixes
        return "e5" in self._model_name.lower()

    @staticmethod
    def _has_task_prefix(text: str) -> bool:
        lowered = text.lstrip().lower()
        return lowered.startswith(_EXISTING_PREFIXES)

    def _prepare(self, text: str, prefix: str) -> str:
        if not self.uses_e5_prefixes() or self._has_task_prefix(text):
            return text
        return f"{prefix}{text}"

    # ── Encoding (runs in a worker thread) ────────────────────────────────────

    def _encode_sync(self, texts: list[str], batch_size: int) -> list[list[float]]:
        model = self._ensure_model_sync()
        try:
            raw = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=self._show_progress_bar,
            )
        except Exception as exc:
            # Inference failures are transient (OOM, bad tensor, device loss).
            raise self._error(
                f"Local embedding model '{self._model_name}' failed to encode "
                f"text: {exc}",
                retriable=True,
            ) from exc

        try:
            matrix = to_float_matrix(raw)
        except (TypeError, ValueError) as exc:
            raise self._error(
                "Local embedding model returned an unexpected output shape.",
                retriable=True,
            ) from exc

        if len(matrix) != len(texts):
            raise self._error(
                f"Local embedding model returned {len(matrix)} vectors for a "
                f"batch of {len(texts)}.",
                retriable=True,
            )

        vectors: list[list[float]] = []
        for vector in matrix:
            try:
                normalized = l2_normalize(vector)
            except ValueError as exc:
                raise self._error(
                    "Local embedding model returned an unusable vector "
                    "(empty, zero-magnitude, or non-finite).",
                    retriable=False,
                ) from exc
            if self._dimensions == 0:
                self._dimensions = len(normalized)
            else:
                # Raises EmbeddingDimensionMismatchError before vector storage.
                self.validate_dimensions(normalized)
            vectors.append(normalized)
        return vectors

    # ── Embedding API ─────────────────────────────────────────────────────────

    async def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            raise ValueError("texts list must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string")

        prepared = [self._prepare(text, _PASSAGE_PREFIX) for text in texts]

        vectors: list[list[float]] = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            # Offload the CPU/GPU work so the event loop stays responsive.
            vectors.extend(await asyncio.to_thread(self._encode_sync, batch, batch_size))
        return vectors

    async def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        prepared = self._prepare(query, _QUERY_PREFIX)
        vectors = await asyncio.to_thread(
            self._encode_sync, [prepared], max(1, self._batch_size)
        )
        return vectors[0]

    # ── Connectivity probe ────────────────────────────────────────────────────

    def _error(self, message: str, *, retriable: bool) -> ProviderError:
        error = ProviderError(
            message,
            provider=self.provider_name(),
            model=self._model_name,
            retriable=retriable,
        )
        error.user_message = message
        return error

    async def test_embedding(self) -> ConnectionTestResult:
        start_time = time.perf_counter()
        try:
            probe_vector = await self.embed_query("Quantiva connection test")
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=True,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                detected_dimensions=len(probe_vector),
            )
        except ProviderError as err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                detected_dimensions=None,
                error_code=type(err).__name__,
                error_message=err.user_message,
            )
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
                detected_dimensions=None,
                error_code="UnknownError",
                error_message=str(exc),
            )
