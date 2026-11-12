"""
app/providers/rerankers/cross_encoder.py
──────────────────────────────────────────────────────────────────────────────
In-process CrossEncoder reranker and the passthrough reranker used when a user
disables reranking in Retrieval Settings.

Like the local embedding provider, the model is loaded lazily (never at import
time), weights are cached under ``<QUANTIVA_DATA_DIR>/models/rerankers``, and
``predict`` runs in ``asyncio.to_thread`` so the event loop is never blocked.

Both classes conform to
:class:`~app.providers.interfaces.reranker.RerankerProvider`.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import settings
from app.providers.interfaces.exceptions import ProviderError
from app.providers.interfaces.reranker import (
    RerankDocument,
    RerankerProvider,
    RerankResult,
)
from app.providers.interfaces.types import ConnectionTestResult
from app.providers.local_models import (
    RERANKERS_SUBDIR,
    configure_offline_mode,
    detect_device,
    model_cache_dir,
    to_float_scores,
)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_UNAVAILABLE_HINT = (
    "Local reranking requires the 'sentence-transformers' package. "
    "Install it with 'pip install sentence-transformers' and restart Quantiva."
)
_LOAD_FAILED_HINT = (
    "The local reranker model could not be loaded. Check that the model name "
    "is correct and that Quantiva can reach it, or download it ahead of time "
    "in Settings > Models."
)


class LocalCrossEncoderReranker(RerankerProvider):
    """Cross-encoder reranker running fully in-process on CPU/GPU."""

    def __init__(
        self,
        model: str | None = None,
        *,
        device: str | None = None,
        cache_dir: str | Path | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        local_files_only: bool | None = None,
        offline: bool | None = None,
        trust_remote_code: bool = False,
        show_progress_bar: bool = False,
        model_instance: Any | None = None,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._model_name = (
            model
            or getattr(settings, "LOCAL_RERANKER_MODEL", None)
            or DEFAULT_RERANKER_MODEL
        )
        self._device_preference = (
            device
            if device is not None
            else getattr(settings, "MODEL_DEVICE", "auto")
        )
        self._cache_dir = model_cache_dir(RERANKERS_SUBDIR, cache_dir)
        self._max_length = int(max_length)
        self._batch_size = int(batch_size)
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

        self._model = model_instance
        self._model_factory = model_factory

        # Separate locks: nested load+predict on one non-reentrant lock would
        # deadlock. Inference is serialised because the CPU model is shared.
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._device = ""

    # ── Identity ──────────────────────────────────────────────────────────────

    def provider_name(self) -> str:
        return "cross_encoder"

    def model_name(self) -> str:
        return self._model_name

    def cache_dir(self) -> Path:
        """Directory that holds this model's downloaded weights."""
        return self._cache_dir

    def device(self) -> str:
        """Resolved device (``""`` until the model is loaded)."""
        return self._device

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Model lifecycle (lazy, thread-safe) ───────────────────────────────────

    def _ensure_model_sync(self) -> Any:
        cached = self._model
        if cached is not None:
            return cached

        with self._load_lock:
            if self._model is not None:
                return self._model

            model = (
                self._model_factory()
                if self._model_factory is not None
                else self._create_model()
            )
            self._model = model
            return model

    def _create_model(self) -> Any:
        configure_offline_mode(self._offline)

        try:
            from sentence_transformers import (  # heavy, lazy by design
                CrossEncoder,
            )
        except Exception as exc:
            raise self._error(_UNAVAILABLE_HINT, retriable=False) from exc

        self._device = detect_device(self._device_preference)
        kwargs: dict[str, Any] = {
            "device": self._device,
            "cache_folder": str(self._cache_dir),
            "trust_remote_code": self._trust_remote_code,
            "max_length": self._max_length,
        }
        if self._local_files_only or self._offline:
            kwargs["local_files_only"] = True

        try:
            return CrossEncoder(self._model_name, **kwargs)
        except TypeError:
            # Older releases may not accept every keyword we pass.
            kwargs.pop("local_files_only", None)
            try:
                return CrossEncoder(self._model_name, **kwargs)
            except Exception as exc:
                # Normalise every load failure into a typed provider error.
                raise self._error(_LOAD_FAILED_HINT, retriable=False) from exc
        except Exception as exc:
            raise self._error(_LOAD_FAILED_HINT, retriable=False) from exc

    async def warmup(self) -> None:
        """Load the model (downloading if necessary) ahead of first retrieval."""
        await asyncio.to_thread(self._ensure_model_sync)

    # ── Scoring (runs in a worker thread) ─────────────────────────────────────

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._ensure_model_sync()
        with self._predict_lock:
            try:
                raw = model.predict(
                    pairs,
                    batch_size=self._batch_size,
                    show_progress_bar=self._show_progress_bar,
                    convert_to_numpy=True,
                )
            except TypeError:
                # Minimal/test doubles and older releases accept pairs only.
                raw = model.predict(pairs)
            except Exception as exc:
                # Scoring failures are usually transient (OOM, device loss).
                raise self._error(
                    f"Local reranker '{self._model_name}' failed to score "
                    f"candidates: {exc}",
                    retriable=True,
                ) from exc

        try:
            return to_float_scores(raw)
        except (TypeError, ValueError) as exc:
            raise self._error(
                "Local reranker returned an unexpected score shape.",
                retriable=True,
            ) from exc

    # ── Reranking API ─────────────────────────────────────────────────────────

    async def rerank(
        self,
        query: str,
        documents: list[RerankDocument],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not documents:
            return []

        pairs = [(query, document.content) for document in documents]
        # Offload import + load + inference so the event loop keeps serving.
        scores = await asyncio.to_thread(self._predict_sync, pairs)

        if len(scores) != len(documents):
            raise self._error(
                f"Local reranker returned {len(scores)} scores for "
                f"{len(documents)} candidates.",
                retriable=True,
            )

        ranked = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RerankResult(
                id=document.id,
                content=document.content,
                metadata=dict(document.metadata),
                rerank_score=float(score),
                initial_score=document.initial_score,
                rank=rank,
            )
            for rank, (document, score) in enumerate(ranked[:top_k])
        ]

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

    async def test_reranker(self) -> ConnectionTestResult:
        start_time = time.perf_counter()
        try:
            results = await self.rerank(
                "Quantiva connection test",
                [
                    RerankDocument(
                        id="probe-1", content="Quantiva connection test"
                    ),
                    RerankDocument(
                        id="probe-2", content="Unrelated candidate content"
                    ),
                ],
                top_k=2,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            if len(results) != 2:
                raise self._error(
                    "Local reranker probe returned an unexpected result count.",
                    retriable=True,
                )
            return ConnectionTestResult(
                success=True,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
            )
        except ProviderError as err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=self.provider_name(),
                model=self._model_name,
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
                error_code="UnknownError",
                error_message=str(exc),
            )


class NoOpReranker(RerankerProvider):
    """Passthrough reranker used when the user disables reranking.

    Preserves upstream retrieval order and scores so the pipeline shape stays
    identical to a reranked run (no branching in the retrieval service).
    """

    def provider_name(self) -> str:
        return "none"

    def model_name(self) -> str:
        return "disabled"

    async def rerank(
        self,
        query: str,
        documents: list[RerankDocument],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return [
            RerankResult(
                id=document.id,
                content=document.content,
                metadata=dict(document.metadata),
                rerank_score=float(document.initial_score),
                initial_score=document.initial_score,
                rank=rank,
            )
            for rank, document in enumerate(documents[:top_k])
        ]

    async def test_reranker(self) -> ConnectionTestResult:
        start_time = time.perf_counter()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return ConnectionTestResult(
            success=True,
            latency_ms=round(latency_ms, 2),
            provider=self.provider_name(),
            model=self.model_name(),
        )
