"""
app/providers/local_models.py
──────────────────────────────────────────────────────────────────────────────
Shared helpers for in-process local model providers (SentenceTransformers
embeddings and CrossEncoder rerankers).

Importing this module must stay cheap: neither :mod:`torch` nor
:mod:`sentence_transformers` is imported here.  Heavy imports happen inside the
provider's model factory, which only runs on first use / explicit warmup.

All downloaded weights are bound to the user's configured Quantiva data
directory so a packaged desktop install never writes into the application
bundle or the roaming profile:

    <QUANTIVA_DATA_DIR>/models/embeddings
    <QUANTIVA_DATA_DIR>/models/rerankers
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

MODELS_DIRNAME = "models"
EMBEDDINGS_SUBDIR = "embeddings"
RERANKERS_SUBDIR = "rerankers"

_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def quantiva_data_dir() -> Path:
    """Return the configured Quantiva data directory (never None)."""
    from app.config import settings

    configured = getattr(settings, "QUANTIVA_DATA_DIR", None)
    if configured:
        return Path(configured)
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "Quantiva"
    return Path.home() / ".quantiva"


def model_cache_dir(kind: str, configured: str | Path | None = None) -> Path:
    """Resolve ``<data_dir>/models/<kind>`` and create it on disk.

    ``kind`` is ``"embeddings"`` or ``"rerankers"``.
    """
    if configured is not None:
        directory = Path(configured).expanduser()
    else:
        directory = quantiva_data_dir() / MODELS_DIRNAME / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def detect_device(preference: str | None = None) -> str:
    """Resolve the inference device without importing torch at module scope.

    ``preference`` may be ``"auto"`` (or ``None``/empty) to probe hardware,
    otherwise an explicit ``"cpu"``, ``"cuda"``, ``"cuda:1"``, or ``"mps"``
    value is returned unchanged (the user's choice is authoritative).
    """
    chosen = (preference or "auto").strip().lower()
    if chosen and chosen != "auto":
        return chosen

    try:
        import torch  # intentionally lazy: torch is heavy and optional
    except Exception:  # noqa: BLE001 - torch is optional until a model is used
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - a broken driver must not crash the app
        # A broken driver or partial CUDA install must never crash the app.
        return "cpu"
    return "cpu"


def configure_offline_mode(offline: bool) -> None:
    """Enable Hugging Face / Transformers offline flags before loading.

    Only ever *enables* offline mode: an operator who set ``HF_HUB_OFFLINE``
    globally keeps that preference even when the app-level flag is off.
    """
    if not offline:
        return
    for name in _OFFLINE_ENV_VARS:
        os.environ[name] = "1"


def _tolist(value: Any) -> Any:
    """Convert numpy/tensor-like values to plain Python containers."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def to_float_list(value: Any) -> list[float]:
    """Coerce a 1-D (or single-row 2-D) numeric value to ``list[float]``."""
    data = _tolist(value)
    if isinstance(data, (int, float)):
        return [float(data)]
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected a numeric sequence, got {type(value).__name__}")

    items = list(data)
    if items and isinstance(_tolist(items[0]), (list, tuple)):
        # e.g. np.array([[0.1, 0.2]]) — unwrap the single row.
        if len(items) != 1:
            raise TypeError("Expected a single vector, got a 2-D matrix")
        items = list(_tolist(items[0]))

    values: list[float] = []
    for item in items:
        scalar = _tolist(item)
        if isinstance(scalar, (list, tuple)):
            raise TypeError("Expected a 1-D vector, got nested values")
        values.append(float(scalar))
    return values


def to_float_matrix(value: Any) -> list[list[float]]:
    """Coerce a 2-D numeric value into ``list[list[float]]``."""
    data = _tolist(value)
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected a numeric matrix, got {type(value).__name__}")
    rows = list(data)
    if not rows:
        return []
    if not isinstance(_tolist(rows[0]), (list, tuple)):
        # A single flat vector was returned for a one-item batch.
        return [to_float_list(rows)]
    return [to_float_list(row) for row in rows]


def to_float_scores(value: Any) -> list[float]:
    """Coerce cross-encoder predictions (possibly ``(n, 1)``) to a flat list."""
    data = _tolist(value)
    if isinstance(data, (int, float)):
        return [float(data)]
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected model scores, got {type(value).__name__}")

    scores: list[float] = []
    for item in list(data):
        scalar = _tolist(item)
        if isinstance(scalar, (list, tuple)):
            # Multi-label head: keep the first logit per pair.
            scalar = list(scalar)
            if not scalar:
                raise TypeError("Received an empty score row")
            scalar = scalar[0]
        scores.append(float(scalar))
    return scores


def l2_normalize(vector: list[float]) -> list[float]:
    """Return a unit-length copy of *vector*.

    Raises ``ValueError`` for a zero-magnitude or non-finite vector so callers
    can convert it into a typed provider error.
    """
    if not vector:
        raise ValueError("empty vector")
    for value in vector:
        if not math.isfinite(value):
            raise ValueError("non-finite vector value")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("zero-magnitude vector")
    if abs(norm - 1.0) < 1e-12:
        return list(vector)
    return [value / norm for value in vector]
