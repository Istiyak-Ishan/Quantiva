"""Reciprocal Rank Fusion (RRF) for dense + sparse retrieval.

The score of a chunk ``d`` is::

    RRF_Score(d) = Σ_m w_m / (k + rank_m(d))

where ``m`` ranges over the dense and sparse result lists, ``rank_m(d)`` is the
1-based rank of ``d`` in that list, ``k`` is the smoothing constant (default
``60``) and ``w_m`` are configurable per-source weights (default ``1.0``).
Chunks appearing in both lists are fused once, keyed by ``chunk_id``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.retrieval.types import RetrievedChunk, chunk_from_hit


def _coerce_result(
    result: RetrievedChunk | Mapping[str, Any],
    *,
    source: str,
) -> RetrievedChunk:
    """Accept typed chunks or legacy mapping-shaped repository results."""
    if isinstance(result, RetrievedChunk):
        return result

    chunk = chunk_from_hit(result)
    generic_score = float(result.get("score", 0.0) or 0.0)
    dense_rank = result.get("dense_rank")
    sparse_rank = result.get("sparse_rank")
    dense_score = result.get("dense_score")
    sparse_score = result.get("sparse_score")
    if dense_rank is not None:
        chunk.dense_rank = int(dense_rank)
    if sparse_rank is not None:
        chunk.sparse_rank = int(sparse_rank)
    if dense_score is not None:
        chunk.dense_score = float(dense_score)
    elif source == "dense":
        chunk.dense_score = generic_score
    if sparse_score is not None:
        chunk.sparse_score = float(sparse_score)
    elif source == "sparse":
        chunk.sparse_score = generic_score
    return chunk


def _merge_metadata(base: dict[str, object], incoming: dict[str, object]) -> None:
    for key, value in incoming.items():
        if key not in base or base[key] is None or base[key] == "":
            base[key] = value


def _merge_chunk(target: RetrievedChunk, incoming: RetrievedChunk) -> None:
    """Preserve the richest content and merge non-empty metadata."""
    if len(incoming.content or "") > len(target.content or ""):
        target.content = incoming.content
    _merge_metadata(target.metadata, incoming.metadata)


def reciprocal_rank_fusion(
    dense_results: Sequence[RetrievedChunk | Mapping[str, Any]],
    sparse_results: Sequence[RetrievedChunk | Mapping[str, Any]],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse dense and sparse result lists using weighted RRF.

    Parameters
    ----------
    dense_results, sparse_results:
        Result lists ordered by descending relevance.  Ranks are derived from
        list position and are therefore 1-based.
    k:
        RRF smoothing constant.  Must be non-negative.
    dense_weight, sparse_weight:
        Per-source weights.  At least one must be positive.
    top_k:
        Optional maximum number of fused chunks to return.

    Returns
    -------
    list[RetrievedChunk]
        Deduplicated chunks ordered by descending ``rrf_score``.  Each chunk
        preserves ``dense_rank``, ``sparse_rank``, ``dense_score`` and
        ``sparse_score`` where available.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("RRF weights must be non-negative")
    if dense_weight == 0 and sparse_weight == 0:
        raise ValueError("At least one RRF weight must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    fused: dict[str, RetrievedChunk] = {}
    seen_dense: set[str] = set()
    seen_sparse: set[str] = set()

    for rank, raw_result in enumerate(dense_results, start=1):
        result = _coerce_result(raw_result, source="dense")
        if result.chunk_id in seen_dense:
            continue
        seen_dense.add(result.chunk_id)

        entry = fused.get(result.chunk_id)
        if entry is None:
            entry = RetrievedChunk(
                chunk_id=result.chunk_id,
                content=result.content,
                metadata=dict(result.metadata),
                dense_rank=rank,
                dense_score=(
                    result.dense_score
                    if result.dense_score is not None
                    else result.score
                ),
            )
            fused[entry.chunk_id] = entry
        else:
            _merge_chunk(entry, result)
            if entry.dense_rank is None:
                entry.dense_rank = rank
            if result.dense_score is not None:
                entry.dense_score = result.dense_score

        entry.rrf_score += dense_weight / (k + rank)

    for rank, raw_result in enumerate(sparse_results, start=1):
        result = _coerce_result(raw_result, source="sparse")
        if result.chunk_id in seen_sparse:
            continue
        seen_sparse.add(result.chunk_id)

        entry = fused.get(result.chunk_id)
        if entry is None:
            entry = RetrievedChunk(
                chunk_id=result.chunk_id,
                content=result.content,
                metadata=dict(result.metadata),
                sparse_rank=rank,
                sparse_score=(
                    result.sparse_score
                    if result.sparse_score is not None
                    else result.score
                ),
            )
            fused[entry.chunk_id] = entry
        else:
            _merge_chunk(entry, result)
            if entry.sparse_rank is None:
                entry.sparse_rank = rank
            if result.sparse_score is not None:
                entry.sparse_score = result.sparse_score

        entry.rrf_score += sparse_weight / (k + rank)

    ordered = sorted(
        fused.values(),
        key=lambda chunk: (-chunk.rrf_score, chunk.chunk_id),
    )
    if top_k is not None:
        return ordered[:top_k]
    return ordered


__all__ = ["reciprocal_rank_fusion"]
