"""Provider-agnostic result and context types for retrieval.

The dataclasses here are the stable boundary between retrieval, reranking and
LLM prompt construction.  They carry provenance from every retrieval stage so
callers can audit why a chunk was selected.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from app.storage.interfaces import SearchHit

RetrievalMode = Literal["hybrid", "dense_only", "sparse_only"]

RETRIEVAL_MODES: frozenset[str] = frozenset({"hybrid", "dense_only", "sparse_only"})


# ── Retrieved chunk ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class RetrievedChunk:
    """A chunk flowing through retrieval, fusion and reranking.

    ``dense_rank`` / ``sparse_rank`` and their scores are preserved from the
    upstream stages.  ``rrf_score`` is populated by rank fusion and
    ``rerank_score`` only by the reranking stage.  Ranks are 1-based.
    """

    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_rank: int | None = None

    @property
    def score(self) -> float:
        """Best available relevance score for downstream consumers."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score:
            return self.rrf_score
        if self.dense_score is not None:
            return self.dense_score
        if self.sparse_score is not None:
            return self.sparse_score
        return 0.0

    def to_payload(self) -> dict[str, Any]:
        """Legacy-shaped payload for agents/API code expecting ``payload``."""
        payload = dict(self.metadata)
        payload.setdefault("chunk_id", self.chunk_id)
        payload.setdefault("text", self.content)
        payload.setdefault("content", self.content)
        return payload


def chunk_from_hit(
    hit: SearchHit | Mapping[str, Any],
    *,
    dense_rank: int | None = None,
    sparse_rank: int | None = None,
) -> RetrievedChunk:
    """Normalise a repository hit into a :class:`RetrievedChunk`.

    The primary contract is :class:`~app.storage.interfaces.SearchHit`, but
    legacy-style mappings with ``id`` / ``payload`` keys are accepted so
    adapters can be migrated incrementally.  ``content`` falls back to common
    metadata keys so adapters that only store text in a payload still produce
    usable reranking input.
    """
    if isinstance(hit, Mapping):
        nested = hit.get("chunk")
        nested_mapping = nested if isinstance(nested, Mapping) else None
        raw_metadata = hit.get("metadata")
        if raw_metadata is None:
            raw_metadata = hit.get("payload")
        if raw_metadata is None and nested_mapping is not None:
            raw_metadata = nested_mapping.get("payload")
            if raw_metadata is None:
                raw_metadata = nested_mapping.get("metadata")
        metadata = dict(raw_metadata or {})
        nested_id = nested_mapping.get("chunk_id") if nested_mapping else None
        if nested_id is None and nested_mapping is not None:
            nested_id = nested_mapping.get("qdrant_id")
        chunk_id = (
            hit.get("chunk_id")
            or hit.get("id")
            or nested_id
            or metadata.get("chunk_id")
            or metadata.get("qdrant_id")
            or ""
        )
        nested_content = None
        if nested_mapping is not None:
            nested_content = nested_mapping.get("content") or nested_mapping.get("text")
        content = (
            hit.get("content")
            or hit.get("text")
            or nested_content
            or metadata.get("content")
            or metadata.get("text")
            or ""
        )
        score = hit.get("score", 0.0)
    else:
        metadata = dict(hit.metadata or {})
        chunk_id = (
            hit.chunk_id
            or metadata.get("chunk_id")
            or metadata.get("qdrant_id")
            or ""
        )
        content = hit.content or metadata.get("content") or metadata.get("text") or ""
        score = hit.score

    chunk_id = str(chunk_id)
    if not chunk_id:
        raise ValueError("Search hit is missing a chunk_id")
    numeric_score = float(score) if score is not None else 0.0
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=str(content or ""),
        metadata=metadata,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        dense_score=numeric_score if dense_rank is not None else None,
        sparse_score=numeric_score if sparse_rank is not None else None,
    )


# ── Citation / context blocks ─────────────────────────────────────────────────

def _first_metadata_value(
    metadata: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value is not None and value != "":
            return value
    return default


@dataclass(frozen=True, slots=True)
class CitationAnchor:
    """Explicit citation anchor embedded in every context block."""

    source_file: str
    page_number: int | str
    chunk_index: int | str

    @classmethod
    def from_chunk(cls, chunk: RetrievedChunk) -> CitationAnchor:
        metadata = chunk.metadata
        source_file = str(
            _first_metadata_value(
                metadata,
                ("source_file", "file_name", "filename", "document_name"),
                "unknown",
            )
        )
        page_number = _first_metadata_value(
            metadata,
            ("page_number", "page", "page_or_row"),
            "unknown",
        )
        chunk_index = _first_metadata_value(
            metadata,
            ("chunk_index",),
            0,
        )
        return cls(
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk_index,
        )

    def format(self) -> str:
        return (
            f"[Doc: {self.source_file}, Page: {self.page_number}, "
            f"Chunk: {self.chunk_index}]"
        )


@dataclass(slots=True)
class ContextBlock:
    """One reranked chunk rendered with its citation anchor."""

    citation: str
    content: str
    chunk: RetrievedChunk

    def render(self) -> str:
        return f"{self.citation}\n{self.content}"


@dataclass(slots=True)
class RetrievalResult:
    """Complete output of :class:`app.retrieval.pipeline.RetrievalPipeline`."""

    query: str
    mode: RetrievalMode
    chunks: list[RetrievedChunk] = field(default_factory=list)
    context_blocks: list[ContextBlock] = field(default_factory=list)
    candidates_considered: int = 0
    reranked: bool = False

    @property
    def context(self) -> str:
        return "\n\n".join(block.render() for block in self.context_blocks)

    @property
    def citations(self) -> list[str]:
        return [block.citation for block in self.context_blocks]

    # Convenience sequence behaviour: callers that only need the chunks can
    # iterate or index the result directly.
    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterator[RetrievedChunk]:
        return iter(self.chunks)

    def __getitem__(self, index: int) -> RetrievedChunk:
        return self.chunks[index]


__all__ = [
    "CitationAnchor",
    "ContextBlock",
    "RETRIEVAL_MODES",
    "RetrievalMode",
    "RetrievalResult",
    "RetrievedChunk",
    "chunk_from_hit",
]
