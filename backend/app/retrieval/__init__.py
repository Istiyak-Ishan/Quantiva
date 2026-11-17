"""Provider-agnostic retrieval package for Quantiva.

Public entry point: :class:`app.retrieval.pipeline.RetrievalPipeline`.
"""
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.pipeline import RetrievalPipeline, build_context_blocks
from app.retrieval.reranking import RerankingStage
from app.retrieval.sparse import (
    SparseRetriever,
    build_sparse_query,
    normalize_query,
    tokenize_query,
)
from app.retrieval.types import (
    CitationAnchor,
    ContextBlock,
    RETRIEVAL_MODES,
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
    chunk_from_hit,
)

__all__ = [
    "CitationAnchor",
    "ContextBlock",
    "DenseRetriever",
    "RETRIEVAL_MODES",
    "RetrievalMode",
    "RetrievalPipeline",
    "RetrievalResult",
    "RetrievedChunk",
    "RerankingStage",
    "SparseRetriever",
    "build_context_blocks",
    "build_sparse_query",
    "chunk_from_hit",
    "normalize_query",
    "reciprocal_rank_fusion",
    "tokenize_query",
]
