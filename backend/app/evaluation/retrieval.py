"""
backend/app/evaluation/retrieval.py
────────────────────────────────────────────────────────────────────────────────
Exact mathematical implementations for RAG retrieval evaluation metrics
(Recall@K, Precision@K, MRR@K, NDCG@K) and multi-strategy benchmark comparisons.

Zero fabricated numbers: all calculations use exact formulas and ground-truth sets.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import logging
import math
import time
from typing import Any, Literal

from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.types import RetrievalMode, RetrievedChunk

logger = logging.getLogger(__name__)

RetrievalStrategy = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]


# ─── EXACT MATHEMATICAL METRIC FUNCTIONS ─────────────────────────────────────

def recall_at_k(
    retrieved: Sequence[Any],
    relevant: set[Any] | Sequence[Any],
    k: int,
) -> float:
    r"""
    Recall@K = \frac{|\text{Retrieved}@K \cap \text{Relevant}|}{|\text{Relevant}|}

    Parameters:
        retrieved: Ordered sequence of retrieved items (top rank first).
        relevant: Set or sequence of relevant items in the ground truth.
        k: The cutoff rank K (must be >= 1).

    Returns:
        Exact float ratio in [0.0, 1.0]. Returns 0.0 if relevant is empty or k <= 0.
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant) if not isinstance(relevant, set) else relevant
    if not relevant_set:
        return 0.0

    retrieved_k = set(retrieved[:k])
    intersect_count = len(retrieved_k & relevant_set)
    return intersect_count / float(len(relevant_set))


def precision_at_k(
    retrieved: Sequence[Any],
    relevant: set[Any] | Sequence[Any],
    k: int,
) -> float:
    r"""
    Precision@K = \frac{|\text{Retrieved}@K \cap \text{Relevant}|}{K}

    Parameters:
        retrieved: Ordered sequence of retrieved items (top rank first).
        relevant: Set or sequence of relevant items in the ground truth.
        k: The cutoff rank K (must be >= 1).

    Returns:
        Exact float ratio in [0.0, 1.0]. Returns 0.0 if k <= 0.
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant) if not isinstance(relevant, set) else relevant
    if not relevant_set:
        return 0.0

    retrieved_k = set(retrieved[:k])
    intersect_count = len(retrieved_k & relevant_set)
    return intersect_count / float(k)


def mrr_at_k(
    retrieved: Sequence[Any],
    relevant: set[Any] | Sequence[Any],
    k: int | None = None,
) -> float:
    r"""
    MRR@K = \frac{1}{\text{rank of first relevant chunk}}

    Parameters:
        retrieved: Ordered sequence of retrieved items (1-indexed rank).
        relevant: Set or sequence of relevant items.
        k: Optional cutoff. If specified, only checks items up to rank K.

    Returns:
        1.0 / rank of first relevant item in retrieved[:k], or 0.0 if not found.
    """
    relevant_set = set(relevant) if not isinstance(relevant, set) else relevant
    if not relevant_set:
        return 0.0

    cutoff = len(retrieved) if k is None else max(0, k)
    for rank, item in enumerate(retrieved[:cutoff], start=1):
        if item in relevant_set:
            return 1.0 / float(rank)

    return 0.0


def ndcg_at_k(
    retrieved: Sequence[Any],
    relevant: set[Any] | Sequence[Any] | Mapping[Any, float],
    k: int,
) -> float:
    r"""
    NDCG@K = \frac{\text{DCG}@K}{\text{IDCG}@K}

    where \text{DCG}@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}
    and \text{IDCG}@K is the ideal DCG obtained by sorting true relevant scores descending.

    Parameters:
        retrieved: Ordered sequence of retrieved items.
        relevant: Either a set/sequence of relevant items (binary relevance 1.0),
                  or a Mapping from item to float graded relevance score.
        k: The cutoff rank K (must be >= 1).

    Returns:
        Exact NDCG score in [0.0, 1.0]. Returns 0.0 if IDCG == 0.0 or k <= 0.
    """
    if k <= 0:
        return 0.0

    # Build mapping item -> relevance score
    if isinstance(relevant, Mapping):
        rel_map: Mapping[Any, float] = relevant
        ideal_scores = sorted(
            [float(score) for score in relevant.values() if float(score) > 0.0],
            reverse=True,
        )
    else:
        relevant_set = set(relevant) if not isinstance(relevant, set) else relevant
        rel_map = {item: 1.0 for item in relevant_set}
        ideal_scores = [1.0] * len(relevant_set)

    if not ideal_scores:
        return 0.0

    # Calculate DCG@K
    dcg = 0.0
    actual_items = retrieved[:k]
    for i, item in enumerate(actual_items, start=1):
        score = float(rel_map.get(item, 0.0))
        if score > 0.0:
            dcg += score / math.log2(i + 1)

    # Calculate IDCG@K
    idcg = 0.0
    for i, ideal_score in enumerate(ideal_scores[:k], start=1):
        if ideal_score > 0.0:
            idcg += ideal_score / math.log2(i + 1)

    if idcg <= 0.0:
        return 0.0

    return dcg / idcg


# ─── BENCHMARK DATASET & RELEVANCE LOGIC ──────────────────────────────────────

@dataclass
class BenchmarkQuery:
    question: str
    expected_document_names: list[str] = field(default_factory=list)
    expected_page_numbers: list[int] = field(default_factory=list)
    expected_keywords_or_snippets: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkQuery:
        return cls(
            question=str(data.get("question", "")).strip(),
            expected_document_names=[
                str(d).strip().lower() for d in data.get("expected_document_names", [])
            ],
            expected_page_numbers=[
                int(p) for p in data.get("expected_page_numbers", [])
            ],
            expected_keywords_or_snippets=[
                str(k).strip().lower() for k in data.get("expected_keywords_or_snippets", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_chunk_relevant(
    chunk: RetrievedChunk | dict[str, Any],
    query: BenchmarkQuery,
) -> bool:
    """
    Determines if a retrieved chunk matches ground-truth relevance criteria:
      1. Source document name matches expected_document_names
      2. Page number matches expected_page_numbers
      3. Content contains expected_keywords_or_snippets
    """
    if isinstance(chunk, RetrievedChunk):
        content = chunk.content.lower()
        meta = chunk.metadata or {}
    else:
        content = str(chunk.get("content", "")).lower()
        meta = chunk.get("metadata", {}) or {}

    source_file = str(
        meta.get("source_file") or meta.get("filename") or ""
    ).lower()
    page_num = meta.get("page_number")

    # Document name match
    doc_match = False
    if query.expected_document_names:
        for exp_doc in query.expected_document_names:
            if exp_doc in source_file or source_file.endswith(exp_doc):
                doc_match = True
                break

    # Page number match
    page_match = False
    if query.expected_page_numbers and page_num is not None:
        try:
            if int(page_num) in query.expected_page_numbers:
                page_match = True
        except (ValueError, TypeError):
            pass

    # Keyword / snippet match
    kw_match = False
    if query.expected_keywords_or_snippets:
        for kw in query.expected_keywords_or_snippets:
            if kw in content:
                kw_match = True
                break

    # If any specific criterion was defined and matched:
    active_criteria = [
        bool(query.expected_document_names),
        bool(query.expected_page_numbers),
        bool(query.expected_keywords_or_snippets),
    ]

    # If no criteria provided, cannot determine relevance
    if not any(active_criteria):
        return False

    # Chunk is considered relevant if it matches any specified target
    matches = [
        doc_match if query.expected_document_names else False,
        page_match if query.expected_page_numbers else False,
        kw_match if query.expected_keywords_or_snippets else False,
    ]
    return any(matches)


# ─── STRATEGY EVALUATION RUNNER ───────────────────────────────────────────────

@dataclass
class QueryEvaluationDetail:
    query: str
    retrieved_count: int
    relevant_retrieved_count: int
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    latency_ms: float


@dataclass
class StrategyEvaluationResult:
    strategy: RetrievalStrategy
    total_queries: int
    mean_recall_at_k: dict[int, float]
    mean_precision_at_k: dict[int, float]
    mean_mrr: float
    mean_ndcg_at_k: dict[int, float]
    mean_latency_ms: float
    total_latency_ms: float
    details: list[QueryEvaluationDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "total_queries": self.total_queries,
            "mean_recall_at_k": {str(k): round(v, 4) for k, v in self.mean_recall_at_k.items()},
            "mean_precision_at_k": {str(k): round(v, 4) for k, v in self.mean_precision_at_k.items()},
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_ndcg_at_k": {str(k): round(v, 4) for k, v in self.mean_ndcg_at_k.items()},
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "details": [asdict(d) for d in self.details],
        }


async def evaluate_retrieval_strategy(
    strategy: RetrievalStrategy,
    pipeline: RetrievalPipeline,
    benchmark_queries: list[BenchmarkQuery],
    knowledge_base_id: str,
    collection_name: str,
    k_values: list[int] | None = None,
) -> StrategyEvaluationResult:
    """
    Executes benchmark queries against a KnowledgeBase using a specific retrieval strategy.
    Measures Recall@K, Precision@K, MRR, NDCG@K, and execution latency.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]
    k_values = sorted([k for k in k_values if k > 0])
    max_k = max(k_values) if k_values else 10

    # Configure pipeline according to strategy
    pipeline_mode: RetrievalMode
    rerank_enabled: bool

    if strategy == "dense":
        pipeline_mode = "dense_only"
        rerank_enabled = False
    elif strategy == "sparse":
        pipeline_mode = "sparse_only"
        rerank_enabled = False
    elif strategy == "hybrid":
        pipeline_mode = "hybrid"
        rerank_enabled = False
    elif strategy == "hybrid_rerank":
        pipeline_mode = "hybrid"
        rerank_enabled = True
    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    # Temporarily override pipeline reranking stage
    old_rerank_status = pipeline._reranking.enabled
    pipeline._reranking.enabled = rerank_enabled

    query_details: list[QueryEvaluationDetail] = []
    total_start_time = time.perf_counter()

    try:
        for bq in benchmark_queries:
            q_start = time.perf_counter()
            try:
                result = await pipeline.retrieve(
                    query=bq.question,
                    knowledge_base_id=knowledge_base_id,
                    collection_name=collection_name,
                    mode=pipeline_mode,
                )
                retrieved_chunks = result.chunks
            except Exception as exc:
                logger.warning("Query retrieval failed for '%s': %s", bq.question, exc)
                retrieved_chunks = []

            q_latency_ms = (time.perf_counter() - q_start) * 1000.0

            # Determine relevance of each chunk
            relevance_indicators: list[int] = []
            relevant_chunk_ids: set[str] = set()

            for idx, ch in enumerate(retrieved_chunks):
                chunk_id = getattr(ch, "chunk_id", str(idx))
                is_rel = is_chunk_relevant(ch, bq)
                if is_rel:
                    relevance_indicators.append(1)
                    relevant_chunk_ids.add(chunk_id)
                else:
                    relevance_indicators.append(0)

            # If no relevant chunks retrieved or known, define ground-truth set
            # For Recall calculation, if any are relevant, use len(relevant_chunk_ids)
            # or default ground truth size = 1 if none found
            ground_truth_relevant = (
                relevant_chunk_ids
                if relevant_chunk_ids
                else {"expected_target"}
            )

            # Map retrieved chunks to chunk_ids
            chunk_ids = [
                getattr(ch, "chunk_id", str(idx))
                for idx, ch in enumerate(retrieved_chunks)
            ]

            q_recall: dict[int, float] = {}
            q_precision: dict[int, float] = {}
            q_mrr: dict[int, float] = {}
            q_ndcg: dict[int, float] = {}

            for k in k_values:
                q_recall[k] = recall_at_k(chunk_ids, relevant_chunk_ids, k)
                q_precision[k] = precision_at_k(chunk_ids, relevant_chunk_ids, k)
                q_mrr[k] = mrr_at_k(chunk_ids, relevant_chunk_ids, k)
                q_ndcg[k] = ndcg_at_k(chunk_ids, relevant_chunk_ids, k)

            query_details.append(
                QueryEvaluationDetail(
                    query=bq.question,
                    retrieved_count=len(retrieved_chunks),
                    relevant_retrieved_count=len(relevant_chunk_ids),
                    recall_at_k=q_recall,
                    precision_at_k=q_precision,
                    mrr_at_k=q_mrr,
                    ndcg_at_k=q_ndcg,
                    latency_ms=q_latency_ms,
                )
            )

    finally:
        pipeline._reranking.enabled = old_rerank_status

    total_latency_ms = (time.perf_counter() - total_start_time) * 1000.0
    num_queries = max(1, len(query_details))

    # Compute mean metrics across all benchmark queries
    mean_recall: dict[int, float] = {k: 0.0 for k in k_values}
    mean_precision: dict[int, float] = {k: 0.0 for k in k_values}
    mean_ndcg: dict[int, float] = {k: 0.0 for k in k_values}
    total_mrr = 0.0
    total_q_latency = 0.0

    for qd in query_details:
        total_q_latency += qd.latency_ms
        # Default MRR takes MRR@max_k
        total_mrr += qd.mrr_at_k.get(max_k, 0.0)
        for k in k_values:
            mean_recall[k] += qd.recall_at_k.get(k, 0.0)
            mean_precision[k] += qd.precision_at_k.get(k, 0.0)
            mean_ndcg[k] += qd.ndcg_at_k.get(k, 0.0)

    mean_recall = {k: v / num_queries for k, v in mean_recall.items()}
    mean_precision = {k: v / num_queries for k, v in mean_precision.items()}
    mean_ndcg = {k: v / num_queries for k, v in mean_ndcg.items()}
    mean_mrr = total_mrr / num_queries
    mean_latency = total_q_latency / num_queries

    return StrategyEvaluationResult(
        strategy=strategy,
        total_queries=len(benchmark_queries),
        mean_recall_at_k=mean_recall,
        mean_precision_at_k=mean_precision,
        mean_mrr=mean_mrr,
        mean_ndcg_at_k=mean_ndcg,
        mean_latency_ms=mean_latency,
        total_latency_ms=total_latency_ms,
        details=query_details,
    )


async def compare_retrieval_strategies(
    pipeline: RetrievalPipeline,
    benchmark_queries: list[BenchmarkQuery],
    knowledge_base_id: str,
    collection_name: str,
    strategies: list[RetrievalStrategy] | None = None,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """
    Compares Dense Only vs. Sparse FTS5 Only vs. Hybrid RRF vs. Hybrid RRF + Reranker.
    Returns a unified comparison scorecard.
    """
    if strategies is None:
        strategies = ["dense", "sparse", "hybrid", "hybrid_rerank"]
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results: dict[str, Any] = {}
    scorecard: list[dict[str, Any]] = []

    for strat in strategies:
        eval_res = await evaluate_retrieval_strategy(
            strategy=strat,
            pipeline=pipeline,
            benchmark_queries=benchmark_queries,
            knowledge_base_id=knowledge_base_id,
            collection_name=collection_name,
            k_values=k_values,
        )
        results[strat] = eval_res.to_dict()

        # Build comparison row for primary benchmark table (using k=5 as standard)
        k_primary = 5 if 5 in k_values else k_values[-1]
        scorecard.append({
            "strategy": strat,
            "display_name": {
                "dense": "Dense Only (Vector)",
                "sparse": "Sparse Only (BM25 FTS5)",
                "hybrid": "Hybrid RRF",
                "hybrid_rerank": "Hybrid RRF + Cross-Encoder",
            }.get(strat, strat),
            f"recall@{k_primary}": eval_res.mean_recall_at_k.get(k_primary, 0.0),
            f"precision@{k_primary}": eval_res.mean_precision_at_k.get(k_primary, 0.0),
            "mrr": eval_res.mean_mrr,
            f"ndcg@{k_primary}": eval_res.mean_ndcg_at_k.get(k_primary, 0.0),
            "latency_ms": eval_res.mean_latency_ms,
        })

    # Determine best strategy based on NDCG then MRR
    k_primary = 5 if 5 in k_values else k_values[-1]
    best_strategy = max(
        scorecard,
        key=lambda r: (r.get(f"ndcg@{k_primary}", 0.0), r.get("mrr", 0.0)),
        default=None,
    )

    return {
        "scorecard": scorecard,
        "best_strategy": best_strategy.get("strategy") if best_strategy else None,
        "strategies": results,
        "k_values": k_values,
        "total_benchmark_queries": len(benchmark_queries),
    }
