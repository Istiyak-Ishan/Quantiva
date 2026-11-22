"""
backend/app/evaluation/__init__.py
────────────────────────────────────────────────────────────────────────────────
Quantiva RAG Evaluation Module.
"""
from app.evaluation.answer_quality import (
    CitationEvaluationDetail,
    CitationEvaluationResult,
    FaithfulnessResult,
    PerformanceTelemetry,
    TelemetryTracker,
    evaluate_answer_faithfulness,
    evaluate_citation_precision_and_groundedness,
    get_current_rss_mb,
)
from app.evaluation.retrieval import (
    BenchmarkQuery,
    QueryEvaluationDetail,
    RetrievalStrategy,
    StrategyEvaluationResult,
    compare_retrieval_strategies,
    evaluate_retrieval_strategy,
    is_chunk_relevant,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "BenchmarkQuery",
    "CitationEvaluationDetail",
    "CitationEvaluationResult",
    "FaithfulnessResult",
    "PerformanceTelemetry",
    "QueryEvaluationDetail",
    "RetrievalStrategy",
    "StrategyEvaluationResult",
    "TelemetryTracker",
    "compare_retrieval_strategies",
    "evaluate_answer_faithfulness",
    "evaluate_citation_precision_and_groundedness",
    "evaluate_retrieval_strategy",
    "get_current_rss_mb",
    "is_chunk_relevant",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
