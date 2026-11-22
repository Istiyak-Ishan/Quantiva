"""
backend/app/api/routes/evaluation.py
────────────────────────────────────────────────────────────────────────────────
FastAPI routes for Quantitative RAG Evaluation & Benchmarking (Prompt P8.1).

Exposes:
  • POST /api/v1/evaluation/run: Run multi-strategy retrieval and generation benchmark.
  • GET  /api/v1/evaluation/runs: List previous evaluation runs and scorecards.
  • GET  /api/v1/evaluation/runs/{run_id}: Get detailed scorecard and query metrics.
  • GET  /api/v1/evaluation/benchmarks/default: Get preconfigured benchmark dataset for a KB.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from app.api.routes.dependencies import verify_desktop_token
from app.core.application_context import get_application_context
from app.core.orchestrator import RAGOrchestrator
from app.evaluation.answer_quality import (
    TelemetryTracker,
    evaluate_answer_faithfulness,
    evaluate_citation_precision_and_groundedness,
)
from app.evaluation.retrieval import (
    BenchmarkQuery,
    RetrievalStrategy,
    compare_retrieval_strategies,
)
from app.providers.registry import IndexMetadata as ProviderIndexMetadata
from app.retrieval.pipeline import RetrievalPipeline
from app.storage.database import SessionLocal, engine as local_database_engine
from app.storage.models import (
    Document,
    DocumentChunk,
    EvaluationRun,
    IndexMetadata,
    KnowledgeBase,
)
from app.storage.sparse_store import SQLiteSparseIndex
from app.storage.vector_store import LocalVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
    dependencies=[Depends(verify_desktop_token)],
)


# ── Pydantic Request & Response Schemas ────────────────────────────────────────

class BenchmarkQuerySchema(BaseModel):
    question: str = Field(min_length=2)
    expected_document_names: list[str] = Field(default_factory=list)
    expected_page_numbers: list[int] = Field(default_factory=list)
    expected_keywords_or_snippets: list[str] = Field(default_factory=list)


class EvaluationRunRequest(BaseModel):
    knowledge_base_id: str
    name: str = Field(default="Benchmark Run", max_length=255)
    strategies: list[str] = Field(
        default=["dense", "sparse", "hybrid", "hybrid_rerank"]
    )
    benchmark_dataset: list[BenchmarkQuerySchema] | None = None
    evaluate_generation: bool = True
    k_values: list[int] = Field(default=[1, 3, 5, 10])


def _serialize_run(run: EvaluationRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "knowledge_base_id": run.knowledge_base_id,
        "name": run.name,
        "status": run.status,
        "strategies_evaluated": run.strategies_evaluated_json or [],
        "strategy_results": run.strategy_results_json or {},
        "generation_metrics": run.generation_metrics_json or {},
        "telemetry": run.telemetry_json or {},
        "benchmark_dataset_count": len(run.benchmark_dataset_json or []),
        "created_at": run.created_at,
        "completed_at": run.completed_at,
    }


# ── Helper: Synthetic / Default Benchmark Generator ───────────────────────────

def _generate_default_benchmark_queries(
    session: Any, knowledge_base_id: str
) -> list[BenchmarkQuery]:
    """
    Generates realistic ground-truth benchmark queries based on indexed documents
    in the knowledge base if no custom JSON dataset is supplied.
    """
    docs = session.scalars(
        select(Document)
        .where(Document.knowledge_base_id == knowledge_base_id)
        .limit(10)
    ).all()

    queries: list[BenchmarkQuery] = []

    for doc in docs:
        filename = getattr(doc, "filename", None) or "document.pdf"
        chunks = session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == doc.id)
            .order_by(DocumentChunk.chunk_index.asc())
            .limit(3)
        ).all()

        for chunk in chunks:
            # Extract distinct key phrase from chunk
            words = [w for w in chunk.content.split() if len(w) > 4][:5]
            snippet = " ".join(words)
            if not snippet:
                continue

            clean_snippet = snippet.replace('"', '').replace("'", "")
            queries.append(
                BenchmarkQuery(
                    question=f"What does {filename} state regarding {clean_snippet}?",
                    expected_document_names=[filename],
                    expected_page_numbers=[chunk.page_number] if chunk.page_number else [],
                    expected_keywords_or_snippets=[words[0].lower()] if words else [],
                )
            )

    if not queries:
        queries = [
            BenchmarkQuery(
                question="What are the main findings and metrics in the indexed documents?",
                expected_document_names=[],
                expected_page_numbers=[],
                expected_keywords_or_snippets=["summary", "key", "report", "findings"],
            ),
            BenchmarkQuery(
                question="Summarize financial performance, revenues, or project timelines.",
                expected_document_names=[],
                expected_page_numbers=[],
                expected_keywords_or_snippets=["financial", "revenue", "timeline", "project"],
            ),
        ]

    return queries[:10]


# ── Evaluation Endpoints ──────────────────────────────────────────────────────

@router.post("/run", status_code=status.HTTP_201_CREATED)
async def run_evaluation(payload: EvaluationRunRequest) -> dict[str, Any]:
    """
    Execute a quantitative evaluation benchmark comparing retrieval strategies
    (Dense Only, Sparse Only, Hybrid RRF, Hybrid RRF + Cross-Encoder) and
    evaluating citation groundedness and faithfulness.
    """
    kb_id = payload.knowledge_base_id

    with SessionLocal() as session:
        kb = session.get(KnowledgeBase, kb_id)
        if kb is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base not found: {kb_id}",
            )

        index_meta = session.scalar(
            select(IndexMetadata).where(IndexMetadata.knowledge_base_id == kb_id)
        )
        if index_meta is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Index metadata missing for knowledge base: {kb_id}",
            )

        collection_name = index_meta.vector_collection_name
        prov_index_meta = ProviderIndexMetadata(
            knowledge_base_id=kb.id,
            embedding_provider=index_meta.embedding_provider,
            embedding_model_name=index_meta.embedding_model_name,
            embedding_dimensions=index_meta.embedding_dimensions,
            collection_name=collection_name,
        )

        # Parse or generate benchmark queries
        if payload.benchmark_dataset and len(payload.benchmark_dataset) > 0:
            benchmark_queries = [
                BenchmarkQuery.from_dict(q.model_dump())
                for q in payload.benchmark_dataset
            ]
        else:
            benchmark_queries = _generate_default_benchmark_queries(session, kb_id)

    app_ctx = get_application_context()
    registry = app_ctx.provider_registry
    vector_store = LocalVectorStore()
    sparse_store = SQLiteSparseIndex(local_database_engine)

    # Resolve providers
    try:
        embedding_provider = registry.get_knowledge_base_embedding_provider(prov_index_meta)
    except Exception:
        embedding_provider = registry.get_embedding_provider()

    reranker = registry.get_reranker_provider()
    llm_provider = registry.get_llm_provider()

    # Build retrieval pipeline
    pipeline = RetrievalPipeline(
        mode="hybrid",
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        sparse_index=sparse_store,
        index_metadata=prov_index_meta,
        reranker=reranker,
        rerank_enabled=True,
    )

    valid_strategies: list[RetrievalStrategy] = []
    for s in payload.strategies:
        if s in ("dense", "sparse", "hybrid", "hybrid_rerank"):
            valid_strategies.append(s)  # type: ignore
    if not valid_strategies:
        valid_strategies = ["dense", "sparse", "hybrid", "hybrid_rerank"]

    # Run benchmark under telemetry tracking
    generation_metrics: dict[str, Any] = {}

    with TelemetryTracker() as tracker:
        retrieval_comparison = await compare_retrieval_strategies(
            pipeline=pipeline,
            benchmark_queries=benchmark_queries,
            knowledge_base_id=kb_id,
            collection_name=collection_name,
            strategies=valid_strategies,
            k_values=payload.k_values,
        )

        # Optional generation & faithfulness evaluation
        if payload.evaluate_generation and benchmark_queries and llm_provider:
            sample_query = benchmark_queries[0]
            try:
                orchestrator = RAGOrchestrator(
                    provider_registry=registry,
                    vector_store=vector_store,
                    sparse_index=sparse_store,
                    index_metadata=prov_index_meta,
                    default_mode="direct",
                )

                rag_resp = await orchestrator.query(
                    query=sample_query.question,
                    knowledge_base_id=kb_id,
                    collection_name=collection_name,
                )

                cit_res = evaluate_citation_precision_and_groundedness(
                    answer=rag_resp.answer,
                    retrieved_chunks=rag_resp.sources,
                )

                faith_res = await evaluate_answer_faithfulness(
                    answer=rag_resp.answer,
                    context_chunks=rag_resp.sources,
                    llm_provider=llm_provider,
                )

                generation_metrics = {
                    "sample_question": sample_query.question,
                    "answer_excerpt": rag_resp.answer[:300] + ("…" if len(rag_resp.answer) > 300 else ""),
                    "citation_precision": cit_res.citation_precision,
                    "citation_groundedness": cit_res.citation_groundedness,
                    "faithfulness_score": faith_res.faithfulness_score,
                    "lexical_overlap_score": faith_res.lexical_overlap_score,
                    "llm_judge_score": faith_res.llm_judge_score,
                    "judge_reasoning": faith_res.judge_reasoning,
                    "ungrounded_claims": faith_res.ungrounded_claims,
                    "citations_detail": cit_res.to_dict(),
                    "generation_latency_ms": rag_resp.latency_ms,
                }
                tracker.telemetry.total_generation_latency_ms = rag_resp.latency_ms
            except Exception as exc:
                logger.warning("Generation evaluation encountered error: %s", exc)
                generation_metrics = {"error": str(exc)}

    # Record best retrieval latency from strategy results into telemetry
    if retrieval_comparison.get("scorecard"):
        tracker.telemetry.retrieval_latency_ms = float(
            retrieval_comparison["scorecard"][0].get("latency_ms", 0.0)
        )

    # Persist EvaluationRun to database
    with SessionLocal() as session:
        eval_run = EvaluationRun(
            knowledge_base_id=kb_id,
            name=payload.name.strip() or "Benchmark Run",
            status="COMPLETED",
            strategies_evaluated_json=valid_strategies,
            strategy_results_json=retrieval_comparison,
            generation_metrics_json=generation_metrics,
            telemetry_json=tracker.telemetry.to_dict(),
            benchmark_dataset_json=[bq.to_dict() for bq in benchmark_queries],
            completed_at=datetime.now(timezone.utc),
        )
        session.add(eval_run)
        session.commit()
        session.refresh(eval_run)
        return _serialize_run(eval_run)


@router.get("/runs", response_model=list[dict[str, Any]])
def list_evaluation_runs(
    knowledge_base_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """List historical evaluation runs ordered by recency."""
    with SessionLocal() as session:
        stmt = select(EvaluationRun).order_by(desc(EvaluationRun.created_at)).limit(limit)
        if knowledge_base_id:
            stmt = stmt.where(EvaluationRun.knowledge_base_id == knowledge_base_id)
        runs = session.scalars(stmt).all()
        return [_serialize_run(r) for r in runs]


@router.get("/runs/{run_id}", response_model=dict[str, Any])
def get_evaluation_run(run_id: str) -> dict[str, Any]:
    """Retrieve detailed scorecard and query metrics for an evaluation run."""
    with SessionLocal() as session:
        run = session.get(EvaluationRun, run_id)
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation run not found: {run_id}",
            )
        return _serialize_run(run)


@router.get("/benchmarks/default", response_model=list[dict[str, Any]])
def get_default_benchmarks(
    knowledge_base_id: str | None = Query(default=None),
) -> list[dict[str, Any]]:
    """Get sample benchmark dataset for a knowledge base."""
    with SessionLocal() as session:
        if knowledge_base_id:
            queries = _generate_default_benchmark_queries(session, knowledge_base_id)
        else:
            first_kb = session.scalar(select(KnowledgeBase).limit(1))
            if first_kb:
                queries = _generate_default_benchmark_queries(session, first_kb.id)
            else:
                queries = [
                    BenchmarkQuery(
                        question="What are the key financial performance numbers and margins?",
                        expected_keywords_or_snippets=["revenue", "margin", "profit", "financial"],
                    ),
                    BenchmarkQuery(
                        question="Summarize the core technical findings and recommendations.",
                        expected_keywords_or_snippets=["recommendation", "finding", "technical", "system"],
                    ),
                ]
        return [q.to_dict() for q in queries]
