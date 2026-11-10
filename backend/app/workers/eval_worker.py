import asyncio
import logging
import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.config import settings
from app.core.ownership import owned_documents
import app.models  # noqa: F401 - ensure all ORM models are registered in Base.metadata
from app.models.eval_result import EvalRun
from app.services.eval.eval_dataset import eval_dataset_generator
from app.services.eval.eval_runner import eval_runner
from app.workers.celery_app import celery_app
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def _evaluate(run_id: str):
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = VectorStore()
    try:
        async with sessions() as db:
            run = await db.scalar(select(EvalRun).where(EvalRun.id == uuid.UUID(run_id)).with_for_update())
            if run is None or run.status == "completed":
                return
            try:
                await owned_documents(db, [run.doc_id], run.user_id)
                dataset = await eval_dataset_generator.generate_qa_pairs(str(run.doc_id), db, n=5, user_id=str(run.user_id))
                if not dataset:
                    raise ValueError("No evaluation questions generated")
                report = await eval_runner.run(dataset, [str(run.doc_id)], db, user_id=str(run.user_id), retriever=HybridRetriever(store))
                run.avg_faithfulness = report.avg_faithfulness
                run.avg_context_relevance = report.avg_relevance
                run.avg_answer_relevance = report.avg_completeness
                run.per_question_scores = report.per_question_scores
                run.status = "completed"
                await db.commit()
            except Exception:
                logger.exception("Evaluation failed for %s", run_id)
                await db.rollback()
                await db.execute(update(EvalRun).where(EvalRun.id == uuid.UUID(run_id)).values(status="failed"))
                await db.commit()
                raise
    finally:
        await store.client.close()
        await engine.dispose()


@celery_app.task(name="run_evaluation", acks_late=True, reject_on_worker_lost=True)
def run_evaluation(run_id: str):
    asyncio.run(_evaluate(run_id))
