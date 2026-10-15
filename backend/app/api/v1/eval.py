import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import asyncio

from app.core.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.eval_result import EvalRun
from app.services.eval.eval_dataset import eval_dataset_generator
from app.services.eval.eval_runner import eval_runner

router = APIRouter()

async def run_eval_background(run_id: uuid.UUID, doc_id_str: str):
    """
    Background task to run the eval pipeline since it can take minutes.
    In production, this should absolutely be a Celery task!
    """
    from app.db.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        run_record = await db.execute(select(EvalRun).where(EvalRun.id == run_id))
        eval_run = run_record.scalar_one_or_none()
        if not eval_run:
            return
            
        try:
            # 1. Generate Dataset
            dataset = await eval_dataset_generator.generate_qa_pairs(doc_id_str, db, n=5)
            
            # 2. Run Eval
            report = await eval_runner.run(dataset, [doc_id_str], db)
            
            # 3. Save Results
            eval_run.avg_faithfulness = report.avg_faithfulness
            eval_run.avg_context_relevance = report.avg_relevance
            eval_run.avg_answer_relevance = report.avg_completeness
            eval_run.per_question_scores = report.per_question_scores
            eval_run.status = "completed"
            
            await db.commit()
        except Exception as e:
            eval_run.status = f"failed: {str(e)}"
            await db.commit()

@router.post("/run")
async def trigger_eval(
    doc_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Triggers an asynchronous evaluation run.
    """
    new_run = EvalRun(
        doc_id=uuid.UUID(doc_id),
        status="pending"
    )
    db.add(new_run)
    await db.commit()
    await db.refresh(new_run)
    
    background_tasks.add_task(run_eval_background, new_run.id, doc_id)
    
    return {"run_id": str(new_run.id), "status": "pending"}

@router.get("/{run_id}")
async def get_eval(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns the result of an evaluation run.
    """
    result = await db.execute(select(EvalRun).where(EvalRun.id == uuid.UUID(run_id)))
    eval_run = result.scalar_one_or_none()
    
    if not eval_run:
        raise HTTPException(status_code=404, detail="Eval run not found")
        
    return {
        "id": str(eval_run.id),
        "doc_id": str(eval_run.doc_id),
        "status": eval_run.status,
        "avg_faithfulness": eval_run.avg_faithfulness,
        "avg_relevance": eval_run.avg_context_relevance,
        "avg_completeness": eval_run.avg_answer_relevance,
        "details": eval_run.per_question_scores
    }

@router.get("/history/list")
async def list_evals(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(EvalRun).order_by(desc(EvalRun.created_at)).limit(10))
    runs = result.scalars().all()
    
    return [{"id": str(r.id), "status": r.status, "faithfulness": r.avg_faithfulness, "created_at": r.created_at} for r in runs]
