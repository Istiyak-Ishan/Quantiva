import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, get_current_user
from app.core.ownership import owned_documents
from app.core.rate_limiter import enforce_rate_limit
from app.models.user import User
from app.models.eval_result import EvalRun
from app.workers.eval_worker import run_evaluation

router = APIRouter(prefix="/eval", tags=["Evaluation"])


@router.post("/run")
async def trigger_eval(doc_id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    docs = await owned_documents(db, [doc_id], current_user.id)
    if docs[0].status != "ready":
        raise HTTPException(status_code=409, detail="Document is not ready")
    await enforce_rate_limit(str(current_user.id), tier=current_user.tier, scope="eval")
    run = EvalRun(doc_id=doc_id, user_id=current_user.id, status="pending")
    db.add(run)
    await db.commit()
    await db.refresh(run)
    try:
        await asyncio.to_thread(run_evaluation.delay, str(run.id))
    except Exception:
        run.status = "failed"
        await db.commit()
        raise HTTPException(status_code=503, detail="Evaluation queue unavailable")
    return {"run_id": str(run.id), "status": "pending"}


@router.get("/history/list")
async def list_evals(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    runs = (await db.scalars(select(EvalRun).where(EvalRun.user_id == current_user.id).order_by(desc(EvalRun.created_at)).limit(10))).all()
    return [{"id": str(r.id), "status": r.status, "faithfulness": r.avg_faithfulness, "created_at": r.created_at} for r in runs]


@router.get("/{run_id}")
async def get_eval(run_id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    run = await db.scalar(select(EvalRun).where(EvalRun.id == run_id, EvalRun.user_id == current_user.id))
    if run is None:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return {"id": str(run.id), "doc_id": str(run.doc_id), "status": run.status, "avg_faithfulness": run.avg_faithfulness, "avg_relevance": run.avg_context_relevance, "avg_completeness": run.avg_answer_relevance, "details": run.per_question_scores}
