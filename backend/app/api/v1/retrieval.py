import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user, get_db
from app.core.ownership import owned_documents, require_current_index
from app.core.rate_limiter import enforce_rate_limit
from app.models.user import User
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.retrieval.hybrid_retriever import hybrid_retriever
from app.services.llm.errors import ProviderError

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=16000)
    doc_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/search")
async def search(req: SearchRequest, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await enforce_rate_limit(str(user.id), tier=user.tier, scope="retrieval")
    docs = await owned_documents(db, req.doc_ids, user.id)
    if not req.doc_ids:
        docs = list((await db.scalars(select(Document).where(Document.user_id == user.id, Document.status == "ready"))).all())
    if any(d.status != "ready" for d in docs):
        raise HTTPException(status_code=409, detail="Selected documents are not ready")
    ids, names = [d.id for d in docs], {str(d.id): d.filename for d in docs}
    chunks = list((await db.scalars(select(Chunk).join(Document, Chunk.doc_id == Document.id).where(Chunk.doc_id.in_(ids), Document.user_id == user.id))).all()) if ids else []
    require_current_index(docs, chunks)
    try:
        results = await hybrid_retriever.search(req.query, [str(d) for d in ids], chunks, top_k=req.top_k, user_id=str(user.id))
    except ProviderError as error:
        raise HTTPException(status_code=error.status_code, detail={"error": str(error), "code": error.code})
    return {"chunks": [{"doc_id": r["payload"]["doc_id"], "filename": names.get(r["payload"]["doc_id"], "Document"), "page_or_row": r["payload"].get("page"), "text": r["payload"]["text"], "score": r.get("rerank_score", r["score"])} for r in results]}
