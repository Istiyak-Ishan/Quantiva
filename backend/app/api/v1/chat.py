import uuid
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.core.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.session import Session
from app.models.message import Message
from app.models.chunk import Chunk
from app.schemas.chat import StreamEvent
from app.schemas.agent import AgentAnswer, Citation
from app.services.retrieval.hybrid_retriever import hybrid_retriever
from app.services.llm.prompt_builder import build_rag_prompt
from app.services.llm.gemini_client import gemini_client

router = APIRouter(prefix="/chat", tags=["Chat"])

class CreateSessionRequest(BaseModel):
    document_ids: List[str]

class ChatMessageRequest(BaseModel):
    session_id: str
    query: str

@router.post("/sessions", response_model=Dict[str, str])
async def create_session(
    req: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    doc_uuids = [uuid.UUID(d) for d in req.document_ids]
    new_session = Session(user_id=current_user.id, document_ids=doc_uuids)
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    return {"session_id": str(new_session.id)}

from app.services.agents.orchestrator import orchestrator

from fastapi import Response, Request
from app.core.rate_limiter import rate_limiter

@router.post("/message", response_model=AgentAnswer)
async def send_message(
    req: ChatMessageRequest,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # 1. Rate Limiting
    tier = "pro" if current_user.email.endswith("@quantiva.ai") else "free"
    limit_status = await rate_limiter.check(str(current_user.id), tier=tier)
    if not limit_status["allowed"]:
        response.headers["Retry-After"] = str(limit_status["reset_in"])
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Try again in {limit_status['reset_in']} seconds.")
        
    session_id_uuid = uuid.UUID(req.session_id)
    result = await db.execute(select(Session).where(Session.id == session_id_uuid, Session.user_id == current_user.id))
    chat_session = result.scalar_one_or_none()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    doc_ids_str = [str(d) for d in chat_session.document_ids]
    
    chunk_res = await db.execute(select(Chunk).where(Chunk.doc_id.in_(chat_session.document_ids)))
    db_chunks = chunk_res.scalars().all()
    
    from app.services.guardrails.guardrails_service import GuardrailBlockError
    try:
        agent_result = await orchestrator.run(query=req.query, session_id=req.session_id, doc_ids=doc_ids_str, db_chunks=db_chunks)
    except GuardrailBlockError as e:
        raise HTTPException(status_code=400, detail={"error": e.message, "categories": e.categories})
    
    # 2. X-Cache-Hit Header
    response.headers["X-Cache-Hit"] = str(agent_result.answer.cache_hit).lower()
    
    from app.services.llm.gemini_client import gemini_client
    
    user_msg = Message(session_id=chat_session.id, role="user", content=req.query)
    
    tokens = agent_result.answer.tokens_used
    input_tokens = int(tokens * 0.8) # rough estimate
    output_tokens = tokens - input_tokens
    cost = gemini_client.calculate_cost(input_tokens, output_tokens) if not agent_result.answer.cache_hit else 0.0
    
    ai_msg = Message(
        session_id=chat_session.id, 
        role="assistant", 
        content=agent_result.answer.answer, 
        citations=[c.model_dump() for c in agent_result.answer.citations],
        tokens=tokens,
        cost_usd=cost
    )
    db.add(user_msg)
    db.add(ai_msg)
    await db.commit()
    
    return agent_result.answer


@router.get("/message/stream")
async def stream_message(
    session_id: str,
    query: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Streaming endpoint using the ReasoningAgent.
    """
    try:
        session_id_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid Session ID")
        
    result = await db.execute(select(Session).where(Session.id == session_id_uuid, Session.user_id == current_user.id))
    chat_session = result.scalar_one_or_none()
    if not chat_session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    doc_ids_str = [str(d) for d in chat_session.document_ids]
    chunk_res = await db.execute(select(Chunk).where(Chunk.doc_id.in_(chat_session.document_ids)))
    db_chunks = chunk_res.scalars().all()
    
    user_msg = Message(session_id=chat_session.id, role="user", content=query)
    db.add(user_msg)
    await db.commit()

    async def event_generator():
        agent_result = await orchestrator.run(query=query, session_id=session_id, doc_ids=doc_ids_str, db_chunks=db_chunks)
        
        # Simulate streaming the answer
        for word in agent_result.answer.answer.split():
            import asyncio
            await asyncio.sleep(0.05)
            event = StreamEvent(token=word + " ", done=False).model_dump_json()
            yield {"data": event}
            
        final_event = StreamEvent(
            token="", 
            done=True, 
            citations=[c.model_dump() for c in agent_result.answer.citations], 
            tokens_used=agent_result.answer.tokens_used
        ).model_dump_json()
        yield {"data": final_event}
        
        ai_msg = Message(session_id=chat_session.id, role="assistant", content=agent_result.answer.answer, citations=[c.model_dump() for c in agent_result.answer.citations])
        db.add(ai_msg)
        await db.commit()

    return EventSourceResponse(event_generator())

@router.get("/sessions")
async def get_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all active sessions for the user."""
    result = await db.execute(select(Session).where(Session.user_id == current_user.id).order_by(Session.created_at.desc()))
    sessions = result.scalars().all()
    return [{"id": str(s.id), "title": s.title, "created_at": s.created_at} for s in sessions]

@router.get("/sessions/{session_id}/history")
async def get_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Fetch hot history from Redis."""
    try:
        session_id_uuid = uuid.UUID(session_id)
    except ValueError:
        return [] # Return empty history for invalid fallback sessions
        
    from app.services.memory.session_memory import session_memory
    
    result = await db.execute(select(Session).where(Session.id == session_id_uuid, Session.user_id == current_user.id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Session not found")
        
    history = await session_memory.load_history(str(session_id), max_turns=50)
    return history

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete session from Postgres and clear Redis memory."""
    from app.services.memory.session_memory import session_memory
    
    result = await db.execute(select(Session).where(Session.id == session_id, Session.user_id == current_user.id))
    chat_session = result.scalar_one_or_none()
    
    if not chat_session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    await session_memory.clear(str(session_id))
    await db.delete(chat_session)
    await db.commit()
    
    return {"status": "deleted"}
