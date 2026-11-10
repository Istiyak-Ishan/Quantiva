import asyncio
import uuid
import json
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.dependencies import get_db, get_current_user
from app.core.ownership import owned_documents, owned_session, session_corpus
from app.core.rate_limiter import enforce_rate_limit
from app.models.user import User
from app.models.session import Session
from app.models.message import Message
from app.schemas.chat import StreamEvent
from app.schemas.agent import AgentAnswer
from app.services.agents.orchestrator import orchestrator
from app.services.guardrails.guardrails_service import GuardrailBlockError
from app.services.llm.errors import ProviderError
from app.services.memory.session_memory import session_memory
from app.core.metrics import ACTIVE_STREAMS

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


class CreateSessionRequest(BaseModel):
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)


class ChatMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: uuid.UUID
    query: str = Field(min_length=1, max_length=16000)


@router.post("/sessions", response_model=dict[str, str])
async def create_session(req: CreateSessionRequest, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    docs = await owned_documents(db, req.document_ids, current_user.id)
    # PostgreSQL UUID arrays accept UUIDs; JSON variant used in tests stores strings.
    session = Session(user_id=current_user.id, document_ids=[str(d.id) for d in docs])
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return {"session_id": str(session.id)}


async def _answer(query, session, current_user, db, *, on_token=None):
    doc_ids, chunks, names = await session_corpus(db, session, current_user.id)
    previous = list((await db.scalars(select(Message).where(Message.session_id == session.id).order_by(Message.created_at.desc()).limit(20))).all())
    history = [{"role": m.role, "content": m.content} for m in reversed(previous)]
    try:
        return await orchestrator.run(query=query, session_id=str(session.id), doc_ids=doc_ids, db_chunks=chunks, user_id=str(current_user.id), history=history, on_token=on_token, document_names=names)
    except GuardrailBlockError as exc:
        raise HTTPException(status_code=400, detail={"error": exc.message, "categories": exc.categories})
    except ProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": str(exc), "code": exc.code})


async def _persist_turn(db, session_id, query, answer):
    tokens = answer.tokens_used
    cost = 0.0 if answer.cache_hit else answer.cost_usd
    created_at = datetime.now(timezone.utc)
    db.add_all([
        Message(session_id=session_id, role="user", content=query, created_at=created_at),
        Message(session_id=session_id, role="assistant", content=answer.answer, citations=[c.model_dump() for c in answer.citations], tokens=tokens, cost_usd=cost, confidence=answer.confidence, chart_url=answer.chart_url, guardrail_flags=answer.guardrail_flags, input_tokens=answer.input_tokens, output_tokens=answer.output_tokens, model_used=answer.model_used, cache_hit=answer.cache_hit, created_at=created_at + timedelta(microseconds=1)),
    ])
    await db.commit()
    await session_memory.save_turn(str(session_id), {"role": "user", "content": query}, {"role": "assistant", "content": answer.answer})


@router.post("/message", response_model=AgentAnswer)
async def send_message(req: ChatMessageRequest, response: Response, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    session = await owned_session(db, req.session_id, current_user.id)
    await enforce_rate_limit(str(current_user.id), tier=current_user.tier)
    result = await _answer(req.query, session, current_user, db)
    await _persist_turn(db, session.id, req.query, result.answer)
    response.headers["X-Cache-Hit"] = str(result.answer.cache_hit).lower()
    return result.answer


@router.get("/message/stream")
async def stream_message(session_id: uuid.UUID, query: str = Query(min_length=1, max_length=16000), current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    session = await owned_session(db, session_id, current_user.id)
    await enforce_rate_limit(str(current_user.id), tier=current_user.tier)
    # Authorization and corpus checks must run before response headers are sent.
    await session_corpus(db, session, current_user.id)

    async def events():
        ACTIVE_STREAMS.inc()
        queue = asyncio.Queue(maxsize=16)

        async def token(delta):
            await queue.put({"data": StreamEvent(token=delta, done=False).model_dump_json()})

        async def produce():
            try:
                result = await _answer(query, session, current_user, db, on_token=token)
                await _persist_turn(db, session.id, query, result.answer)
                await queue.put({"data": StreamEvent(token="", done=True, **result.answer.model_dump(exclude={"structured_data", "reasoning_trace", "latency_ms", "model_used"})).model_dump_json()})
            except HTTPException as exc:
                await queue.put({"event": "error", "data": json.dumps({"error": exc.detail, "done": True})})
            except Exception:
                logger.exception("Chat stream failed")
                await db.rollback()
                await queue.put({"event": "error", "data": json.dumps({"error": "Chat could not be completed", "done": True})})
            finally:
                if not asyncio.current_task().cancelling():
                    await queue.put(None)

        task = asyncio.create_task(produce())
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            ACTIVE_STREAMS.dec()

    return EventSourceResponse(events())


@router.get("/sessions")
async def get_sessions(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sessions = (await db.scalars(select(Session).where(Session.user_id == current_user.id).order_by(Session.created_at.desc()))).all()
    return [{"id": str(s.id), "title": s.title, "created_at": s.created_at} for s in sessions]


@router.get("/sessions/{session_id}/history")
async def get_history(session_id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await owned_session(db, session_id, current_user.id)
    # Durable source of truth retains IDs, citations and timestamps. Redis is only
    # prompt memory, never an authorization boundary or an alternate conversation.
    messages = (await db.scalars(select(Message).where(Message.session_id == session_id).order_by(Message.created_at, Message.id))).all()
    return [{"id": str(m.id), "role": m.role, "content": m.content, "citations": m.citations or [], "confidence": m.confidence, "chart_url": m.chart_url, "guardrail_flags": m.guardrail_flags or [], "tokens": m.tokens, "input_tokens": m.input_tokens, "output_tokens": m.output_tokens, "model_used": m.model_used, "cache_hit": m.cache_hit, "cost_usd": m.cost_usd, "created_at": m.created_at.isoformat() if m.created_at else None} for m in messages]


@router.get("/sessions/{session_id}")
async def get_session(session_id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    session = await owned_session(db, session_id, current_user.id)
    return {"id": str(session.id), "document_ids": [str(d) for d in session.document_ids or []], "title": session.title}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    session = await owned_session(db, session_id, current_user.id)
    await db.delete(session)
    await db.commit()
    await session_memory.clear(str(session_id))
    return {"status": "deleted"}
