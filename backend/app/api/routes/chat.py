"""
backend/app/api/routes/chat.py
────────────────────────────────────────────────────────────────────────────────
FastAPI routes for Conversation & RAG Chat management (Prompt P5.3).
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.routes.dependencies import verify_desktop_token
from app.core.application_context import get_application_context
from app.core.orchestrator import (
    RAGOrchestrator,
    RAGResponse,
    format_sse,
)
from app.providers.interfaces.types import ChatMessage, ChatRole
from app.providers.registry import IndexMetadata as ProviderIndexMetadata
from app.storage.database import SessionLocal, engine as local_database_engine
from app.storage.models import Conversation, IndexMetadata, KnowledgeBase, Message, ModelConfiguration
from app.storage.sparse_store import SQLiteSparseIndex
from app.storage.vector_store import LocalVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Chat"], dependencies=[Depends(verify_desktop_token)])


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    knowledge_base_id: str
    title: str = Field(default="New Conversation", max_length=512)
    model_config_id: str | None = None
    is_pinned: bool = False


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    stream: bool = False
    mode: str | None = Field(default=None, description="'direct' or 'agentic'")
    model_config_id: str | None = None
    filters: dict[str, Any] | None = None


def _serialize_conversation(conv: Conversation) -> dict[str, Any]:
    return {
        "id": conv.id,
        "knowledge_base_id": conv.knowledge_base_id,
        "title": conv.title,
        "model_config_id": conv.model_config_id,
        "is_pinned": conv.is_pinned,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "message_count": len(conv.messages) if conv.messages else 0,
    }


def _serialize_message(msg: Message) -> dict[str, Any]:
    return {
        "id": msg.id,
        "conversation_id": msg.conversation_id,
        "role": msg.role,
        "content": msg.content,
        "citations": msg.citations_json or [],
        "reasoning_trace": msg.reasoning_trace_json,
        "retrieved_chunks": msg.retrieved_chunks_json or [],
        "token_usage": msg.token_usage_json or {},
        "latency_ms": msg.latency_ms,
        "provider_used": msg.provider_used,
        "model_used": msg.model_used,
        "created_at": msg.created_at,
    }


# ── Conversation CRUD Endpoints ───────────────────────────────────────────────

@router.get("", response_model=list[dict[str, Any]])
def list_conversations(
    kb_id: str | None = Query(default=None, alias="kb_id"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    """List conversations, optionally filtered by knowledge_base_id, ordered by pin status & activity."""
    with SessionLocal() as session:
        stmt = (
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .order_by(Conversation.is_pinned.desc(), Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        if kb_id:
            stmt = stmt.where(Conversation.knowledge_base_id == kb_id)

        convs = session.scalars(stmt).all()
        return [_serialize_conversation(c) for c in convs]


@router.post("", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
def create_conversation(payload: ConversationCreate) -> dict[str, Any]:
    """Create a new conversation bound to a knowledge base."""
    with SessionLocal() as session:
        kb = session.get(KnowledgeBase, payload.knowledge_base_id)
        if kb is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base not found: {payload.knowledge_base_id}",
            )

        model_id = payload.model_config_id
        if not model_id:
            default_model = session.scalar(
                select(ModelConfiguration).where(ModelConfiguration.is_default == True)
            )
            if default_model:
                model_id = default_model.id

        conv = Conversation(
            knowledge_base_id=payload.knowledge_base_id,
            title=payload.title.strip() or "New Conversation",
            model_config_id=model_id,
            is_pinned=payload.is_pinned,
        )
        session.add(conv)
        session.commit()
        session.refresh(conv)
        return _serialize_conversation(conv)


@router.get("/{conv_id}/messages", response_model=list[dict[str, Any]])
def get_conversation_messages(
    conv_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    """Retrieve full message history for a conversation in chronological order."""
    with SessionLocal() as session:
        conv = session.get(Conversation, conv_id)
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conv_id}",
            )

        messages = session.scalars(
            select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(Message.created_at.asc())
            .offset(offset)
            .limit(limit)
        ).all()
        return [_serialize_message(m) for m in messages]


@router.delete("/{conv_id}")
def delete_conversation(conv_id: str) -> dict[str, Any]:
    """Delete a conversation and its messages."""
    with SessionLocal() as session:
        conv = session.get(Conversation, conv_id)
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conv_id}",
            )
        session.delete(conv)
        session.commit()
    return {"success": True, "id": conv_id, "message": "Conversation deleted."}


# ── Chat Execution Endpoint ───────────────────────────────────────────────────

@router.post("/{conv_id}/ask")
async def ask_question(
    conv_id: str,
    payload: AskRequest,
) -> Any:
    """Execute a RAG query within a conversation, persisting citations and turns."""
    query_clean = payload.query.strip()
    if not query_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    with SessionLocal() as session:
        conv = session.get(Conversation, conv_id)
        if conv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conv_id}",
            )
        kb_id = conv.knowledge_base_id

        kb = session.get(KnowledgeBase, kb_id)
        if kb is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Associated knowledge base not found: {kb_id}",
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

        # Load conversation history for contextual memory
        history_records = session.scalars(
            select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(Message.created_at.asc())
        ).all()
        history = [
            ChatMessage(
                role=ChatRole.USER if m.role == "user" else ChatRole.ASSISTANT,
                content=m.content,
            )
            for m in history_records
            if m.role in ("user", "assistant")
        ]

        # Record user turn immediately
        user_msg = Message(
            conversation_id=conv_id,
            role="user",
            content=query_clean,
        )
        session.add(user_msg)
        session.commit()

    # Build RAG Orchestrator
    app_ctx = get_application_context()
    registry = app_ctx.provider_registry
    vector_store = LocalVectorStore()
    sparse_store = SQLiteSparseIndex(local_database_engine)

    orchestrator = RAGOrchestrator(
        provider_registry=registry,
        vector_store=vector_store,
        sparse_index=sparse_store,
        index_metadata=prov_index_meta,
        default_mode=payload.mode or "direct",
    )

    llm = orchestrator.get_llm_provider()
    provider_name = llm.provider_name() if hasattr(llm, "provider_name") else "unknown"
    model_name = llm.model_name() if hasattr(llm, "model_name") else "unknown"

    # Non-streaming JSON response
    if not payload.stream:
        response: RAGResponse = await orchestrator.query(
            query=query_clean,
            knowledge_base_id=kb_id,
            collection_name=collection_name,
            filters=payload.filters,
            history=history,
            mode=payload.mode or "direct",
        )

        citations_list = [
            asdict(c) if hasattr(c, "__dataclass_fields__") else dict(c)
            for c in response.citations
        ]
        sources_list = [
            asdict(s) if hasattr(s, "__dataclass_fields__") else dict(s)
            for s in response.sources
        ]
        usage_dict = asdict(response.usage) if response.usage else {}

        with SessionLocal() as session:
            asst_msg = Message(
                conversation_id=conv_id,
                role="assistant",
                content=response.answer,
                citations_json=citations_list,
                retrieved_chunks_json=sources_list,
                reasoning_trace_json={"reasoning": response.reasoning} if response.reasoning else None,
                token_usage_json=usage_dict,
                latency_ms=int(response.latency_ms),
                provider_used=provider_name,
                model_used=model_name,
            )
            session.add(asst_msg)

            target_conv = session.get(Conversation, conv_id)
            if target_conv:
                target_conv.updated_at = datetime.now(timezone.utc)
            session.commit()

        return {
            "conversation_id": conv_id,
            "answer": response.answer,
            "citations": citations_list,
            "sources": sources_list,
            "reasoning": response.reasoning,
            "usage": usage_dict,
            "latency_ms": response.latency_ms,
            "provider_used": provider_name,
            "model_used": model_name,
        }

    # Streaming SSE response
    async def sse_event_stream():
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        collected_sources: list[dict[str, Any]] = []
        collected_citations: list[dict[str, Any]] = []
        collected_usage: dict[str, Any] = {}
        latency_ms = 0.0

        try:
            async for event in orchestrator.stream_query(
                query=query_clean,
                knowledge_base_id=kb_id,
                collection_name=collection_name,
                filters=payload.filters,
                history=history,
                mode=payload.mode or "direct",
            ):
                yield format_sse(event)
                event_type = event.get("event")
                data = event.get("data", {})
                if event_type == "token_delta":
                    answer_parts.append(data.get("text", ""))
                elif event_type == "reasoning_delta":
                    reasoning_parts.append(data.get("text", ""))
                elif event_type == "sources":
                    collected_sources = data if isinstance(data, list) else []
                elif event_type == "citations":
                    collected_citations = data if isinstance(data, list) else []
                elif event_type == "done":
                    collected_usage = data.get("usage", {})
                    latency_ms = float(data.get("latency_ms", 0.0))

            full_answer = "".join(answer_parts)
            full_reasoning = "".join(reasoning_parts) if reasoning_parts else None

            # Persist assistant turn to database
            with SessionLocal() as session:
                asst_msg = Message(
                    conversation_id=conv_id,
                    role="assistant",
                    content=full_answer,
                    citations_json=collected_citations,
                    retrieved_chunks_json=collected_sources,
                    reasoning_trace_json={"reasoning": full_reasoning} if full_reasoning else None,
                    token_usage_json=collected_usage,
                    latency_ms=int(latency_ms),
                    provider_used=provider_name,
                    model_used=model_name,
                )
                session.add(asst_msg)

                target_conv = session.get(Conversation, conv_id)
                if target_conv:
                    target_conv.updated_at = datetime.now(timezone.utc)
                session.commit()
        except Exception as exc:
            logger.exception("Error in SSE chat stream: %s", exc)
            yield format_sse({"event": "error", "data": {"message": str(exc)}})

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")
