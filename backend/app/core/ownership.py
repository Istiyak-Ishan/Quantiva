"""Database authorization helpers. All IDs are validated before reaching these."""
import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.chunk import Chunk
from app.models.session import Session
from app.config import settings


def require_current_index(docs, chunks):
    if any(c.full_text is None for c in chunks):
        raise HTTPException(status_code=409, detail="Legacy document chunks require reindexing with the supported embedding model")
    populated = {c.doc_id for c in chunks}
    for doc in docs:
        metadata = doc.metadata_json or {}
        if doc.id in populated and metadata.get("embedding_collection") != settings.embedding_collection:
            raise HTTPException(status_code=409, detail="Document embedding configuration changed; explicit reindexing is required")


async def owned_session(db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID) -> Session:
    session = await db.scalar(select(Session).where(Session.id == session_id, Session.user_id == user_id))
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


async def owned_documents(db: AsyncSession, doc_ids: list[uuid.UUID], user_id: uuid.UUID) -> list[Document]:
    if not doc_ids:
        return []
    doc_ids = [uuid.UUID(str(d)) for d in doc_ids]
    docs = list((await db.scalars(select(Document).where(Document.id.in_(doc_ids), Document.user_id == user_id))).all())
    if {d.id for d in docs} != set(doc_ids):
        raise HTTPException(status_code=404, detail="Document not found")
    return docs


async def session_corpus(db: AsyncSession, session: Session, user_id: uuid.UUID):
    docs = await owned_documents(db, session.document_ids or [], user_id)
    if docs and any(d.status != "ready" for d in docs):
        raise HTTPException(status_code=409, detail="Selected documents are not ready")
    if not session.document_ids:
        docs = list((await db.scalars(select(Document).where(Document.user_id == user_id, Document.status == "ready"))).all())
    ids = [d.id for d in docs]
    chunks = list((await db.scalars(select(Chunk).join(Document, Chunk.doc_id == Document.id).where(Chunk.doc_id.in_(ids), Document.user_id == user_id).order_by(Chunk.doc_id, Chunk.chunk_index))).all()) if ids else []
    require_current_index(docs, chunks)
    return [str(d) for d in ids], chunks, {str(d.id): d.filename for d in docs}
