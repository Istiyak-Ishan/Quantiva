import uuid
import asyncio
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import Response
from urllib.parse import quote

from app.config import settings
from app.core.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.document import Document
from app.models.chunk import Chunk
from app.schemas.document import DocumentResponse, DocumentUploadResponse, DocumentStatusResponse
from app.core.storage import delete_file, StorageService
from app.workers.ingestion_worker import process_document
from app.core.rate_limiter import enforce_rate_limit
from app.services.ingestion.upload_validation import read_validated_upload
from app.services.ingestion.status_monitor import (
    WORKER_UNAVAILABLE_REASON,
    fail_stale_documents,
    mark_failed,
    processing_age_seconds,
    worker_is_available,
)
from app.services.retrieval.vector_store import vector_store
from app.services.memory.semantic_cache import semantic_cache

ingest_document_task = process_document

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_TYPES = [
    "application/pdf", 
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", # xlsx
    "text/csv", 
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # docx
    "text/plain"
]

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    await enforce_rate_limit(str(current_user.id), tier=current_user.tier, scope="upload")
    file_bytes, filename, content_type = await read_validated_upload(file)
    object_key = await asyncio.to_thread(StorageService.upload_file, file_bytes, filename, content_type)
    
    # 2. Save metadata to PostgreSQL
    new_doc = Document(
        user_id=current_user.id,
        filename=filename,
        file_type=content_type,
        file_size=len(file_bytes),
        storage_path=object_key,
        status="processing",
        processing_started_at=datetime.now(timezone.utc),
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)
    
    # 3. Trigger Background Ingestion Task
    try:
        await asyncio.to_thread(process_document.delay, str(new_doc.id))
    except Exception:
        new_doc.status = "failed"
        new_doc.failure_reason = "Ingestion queue unavailable; document was not processed. Retry indexing."
        new_doc.processing_started_at = None
        await db.commit()
        raise HTTPException(status_code=503, detail={"error": "Ingestion queue unavailable; document was not processed", "document_id": str(new_doc.id)})
    
    return DocumentUploadResponse(
        id=new_doc.id,
        filename=new_doc.filename,
        status=new_doc.status
    )


@router.post("/{doc_id}/reindex", response_model=DocumentUploadResponse)
async def reindex_document(doc_id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    doc = await db.scalar(select(Document).where(Document.id == doc_id, Document.user_id == current_user.id).with_for_update())
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status not in {"ready", "failed"}:
        raise HTTPException(status_code=409, detail="Document is already processing or deleting")
    await enforce_rate_limit(str(current_user.id), tier=current_user.tier, scope="upload")
    doc.status = "processing"
    doc.failure_reason = None
    doc.processing_started_at = datetime.now(timezone.utc)
    await db.commit()
    try:
        await asyncio.to_thread(process_document.delay, str(doc.id))
    except Exception:
        doc.status = "failed"
        doc.failure_reason = "Ingestion queue unavailable; retry indexing."
        doc.processing_started_at = None
        await db.commit()
        raise HTTPException(status_code=503, detail="Ingestion queue unavailable")
    return DocumentUploadResponse(id=doc.id, filename=doc.filename, status=doc.status)

@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Recover rows orphaned by a stopped/crashed worker so the UI never shows
    # an endless "Indexing…" badge after a reload.
    await fail_stale_documents(db, user_id=current_user.id)
    result = await db.execute(select(Document).where(Document.user_id == current_user.id))
    documents = result.scalars().all()
    return documents

@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    await fail_stale_documents(db, user_id=current_user.id)
    result = await db.execute(select(Document).where(Document.id == doc_id, Document.user_id == current_user.id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Document).where(Document.id == doc_id, Document.user_id == current_user.id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Mark unavailable before external cleanup. Failure leaves a retryable row,
    # not a ready document that can continue serving deleted content.
    doc.status = "deleting"
    await db.commit()
    try:
        await vector_store.delete_document(str(doc_id), user_id=str(current_user.id))
        await semantic_cache.invalidate(str(doc_id), user_id=str(current_user.id))
        await asyncio.to_thread(delete_file, doc.storage_path)
    except Exception:
        raise HTTPException(status_code=503, detail="Document cleanup incomplete; retry deletion")
    await db.delete(doc)
    await db.commit()
    return None


@router.get("/{doc_id}/download")
async def download_document(doc_id: uuid.UUID, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    doc = await db.scalar(select(Document).where(Document.id == doc_id, Document.user_id == current_user.id))
    if doc is None or doc.status == "deleting":
        raise HTTPException(status_code=404, detail="Document not found")
    await enforce_rate_limit(str(current_user.id), tier=current_user.tier, scope="download")
    try:
        data = await asyncio.to_thread(StorageService.get_file, doc.storage_path)
    except Exception:
        raise HTTPException(status_code=503, detail="Document storage unavailable")
    return Response(data, media_type=doc.file_type, headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(doc.filename, safe='')}", "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})

@router.get("/{doc_id}/status")
async def get_document_status(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Server-Sent Events (SSE) endpoint to stream document processing status."""

    exists = await db.scalar(select(Document).where(Document.id == doc_id, Document.user_id == current_user.id))
    if exists is None:
        raise HTTPException(status_code=404, detail="Document not found")

    async def event_generator():
        from app.db.database import AsyncSessionLocal

        # Probe worker liveness after the grace period and then once a minute.
        # This turns a missing worker into a prompt, actionable failure instead
        # of an endless spinner. Active rows are additionally protected by the
        # worker's row lock, so the timeout sweep never interrupts real work.
        next_worker_probe = float(settings.INGESTION_WORKER_GRACE_SECONDS)
        last_sweep = 0.0
        while True:
            async with AsyncSessionLocal() as fresh_db:
                result = await fresh_db.execute(
                    select(Document).where(
                        Document.id == doc_id,
                        Document.user_id == current_user.id
                    )
                )
                doc = result.scalar_one_or_none()

                if doc is None:
                    yield {"data": '{"error": "Document not found"}'}
                    break

                if doc.status == "processing":
                    age = processing_age_seconds(doc)
                    if age >= next_worker_probe:
                        next_worker_probe = age + 60
                        if not await worker_is_available():
                            await mark_failed(fresh_db, doc, WORKER_UNAVAILABLE_REASON)

                    now = asyncio.get_running_loop().time()
                    if now - last_sweep >= 15:
                        last_sweep = now
                        if await fail_stale_documents(fresh_db, user_id=current_user.id):
                            doc = await fresh_db.scalar(
                                select(Document).where(
                                    Document.id == doc_id,
                                    Document.user_id == current_user.id,
                                )
                            )

                status, chunk_count, failure_reason = doc.status, doc.chunk_count, doc.failure_reason

            data = DocumentStatusResponse(
                id=doc.id,
                status=status,
                chunk_count=chunk_count,
                failure_reason=failure_reason,
            ).model_dump_json()
            yield {"data": data}

            if status in ["ready", "failed"]:
                break

            await asyncio.sleep(1)  # Poll every 1 second

    return EventSourceResponse(event_generator())

@router.get("/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Debug endpoint to list all chunks for a document."""
    # First verify document belongs to user
    doc_result = await db.execute(select(Document).where(Document.id == doc_id, Document.user_id == current_user.id))
    if not doc_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found")
        
    result = await db.execute(select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index))
    chunks = result.scalars().all()
    
    return [
        {
            "id": c.id,
            "chunk_index": c.chunk_index,
            "text_preview": c.text_preview,
            "page_number": c.page_number,
            "chunk_type": c.chunk_type,
            "qdrant_id": c.qdrant_id
        }
        for c in chunks
    ]
