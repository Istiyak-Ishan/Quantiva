import uuid
import asyncio
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.core.dependencies import get_db, get_current_user
from app.models.user import User
from app.models.document import Document
from app.models.chunk import Chunk
from app.schemas.document import DocumentResponse, DocumentUploadResponse, DocumentStatusResponse
from app.core.storage import upload_file, delete_file
from app.workers.ingestion_worker import process_document

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
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"File type {file.content_type} not supported")

    # Read file bytes (FastAPI does this asynchronously)
    file_bytes = await file.read()
    
    # 1. Upload to S3 (s3mock) synchronously
    object_key = upload_file(file_bytes, file.filename, file.content_type)
    
    # 2. Save metadata to PostgreSQL
    new_doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        file_type=file.content_type,
        file_size=len(file_bytes),
        storage_path=object_key,
        status="processing"
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)
    
    # 3. Trigger Background Celery Task
    process_document.delay(str(new_doc.id))
    
    return DocumentUploadResponse(
        id=new_doc.id,
        filename=new_doc.filename,
        status=new_doc.status
    )

@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Document).where(Document.user_id == current_user.id))
    documents = result.scalars().all()
    return documents

@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
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
        
    # Delete from S3
    delete_file(doc.storage_path)
    
    # Delete from DB
    await db.delete(doc)
    await db.commit()
    return None

@router.get("/{doc_id}/status")
async def get_document_status(
    doc_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    """Server-Sent Events (SSE) endpoint to stream document processing status."""

    async def event_generator():
        from app.db.database import AsyncSessionLocal
        while True:
            # Use a fresh session each iteration to avoid "another operation in progress" error
            async with AsyncSessionLocal() as fresh_db:
                result = await fresh_db.execute(
                    select(Document).where(
                        Document.id == doc_id,
                        Document.user_id == current_user.id
                    )
                )
                doc = result.scalar_one_or_none()

            if not doc:
                yield {"data": '{"error": "Document not found"}'}
                break

            data = DocumentStatusResponse(
                id=doc.id, status=doc.status, chunk_count=doc.chunk_count
            ).model_dump_json()
            yield {"data": data}

            if doc.status in ["ready", "failed"]:
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
