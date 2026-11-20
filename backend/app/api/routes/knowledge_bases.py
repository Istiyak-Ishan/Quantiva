"""
backend/app/api/routes/knowledge_bases.py
────────────────────────────────────────────────────────────────────────────────
FastAPI routes for Quantiva Knowledge Base management (Prompt P5.3).
"""
from __future__ import annotations

from datetime import datetime
import io
import json
import logging
from pathlib import Path
from typing import Any
import zipfile

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.api.routes.dependencies import verify_desktop_token
from app.core.knowledge_bases import (
    KnowledgeBaseDetails,
    KnowledgeBaseService,
    KnowledgeBaseSummary,
    get_knowledge_base_service,
)
from app.storage.database import SessionLocal
from app.storage.models import Document, IndexMetadata, KnowledgeBase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Bases"], dependencies=[Depends(verify_desktop_token)])


# ── Pydantic Request & Response Schemas ────────────────────────────────────────

class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    embedding_config_id: str | None = None
    retrieval_config_id: str | None = None


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    description: str | None = None
    embedding_config_id: str | None = None
    retrieval_config_id: str | None = None
    trigger_reindex: bool = False


class ReindexRequest(BaseModel):
    new_embedding_config_id: str | None = None


# ── Knowledge Base Endpoints ───────────────────────────────────────────────────

@router.get("", response_model=list[dict[str, Any]])
def list_knowledge_bases(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> list[dict[str, Any]]:
    """List all knowledge bases with aggregated statistics."""
    items = service.list_knowledge_bases()
    return [item.to_dict() for item in items[offset : offset + limit]]


@router.post("", response_model=dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> dict[str, Any]:
    """Create a new Knowledge Base with locked embedding dimensions and dedicated vector collection."""
    try:
        kb = await service.create_knowledge_base(
            name=payload.name,
            description=payload.description,
            embedding_config_id=payload.embedding_config_id,
            retrieval_config_id=payload.retrieval_config_id,
        )
        details = await service.get_knowledge_base_details(kb.id)
        return details.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("/{kb_id}", response_model=dict[str, Any])
async def get_knowledge_base(
    kb_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> dict[str, Any]:
    """Retrieve detailed diagnostics and index health for a specific knowledge base."""
    try:
        details = await service.get_knowledge_base_details(kb_id)
        return details.to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Knowledge base not found: {kb_id}")


@router.put("/{kb_id}", response_model=dict[str, Any])
async def update_knowledge_base(
    kb_id: str,
    payload: KnowledgeBaseUpdate,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> dict[str, Any]:
    """Update Knowledge Base metadata and detect embedding configuration divergence."""
    try:
        kb = await service.update_knowledge_base(
            knowledge_base_id=kb_id,
            name=payload.name,
            description=payload.description,
            embedding_config_id=payload.embedding_config_id,
            retrieval_config_id=payload.retrieval_config_id,
            trigger_reindex=payload.trigger_reindex,
        )
        details = await service.get_knowledge_base_details(kb_id)
        return details.to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> dict[str, Any]:
    """Completely purge a knowledge base across vector store, FTS5, files, and SQLite records."""
    success = await service.delete_knowledge_base(kb_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Knowledge base not found: {kb_id}")
    return {"success": True, "id": kb_id, "message": "Knowledge base and associated assets deleted cleanly."}


@router.post("/{kb_id}/reindex")
async def reindex_knowledge_base(
    kb_id: str,
    payload: ReindexRequest = ReindexRequest(),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> dict[str, Any]:
    """Trigger an asynchronous shadow-collection blue-green reindex job."""
    try:
        job_id = await service.trigger_reindex(
            knowledge_base_id=kb_id,
            new_embedding_config_id=payload.new_embedding_config_id,
        )
        return {
            "job_id": job_id,
            "knowledge_base_id": kb_id,
            "status": "QUEUED",
            "message": "Reindex job submitted successfully.",
        }
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


# ── Knowledge Base Export & Import ─────────────────────────────────────────────

@router.post("/{kb_id}/export")
async def export_knowledge_base(
    kb_id: str,
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> Response:
    """Export knowledge base metadata and original documents into a portable ZIP package."""
    try:
        details = await service.get_knowledge_base_details(kb_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Knowledge base not found: {kb_id}")

    docs = service.list_documents(kb_id, limit=10000)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write manifest
        manifest_data = {
            "version": 1,
            "exported_at": datetime.now().isoformat(),
            "knowledge_base": details.to_dict(),
            "documents": [d.to_dict() for d in docs],
        }
        zf.writestr("manifest.json", json.dumps(manifest_data, indent=2, default=str))

        # Write original files
        for doc in docs:
            try:
                file_bytes = service.file_store.read_original(kb_id, doc.id, doc.filename)
                zf.writestr(f"files/{doc.id}_{doc.filename}", file_bytes)
            except Exception as exc:
                logger.warning("Could not read original file for doc %s during export: %s", doc.id, exc)

    zip_bytes = zip_buffer.getvalue()
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in details.name)
    filename = f"quantiva_kb_{safe_name}_{kb_id[:8]}.zip"

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/import", status_code=status.HTTP_201_CREATED)
async def import_knowledge_base(
    file: UploadFile = File(...),
    service: KnowledgeBaseService = Depends(get_knowledge_base_service),
) -> dict[str, Any]:
    """Import a knowledge base package from a ZIP archive."""
    contents = await file.read()
    if not zipfile.is_zipfile(io.BytesIO(contents)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is not a valid ZIP archive.")

    try:
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zf:
            if "manifest.json" not in zf.namelist():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid package: manifest.json missing.")

            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            kb_data = manifest.get("knowledge_base", {})
            base_name = kb_data.get("name", "Imported Knowledge Base")
            target_name = f"{base_name} (Imported)"

            # Ensure unique name
            existing_names = {s.name for s in service.list_knowledge_bases()}
            candidate_name = target_name
            counter = 1
            while candidate_name in existing_names:
                candidate_name = f"{target_name} {counter}"
                counter += 1

            new_kb = await service.create_knowledge_base(
                name=candidate_name,
                description=kb_data.get("description"),
                embedding_config_id=kb_data.get("embedding_config_id"),
                retrieval_config_id=kb_data.get("retrieval_config_id"),
            )

            # Restore documents
            restored_docs = 0
            for doc_item in manifest.get("documents", []):
                doc_id = doc_item.get("id")
                filename = doc_item.get("filename", "document.bin")
                file_key = f"files/{doc_id}_{filename}"
                if file_key in zf.namelist():
                    doc_bytes = zf.read(file_key)
                    try:
                        await service.upload_document(
                            knowledge_base_id=new_kb.id,
                            file_content=doc_bytes,
                            filename=filename,
                            file_type=doc_item.get("file_type"),
                            allow_overwrite=True,
                            metadata_json=doc_item.get("metadata", {}),
                        )
                        restored_docs += 1
                    except Exception as exc:
                        logger.warning("Failed restoring document %s during KB import: %s", filename, exc)

        details = await service.get_knowledge_base_details(new_kb.id)
        result = details.to_dict()
        result["restored_documents"] = restored_docs
        return result
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        logger.exception("Error importing knowledge base: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to import knowledge base: {exc}")
