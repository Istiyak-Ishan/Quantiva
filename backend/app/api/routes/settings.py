"""
backend/app/api/routes/settings.py
────────────────────────────────────────────────────────────────────────────────
FastAPI routes for Retrieval Settings, Storage & Backup Operations (Prompt P4.3).
"""
from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.routes.dependencies import verify_desktop_token
from app.config import LocalDataPaths, settings
from app.core.model_studio import (
    ModelStudioService,
    get_model_studio_service,
)
from app.storage.backup import (
    create_backup,
    migrate_data_directory,
    restore_backup,
    verify_backup_archive,
)
from app.storage.database import SessionLocal, sqlite_database_path
from app.storage.models import Document, RetrievalConfiguration

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_desktop_token)])


# ── Pydantic Request & Response Schemas ────────────────────────────────────────

class RetrievalConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    search_mode: str = "hybrid"
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    enable_reranking: bool = True
    reranker_provider: str | None = None
    reranker_model: str | None = None
    final_top_k: int = 8
    enable_agentic_mode: bool = True
    max_agent_iterations: int = 5
    is_default: bool = True


class RetrievalConfigUpdate(BaseModel):
    name: str | None = None
    search_mode: str | None = None
    dense_top_k: int | None = Field(default=None, ge=1)
    sparse_top_k: int | None = Field(default=None, ge=1)
    rrf_k: int | None = Field(default=None, ge=1)
    dense_weight: float | None = Field(default=None, ge=0.0)
    sparse_weight: float | None = Field(default=None, ge=0.0)
    enable_reranking: bool | None = None
    reranker_provider: str | None = None
    reranker_model: str | None = None
    final_top_k: int | None = Field(default=None, ge=1)
    enable_agentic_mode: bool | None = None
    max_agent_iterations: int | None = Field(default=None, ge=1)


class StorageStatusResponse(BaseModel):
    current_data_dir: str
    database_path: str
    disk_total_bytes: int
    disk_free_bytes: int
    documents_count: int
    database_size_bytes: int
    vector_store_size_bytes: int


class StorageMigrateRequest(BaseModel):
    new_path: str = Field(min_length=1)
    verify_space: bool = True


class StorageMigrateResponse(BaseModel):
    status: str = "migrated"
    new_data_dir: str


class BackupCreateRequest(BaseModel):
    include_models: bool = False


class BackupCreateResponse(BaseModel):
    status: str = "created"
    archive_path: str
    filename: str
    size_bytes: int


class BackupInfoResponse(BaseModel):
    filename: str
    path: str
    size_bytes: int
    created_at: str
    schema_version: int | None = None
    app_version: str | None = None


class BackupRestoreRequest(BaseModel):
    archive_path: str = Field(min_length=1)
    create_safety_snapshot: bool = True


class BackupRestoreResponse(BaseModel):
    status: str = "restored"
    archive_path: str


# ── Retrieval Settings Endpoints ──────────────────────────────────────────────

@router.get(
    "/retrieval",
    response_model=RetrievalConfigResponse,
    summary="Get Active Retrieval & Reranker Configuration",
)
async def get_retrieval_settings(
    service: ModelStudioService = Depends(get_model_studio_service),
) -> RetrievalConfigResponse:
    service.ensure_presets()
    cfg = service.get_default_retrieval_config()
    if cfg is None:
        configs = service.list_retrieval_configs()
        if configs:
            cfg = configs[0]
        else:
            raise HTTPException(status_code=404, detail="No retrieval configuration found.")
    return RetrievalConfigResponse.model_validate(cfg)


@router.put(
    "/retrieval",
    response_model=RetrievalConfigResponse,
    summary="Update Active Retrieval & Reranker Configuration",
)
async def update_retrieval_settings(
    payload: RetrievalConfigUpdate,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> RetrievalConfigResponse:
    service.ensure_presets()
    cfg = service.get_default_retrieval_config()
    if cfg is None:
        configs = service.list_retrieval_configs()
        if configs:
            cfg = configs[0]
        else:
            raise HTTPException(status_code=404, detail="No retrieval configuration found to update.")

    updated = service.update_retrieval_config(cfg.id, payload.model_dump(exclude_unset=True))
    return RetrievalConfigResponse.model_validate(updated)


# ── Storage Settings Endpoints ────────────────────────────────────────────────

@router.get(
    "/storage",
    response_model=StorageStatusResponse,
    summary="Get Workspace Storage Usage & Directory Status",
)
async def get_storage_settings() -> StorageStatusResponse:
    root = Path(settings.QUANTIVA_DATA_DIR).expanduser().resolve()
    paths = LocalDataPaths(root)

    # Disk usage
    total_bytes = 0
    free_bytes = 0
    try:
        usage = shutil.disk_usage(str(root))
        total_bytes = usage.total
        free_bytes = usage.free
    except Exception:
        pass

    # Database file size
    db_size = 0
    if paths.database_file.is_file():
        try:
            db_size = paths.database_file.stat().st_size
        except OSError:
            pass

    # Vector store directory size
    vector_size = 0
    if paths.vector_store.is_dir():
        for p in paths.vector_store.rglob("*"):
            if p.is_file():
                try:
                    vector_size += p.stat().st_size
                except OSError:
                    pass

    # Documents count
    doc_count = 0
    try:
        with SessionLocal() as session:
            doc_count = session.query(Document).count()
    except Exception:
        pass

    return StorageStatusResponse(
        current_data_dir=str(root),
        database_path=str(paths.database_file),
        disk_total_bytes=total_bytes,
        disk_free_bytes=free_bytes,
        documents_count=doc_count,
        database_size_bytes=db_size,
        vector_store_size_bytes=vector_size,
    )


@router.post(
    "/storage/migrate",
    response_model=StorageMigrateResponse,
    summary="Migrate Workspace Data Directory to New Location",
)
async def migrate_storage_location(
    payload: StorageMigrateRequest,
) -> StorageMigrateResponse:
    try:
        new_dir = migrate_data_directory(
            payload.new_path,
            verify_space=payload.verify_space,
        )
        return StorageMigrateResponse(
            status="migrated",
            new_data_dir=str(new_dir),
        )
    except Exception as exc:
        logger.exception("Storage migration failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Backup & Restore Endpoints ────────────────────────────────────────────────

@router.post(
    "/backups/create",
    response_model=BackupCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Atomic Workspace Backup Archive",
)
async def create_workspace_backup(
    payload: BackupCreateRequest,
) -> BackupCreateResponse:
    try:
        archive_path = create_backup(include_models=payload.include_models)
        return BackupCreateResponse(
            status="created",
            archive_path=str(archive_path),
            filename=archive_path.name,
            size_bytes=archive_path.stat().st_size,
        )
    except Exception as exc:
        logger.exception("Failed creating workspace backup")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/backups",
    response_model=list[BackupInfoResponse],
    summary="List Available Workspace Backup Archives",
)
async def list_workspace_backups() -> list[BackupInfoResponse]:
    root = Path(settings.QUANTIVA_DATA_DIR).expanduser().resolve()
    backups_dir = LocalDataPaths(root).backups

    if not backups_dir.is_dir():
        return []

    results: list[BackupInfoResponse] = []
    for archive in sorted(backups_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not archive.is_file():
            continue

        schema_ver = None
        app_ver = None
        created_at_str = None
        try:
            manifest = verify_backup_archive(archive)
            schema_ver = manifest.get("schema_version")
            app_ver = manifest.get("app_version")
            created_at_str = manifest.get("created_at")
        except Exception:
            pass

        if not created_at_str:
            from datetime import datetime, timezone
            mtime = archive.stat().st_mtime
            created_at_str = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        results.append(
            BackupInfoResponse(
                filename=archive.name,
                path=str(archive),
                size_bytes=archive.stat().st_size,
                created_at=created_at_str,
                schema_version=schema_ver,
                app_version=app_ver,
            )
        )

    return results


@router.post(
    "/backups/restore",
    response_model=BackupRestoreResponse,
    summary="Restore Workspace from Verified Backup Archive",
)
async def restore_workspace_backup(
    payload: BackupRestoreRequest,
) -> BackupRestoreResponse:
    archive = Path(payload.archive_path).expanduser().resolve()
    if not archive.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Backup archive not found: {payload.archive_path}",
        )

    try:
        restored_path = restore_backup(
            archive,
            create_safety_snapshot=payload.create_safety_snapshot,
        )
        return BackupRestoreResponse(
            status="restored",
            archive_path=str(restored_path),
        )
    except Exception as exc:
        logger.exception("Restore operation failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
