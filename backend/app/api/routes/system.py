"""
backend/app/api/routes/system.py
────────────────────────────────────────────────────────────────────────────────
FastAPI routes for System Health and Hardware Inspection (Prompt P4.3).
"""
from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.api.routes.dependencies import verify_desktop_token
from app.config import settings
from app.core.application_context import get_application_context
from app.core.hardware import HardwareInspector, HardwareProfile, ModelTierRecommendation
from app.storage.database import sqlite_database_path

router = APIRouter()


# ── Pydantic Request & Response Schemas ────────────────────────────────────────

class SystemHealthResponse(BaseModel):
    status: str
    app_version: str
    startup_status: str
    sqlite_status: str
    vector_store_status: str
    data_directory: str
    database_path: str


class GPUInfoResponse(BaseModel):
    name: str
    vram_total_mb: int
    vram_total_gb: float
    vram_free_mb: int | None = None


class HardwareProfileResponse(BaseModel):
    cpu_model: str
    physical_cores: int
    logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    gpus: list[GPUInfoResponse] = Field(default_factory=list)
    detection_errors: list[str] = Field(default_factory=list)


class ModelTierRecommendationResponse(BaseModel):
    tier: str
    description: str
    llm_models: list[str] = Field(default_factory=list)
    embedding_models: list[str] = Field(default_factory=list)
    reranker_models: list[str] = Field(default_factory=list)
    rationale: str


class SystemHardwareResponse(BaseModel):
    hardware: HardwareProfileResponse
    recommendation: ModelTierRecommendationResponse


# ── System Endpoints ──────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=SystemHealthResponse,
    summary="Desktop Backend Process & Subsystem Health Check",
)
async def get_system_health() -> SystemHealthResponse:
    """Check desktop subsystem health.

    EXEMPT from X-Quantiva-Token header per AGENTS.md Rule 6.
    """
    db_path = sqlite_database_path()
    sqlite_status = "unknown"
    try:
        if db_path.is_file():
            with sqlite3.connect(str(db_path), timeout=3.0) as conn:
                conn.execute("SELECT 1;").fetchone()
            sqlite_status = "connected"
        else:
            sqlite_status = "not_initialized"
    except Exception as exc:  # noqa: BLE001
        sqlite_status = f"error: {exc}"

    # Vector store status
    app_ctx = get_application_context()
    vector_client = app_ctx.get_vector_client()
    vector_status = "connected" if vector_client is not None else "ready"

    overall_status = "ok" if sqlite_status in ("connected", "not_initialized") else "degraded"

    return SystemHealthResponse(
        status=overall_status,
        app_version=settings.APP_VERSION,
        startup_status="ready",
        sqlite_status=sqlite_status,
        vector_store_status=vector_status,
        data_directory=str(settings.QUANTIVA_DATA_DIR),
        database_path=str(db_path),
    )


@router.get(
    "/hardware",
    response_model=SystemHardwareResponse,
    dependencies=[Depends(verify_desktop_token)],
    summary="Local Hardware Profile & Recommended Model Tier",
)
async def get_system_hardware() -> SystemHardwareResponse:
    """Inspect local CPU, RAM, disk, GPU/VRAM, and recommend model tiers."""
    inspector = HardwareInspector(data_dir=settings.QUANTIVA_DATA_DIR)
    report = inspector.full_report()

    hw = report["hardware"]
    rec = report["recommendation"]

    gpu_list = [
        GPUInfoResponse(
            name=g["name"],
            vram_total_mb=g["vram_total_mb"],
            vram_total_gb=g["vram_total_gb"],
            vram_free_mb=g.get("vram_free_mb"),
        )
        for g in hw.get("gpus", [])
    ]

    hw_resp = HardwareProfileResponse(
        cpu_model=hw["cpu_model"],
        physical_cores=hw["physical_cores"],
        logical_cores=hw["logical_cores"],
        ram_total_gb=hw["ram_total_gb"],
        ram_available_gb=hw["ram_available_gb"],
        disk_free_gb=hw["disk_free_gb"],
        gpus=gpu_list,
        detection_errors=hw.get("detection_errors", []),
    )

    rec_resp = ModelTierRecommendationResponse(
        tier=rec["tier"],
        description=rec["description"],
        llm_models=rec.get("llm_models", []),
        embedding_models=rec.get("embedding_models", []),
        reranker_models=rec.get("reranker_models", []),
        rationale=rec["rationale"],
    )

    return SystemHardwareResponse(
        hardware=hw_resp,
        recommendation=rec_resp,
    )


__all__ = ["router"]
