"""
backend/app/api/routes/models.py
────────────────────────────────────────────────────────────────────────────────
FastAPI routes for Quantiva Model Studio (Prompt P4.3).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.routes.dependencies import verify_desktop_token
from app.core.model_studio import (
    ModelStudioService,
    get_model_studio_service,
)
from app.providers.interfaces.types import ConnectionTestResult

router = APIRouter(dependencies=[Depends(verify_desktop_token)])


# ── Pydantic Request & Response Schemas ────────────────────────────────────────

class LLMConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    provider_type: str
    model_name: str
    base_url: str | None = None
    has_api_key: bool = False
    temperature: float = 0.2
    max_tokens: int = 4096
    context_window: int | None = None
    capabilities_json: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class LLMConfigCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    provider_type: str = Field(min_length=1, max_length=64)
    model_name: str = Field(min_length=1, max_length=255)
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    context_window: int | None = None
    capabilities_json: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class LLMConfigUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    provider_type: str | None = Field(default=None, max_length=64)
    model_name: str | None = Field(default=None, max_length=255)
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    context_window: int | None = None
    capabilities_json: dict[str, Any] | None = None
    is_default: bool | None = None


class LLMTestRequest(BaseModel):
    config_id: str | None = None
    provider_type: str | None = None
    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class ProviderCapabilitiesResponse(BaseModel):
    supports_streaming: bool
    supports_function_calling: bool
    supports_structured_output: bool
    supports_vision: bool
    supports_system_prompt: bool
    max_context_window: int
    is_local: bool


class ConnectionTestResponse(BaseModel):
    success: bool
    latency_ms: float
    provider: str
    model: str
    capabilities: ProviderCapabilitiesResponse | None = None
    error_code: str | None = None
    error_message: str | None = None


class EmbeddingConfigResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    provider_type: str
    model_name: str
    base_url: str | None = None
    has_api_key: bool = False
    dimensions: int
    batch_size: int = 32
    is_default: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class EmbeddingConfigCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    provider_type: str = Field(min_length=1, max_length=64)
    model_name: str = Field(min_length=1, max_length=255)
    dimensions: int = Field(gt=0)
    base_url: str | None = None
    api_key: str | None = None
    batch_size: int = Field(default=32, gt=0)
    is_default: bool = False


class EmbeddingConfigUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    provider_type: str | None = Field(default=None, max_length=64)
    model_name: str | None = Field(default=None, max_length=255)
    dimensions: int | None = Field(default=None, gt=0)
    base_url: str | None = None
    api_key: str | None = None
    batch_size: int | None = Field(default=None, gt=0)
    is_default: bool | None = None


class EmbeddingTestRequest(BaseModel):
    config_id: str | None = None
    provider_type: str | None = None
    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    dimensions: int | None = None


class EmbeddingTestResponse(BaseModel):
    success: bool
    latency_ms: float
    provider: str
    model: str
    detected_dimensions: int | None = None
    error_code: str | None = None
    error_message: str | None = None


class RerankerTestRequest(BaseModel):
    config_id: str | None = None
    provider_type: str | None = None
    reranker_provider: str | None = None
    model_name: str | None = None
    reranker_model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class RerankerTestResponse(BaseModel):
    success: bool
    latency_ms: float
    provider: str
    model: str
    error_code: str | None = None
    error_message: str | None = None


class OllamaModelInfo(BaseModel):
    name: str
    size_bytes: int
    family: str
    parameter_size: str
    quantization_level: str


class OllamaStatusResponse(BaseModel):
    is_installed: bool
    is_running: bool
    binary_path: str | None = None
    base_url: str
    version: str | None = None
    models: list[OllamaModelInfo] = Field(default_factory=list)


class OllamaPullRequest(BaseModel):
    model_name: str = Field(min_length=1)
    base_url: str | None = None


class LocalWeightDownloadRequest(BaseModel):
    model_name: str = Field(min_length=1)
    kind: str = Field(default="embeddings", pattern="^(embeddings|rerankers)$")


# ── LLM Configuration Endpoints ───────────────────────────────────────────────

@router.get("/llm", response_model=list[LLMConfigResponse], summary="List LLM Configurations")
async def list_llm_configs(
    service: ModelStudioService = Depends(get_model_studio_service),
) -> list[LLMConfigResponse]:
    service.ensure_presets()
    configs = service.list_llm_configs()
    return [LLMConfigResponse.model_validate(c) for c in configs]


@router.post(
    "/llm",
    response_model=LLMConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create LLM Configuration",
)
async def create_llm_config(
    payload: LLMConfigCreate,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> LLMConfigResponse:
    try:
        config = service.create_llm_config(payload.model_dump())
        return LLMConfigResponse.model_validate(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/llm/{id}", response_model=LLMConfigResponse, summary="Update LLM Configuration")
async def update_llm_config(
    id: str,
    payload: LLMConfigUpdate,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> LLMConfigResponse:
    try:
        config = service.update_llm_config(id, payload.model_dump(exclude_unset=True))
        return LLMConfigResponse.model_validate(config)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/llm/{id}", summary="Delete LLM Configuration")
async def delete_llm_config(
    id: str,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> dict[str, Any]:
    deleted = service.delete_llm_config(id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"LLM configuration not found: {id}")
    return {"status": "deleted", "id": id}


@router.post(
    "/llm/test",
    response_model=ConnectionTestResponse,
    summary="Test LLM Provider Connectivity & Capabilities",
)
async def test_llm_connection(
    payload: LLMTestRequest,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> ConnectionTestResponse:
    res: ConnectionTestResult = await service.test_llm_config(payload.model_dump(exclude_unset=True))
    caps = None
    if res.capabilities:
        caps = ProviderCapabilitiesResponse(
            supports_streaming=res.capabilities.supports_streaming,
            supports_function_calling=res.capabilities.supports_function_calling,
            supports_structured_output=res.capabilities.supports_structured_output,
            supports_vision=res.capabilities.supports_vision,
            supports_system_prompt=res.capabilities.supports_system_prompt,
            max_context_window=res.capabilities.max_context_window,
            is_local=res.capabilities.is_local,
        )
    return ConnectionTestResponse(
        success=res.success,
        latency_ms=res.latency_ms,
        provider=res.provider,
        model=res.model,
        capabilities=caps,
        error_code=res.error_code,
        error_message=res.error_message,
    )


# ── Embedding Configuration Endpoints ─────────────────────────────────────────

@router.get("/embeddings", response_model=list[EmbeddingConfigResponse], summary="List Embedding Configurations")
async def list_embedding_configs(
    service: ModelStudioService = Depends(get_model_studio_service),
) -> list[EmbeddingConfigResponse]:
    service.ensure_presets()
    configs = service.list_embedding_configs()
    return [EmbeddingConfigResponse.model_validate(c) for c in configs]


@router.post(
    "/embeddings",
    response_model=EmbeddingConfigResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Embedding Configuration",
)
async def create_embedding_config(
    payload: EmbeddingConfigCreate,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> EmbeddingConfigResponse:
    try:
        config = service.create_embedding_config(payload.model_dump())
        return EmbeddingConfigResponse.model_validate(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/embeddings/{id}", response_model=EmbeddingConfigResponse, summary="Update Embedding Configuration")
async def update_embedding_config(
    id: str,
    payload: EmbeddingConfigUpdate,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> EmbeddingConfigResponse:
    try:
        config = service.update_embedding_config(id, payload.model_dump(exclude_unset=True))
        return EmbeddingConfigResponse.model_validate(config)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/embeddings/{id}", summary="Delete Embedding Configuration")
async def delete_embedding_config(
    id: str,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> dict[str, Any]:
    deleted = service.delete_embedding_config(id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Embedding configuration not found: {id}")
    return {"status": "deleted", "id": id}


@router.post(
    "/embeddings/test",
    response_model=EmbeddingTestResponse,
    summary="Test Embedding Connectivity & Auto-Detect Dimensions",
)
async def test_embedding_connection(
    payload: EmbeddingTestRequest,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> EmbeddingTestResponse:
    res: ConnectionTestResult = await service.test_embedding_config(payload.model_dump(exclude_unset=True))
    return EmbeddingTestResponse(
        success=res.success,
        latency_ms=res.latency_ms,
        provider=res.provider,
        model=res.model,
        detected_dimensions=res.detected_dimensions,
        error_code=res.error_code,
        error_message=res.error_message,
    )


# ── Reranker Configuration Endpoints ──────────────────────────────────────────

@router.post(
    "/rerankers/test",
    response_model=RerankerTestResponse,
    summary="Test Reranker Relevance Ordering & Latency",
)
async def test_reranker_connection(
    payload: RerankerTestRequest,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> RerankerTestResponse:
    res: ConnectionTestResult = await service.test_reranker_config(payload.model_dump(exclude_unset=True))
    return RerankerTestResponse(
        success=res.success,
        latency_ms=res.latency_ms,
        provider=res.provider,
        model=res.model,
        error_code=res.error_code,
        error_message=res.error_message,
    )


# ── Ollama Runtime & Model Management ─────────────────────────────────────────

@router.get("/ollama/status", response_model=OllamaStatusResponse, summary="Get Ollama Daemon & Installed Models Status")
async def get_ollama_status(
    service: ModelStudioService = Depends(get_model_studio_service),
) -> OllamaStatusResponse:
    detection = await service.detect_ollama()
    models_data = []
    if detection.is_running:
        raw_models = await service.list_ollama_models(detection.base_url)
        models_data = [
            OllamaModelInfo(
                name=m["name"],
                size_bytes=m.get("size_bytes", 0),
                family=m.get("family", ""),
                parameter_size=m.get("parameter_size", ""),
                quantization_level=m.get("quantization_level", ""),
            )
            for m in raw_models
        ]
    return OllamaStatusResponse(
        is_installed=detection.is_installed,
        is_running=detection.is_running,
        binary_path=detection.binary_path,
        base_url=detection.base_url,
        version=detection.version,
        models=models_data,
    )


@router.post("/ollama/pull", summary="Stream Ollama Model Pull Progress (SSE)")
async def pull_ollama_model(
    payload: OllamaPullRequest,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> StreamingResponse:
    async def sse_stream():
        try:
            async for event in service.pull_ollama_model(payload.model_name, payload.base_url):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            err_event = {"status": "error", "error": str(exc), "percent": 0.0}
            yield f"data: {json.dumps(err_event)}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@router.post("/local-weights/download", summary="Pre-download Local HuggingFace Weights (SSE)")
async def download_local_weights(
    payload: LocalWeightDownloadRequest,
    service: ModelStudioService = Depends(get_model_studio_service),
) -> StreamingResponse:
    async def sse_stream():
        try:
            async for event in service.download_huggingface_model(payload.model_name, payload.kind):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # noqa: BLE001
            err_event = {"status": "error", "error": str(exc), "percent": 0.0}
            yield f"data: {json.dumps(err_event)}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


__all__ = ["router"]
