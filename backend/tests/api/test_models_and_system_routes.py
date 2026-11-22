"""
backend/tests/api/test_models_and_system_routes.py
────────────────────────────────────────────────────────────────────────────────
Integration tests for Model Studio, System Health/Hardware, and Settings Routes.
Prompt P4.3 verification suite.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api.routes.dependencies import verify_desktop_token
from app.config import settings
from app.core.application_context import get_application_context
from app.core.model_studio import (
    OllamaDetectionResult,
    get_model_studio_service,
    reset_model_studio_service,
)
from app.main import app
from app.providers.interfaces.types import (
    ConnectionTestResult,
    ProviderCapabilities,
)
from app.storage.database import (
    engine as local_db_engine,
    initialize_database as init_local_db,
)

AUTH_HEADER = {"X-Quantiva-Token": "quantiva-test-token"}


@pytest_asyncio.fixture
async def api_client(tmp_path: Path):
    """AsyncClient wired to the FastAPI application with initialized local database."""
    init_local_db(local_db_engine)
    reset_model_studio_service()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── System Health & Hardware Tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_system_health_no_token_required(api_client: AsyncClient):
    """GET /api/v1/system/health is exempt from X-Quantiva-Token header."""
    resp = await api_client.get("/api/v1/system/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "app_version" in data
    assert data["startup_status"] == "ready"
    assert "sqlite_status" in data
    assert "data_directory" in data
    assert "database_path" in data


@pytest.mark.asyncio
async def test_system_hardware_requires_token(api_client: AsyncClient):
    """GET /api/v1/system/hardware fails with 401 when missing X-Quantiva-Token."""
    resp = await api_client.get("/api/v1/system/hardware")
    assert resp.status_code == 401
    assert "X-Quantiva-Token" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_system_hardware_with_token(api_client: AsyncClient):
    """GET /api/v1/system/hardware returns hardware profile and tier recommendation."""
    resp = await api_client.get("/api/v1/system/hardware", headers=AUTH_HEADER)
    assert resp.status_code == 200
    data = resp.json()

    assert "hardware" in data
    hw = data["hardware"]
    assert "cpu_model" in hw
    assert "physical_cores" in hw
    assert "logical_cores" in hw
    assert "ram_total_gb" in hw
    assert "disk_free_gb" in hw
    assert isinstance(hw["gpus"], list)

    assert "recommendation" in data
    rec = data["recommendation"]
    assert "tier" in rec
    assert rec["tier"] in ("cloud_api", "ultra_light", "balanced", "powerful")
    assert "rationale" in rec
    assert "description" in rec
    assert isinstance(rec["llm_models"], list)


# ── LLM Configuration Routes Tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_routes_require_token(api_client: AsyncClient):
    """LLM endpoints reject requests without X-Quantiva-Token."""
    resp = await api_client.get("/api/v1/models/llm")
    assert resp.status_code == 401

    resp = await api_client.post("/api/v1/models/llm", json={"name": "test"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_llm_configs_seeds_presets(api_client: AsyncClient):
    """GET /api/v1/models/llm returns the seeded out-of-the-box presets."""
    resp = await api_client.get("/api/v1/models/llm", headers=AUTH_HEADER)
    assert resp.status_code == 200
    configs = resp.json()
    assert isinstance(configs, list)
    assert len(configs) >= 2

    names = [c["name"] for c in configs]
    assert "Default Local LLM" in names
    assert "Default Cloud LLM" in names

    local_preset = next(c for c in configs if c["name"] == "Default Local LLM")
    assert local_preset["provider_type"] == "ollama"
    assert local_preset["model_name"] == "qwen2.5:7b"
    assert local_preset["temperature"] == 0.2
    assert local_preset["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_llm_crud_lifecycle(api_client: AsyncClient):
    """POST / PUT / DELETE lifecycle for LLM configurations."""
    # 1. Create
    payload = {
        "name": f"API Test OpenAI {Path.cwd().name}",
        "provider_type": "openai",
        "model_name": "gpt-4o",
        "api_key": "sk-secret-test-key",
        "temperature": 0.3,
        "max_tokens": 2048,
        "is_default": False,
    }
    resp = await api_client.post("/api/v1/models/llm", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 201
    created = resp.json()
    config_id = created["id"]
    assert created["name"] == payload["name"]
    assert created["has_api_key"] is True
    assert "api_key" not in created  # Never leak plaintext

    # 2. Update
    update_payload = {
        "temperature": 0.8,
        "max_tokens": 8192,
    }
    resp = await api_client.put(f"/api/v1/models/llm/{config_id}", json=update_payload, headers=AUTH_HEADER)
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["temperature"] == 0.8
    assert updated["max_tokens"] == 8192

    # 3. Delete
    resp = await api_client.delete(f"/api/v1/models/llm/{config_id}", headers=AUTH_HEADER)
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # 4. Confirm deleted
    resp = await api_client.put(f"/api/v1/models/llm/{config_id}", json=update_payload, headers=AUTH_HEADER)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_llm_test_connection_endpoint(api_client: AsyncClient):
    """POST /api/v1/models/llm/test executes connection probe."""
    fake_result = ConnectionTestResult(
        success=True,
        latency_ms=42.0,
        provider="ollama",
        model="qwen2.5:7b",
        capabilities=ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_vision=False,
            supports_system_prompt=True,
            max_context_window=32768,
            is_local=True,
        ),
    )

    with patch.object(
        get_model_studio_service(),
        "test_llm_config",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        resp = await api_client.post(
            "/api/v1/models/llm/test",
            json={"provider_type": "ollama", "model_name": "qwen2.5:7b"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["latency_ms"] == 42.0
        assert data["capabilities"]["supports_function_calling"] is True
        assert data["capabilities"]["max_context_window"] == 32768


# ── Embedding Configuration Routes Tests ──────────────────────────────────────

@pytest.mark.asyncio
async def test_embedding_crud_lifecycle(api_client: AsyncClient):
    """CRUD lifecycle for Embedding configurations."""
    # 1. List presets
    resp = await api_client.get("/api/v1/models/embeddings", headers=AUTH_HEADER)
    assert resp.status_code == 200
    configs = resp.json()
    assert len(configs) >= 1
    assert any(c["name"] == "Default Local Embedding" for c in configs)

    # 2. Create
    payload = {
        "name": "Custom MiniLM Embedding",
        "provider_type": "sentence_transformers",
        "model_name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "batch_size": 64,
        "is_default": False,
    }
    resp = await api_client.post("/api/v1/models/embeddings", json=payload, headers=AUTH_HEADER)
    assert resp.status_code == 201
    created = resp.json()
    config_id = created["id"]
    assert created["dimensions"] == 384

    # 3. Update
    resp = await api_client.put(
        f"/api/v1/models/embeddings/{config_id}",
        json={"batch_size": 128},
        headers=AUTH_HEADER,
    )
    assert resp.status_code == 200
    assert resp.json()["batch_size"] == 128

    # 4. Delete
    resp = await api_client.delete(f"/api/v1/models/embeddings/{config_id}", headers=AUTH_HEADER)
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


@pytest.mark.asyncio
async def test_embedding_test_connection_endpoint(api_client: AsyncClient):
    """POST /api/v1/models/embeddings/test returns auto-detected dimensions."""
    fake_result = ConnectionTestResult(
        success=True,
        latency_ms=18.5,
        provider="sentence_transformers",
        model="intfloat/multilingual-e5-base",
        detected_dimensions=768,
    )

    with patch.object(
        get_model_studio_service(),
        "test_embedding_config",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        resp = await api_client.post(
            "/api/v1/models/embeddings/test",
            json={
                "provider_type": "sentence_transformers",
                "model_name": "intfloat/multilingual-e5-base",
            },
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["detected_dimensions"] == 768


# ── Reranker Test Route Tests ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reranker_test_connection_endpoint(api_client: AsyncClient):
    """POST /api/v1/models/rerankers/test executes relevance ordering test."""
    fake_result = ConnectionTestResult(
        success=True,
        latency_ms=14.2,
        provider="cross_encoder",
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    with patch.object(
        get_model_studio_service(),
        "test_reranker_config",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        resp = await api_client.post(
            "/api/v1/models/rerankers/test",
            json={
                "provider_type": "cross_encoder",
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["latency_ms"] == 14.2


# ── Ollama & Local Weights Routes Tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_ollama_status_endpoint(api_client: AsyncClient):
    """GET /api/v1/models/ollama/status returns daemon state and model list."""
    fake_detection = OllamaDetectionResult(
        is_installed=True,
        is_running=True,
        binary_path=r"C:\Users\User\AppData\Local\Programs\Ollama\ollama.exe",
        base_url="http://127.0.0.1:11434",
        version="0.5.4",
    )
    fake_models = [
        {"name": "qwen2.5:7b", "size_bytes": 4700000000, "family": "qwen2", "parameter_size": "7B", "quantization_level": "Q4_K_M"}
    ]

    service = get_model_studio_service()
    with (
        patch.object(service, "detect_ollama", new_callable=AsyncMock, return_value=fake_detection),
        patch.object(service, "list_ollama_models", new_callable=AsyncMock, return_value=fake_models),
    ):
        resp = await api_client.get("/api/v1/models/ollama/status", headers=AUTH_HEADER)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_installed"] is True
        assert data["is_running"] is True
        assert data["version"] == "0.5.4"
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "qwen2.5:7b"


@pytest.mark.asyncio
async def test_ollama_pull_stream_endpoint(api_client: AsyncClient):
    """POST /api/v1/models/ollama/pull returns SSE stream with real-time progress."""
    async def fake_stream(model_name, base_url=None):
        yield {"status": "downloading", "completed": 25, "total": 100, "percent": 25.0}
        yield {"status": "success", "completed": 100, "total": 100, "percent": 100.0}

    service = get_model_studio_service()
    with patch.object(service, "pull_ollama_model", side_effect=fake_stream):
        resp = await api_client.post(
            "/api/v1/models/ollama/pull",
            json={"model_name": "llama3.2:1b"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert "data: " in body
        assert "25.0" in body
        assert "100.0" in body


@pytest.mark.asyncio
async def test_local_weights_download_stream_endpoint(api_client: AsyncClient):
    """POST /api/v1/models/local-weights/download streams weight download progress."""
    async def fake_stream(model_name, kind):
        yield {"status": "starting", "model": model_name, "percent": 0.0}
        yield {"status": "completed", "model": model_name, "percent": 100.0}

    service = get_model_studio_service()
    with patch.object(service, "download_huggingface_model", side_effect=fake_stream):
        resp = await api_client.post(
            "/api/v1/models/local-weights/download",
            json={"model_name": "intfloat/multilingual-e5-base", "kind": "embeddings"},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "completed" in resp.text


# ── Settings & Storage Routes Tests ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_and_update_retrieval_settings(api_client: AsyncClient):
    """GET / PUT /api/v1/settings/retrieval reads and modifies active retrieval parameters."""
    # 1. Get default
    resp = await api_client.get("/api/v1/settings/retrieval", headers=AUTH_HEADER)
    assert resp.status_code == 200
    data = resp.json()
    assert data["enable_reranking"] is True
    assert data.get("reranker_provider") in ("cross_encoder", None)

    # 2. Update
    update_payload = {
        "dense_weight": 0.8,
        "sparse_weight": 0.2,
        "final_top_k": 12,
    }
    resp = await api_client.put(
        "/api/v1/settings/retrieval",
        json=update_payload,
        headers=AUTH_HEADER,
    )
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["dense_weight"] == 0.8
    assert updated["sparse_weight"] == 0.2
    assert updated["final_top_k"] == 12


@pytest.mark.asyncio
async def test_get_storage_settings(api_client: AsyncClient):
    """GET /api/v1/settings/storage reports directory paths and disk stats."""
    resp = await api_client.get("/api/v1/settings/storage", headers=AUTH_HEADER)
    assert resp.status_code == 200
    data = resp.json()
    assert "current_data_dir" in data
    assert "database_path" in data
    assert "disk_total_bytes" in data
    assert "disk_free_bytes" in data
    assert "database_size_bytes" in data


@pytest.mark.asyncio
async def test_storage_migrate_endpoint(api_client: AsyncClient, tmp_path: Path):
    """POST /api/v1/settings/storage/migrate migrates data directory."""
    new_dir = tmp_path / "migrated_quantiva_data"

    with patch("app.api.routes.settings.migrate_data_directory", return_value=new_dir):
        resp = await api_client.post(
            "/api/v1/settings/storage/migrate",
            json={"new_path": str(new_dir), "verify_space": False},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "migrated"
        assert str(new_dir) in data["new_data_dir"]


# ── Backup & Restore Routes Tests ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backup_create_list_restore_lifecycle(api_client: AsyncClient, tmp_path: Path):
    """Full lifecycle for backup creation, listing, and restore."""
    fake_archive = tmp_path / "quantiva_backup_test.zip"
    fake_archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty zip

    # 1. Create backup
    with patch("app.api.routes.settings.create_backup", return_value=fake_archive):
        resp = await api_client.post(
            "/api/v1/settings/backups/create",
            json={"include_models": False},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "created"
        assert data["filename"] == "quantiva_backup_test.zip"

    # 2. List backups
    with patch("app.api.routes.settings.LocalDataPaths.backups", new_callable=lambda: tmp_path):
        resp = await api_client.get("/api/v1/settings/backups", headers=AUTH_HEADER)
        assert resp.status_code == 200
        backups = resp.json()
        assert isinstance(backups, list)

    # 3. Restore backup
    with patch("app.api.routes.settings.restore_backup", return_value=fake_archive):
        resp = await api_client.post(
            "/api/v1/settings/backups/restore",
            json={"archive_path": str(fake_archive), "create_safety_snapshot": False},
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "restored"
