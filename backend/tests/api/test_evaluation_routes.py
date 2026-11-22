"""
backend/tests/api/test_evaluation_routes.py
────────────────────────────────────────────────────────────────────────────────
Integration tests for RAG Evaluation FastAPI routes (Prompt P8.1).
"""
from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.config import settings
from app.main import app
from app.storage.database import SessionLocal, engine as local_db_engine, initialize_database
from app.storage.models import (
    Document,
    DocumentChunk,
    EmbeddingConfiguration,
    IndexMetadata,
    KnowledgeBase,
)


@pytest.fixture(autouse=True)
def setup_eval_db(tmp_path):
    settings.QUANTIVA_DATA_DIR = str(tmp_path)
    initialize_database(local_db_engine)
    yield


@pytest.mark.asyncio
async def test_evaluation_routes_lifecycle(monkeypatch):
    token = "test-eval-token"
    monkeypatch.setenv("QUANTIVA_IPC_TOKEN", token)

    # Query or create an embedding config
    with SessionLocal() as session:
        emb_cfg = session.scalar(select(EmbeddingConfiguration))
        if emb_cfg is None:
            emb_cfg = EmbeddingConfiguration(
                name="Eval Embeddings Unique",
                provider_type="local",
                model_name="all-MiniLM-L6-v2",
                dimensions=384,
            )
            session.add(emb_cfg)
            session.commit()
            session.refresh(emb_cfg)

        kb = KnowledgeBase(
            name=f"Evaluation KB {uuid.uuid4().hex[:6]}",
            embedding_config_id=emb_cfg.id,
        )
        session.add(kb)
        session.commit()
        session.refresh(kb)

        idx_meta = IndexMetadata(
            knowledge_base_id=kb.id,
            vector_collection_name=f"kb_{kb.id}_vectors",
            embedding_provider="local",
            embedding_model_name="all-MiniLM-L6-v2",
            embedding_dimensions=384,
        )
        session.add(idx_meta)

        doc = Document(
            knowledge_base_id=kb.id,
            filename="financials.pdf",
            original_rel_path="financials.pdf",
            file_type="pdf",
            file_size_bytes=1024,
            sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            status="READY",
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)

        chunk = DocumentChunk(
            knowledge_base_id=kb.id,
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Total net revenue grew 24 percent in Q3 to reach 100M.",
            char_count=55,
            content_hash="abc12345",
        )
        session.add(chunk)
        session.commit()

        kb_id = kb.id

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Unauthenticated request must fail with 401
        unauth_resp = await client.get("/api/v1/evaluation/runs")
        assert unauth_resp.status_code == 401

        # 2. Get default benchmarks
        auth_headers = {"X-Quantiva-Token": token}
        bench_resp = await client.get(
            f"/api/v1/evaluation/benchmarks/default?knowledge_base_id={kb_id}",
            headers=auth_headers,
        )
        assert bench_resp.status_code == 200
        benchmarks = bench_resp.json()
        assert len(benchmarks) >= 1

        # 3. Run evaluation
        run_payload = {
            "knowledge_base_id": kb_id,
            "name": "Integration Test Benchmark",
            "strategies": ["sparse"],
            "benchmark_dataset": [
                {
                    "question": "What was the total net revenue?",
                    "expected_document_names": ["financials.pdf"],
                    "expected_page_numbers": [1],
                    "expected_keywords_or_snippets": ["revenue", "100m"],
                }
            ],
            "evaluate_generation": False,
            "k_values": [1, 3, 5],
        }

        run_resp = await client.post(
            "/api/v1/evaluation/run",
            json=run_payload,
            headers=auth_headers,
        )
        assert run_resp.status_code == 201
        run_data = run_resp.json()
        assert run_data["name"] == "Integration Test Benchmark"
        assert run_data["status"] == "COMPLETED"
        assert "strategy_results" in run_data
        run_id = run_data["id"]

        # 4. List runs
        list_resp = await client.get("/api/v1/evaluation/runs", headers=auth_headers)
        assert list_resp.status_code == 200
        runs = list_resp.json()
        assert any(r["id"] == run_id for r in runs)

        # 5. Get run details
        detail_resp = await client.get(f"/api/v1/evaluation/runs/{run_id}", headers=auth_headers)
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["id"] == run_id
