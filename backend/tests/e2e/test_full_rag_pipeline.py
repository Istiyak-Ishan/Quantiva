import pytest
import io
import uuid
from sqlalchemy import update
from app.models.document import Document
from unittest.mock import patch, AsyncMock, MagicMock
from app.schemas.agent import AgentAnswer, Citation

FAKE_PDF = b"%PDF-1.4 1 0 obj << /Type /Catalog >> endobj"

MOCK_ANSWER = AgentAnswer(
    answer="Based on the report, total revenue for Q3 was $5.2 million, up 15.3% from Q2.",
    citations=[Citation(
        doc_id="doc-e2e-test",
        filename="q3_financial_report.pdf",
        page_or_row=7,
        text_excerpt="Q3 total revenue: $5.2M (Q2: $4.5M, +15.3%)",
        relevance_score=0.97,
    )],
    confidence=0.97,
    tokens_used=342,
)

MOCK_RESULT = MagicMock()
MOCK_RESULT.answer = MOCK_ANSWER
MOCK_RESULT.tool_calls_made = 2
MOCK_RESULT.iterations = 3

class TestFullRAGPipeline:
    async def test_register_login_upload_ask(self, client, db_session):
        reg_resp = await client.post("/api/v1/auth/register", json={
            "email": "e2e_user@quantiva.ai",
            "password": "E2ETestPass1!",
        })
        assert reg_resp.status_code in (200, 201)
        token = reg_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        with (
            patch("app.api.v1.documents.ingest_document_task.delay", return_value=MagicMock(id="celery-task-1")),
            patch("app.core.storage.StorageService.upload_file", return_value="s3://q/doc.pdf"),
        ):
            upload_resp = await client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("q3_report.pdf", io.BytesIO(FAKE_PDF), "application/pdf")},
            )

        assert upload_resp.status_code == 200
        doc_id = upload_resp.json()["id"]
        # This is an API contract test, not real ingestion: explicitly model the
        # worker's completed transition instead of querying a processing document.
        await db_session.execute(update(Document).where(Document.id == uuid.UUID(doc_id)).values(status="ready"))
        await db_session.commit()

        sess_resp = await client.post(
            "/api/v1/chat/sessions",
            json={"document_ids": [doc_id]},
            headers=headers,
        )
        assert sess_resp.status_code == 200
        session_id = sess_resp.json()["session_id"]

        with (
            patch("app.api.v1.chat.orchestrator.run", new_callable=AsyncMock, return_value=MOCK_RESULT),
            patch("app.services.memory.session_memory.session_memory.load_history", new_callable=AsyncMock, return_value=[]),
            patch("app.services.memory.session_memory.session_memory.save_turn", new_callable=AsyncMock),
            patch("app.services.memory.semantic_cache.semantic_cache.get", new_callable=AsyncMock, return_value=None),
            patch("app.services.memory.semantic_cache.semantic_cache.set", new_callable=AsyncMock),
        ):
            answer_resp = await client.post(
                "/api/v1/chat/message",
                json={
                    "session_id": session_id,
                    "query": "What was the Q3 revenue?",
                },
                headers=headers,
            )

        assert answer_resp.status_code == 200
        data = answer_resp.json()
        assert "answer" in data
        assert len(data["citations"]) >= 1
        assert 0.0 <= data["confidence"] <= 1.0
