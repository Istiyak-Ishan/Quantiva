from unittest.mock import AsyncMock, patch

from sqlalchemy import select

from app.config import settings
from app.models.document import Document


async def test_upload_rejects_spoofed_pdf_before_storage(client, auth_headers):
    with patch("app.api.v1.documents.StorageService.upload_file") as storage:
        response = await client.post("/api/v1/documents/upload", headers=auth_headers, files={"file": ("fake.pdf", b"not a PDF", "application/pdf")})
        assert response.status_code == 400
        storage.assert_not_called()


async def test_upload_enforces_limit(client, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 10)
    response = await client.post("/api/v1/documents/upload", headers=auth_headers, files={"file": ("large.csv", b"a,b\n" + b"1,2\n" * 10, "text/csv")})
    assert response.status_code == 413


async def test_queue_failure_is_visible_and_never_launches_local_fallback(client, auth_headers, db_session):
    with patch("app.api.v1.documents.StorageService.upload_file", return_value="upload.pdf"), patch("app.api.v1.documents.process_document.delay", side_effect=RuntimeError("broker down")):
        response = await client.post("/api/v1/documents/upload", headers=auth_headers, files={"file": ("valid.pdf", b"%PDF-1.4\n", "application/pdf")})
    assert response.status_code == 503
    doc = await db_session.scalar(select(Document))
    assert doc.status == "failed"
    assert doc.processing_started_at is None
    assert doc.failure_reason == "Ingestion queue unavailable; document was not processed. Retry indexing."


async def test_delete_removes_vectors_cache_storage_and_row(client, tenants, db_session):
    alice, _, doc, _, _ = tenants
    with patch("app.api.v1.documents.vector_store.delete_document", new_callable=AsyncMock) as vectors, patch("app.api.v1.documents.semantic_cache.invalidate", new_callable=AsyncMock) as cache, patch("app.api.v1.documents.delete_file") as storage:
        response = await client.delete(f"/api/v1/documents/{doc.id}", headers=alice)
    assert response.status_code == 204, response.text
    vectors.assert_awaited_once_with(str(doc.id), user_id=str(doc.user_id))
    cache.assert_awaited_once_with(str(doc.id), user_id=str(doc.user_id))
    storage.assert_called_once_with(doc.storage_path)
    assert await db_session.scalar(select(Document).where(Document.id == doc.id)) is None


async def test_delete_failure_leaves_retryable_unqueryable_document(client, tenants, db_session):
    alice, _, doc, _, session_id = tenants
    with patch("app.api.v1.documents.vector_store.delete_document", new_callable=AsyncMock, side_effect=RuntimeError("qdrant down")):
        response = await client.delete(f"/api/v1/documents/{doc.id}", headers=alice)
    assert response.status_code == 503
    await db_session.refresh(doc)
    assert doc.status == "deleting"
    response = await client.post("/api/v1/chat/message", headers=alice, json={"session_id": session_id, "query": "private"})
    assert response.status_code == 409


async def test_download_is_owner_scoped_and_does_not_expose_storage_keys(client, tenants):
    alice, bob, doc, _, _ = tenants
    with patch("app.api.v1.documents.StorageService.get_file", return_value=b"%PDF-test") as storage:
        response = await client.get(f"/api/v1/documents/{doc.id}/download", headers=bob)
        assert response.status_code == 404
        storage.assert_not_called()
        response = await client.get(f"/api/v1/documents/{doc.id}/download", headers=alice)
    assert response.status_code == 200
    assert response.content == b"%PDF-test"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
