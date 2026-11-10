from unittest.mock import AsyncMock, MagicMock, patch

from app.config import settings


async def test_public_liveness_does_not_expose_connection_details(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "services" not in response.json()


async def test_readiness_returns_503_and_redacts_errors(client):
    context = AsyncMock()
    context.__aenter__.side_effect = RuntimeError("secret_password@internal-db:5432")
    with patch("app.main.engine", MagicMock(connect=MagicMock(return_value=context))), patch("app.main.rate_limiter.client.ping", new_callable=AsyncMock, side_effect=RuntimeError("secret")), patch("app.main.vector_store.client.get_collections", new_callable=AsyncMock, side_effect=RuntimeError("private-host")), patch("app.main.s3_client.head_bucket", side_effect=RuntimeError("secret-key")):
        response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "services": {"postgres": "down", "redis": "down", "qdrant": "down", "s3": "down"}}
    assert "secret" not in response.text
    assert "internal" not in response.text


async def test_readiness_up_checks_all_dependencies(client):
    with patch("app.main.engine", MagicMock(connect=MagicMock(return_value=AsyncMock()))), patch("app.main.rate_limiter.client.ping", new_callable=AsyncMock), patch("app.main.vector_store.client.get_collections", new_callable=AsyncMock), patch("app.main.s3_client.head_bucket"):
        response = await client.get("/ready")
    assert response.status_code == 200
    assert set(response.json()["services"].values()) == {"up"}


async def test_security_headers_are_applied_but_not_to_openapi_assets(client):
    response = await client.get("/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "strict-transport-security" not in response.headers  # development HTTP
    openapi = await client.get("/openapi.json")
    assert "content-security-policy" not in openapi.headers


async def test_metrics_disabled_without_operator_secret(client, monkeypatch):
    monkeypatch.setattr(settings, "METRICS_TOKEN", None)
    assert (await client.get("/metrics")).status_code == 404


async def test_metrics_requires_correct_secret(client, monkeypatch):
    monkeypatch.setattr(settings, "METRICS_TOKEN", "metrics-test-only")
    assert (await client.get("/metrics")).status_code == 401
    assert (await client.get("/metrics", headers={"Authorization": "Bearer wrong"})).status_code == 401
    await client.get("/api/v1/chat/sessions")
    response = await client.get("/metrics", headers={"Authorization": "Bearer metrics-test-only"})
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text
    assert "llm_tokens_total" in response.text
