import pytest

class TestHealthEndpoint:
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_health_has_status_key(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    async def test_health_lists_services(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "services" in data
        services = data["services"]
        assert "postgres" in services
        assert "redis" in services
        assert "qdrant" in services

class TestMetricsEndpoint:
    async def test_metrics_returns_200(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200

    async def test_metrics_content_type(self, client):
        resp = await client.get("/metrics")
        assert "text/plain" in resp.headers.get("content-type", "")

    async def test_metrics_contains_http_counter(self, client):
        await client.get("/health")
        resp = await client.get("/metrics")
        assert "http_requests_total" in resp.text

    async def test_metrics_contains_llm_tokens(self, client):
        resp = await client.get("/metrics")
        assert "llm_tokens_total" in resp.text

