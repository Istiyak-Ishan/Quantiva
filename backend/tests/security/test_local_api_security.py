"""
backend/tests/security/test_local_api_security.py
────────────────────────────────────────────────────────────────────────────────
Tests for the desktop security middleware (Prompt P6.2).

Verifies:
  • IPC token authentication (401 without / wrong token, 200 with correct)
  • Health endpoint exemption from token auth
  • Loopback guard rejects non-loopback IPs
  • CORS origins constant completeness

Uses standalone Starlette apps to avoid the "Cannot add middleware after an
application has started" limitation of the main FastAPI app singleton.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncBaseTransport, AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from app.security.local_api import (
    DESKTOP_CORS_ORIGINS,
    IPCTokenAuthMiddleware,
    LoopbackGuardMiddleware,
)

TEST_TOKEN = "test-secret-token-p62"


# ── Helper: build a mini app with the security middleware ─────────────────────

def _build_secured_app(*, token: str = TEST_TOKEN) -> Starlette:
    """Build a Starlette app that mirrors the Quantiva health / protected structure."""

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def system_health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "app_version": "1.0.0-test"})

    async def protected(request: Request) -> JSONResponse:
        return JSONResponse({"data": "secret"})

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/api/v1/system/health", system_health),
            Route("/api/v1/system/hardware", protected),
            Route("/api/v1/data", protected),
        ],
    )
    # Add middleware: IPCTokenAuth first (runs second), LoopbackGuard second (runs first).
    app.add_middleware(IPCTokenAuthMiddleware, token=token)
    app.add_middleware(LoopbackGuardMiddleware)
    return app


@pytest_asyncio.fixture
async def secured_client():
    """ASGI test client wrapping a secured mini-app."""
    app = _build_secured_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as ac:
        yield ac


# ── Token Authentication Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_request_without_token_returns_401(secured_client: AsyncClient):
    """Requests lacking the X-Quantiva-Token header should be rejected."""
    resp = await secured_client.get("/api/v1/system/hardware")
    assert resp.status_code == 401
    assert "X-Quantiva-Token" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_request_with_wrong_token_returns_401(secured_client: AsyncClient):
    """An incorrect token should be rejected."""
    resp = await secured_client.get(
        "/api/v1/system/hardware",
        headers={"X-Quantiva-Token": "wrong-token"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_request_with_valid_token_succeeds(secured_client: AsyncClient):
    """A correct token should pass the middleware and reach the route."""
    resp = await secured_client.get(
        "/api/v1/data",
        headers={"X-Quantiva-Token": TEST_TOKEN},
    )
    assert resp.status_code == 200
    assert resp.json() == {"data": "secret"}


@pytest.mark.asyncio
async def test_health_endpoint_exempt_from_token(secured_client: AsyncClient):
    """The /api/v1/system/health endpoint must be accessible without any token."""
    resp = await secured_client.get("/api/v1/system/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_root_health_endpoint_exempt(secured_client: AsyncClient):
    """The root /health endpoint must also be exempt."""
    resp = await secured_client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_with_token_still_works(secured_client: AsyncClient):
    """Health endpoint should also work WITH a token (redundant but valid)."""
    resp = await secured_client.get(
        "/api/v1/system/health",
        headers={"X-Quantiva-Token": TEST_TOKEN},
    )
    assert resp.status_code == 200


# ── Loopback Guard Tests ─────────────────────────────────────────────────────

class _FakeClientTransport(AsyncBaseTransport):
    """ASGI transport that injects a custom client IP into the ASGI scope."""

    def __init__(self, app: Starlette, client_host: str) -> None:
        self._app = app
        self._client_host = client_host

    async def handle_async_request(self, request):
        async def scope_wrapper(scope, receive, send):
            scope["client"] = (self._client_host, 12345)
            await self._app(scope, receive, send)

        inner = ASGITransport(app=scope_wrapper)
        return await inner.handle_async_request(request)


@pytest.mark.asyncio
async def test_loopback_guard_allows_127_0_0_1():
    """Requests from 127.0.0.1 should be allowed."""
    app = _build_secured_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as ac:
        resp = await ac.get(
            "/api/v1/data",
            headers={"X-Quantiva-Token": TEST_TOKEN},
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_loopback_guard_rejects_non_loopback():
    """Requests from external IPs should be rejected with 403."""
    app = _build_secured_app()
    transport = _FakeClientTransport(app, "192.168.1.100")
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/v1/data",
            headers={"X-Quantiva-Token": TEST_TOKEN},
        )
        assert resp.status_code == 403
        assert "loopback" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_loopback_guard_rejects_public_ip():
    """Public IPs should be rejected."""
    app = _build_secured_app()
    transport = _FakeClientTransport(app, "8.8.8.8")
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 403


# ── CORS Origins Constant ─────────────────────────────────────────────────────

def test_desktop_cors_origins_includes_tauri():
    """Verify the DESKTOP_CORS_ORIGINS list covers all required Tauri origins."""
    assert "tauri://localhost" in DESKTOP_CORS_ORIGINS
    assert "http://tauri.local" in DESKTOP_CORS_ORIGINS
    assert "https://tauri.local" in DESKTOP_CORS_ORIGINS
    assert "http://localhost:3000" in DESKTOP_CORS_ORIGINS
    assert "http://127.0.0.1:3000" in DESKTOP_CORS_ORIGINS


# ── IPCTokenAuthMiddleware Unit Tests ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_ipc_middleware_noop_when_no_token():
    """When no token is configured, middleware should let everything through."""

    async def home(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", home)])
    app.add_middleware(IPCTokenAuthMiddleware, token=None)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_ipc_middleware_allows_options_preflight():
    """CORS preflight (OPTIONS) must pass without a token."""

    async def data(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/data", data, methods=["GET", "OPTIONS"])])
    app.add_middleware(IPCTokenAuthMiddleware, token="secret123")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.options("/data")
        # Must NOT be 401 — OPTIONS is always exempt.
        assert resp.status_code != 401


@pytest.mark.asyncio
async def test_empty_token_header_rejected(secured_client: AsyncClient):
    """An empty X-Quantiva-Token header should be rejected."""
    resp = await secured_client.get(
        "/api/v1/data",
        headers={"X-Quantiva-Token": ""},
    )
    assert resp.status_code == 401
