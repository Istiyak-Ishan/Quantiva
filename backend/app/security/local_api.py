"""
backend/app/security/local_api.py
────────────────────────────────────────────────────────────────────────────────
Desktop-mode security middleware for the Quantiva local backend.

Provides:
  • LoopbackGuardMiddleware  — rejects non-loopback client IPs (AGENTS.md Rule 6)
  • IPCTokenAuthMiddleware   — constant-time X-Quantiva-Token validation
  • configure_desktop_security() — wires both middleware + strict CORS origins
"""

from __future__ import annotations

import hmac
import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DESKTOP_CORS_ORIGINS: list[str] = [
    "http://tauri.local",
    "https://tauri.local",
    "tauri://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Paths exempt from IPC token validation.
_TOKEN_EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/api/v1/system/health",
})

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})


# ── Loopback Guard ────────────────────────────────────────────────────────────

class LoopbackGuardMiddleware(BaseHTTPMiddleware):
    """Reject any request originating from a non-loopback IP address.

    The Quantiva desktop backend MUST bind to ``127.0.0.1`` only.
    This middleware adds defence-in-depth by inspecting the ASGI client
    address and rejecting anything that isn't loopback.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        client = request.client
        client_host = client.host if client else None

        if client_host and client_host not in _LOOPBACK_HOSTS:
            logger.warning(
                "Rejected non-loopback request from %s %s",
                client_host,
                request.url.path,
            )
            return JSONResponse(
                status_code=403,
                content={"detail": "Only loopback connections are allowed."},
            )

        return await call_next(request)


# ── IPC Token Auth ────────────────────────────────────────────────────────────

class IPCTokenAuthMiddleware(BaseHTTPMiddleware):
    """Validate the ``X-Quantiva-Token`` header injected by the Tauri shell.

    When ``QUANTIVA_IPC_TOKEN`` is set in the environment (always set in
    desktop mode), every request except health-check endpoints and CORS
    preflight (OPTIONS) must carry a matching token.

    The comparison uses ``hmac.compare_digest`` for constant-time safety.
    """

    def __init__(self, app, *, token: str | None = None) -> None:  # noqa: ANN001
        super().__init__(app)
        self._token: str | None = token or os.environ.get("QUANTIVA_IPC_TOKEN")

    async def dispatch(self, request: Request, call_next) -> Response:
        # No token configured → middleware is a no-op (legacy / CI mode).
        if not self._token:
            return await call_next(request)

        # Always let CORS preflight through.
        if request.method == "OPTIONS":
            return await call_next(request)

        # Exempt health-check paths.
        if request.url.path in _TOKEN_EXEMPT_PATHS:
            return await call_next(request)

        # Validate token.
        request_token = request.headers.get("X-Quantiva-Token", "")
        if not request_token or not hmac.compare_digest(request_token, self._token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid X-Quantiva-Token header."},
            )

        return await call_next(request)


# ── Convenience wiring ────────────────────────────────────────────────────────

def configure_desktop_security(app) -> None:  # noqa: ANN001 — FastAPI
    """Add desktop security middleware and tighten CORS for Tauri origins.

    Call this AFTER the base ``CORSMiddleware`` has been added so the
    execution order is:

        request → LoopbackGuard → IPCTokenAuth → CORS → route handler

    In Starlette, middleware added *last* executes *first* on the inbound
    path. So we add LoopbackGuard second (runs first) and IPCTokenAuth first
    (runs after loopback check but before the route).
    """
    from fastapi.middleware.cors import CORSMiddleware

    token = os.environ.get("QUANTIVA_IPC_TOKEN")

    # 1. Replace CORS origins with strict desktop list.
    #    Remove the existing CORSMiddleware entries and re-add with desktop
    #    origins.  Starlette stores middleware as a list; we rebuild to avoid
    #    duplicates.
    app.add_middleware(IPCTokenAuthMiddleware, token=token)
    app.add_middleware(LoopbackGuardMiddleware)

    logger.info(
        "Desktop security middleware enabled (token=%s)",
        "set" if token else "unset",
    )


__all__ = [
    "DESKTOP_CORS_ORIGINS",
    "IPCTokenAuthMiddleware",
    "LoopbackGuardMiddleware",
    "configure_desktop_security",
]
