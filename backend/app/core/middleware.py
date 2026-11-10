"""
backend/app/core/middleware.py
────────────────────────────────────────────────────────────────────────────────
Prometheus instrumentation middleware.

Automatically records:
  • http_requests_total          (method, path, status_code)
  • http_request_duration_seconds (method, path)

for every inbound HTTP request.

Paths listed in EXCLUDE_PATHS are skipped to avoid polluting the metrics
with high-frequency, low-value scrape and health-check traffic.

Path normalisation keeps label cardinality bounded by replacing UUID-style
and purely numeric path segments with a generic ``{id}`` placeholder before
the label values are recorded.
"""

import re
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings
from app.core.metrics import HTTP_LATENCY, HTTP_REQUESTS


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add browser hardening headers without breaking the OpenAPI UI.

    The API only returns JSON/SSE/files, so a deny-all content policy is safe
    outside the Swagger/ReDoc asset routes. HSTS is only emitted outside
    development because local HTTP is intentionally supported.
    """

    DOCS_PREFIXES = ("/docs", "/redoc", "/openapi.json")

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        if not request.url.path.startswith(self.DOCS_PREFIXES):
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            )
        if settings.APP_ENV in {"production", "staging"}:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for every HTTP call."""

    # Paths we skip to avoid high-cardinality noise
    EXCLUDE_PATHS: frozenset[str] = frozenset({"/metrics", "/health", "/favicon.ico"})

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Normalise dynamic segments to keep label cardinality bounded
        normalized = self._normalize_path(path)

        if normalized in self.EXCLUDE_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=normalized,
            status_code=str(response.status_code),
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            endpoint=normalized,
        ).observe(duration)

        return response

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Replace UUID-like and numeric path segments with {id} placeholders.

        Examples
        --------
        /documents/abc-123/status          → /documents/{id}/status
        /documents/550e8400-e29b-41d4-...  → /documents/{id}
        /users/42/profile                  → /users/{id}/profile
        """
        # Match standard UUIDs (8-4-4-4-12 hex groups)
        path = re.sub(
            r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "/{id}",
            path,
            flags=re.IGNORECASE,
        )
        # Match purely numeric IDs
        path = re.sub(r"/\d+", "/{id}", path)
        return path
