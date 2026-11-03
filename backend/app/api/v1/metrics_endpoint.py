"""
backend/app/api/v1/metrics_endpoint.py
────────────────────────────────────────────────────────────────────────────────
Exposes the /metrics endpoint that Prometheus scrapes.

Returns metrics in the OpenMetrics / Prometheus text exposition format
(text/plain; version=0.0.4).

Security note
─────────────
Access is intentionally unauthenticated so Prometheus can scrape without a
bearer token.  In production, restrict this path at the network / reverse-proxy
/ firewall level (e.g., allow only the Prometheus server's IP) rather than
adding application-layer auth that would complicate scrape configuration.
"""

from fastapi import APIRouter
from fastapi.responses import Response

import prometheus_client

router = APIRouter()


@router.get(
    "/metrics",
    include_in_schema=False,   # keep Swagger / ReDoc docs uncluttered
    response_class=Response,
    summary="Prometheus scrape endpoint",
)
async def prometheus_metrics() -> Response:
    """Return all registered Prometheus metrics in the text exposition format.

    Prometheus should be configured to scrape this endpoint at the desired
    interval (e.g., every 15 s).  The ``Content-Type`` header is set to the
    value exported by the ``prometheus_client`` library so that Prometheus and
    OpenMetrics-compatible scrapers both work correctly.
    """
    data = prometheus_client.generate_latest()
    return Response(
        content=data,
        media_type=prometheus_client.CONTENT_TYPE_LATEST,
    )
