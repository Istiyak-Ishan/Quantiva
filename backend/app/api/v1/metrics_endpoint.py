"""Prometheus endpoint protected by an operator-managed bearer credential."""
import secrets

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
import prometheus_client

from app.config import settings

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(authorization: str | None = Header(default=None)):
    if not settings.METRICS_TOKEN:
        raise HTTPException(status_code=404, detail="Not found")
    if not authorization or not secrets.compare_digest(authorization, f"Bearer {settings.METRICS_TOKEN}"):
        raise HTTPException(status_code=401, detail="Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return Response(content=prometheus_client.generate_latest(), media_type=prometheus_client.CONTENT_TYPE_LATEST)
