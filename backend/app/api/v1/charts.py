"""Owner-scoped, renewable chart delivery.

Generated messages persist a stable API path instead of an expiring S3 URL.
Loading the chart re-signs the underlying object, so history remains usable
after the original presign would have expired.
"""
import asyncio
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from app.core.dependencies import get_current_user
from app.core.storage import presigned_url
from app.models.user import User

router = APIRouter(prefix="/charts", tags=["Charts"])

_CHART_NAME = re.compile(r"^[0-9a-f]{32}\.png$")


@router.get("/{owner_id}/{chart_name}")
async def get_chart(
    owner_id: uuid.UUID,
    chart_name: str,
    current_user: User = Depends(get_current_user),
):
    # Charts are derived from a private conversation; never serve another
    # tenant's chart even if the object key is guessed.
    if owner_id != current_user.id or not _CHART_NAME.fullmatch(chart_name):
        raise HTTPException(status_code=404, detail="Chart not found")
    try:
        url = await asyncio.to_thread(
            presigned_url, f"charts/{owner_id}/{chart_name}", 3600
        )
    except Exception:
        raise HTTPException(status_code=503, detail="Chart storage unavailable")
    return RedirectResponse(
        url,
        status_code=307,
        headers={"Cache-Control": "private, max-age=600", "X-Content-Type-Options": "nosniff"},
    )
