"""
backend/app/api/routes/dependencies.py
────────────────────────────────────────────────────────────────────────────────
Shared dependencies for Quantiva desktop FastAPI routes.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Header, HTTPException

from app.config import settings

logger = logging.getLogger(__name__)


def verify_desktop_token(
    x_quantiva_token: Annotated[str | None, Header(alias="X-Quantiva-Token")] = None,
) -> str:
    """Validate X-Quantiva-Token injected by the Tauri desktop shell (AGENTS.md Rule 6).

    All desktop API endpoints (except /api/v1/system/health) require this token.
    """
    expected = getattr(settings, "QUANTIVA_TOKEN", None)
    if expected:
        if not x_quantiva_token or x_quantiva_token != expected:
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid X-Quantiva-Token header.",
            )
        return x_quantiva_token

    # When no explicit secret token is specified in environment, require non-empty header
    if not x_quantiva_token or not x_quantiva_token.strip():
        raise HTTPException(
            status_code=401,
            detail="Missing required X-Quantiva-Token header.",
        )
    return x_quantiva_token
