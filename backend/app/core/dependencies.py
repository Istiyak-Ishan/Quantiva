from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.database import AsyncSessionLocal
from app.core.security import decode_token
from app.config import settings
from app.models.auth_session import AuthSession
from datetime import datetime, timezone
from app.models.user import User
import jwt

# HTTPBearer dependency for accepting JWTs in the Authorization header
security = HTTPBearer()

async def get_db():
    """Dependency for getting async database session in FastAPI routes"""
    async with AsyncSessionLocal() as session:
        yield session

from fastapi import Request

import uuid


from urllib.parse import urlsplit

def require_trusted_origin(request: Request) -> None:
    """Cookie-authenticated mutations and streams require a browser-supplied trusted Origin or Referer.

    Non-browser API clients use Authorization headers and are not subject to CSRF.
    Query-string credentials are never accepted.
    """
    origin = request.headers.get("origin")
    if not origin:
        referer = request.headers.get("referer")
        if referer:
            parsed = urlsplit(referer)
            if parsed.scheme and parsed.netloc:
                origin = f"{parsed.scheme}://{parsed.netloc}"
        elif request.headers.get("sec-fetch-site") == "same-origin":
            host = request.headers.get("x-forwarded-host") or request.headers.get("host")
            proto = request.headers.get("x-forwarded-proto", "http")
            if host:
                origin = f"{proto}://{host}"

    allowed_origins = set(settings.CORS_ORIGINS)
    if settings.APP_ENV == "development":
        if "http://localhost:3000" in allowed_origins:
            allowed_origins.add("http://127.0.0.1:3000")
        if "http://127.0.0.1:3000" in allowed_origins:
            allowed_origins.add("http://localhost:3000")

    if origin not in allowed_origins:
        raise HTTPException(status_code=403, detail="Untrusted request origin")

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Dependency that verifies the JWT token and fetches the current user from the database."""
    token = credentials.credentials if credentials else request.cookies.get("quantiva_access")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not credentials and (request.method not in {"GET", "HEAD", "OPTIONS"} or request.url.path.endswith("/message/stream")):
        require_trusted_origin(request)
    try:
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        try:
            user_uuid = uuid.UUID(user_id)
            auth_session_id = uuid.UUID(payload["sid"])
        except (ValueError, TypeError):
            raise credentials_exception
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise credentials_exception

    # Fetch user from DB
    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exception
    login_session = await db.scalar(select(AuthSession).where(
        AuthSession.id == auth_session_id,
        AuthSession.user_id == user.id,
        AuthSession.revoked_at.is_(None),
        AuthSession.expires_at > datetime.now(timezone.utc),
    ))
    if login_session is None:
        raise credentials_exception
    return user
