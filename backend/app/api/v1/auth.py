from datetime import datetime, timedelta, timezone
import uuid

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select, update, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.dependencies import get_db, get_current_user, require_trusted_origin
from app.core.security import hash_password, verify_password, create_access_token, decode_token, token_digest
from app.models.auth_session import AuthSession
from app.core.rate_limiter import enforce_rate_limit
from app.models.user import User
from app.schemas.auth import RegisterRequest, LoginRequest, TokenResponse, UserResponse, RefreshRequest

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _auth_origin(request: Request):
    # Reject cross-site login CSRF too; CLI callers without Origin remain supported.
    if request.headers.get("origin") is not None:
        require_trusted_origin(request)


def _set_cookies(response: Response, tokens: TokenResponse):
    secure = settings.APP_ENV in {"production", "staging"}
    response.set_cookie("quantiva_access", tokens.access_token, httponly=True, secure=secure, samesite="lax", path="/api/v1", max_age=settings.ACCESS_TOKEN_MINUTES * 60)
    response.set_cookie("quantiva_refresh", tokens.refresh_token, httponly=True, secure=secure, samesite="lax", path="/api/v1/auth", max_age=settings.REFRESH_TOKEN_DAYS * 86400)
    response.headers["Cache-Control"] = "no-store"


def _tokens(user_id: uuid.UUID, session_id: uuid.UUID) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(str(user_id), timedelta(minutes=settings.ACCESS_TOKEN_MINUTES), session_id=str(session_id)),
        refresh_token=create_access_token(str(user_id), timedelta(days=settings.REFRESH_TOKEN_DAYS), session_id=str(session_id), token_type="refresh"),
    )


async def _login_session(user: User, response: Response, db: AsyncSession) -> TokenResponse:
    session_id = uuid.uuid4()
    tokens = _tokens(user.id, session_id)
    db.add(AuthSession(id=session_id, user_id=user.id, refresh_digest=token_digest(tokens.refresh_token), expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_DAYS)))
    await db.commit()
    _set_cookies(response, tokens)
    return tokens


@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    _auth_origin(request)
    await enforce_rate_limit(request.client.host if request.client else "unknown", scope="register", limit=5)
    email = str(body.email).lower()
    if await db.scalar(select(User).where(func.lower(User.email) == email)):
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=email, hashed_password=hash_password(body.password))
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Email already registered")
    return await _login_session(user, response, db)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    _auth_origin(request)
    await enforce_rate_limit(request.client.host if request.client else "unknown", scope="login", limit=10)
    # Preserve exact-case legacy accounts without rewriting user data. New
    # accounts are canonical lowercase; a unique legacy account can also log in
    # case-insensitively. Ambiguous legacy duplicates require the exact address.
    user = await db.scalar(select(User).where(User.email == str(body.email)))
    if user is None:
        candidates = list((await db.scalars(select(User).where(func.lower(User.email) == str(body.email).lower()).limit(2))).all())
        user = candidates[0] if len(candidates) == 1 else None
    if not user or not user.is_active or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    return await _login_session(user, response, db)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response, body: RefreshRequest | None = None, db: AsyncSession = Depends(get_db)):
    _auth_origin(request)
    await enforce_rate_limit(request.client.host if request.client else "unknown", scope="refresh", limit=30)
    token = body.refresh_token if body and body.refresh_token else request.cookies.get("quantiva_refresh")
    if not (body and body.refresh_token):
        require_trusted_origin(request)
    try:
        payload = decode_token(token or "", token_type="refresh")
        user_id, session_id = uuid.UUID(payload["sub"]), uuid.UUID(payload["sid"])
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = await db.scalar(select(User).where(User.id == user_id, User.is_active.is_(True)))
    if not user:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    tokens = _tokens(user_id, session_id)
    # Compare-and-swap prevents replay and concurrent refreshes, including on SQLite.
    changed = await db.execute(update(AuthSession).where(
        AuthSession.id == session_id, AuthSession.user_id == user_id,
        AuthSession.refresh_digest == token_digest(token), AuthSession.revoked_at.is_(None),
        AuthSession.expires_at > datetime.now(timezone.utc),
    ).values(refresh_digest=token_digest(tokens.refresh_token)).returning(AuthSession.id))
    if changed.scalar_one_or_none() is None:
        await db.rollback()
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    await db.commit()
    _set_cookies(response, tokens)
    return tokens


@router.post("/logout", status_code=204)
async def logout(request: Request, response: Response, body: RefreshRequest | None = None, db: AsyncSession = Depends(get_db)):
    _auth_origin(request)
    token = body.refresh_token if body and body.refresh_token else request.cookies.get("quantiva_refresh")
    if not (body and body.refresh_token):
        require_trusted_origin(request)
    try:
        payload = decode_token(token or "", token_type="refresh")
        # Any valid refresh from this login can revoke it, even after rotation.
        await db.execute(update(AuthSession).where(AuthSession.id == uuid.UUID(payload["sid"]), AuthSession.user_id == uuid.UUID(payload["sub"])).values(revoked_at=datetime.now(timezone.utc)))
        await db.commit()
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        pass
    response.delete_cookie("quantiva_access", path="/api/v1")
    response.delete_cookie("quantiva_refresh", path="/api/v1/auth")
    response.headers["Cache-Control"] = "no-store"


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user
