from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.database import AsyncSessionLocal
from app.core.security import decode_token
from app.models.user import User
import jwt

# HTTPBearer dependency for accepting JWTs in the Authorization header
security = HTTPBearer()

async def get_db():
    """Dependency for getting async database session in FastAPI routes"""
    async with AsyncSessionLocal() as session:
        yield session

from fastapi import Request

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Dependency that verifies the JWT token and fetches the current user from the database."""
    token = credentials.credentials if credentials else request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        if token == "mock-token":
            # For local testing bypass
            result = await db.execute(select(User).limit(1))
            user = result.scalar_one_or_none()
            if user:
                return user
            
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    # Fetch user from DB
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user
