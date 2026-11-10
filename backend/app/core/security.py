import bcrypt
import jwt
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from app.config import settings

ALGORITHM = "HS256"

def hash_password(plain: str) -> str:
    """Hashes a plain text password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(plain.encode('utf-8'), salt)
    return hashed_bytes.decode('utf-8')

def verify_password(plain: str, hashed: str) -> bool:
    """Verifies a plain text password against a hashed password."""
    try:
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except (ValueError, TypeError):
        return False

def create_access_token(user_id: str, expires_delta: timedelta | None = None, *, session_id: str | None = None, token_type: str = "access") -> str:
    """Creates a JWT access token encoding the user_id and expiration time."""
    expire = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode = {"exp": expire, "iat": datetime.now(timezone.utc), "sub": str(user_id), "type": token_type, "jti": str(uuid.uuid4())}
    if session_id is not None:
        to_encode["sid"] = str(session_id)
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str, token_type: str = "access") -> dict:
    """Decodes a JWT token and returns the payload dict. Raises jwt.PyJWTError if invalid."""
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM], options={"require": ["sub", "exp", "iat", "jti", "type"]})
    if payload["type"] != token_type:
        raise jwt.InvalidTokenError("Incorrect token type")
    return payload


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
