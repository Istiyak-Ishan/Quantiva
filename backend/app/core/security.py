import bcrypt
import jwt
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
    return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(user_id: str, expires_delta: timedelta | None = None) -> str:
    """Creates a JWT access token encoding the user_id and expiration time."""
    expire = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode = {"exp": expire, "sub": str(user_id)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> dict:
    """Decodes a JWT token and returns the payload dict. Raises jwt.PyJWTError if invalid."""
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
