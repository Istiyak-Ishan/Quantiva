from pydantic import BaseModel, EmailStr, ConfigDict, field_validator, Field
import uuid
from datetime import datetime

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_policy(cls, value: str) -> str:
        if len(value) < 8 or len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at least 8 characters and at most 72 UTF-8 bytes")
        return value

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class RefreshRequest(BaseModel):
    refresh_token: str | None = None

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: uuid.UUID
    email: EmailStr
    tier: str
    is_active: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
