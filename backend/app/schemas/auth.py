from pydantic import BaseModel, EmailStr, ConfigDict
import uuid
from datetime import datetime

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

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
