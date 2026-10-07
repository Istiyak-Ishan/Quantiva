from pydantic import BaseModel, ConfigDict
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class DocumentUploadResponse(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    message: str = "File uploaded successfully and is being processed."

class DocumentStatusResponse(BaseModel):
    id: uuid.UUID
    status: str
    chunk_count: int
