from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class StreamEvent(BaseModel):
    token: str = ""
    done: bool = False
    citations: Optional[List[Dict[str, Any]]] = None
    tokens_used: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
