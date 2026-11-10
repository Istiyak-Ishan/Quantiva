from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class StreamEvent(BaseModel):
    token: str = ""
    done: bool = False
    citations: Optional[List[Dict[str, Any]]] = None
    tokens_used: Optional[int] = None
    answer: Optional[str] = None
    confidence: Optional[float] = None
    chart_url: Optional[str] = None
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_hit: bool = False
    guardrail_flags: List[str] = []

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
