from typing import List, Dict, Any, Tuple
from pydantic import BaseModel

class ContextPlan(BaseModel):
    messages_kept: int
    chunks_kept: int
    tokens_used: int
    tokens_remaining: int

class ContextBudgetManager:
    def __init__(self):
        # Gemini 1.5 Flash has a massive 1M window, but Llama is 128k. 
        # We enforce limits to prevent latency/cost spikes.
        self.MODEL_LIMITS = {
            "gemini-1.5-flash": 1_000_000,
            "llama3.1:8b": 128_000,
            "default": 128_000
        }
        
        # We strictly allocate percentages of the total budget to avoid one area dominating
        self.BUDGET_ALLOCATION = {
            "system": 0.05,  # 5% for rules
            "tools": 0.10,   # 10% for tool schemas
            "history": 0.20, # 20% for past conversation context
            "context": 0.50, # 50% for the physical document chunks
            "output": 0.15   # 15% reserved space for the LLM to write the answer
        }

    def count_tokens(self, text: str) -> int:
        """
        Estimates the number of tokens in a string.
        A standard heuristic is 4 characters per token.
        For production accuracy, use the `tiktoken` library.
        """
        if not text:
            return 0
        return len(text) // 4

    def _sliding_window(self, messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Keeps only the most recent messages that fit within the history budget.
        Trims from the oldest (start of list).
        """
        kept_messages = []
        current_tokens = 0
        
        # Iterate backwards (newest to oldest)
        for msg in reversed(messages):
            msg_text = msg.get("content", "")
            if isinstance(msg_text, str):
                tokens = self.count_tokens(msg_text)
            else:
                tokens = 100 # Tool calls rough estimate
                
            if current_tokens + tokens > max_tokens:
                break
                
            current_tokens += tokens
            kept_messages.insert(0, msg) # Insert at beginning to maintain chronological order
            
        return kept_messages

    def _fit_chunks(self, chunks: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Keeps chunks that fit within the context budget, prioritizing the highest scored ones.
        """
        kept_chunks = []
        current_tokens = 0
        
        # Chunks are assumed to be pre-sorted by Rerank/RRF score
        for chunk in chunks:
            text = chunk.get("payload", {}).get("text", "")
            tokens = self.count_tokens(text)
            
            if current_tokens + tokens > max_tokens:
                break
                
            current_tokens += tokens
            kept_chunks.append(chunk)
            
        return kept_chunks

    def allocate(self, model: str, history: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Tuple[ContextPlan, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Enforces the budget constraints, trimming history and chunks as needed.
        """
        limit = self.MODEL_LIMITS.get(model, self.MODEL_LIMITS["default"])
        
        # Calculate maximum allowed tokens for each category
        history_budget = int(limit * self.BUDGET_ALLOCATION["history"])
        context_budget = int(limit * self.BUDGET_ALLOCATION["context"])
        
        trimmed_history = self._sliding_window(history, history_budget)
        trimmed_chunks = self._fit_chunks(chunks, context_budget)
        
        # Calculate used
        history_used = sum(self.count_tokens(str(m.get("content", ""))) for m in trimmed_history)
        chunks_used = sum(self.count_tokens(c.get("payload", {}).get("text", "")) for c in trimmed_chunks)
        
        plan = ContextPlan(
            messages_kept=len(trimmed_history),
            chunks_kept=len(trimmed_chunks),
            tokens_used=history_used + chunks_used,
            tokens_remaining=limit - (history_used + chunks_used)
        )
        
        return plan, trimmed_history, trimmed_chunks

context_budget_manager = ContextBudgetManager()
