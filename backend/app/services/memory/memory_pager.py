from typing import List, Dict, Any
from app.services.llm.gemini_client import gemini_client
from app.services.memory.session_memory import session_memory

class MemoryPager:
    def __init__(self):
        pass

    async def summarize(self, session_id: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        If a conversation gets too long for the LLM context window, 
        we use the LLM itself to compress the oldest turns into a dense summary.
        """
        
        system_prompt = (
            "You are an AI Memory Compressor. Your job is to take the following conversation history "
            "and compress it into a dense, factual summary. Preserve all entities, dates, numbers, "
            "and key decisions. Do not write fluff. Write it as context for a future AI."
        )
        
        # Convert history into a string for the prompt
        history_text = ""
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            history_text += f"{role.upper()}: {content}\n\n"
            
        messages = [{"role": "user", "content": f"Summarize this conversation:\n\n{history_text}"}]
        
        # Call the LLM to generate the summary
        summary_text = gemini_client.complete(messages, system_prompt)
        
        # This summary message will replace the raw history
        summary_msg = {
            "role": "system",
            "content": f"Previous conversation summary: {summary_text}"
        }
        
        return summary_msg

memory_pager = MemoryPager()
