import asyncio
from typing import List, Dict, Any

from app.services.llm.gemini_client import gemini_client


class MemoryPager:
    """Compress history that falls outside the deterministic context budget."""

    async def summarize(self, session_id: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Replace the oldest turns with a dense factual summary when a
        conversation exceeds the model-context budget.
        """
        system_prompt = (
            "You are an AI Memory Compressor. Your job is to take the following conversation history "
            "and compress it into a dense, factual summary. Preserve all entities, dates, numbers, "
            "and key decisions. Do not write fluff. Write it as context for a future AI."
        )

        history_text = ""
        for message in history:
            role = message.get("role", "user")
            content = message.get("content", "")
            history_text += f"{role.upper()}: {content}\n\n"

        messages = [{"role": "user", "content": f"Summarize this conversation:\n\n{history_text}"}]
        # Use the shared async transport rather than blocking the event loop.
        summary_text = await asyncio.to_thread(gemini_client.complete, messages, system_prompt)

        return {
            "role": "system",
            "content": f"Previous conversation summary: {summary_text}",
        }


memory_pager = MemoryPager()
