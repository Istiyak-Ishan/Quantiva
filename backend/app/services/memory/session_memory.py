import json
from typing import List, Dict, Any
import redis.asyncio as redis
from app.config import settings

class SessionMemory:
    def __init__(self):
        # We use Redis because conversation memory is "hot" data.
        # It needs to be read and written in sub-millisecond time.
        # Postgres is used for durable, long-term storage of these chats.
        self.client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.PREFIX = "chat:memory:"

    async def load_history(self, session_id: str, max_turns: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the conversation history from Redis.
        max_turns=10 means we only pull the last 10 messages from fast cache.
        """
        key = f"{self.PREFIX}{session_id}"
        # LRANGE retrieves elements from a list. 0 to max_turns-1 gets the most recent.
        raw_msgs = await self.client.lrange(key, 0, max_turns - 1)
        
        # Redis lists pull newest first if we LPUSH, so we need to reverse them
        # back to chronological order (oldest -> newest) for the LLM
        messages = [json.loads(m) for m in raw_msgs]
        return list(reversed(messages))

    async def save_turn(self, session_id: str, user_msg: Dict[str, Any], assistant_msg: Dict[str, Any]):
        """
        Saves the turn (user query + ai response) to Redis.
        We use LPUSH (prepend) so the newest messages are at index 0.
        We use LTRIM to strictly maintain a maximum history size (sliding window in cache).
        """
        key = f"{self.PREFIX}{session_id}"
        
        # Push both messages. LPUSH pushes to the front.
        # So we push assistant first, then user, so user ends up at [0] and assistant at [1]
        await self.client.lpush(key, json.dumps(assistant_msg), json.dumps(user_msg))
        
        # Keep only the last 100 messages (50 turns) in hot memory
        await self.client.ltrim(key, 0, 99)

    async def clear(self, session_id: str):
        """Clears the session from Redis."""
        key = f"{self.PREFIX}{session_id}"
        await self.client.delete(key)

session_memory = SessionMemory()
