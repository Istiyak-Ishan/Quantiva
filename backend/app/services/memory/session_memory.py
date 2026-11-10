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

    async def load_history(self, session_id: str, max_turns: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves the conversation history from Redis in chronological order.
        """
        key = f"{self.PREFIX}{session_id}"
        try:
            # Fetch the most recent messages (up to max_turns)
            raw_msgs = await self.client.lrange(key, -max_turns, -1)
            messages = [json.loads(m) for m in raw_msgs]
            return messages
        except Exception:
            return []

    async def save_turn(self, session_id: str, user_msg: Dict[str, Any], assistant_msg: Dict[str, Any]):
        """
        Saves the turn (user query + ai response) to Redis in chronological order using RPUSH.
        """
        key = f"{self.PREFIX}{session_id}"
        try:
            await self.client.rpush(key, json.dumps(user_msg), json.dumps(assistant_msg))
            await self.client.ltrim(key, -100, -1)
        except Exception:
            pass

    async def clear(self, session_id: str):
        """Clears the session from Redis."""
        key = f"{self.PREFIX}{session_id}"
        try:
            await self.client.delete(key)
        except Exception:
            pass

session_memory = SessionMemory()
