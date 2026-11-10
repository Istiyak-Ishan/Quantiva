"""Atomic Redis sliding windows. Expensive/auth operations fail closed."""
import time
import uuid

from fastapi import HTTPException
import redis.asyncio as redis

from app.config import settings

_WINDOW = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1] - 60)
local count = redis.call('ZCARD', KEYS[1])
if count >= tonumber(ARGV[2]) then
    local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
    return {0, 0, math.max(1, math.ceil(60 - (ARGV[1] - oldest[2])))}
end
redis.call('ZADD', KEYS[1], ARGV[1], ARGV[3])
redis.call('EXPIRE', KEYS[1], 60)
return {1, tonumber(ARGV[2]) - count - 1, 0}
"""


class RateLimiter:
    LIMITS = {"free": 10, "pro": 60, "enterprise": 600}

    def __init__(self):
        self.client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)

    async def check(self, user_id: str, tier: str = "free", *, scope: str = "chat", limit: int | None = None) -> dict:
        ceiling = limit if limit is not None else self.LIMITS.get(tier, self.LIMITS["free"])
        try:
            allowed, remaining, reset_in = await self.client.eval(_WINDOW, 1, f"ratelimit:{scope}:{user_id}", time.time(), ceiling, uuid.uuid4().hex)
        except redis.RedisError:
            raise HTTPException(status_code=503, detail="Rate limiting temporarily unavailable", headers={"Retry-After": "5"})
        return {"allowed": bool(allowed), "remaining": int(remaining), "reset_in": int(reset_in)}


rate_limiter = RateLimiter()


async def enforce_rate_limit(subject: str, *, tier: str = "free", scope: str = "chat", limit: int | None = None):
    result = await rate_limiter.check(subject, tier, scope=scope, limit=limit)
    if not result["allowed"]:
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": str(result["reset_in"])})
