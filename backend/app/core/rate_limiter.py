import time
import redis.asyncio as redis
from typing import Dict, Any

from app.config import settings

class RateLimiter:
    def __init__(self):
        self.client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.PREFIX = "ratelimit:"
        
        # RPM = Requests Per Minute
        self.LIMITS = {
            "free": 10,
            "pro": 60,
            "enterprise": 1000000
        }

    async def check(self, user_id: str, tier: str = "free") -> Dict[str, Any]:
        """
        Implements a Sliding Window rate limit using Redis Sorted Sets.
        Returns: {allowed: bool, remaining: int, reset_in: int}
        """
        key = f"{self.PREFIX}{user_id}"
        limit = self.LIMITS.get(tier, self.LIMITS["free"])
        
        now = time.time()
        window_start = now - 60.0  # 1 minute sliding window
        
        # 1. Remove timestamps older than the sliding window (clean up)
        await self.client.zremrangebyscore(key, 0, window_start)
        
        # 2. Count how many requests occurred in the last minute
        request_count = await self.client.zcard(key)
        
        if request_count >= limit:
            # Look up the oldest timestamp to determine when the user gets 1 token back
            oldest_requests = await self.client.zrange(key, 0, 0, withscores=True)
            reset_in = 60
            if oldest_requests:
                oldest_ts = oldest_requests[0][1]
                reset_in = max(1, int(60 - (now - oldest_ts)))
                
            return {
                "allowed": False,
                "remaining": 0,
                "reset_in": reset_in
            }
            
        # 3. Add the current request timestamp to the sorted set
        # Score = timestamp, Member = timestamp_uuid
        member = f"{now}"
        await self.client.zadd(key, {member: now})
        
        # 4. Set expiry on the key to prevent orphaned keys from consuming memory
        await self.client.expire(key, 60)
        
        return {
            "allowed": True,
            "remaining": (limit - request_count - 1),
            "reset_in": 0
        }

rate_limiter = RateLimiter()
