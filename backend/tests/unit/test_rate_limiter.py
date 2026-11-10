from unittest.mock import AsyncMock

import pytest
import redis.exceptions
from fastapi import HTTPException

from app.core.rate_limiter import RateLimiter


async def test_redis_outage_does_not_disable_limits():
    limiter = RateLimiter()
    limiter.client = AsyncMock()
    limiter.client.eval.side_effect = redis.exceptions.ConnectionError("unavailable")
    with pytest.raises(HTTPException) as error:
        await limiter.check("user")
    assert error.value.status_code == 503


async def test_rate_limit_uses_atomic_script_and_tier():
    limiter = RateLimiter()
    limiter.client = AsyncMock()
    limiter.client.eval.return_value = [0, 0, 42]
    result = await limiter.check("user", "pro")
    assert result == {"allowed": False, "remaining": 0, "reset_in": 42}
    args = limiter.client.eval.call_args.args
    assert args[1] == 1
    assert args[2] == "ratelimit:chat:user"
    assert args[4] == 60
