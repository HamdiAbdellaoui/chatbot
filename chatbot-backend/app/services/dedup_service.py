"""Webhook idempotence: process each Chatwoot message id only once.

Chatwoot may deliver the same `message_created` webhook more than once
(retries after a timeout, replays). Without this guard the customer would get
duplicate replies.

- Redis (REDIS_URL set): SET dedup:msg:{message_id} 1 NX EX 600, shared by all workers.
- Otherwise, or if Redis fails: bounded in-memory cache (per process).
Never raises.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

from app.config import settings
from app.services.session_service import get_redis_client

logger = logging.getLogger(__name__)

DEDUP_TTL_S = 600
_REDIS_TIMEOUT_S = 1.0
_MEMORY_MAX_ENTRIES = 10_000

# message key -> expiry (monotonic seconds), oldest first.
_memory_cache: "OrderedDict[str, float]" = OrderedDict()


def _key(message_id: int) -> str:
    return f"dedup:msg:{message_id}"


def _seen_in_memory(key: str) -> bool:
    now = time.monotonic()
    # Drop expired entries (the dict is ordered by insertion = expiry order).
    while _memory_cache:
        oldest_key, expiry = next(iter(_memory_cache.items()))
        if expiry > now:
            break
        _memory_cache.popitem(last=False)

    if key in _memory_cache:
        return True

    _memory_cache[key] = now + DEDUP_TTL_S
    while len(_memory_cache) > _MEMORY_MAX_ENTRIES:
        _memory_cache.popitem(last=False)
    return False


def clear_memory_cache() -> None:
    _memory_cache.clear()


async def is_duplicate_message(message_id: int | None) -> bool:
    """Return True if this message id was already seen within DEDUP_TTL_S, else mark it as seen."""
    if message_id is None:
        return False

    key = _key(message_id)

    if (settings.REDIS_URL or "").strip():
        try:
            client = await asyncio.wait_for(get_redis_client(), timeout=_REDIS_TIMEOUT_S)
            created = await asyncio.wait_for(client.set(key, "1", nx=True, ex=DEDUP_TTL_S), timeout=_REDIS_TIMEOUT_S)
            return not created
        except Exception as e:
            logger.warning("Redis dedup unavailable (%s); falling back to in-memory dedup", type(e).__name__)

    return _seen_in_memory(key)
