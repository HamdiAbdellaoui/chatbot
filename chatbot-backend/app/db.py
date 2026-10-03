"""Shared PostgreSQL access: asyncpg module loading and per-DSN pool caching.

Several services (active learning flags, conversation logs, etc.) each need a
PostgreSQL connection pool. This module centralizes creation so that services
sharing the same DSN reuse the same underlying asyncpg pool instead of each
opening its own.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def get_asyncpg():
    try:
        import asyncpg  # type: ignore
    except Exception as exc:
        raise RuntimeError("asyncpg is required for PostgreSQL access") from exc
    return asyncpg


_pools: Dict[str, object] = {}
_pools_lock = asyncio.Lock()


async def get_pool(dsn: str):
    """Return a cached asyncpg pool for this DSN, creating it on first use."""
    dsn = (dsn or "").strip()
    if not dsn:
        raise RuntimeError("A PostgreSQL DSN is required to get a connection pool")

    if dsn in _pools:
        return _pools[dsn]

    async with _pools_lock:
        if dsn in _pools:
            return _pools[dsn]
        asyncpg = get_asyncpg()
        pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5, command_timeout=10)
        _pools[dsn] = pool
        return pool
