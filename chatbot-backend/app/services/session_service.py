"""Conversation session memory: short multi-turn history per Chatwoot conversation.

Design goals:
- Optional and resilient: never raise to the caller. On any storage error,
  log and behave as if there is no history.
- Backend selection: Redis (fast, has native TTL) if REDIS_URL is configured,
  else PostgreSQL (shared asyncpg pool from app.db) if APP_DATABASE_URL is
  configured (it falls back to ACTIVE_LEARNING_DATABASE_URL), else a no-op
  in-memory-empty mode.
- History is stored as individual turns (one role/content entry per call to
  append_turn), truncated FIFO to the last SESSION_HISTORY_TURNS entries.
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any, Dict, List

from app.config import settings
from app.db import get_pool

logger = logging.getLogger(__name__)


def _history_key(conversation_id: int) -> str:
    return f"session:history:{conversation_id}"


# --- Redis backend -----------------------------------------------------

@lru_cache(maxsize=1)
def _get_redis_module():
    try:
        import redis.asyncio as redis  # type: ignore
    except Exception as exc:
        raise RuntimeError("redis is required for Redis-backed session history") from exc
    return redis


_redis_client = None
_redis_lock = asyncio.Lock()


async def _get_redis_client():
    redis = _get_redis_module()
    url = (settings.REDIS_URL or "").strip()
    if not url:
        raise RuntimeError("REDIS_URL is not configured")

    global _redis_client
    if _redis_client is not None:
        return _redis_client

    async with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        _redis_client = redis.from_url(url, decode_responses=True)
        return _redis_client


async def get_redis_client():
    """Shared Redis client (REDIS_URL). Raises if Redis is not configured/installed."""
    return await _get_redis_client()


async def _get_history_redis(conversation_id: int) -> List[Dict[str, str]]:
    client = await _get_redis_client()
    raw_entries = await client.lrange(_history_key(conversation_id), 0, -1)
    history: List[Dict[str, str]] = []
    for raw in raw_entries:
        try:
            entry = json.loads(raw)
            history.append({"role": entry["role"], "content": entry["content"]})
        except Exception:
            logger.warning("Skipping malformed session history entry (conversation_id=%s)", conversation_id)
    return history


async def _append_turn_redis(conversation_id: int, role: str, content: str) -> None:
    client = await _get_redis_client()
    key = _history_key(conversation_id)
    turns = max(1, int(settings.SESSION_HISTORY_TURNS))
    entry = json.dumps({"role": role, "content": content}, ensure_ascii=False)
    async with client.pipeline(transaction=True) as pipe:
        pipe.rpush(key, entry)
        pipe.ltrim(key, -turns, -1)
        pipe.expire(key, max(1, int(settings.SESSION_TTL_SECONDS)))
        await pipe.execute()


# --- PostgreSQL fallback backend ---------------------------------------

_pg_schema_ready = False
_pg_schema_lock = asyncio.Lock()


async def _get_pg_pool():
    dsn = (settings.APP_DATABASE_URL or "").strip()
    if not dsn:
        raise RuntimeError("APP_DATABASE_URL is not configured")

    pool = await get_pool(dsn)

    global _pg_schema_ready
    if not _pg_schema_ready:
        async with _pg_schema_lock:
            if not _pg_schema_ready:
                await _init_pg_schema(pool)
                _pg_schema_ready = True
    return pool


async def _init_pg_schema(pool) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS conversation_history (
        id BIGSERIAL PRIMARY KEY,
        conversation_id BIGINT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_conversation_history_conversation_id
        ON conversation_history (conversation_id, id);
    """
    stmts = [s.strip() for s in ddl.split(";") if s.strip()]
    async with pool.acquire() as conn:
        for stmt in stmts:
            await conn.execute(stmt)


async def _get_history_pg(conversation_id: int) -> List[Dict[str, str]]:
    pool = await _get_pg_pool()
    turns = max(1, int(settings.SESSION_HISTORY_TURNS))
    ttl_seconds = max(1, int(settings.SESSION_TTL_SECONDS))

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content FROM (
                SELECT id, role, content
                FROM conversation_history
                WHERE conversation_id = $1
                  AND created_at > now() - ($2 || ' seconds')::interval
                ORDER BY id DESC
                LIMIT $3
            ) sub
            ORDER BY id ASC
            """,
            conversation_id,
            str(ttl_seconds),
            turns,
        )
    return [{"role": r["role"], "content": r["content"]} for r in rows]


async def _append_turn_pg(conversation_id: int, role: str, content: str) -> None:
    pool = await _get_pg_pool()
    turns = max(1, int(settings.SESSION_HISTORY_TURNS))

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO conversation_history (conversation_id, role, content) VALUES ($1, $2, $3)",
            conversation_id,
            role,
            content,
        )
        await conn.execute(
            """
            DELETE FROM conversation_history
            WHERE conversation_id = $1
              AND id NOT IN (
                  SELECT id FROM conversation_history
                  WHERE conversation_id = $1
                  ORDER BY id DESC
                  LIMIT $2
              )
            """,
            conversation_id,
            turns,
        )


# --- Public API ----------------------------------------------------------

@lru_cache(maxsize=1)
def _warned_no_backend() -> Dict[str, Any]:
    return {"warned": False}


def _backend_configured() -> str | None:
    if (settings.REDIS_URL or "").strip():
        return "redis"
    if (settings.APP_DATABASE_URL or "").strip():
        return "postgres"
    return None


async def get_history(conversation_id: int) -> List[Dict[str, str]]:
    """Return the last SESSION_HISTORY_TURNS turns for this conversation.

    Never raises: any storage error results in an empty history so the
    caller can continue processing the request.
    """
    backend = _backend_configured()
    if backend is None:
        state = _warned_no_backend()
        if not state["warned"]:
            logger.info("No session history backend configured (REDIS_URL/APP_DATABASE_URL unset); running without conversation memory")
            state["warned"] = True
        return []

    try:
        if backend == "redis":
            return await _get_history_redis(conversation_id)
        return await _get_history_pg(conversation_id)
    except Exception:
        logger.exception("Failed to load session history (conversation_id=%s); continuing with empty history", conversation_id)
        return []


async def append_turn(conversation_id: int, role: str, content: str) -> None:
    """Append one turn to the conversation history, truncated FIFO.

    Never raises: any storage error is logged and swallowed.
    """
    backend = _backend_configured()
    if backend is None:
        return

    try:
        if backend == "redis":
            await _append_turn_redis(conversation_id, role, content)
        else:
            await _append_turn_pg(conversation_id, role, content)
    except Exception:
        logger.exception("Failed to append session turn (conversation_id=%s, role=%s); continuing", conversation_id, role)
