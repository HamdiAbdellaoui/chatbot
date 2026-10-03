"""Active learning loop: flag low-confidence interactions for review.

Design goals:
- Optional (safe no-op when not configured)
- Store only masked user text + retrieval metadata (avoid raw PII)
- Use PostgreSQL (same DB as Chatwoot is fine) via asyncpg
"""

from __future__ import annotations

import json
import logging
import asyncio
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.config import settings
from app.db import get_asyncpg as _get_asyncpg, get_pool as _get_shared_pool

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _warned_missing_dsn() -> set[str]:
    return set()


@lru_cache(maxsize=1)
def _warned_disabled() -> bool:
    return False


_schema_ready = False
_schema_lock = asyncio.Lock()


async def _get_pool():
    dsn = (settings.ACTIVE_LEARNING_DATABASE_URL or "").strip()
    if not dsn:
        raise RuntimeError("ACTIVE_LEARNING_DATABASE_URL is not configured")

    pool = await _get_shared_pool(dsn)

    global _schema_ready
    if not _schema_ready:
        async with _schema_lock:
            if not _schema_ready:
                await _init_schema(pool)
                _schema_ready = True

    return pool


async def _init_schema(pool) -> None:
    # This table is intentionally simple and append-only.
    # Avoid storing raw PII: use masked_user_message.
    ddl = """
    CREATE TABLE IF NOT EXISTS active_learning_flags (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        event_type TEXT NOT NULL,
        store_key TEXT NULL,
        inbox_id BIGINT NULL,
        conversation_id BIGINT NULL,
        reason TEXT NULL,
        top_score DOUBLE PRECISION NULL,
        hits_count INTEGER NULL,
        sources JSONB NULL,
        masked_user_message TEXT NULL,
        model TEXT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_active_learning_flags_created_at
        ON active_learning_flags (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_active_learning_flags_conversation_id
        ON active_learning_flags (conversation_id);
    CREATE INDEX IF NOT EXISTS idx_active_learning_flags_store_key
        ON active_learning_flags (store_key);
    """

    # asyncpg executes only one statement at a time; split.
    stmts = [s.strip() for s in ddl.split(";") if s.strip()]
    async with pool.acquire() as conn:
        for stmt in stmts:
            await conn.execute(stmt)


async def log_low_confidence_flag(
    *,
    store_key: str | None,
    inbox_id: int | None,
    conversation_id: int | None,
    reason: str | None,
    top_score: float | None,
    hits_count: int,
    sources: List[str] | None,
    masked_user_message: str | None,
) -> None:
    """Persist a low-confidence event for later review.

    Safe behavior:
    - If disabled or not configured, do nothing.
    - Never raises to caller.
    """
    if not settings.ACTIVE_LEARNING_ENABLED:
        # avoid spamming logs
        return

    dsn = (settings.ACTIVE_LEARNING_DATABASE_URL or "").strip()
    if not dsn:
        if "missing_dsn" not in _warned_missing_dsn():
            logger.warning("Active learning enabled but ACTIVE_LEARNING_DATABASE_URL is not set; skipping logging")
            _warned_missing_dsn().add("missing_dsn")
        return

    try:
        pool = await _get_pool()
        payload_sources = sources or []
        asyncpg = _get_asyncpg()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO active_learning_flags (
                    event_type,
                    store_key,
                    inbox_id,
                    conversation_id,
                    reason,
                    top_score,
                    hits_count,
                    sources,
                    masked_user_message,
                    model
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10
                )
                """,
                "low_confidence",
                store_key,
                inbox_id,
                conversation_id,
                reason,
                top_score,
                int(hits_count),
                asyncpg.types.Json(payload_sources),
                masked_user_message,
                settings.OPENAI_MODEL,
            )
    except Exception:
        logger.exception("Active learning logging failed")
        return


async def list_flags(*, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """List active learning flags (most recent first)."""
    pool = await _get_pool()
    safe_limit = max(1, min(int(limit), 200))
    safe_offset = max(0, int(offset))

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id,
                created_at,
                event_type,
                store_key,
                inbox_id,
                conversation_id,
                reason,
                top_score,
                hits_count,
                sources,
                masked_user_message,
                model
            FROM active_learning_flags
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            """,
            safe_limit,
            safe_offset,
        )

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": int(r["id"]),
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "event_type": r["event_type"],
                "store_key": r["store_key"],
                "inbox_id": r["inbox_id"],
                "conversation_id": r["conversation_id"],
                "reason": r["reason"],
                "top_score": r["top_score"],
                "hits_count": r["hits_count"],
                "sources": r["sources"],
                "masked_user_message": r["masked_user_message"],
                "model": r["model"],
            }
        )
    return out


async def get_flag(*, flag_id: int) -> Optional[Dict[str, Any]]:
    """Get a single flag by id."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        r = await conn.fetchrow(
            """
            SELECT
                id,
                created_at,
                event_type,
                store_key,
                inbox_id,
                conversation_id,
                reason,
                top_score,
                hits_count,
                sources,
                masked_user_message,
                model
            FROM active_learning_flags
            WHERE id = $1
            """,
            int(flag_id),
        )
    if not r:
        return None

    return {
        "id": int(r["id"]),
        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        "event_type": r["event_type"],
        "store_key": r["store_key"],
        "inbox_id": r["inbox_id"],
        "conversation_id": r["conversation_id"],
        "reason": r["reason"],
        "top_score": r["top_score"],
        "hits_count": r["hits_count"],
        "sources": r["sources"],
        "masked_user_message": r["masked_user_message"],
        "model": r["model"],
    }
