"""Full conversation logging: persist every inbound/outbound turn for observability.

Design goals (mirrors active_learning_service.py):
- Optional (safe no-op when APP_DATABASE_URL is not configured)
- Store only masked text (content_masked); never raw user text
- Use PostgreSQL via asyncpg, pool shared through app.db.get_pool
- Never raises to the caller: log the error and return
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import List, Optional

from app.config import settings
from app.db import get_asyncpg, get_pool

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _warned_missing_dsn() -> set[str]:
    return set()


_schema_ready = False
_schema_lock = asyncio.Lock()


async def _get_pool():
    dsn = (settings.APP_DATABASE_URL or "").strip()
    if not dsn:
        raise RuntimeError("APP_DATABASE_URL is not configured")

    pool = await get_pool(dsn)

    global _schema_ready
    if not _schema_ready:
        async with _schema_lock:
            if not _schema_ready:
                await _init_schema(pool)
                _schema_ready = True

    return pool


async def _init_schema(pool) -> None:
    # Append-only log. Only masked content is ever stored (see log_turn).
    ddl = """
    CREATE TABLE IF NOT EXISTS conversation_logs (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        conversation_id BIGINT NULL,
        inbox_id BIGINT NULL,
        store_key TEXT NULL,
        direction TEXT NOT NULL,
        content_masked TEXT NULL,
        pii_types JSONB NULL,
        rag_top_score DOUBLE PRECISION NULL,
        rag_hits_count INTEGER NULL,
        confidence_score DOUBLE PRECISION NULL,
        escalated BOOLEAN NOT NULL DEFAULT false,
        escalation_reason TEXT NULL,
        model TEXT NULL,
        latency_ms INTEGER NULL
    );

    CREATE INDEX IF NOT EXISTS idx_conversation_logs_created_at
        ON conversation_logs (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_conversation_logs_conversation_id
        ON conversation_logs (conversation_id);
    CREATE INDEX IF NOT EXISTS idx_conversation_logs_store_key
        ON conversation_logs (store_key);
    """

    # asyncpg executes only one statement at a time; split.
    stmts = [s.strip() for s in ddl.split(";") if s.strip()]
    async with pool.acquire() as conn:
        for stmt in stmts:
            await conn.execute(stmt)


async def log_turn(
    *,
    conversation_id: Optional[int],
    inbox_id: Optional[int],
    store_key: Optional[str],
    direction: str,
    content_masked: Optional[str],
    pii_types: Optional[List[str]] = None,
    rag_top_score: Optional[float] = None,
    rag_hits_count: Optional[int] = None,
    confidence_score: Optional[float] = None,
    escalated: bool = False,
    escalation_reason: Optional[str] = None,
    model: Optional[str] = None,
    latency_ms: Optional[int] = None,
) -> None:
    """Persist one conversational turn ("in" or "out") for observability.

    Safe behavior:
    - If APP_DATABASE_URL is not configured, do nothing (warn once).
    - Never raises to caller.
    - Only ever stores already-masked content; callers must never pass raw text.
    """
    dsn = (settings.APP_DATABASE_URL or "").strip()
    if not dsn:
        if "missing_dsn" not in _warned_missing_dsn():
            logger.warning("APP_DATABASE_URL is not configured; skipping conversation logging")
            _warned_missing_dsn().add("missing_dsn")
        return

    try:
        pool = await _get_pool()
        asyncpg = get_asyncpg()
        pii_types_json = asyncpg.types.Json(pii_types) if pii_types is not None else None

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversation_logs (
                    conversation_id,
                    inbox_id,
                    store_key,
                    direction,
                    content_masked,
                    pii_types,
                    rag_top_score,
                    rag_hits_count,
                    confidence_score,
                    escalated,
                    escalation_reason,
                    model,
                    latency_ms
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13
                )
                """,
                conversation_id,
                inbox_id,
                store_key,
                direction,
                content_masked,
                pii_types_json,
                rag_top_score,
                rag_hits_count,
                confidence_score,
                bool(escalated),
                escalation_reason,
                model,
                latency_ms,
            )
    except Exception:
        logger.exception(
            "Conversation logging failed (conversation_id=%s direction=%s)",
            conversation_id,
            direction,
        )
        return
