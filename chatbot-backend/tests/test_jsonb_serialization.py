"""JSONB parameters must be sent to asyncpg as JSON strings.

asyncpg has no ``asyncpg.types.Json`` wrapper: its default codec for ``json``
and ``jsonb`` columns expects a ``str``. The fake asyncpg module used here has
no ``types`` attribute at all, so any regression back to ``asyncpg.types.Json``
fails loudly instead of being hidden by a permissive mock.
"""

import asyncio
import inspect
import json

from app.config import settings
from app.services import active_learning_service, conversation_log_service


class _AsyncpgWithoutTypes:
    """Stand-in for the asyncpg module that deliberately lacks ``types``."""


class _RecordingConnection:
    def __init__(self, calls: list):
        self._calls = calls

    async def execute(self, query, *args):
        self._calls.append((" ".join(query.split()), args))


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _RecordingPool:
    def __init__(self):
        self.calls: list = []
        self._conn = _RecordingConnection(self.calls)

    def acquire(self):
        return _AcquireCtx(self._conn)

    def inserts_into(self, table: str):
        return [args for query, args in self.calls if query.startswith(f"INSERT INTO {table}")]


def test_log_turn_sends_pii_types_as_json_string(monkeypatch):
    pool = _RecordingPool()

    async def fake_get_pool(dsn):
        return pool

    monkeypatch.setattr(conversation_log_service, "get_pool", fake_get_pool)
    monkeypatch.setattr(conversation_log_service, "get_asyncpg", lambda: _AsyncpgWithoutTypes)
    monkeypatch.setattr(conversation_log_service, "_schema_ready", False)
    monkeypatch.setattr(settings, "APP_DATABASE_URL", "postgresql://fake:5432/fake_db")

    asyncio.run(
        conversation_log_service.log_turn(
            conversation_id=1,
            inbox_id=2,
            store_key="default",
            direction="in",
            content_masked="Mon e-mail est [EMAIL]",
            pii_types=["EMAIL", "PHONE"],
        )
    )

    inserts = pool.inserts_into("conversation_logs")
    assert len(inserts) == 1
    pii_types_param = inserts[0][5]
    assert isinstance(pii_types_param, str)
    assert json.loads(pii_types_param) == ["EMAIL", "PHONE"]


def test_log_turn_sends_null_pii_types_when_absent(monkeypatch):
    pool = _RecordingPool()

    async def fake_get_pool(dsn):
        return pool

    monkeypatch.setattr(conversation_log_service, "get_pool", fake_get_pool)
    monkeypatch.setattr(conversation_log_service, "get_asyncpg", lambda: _AsyncpgWithoutTypes)
    monkeypatch.setattr(conversation_log_service, "_schema_ready", False)
    monkeypatch.setattr(settings, "APP_DATABASE_URL", "postgresql://fake:5432/fake_db")

    asyncio.run(
        conversation_log_service.log_turn(
            conversation_id=1,
            inbox_id=2,
            store_key="default",
            direction="out",
            content_masked="Bonjour !",
        )
    )

    inserts = pool.inserts_into("conversation_logs")
    assert len(inserts) == 1
    assert inserts[0][5] is None


def test_low_confidence_flag_sends_sources_as_json_string(monkeypatch):
    pool = _RecordingPool()

    async def fake_get_pool(dsn):
        return pool

    monkeypatch.setattr(active_learning_service, "_get_shared_pool", fake_get_pool)
    monkeypatch.setattr(active_learning_service, "_get_asyncpg", lambda: _AsyncpgWithoutTypes)
    monkeypatch.setattr(active_learning_service, "_schema_ready", False)
    monkeypatch.setattr(settings, "ACTIVE_LEARNING_ENABLED", True)
    monkeypatch.setattr(settings, "ACTIVE_LEARNING_DATABASE_URL", "postgresql://fake:5432/fake_db")

    for sources, expected in ((["faq.md", "retours.md"], ["faq.md", "retours.md"]), (None, [])):
        pool.calls.clear()
        asyncio.run(
            active_learning_service.log_low_confidence_flag(
                store_key="default",
                inbox_id=2,
                conversation_id=1,
                reason="low_rag_score",
                top_score=0.12,
                hits_count=2,
                sources=sources,
                masked_user_message="Bonjour [NAME]",
            )
        )

        inserts = pool.inserts_into("active_learning_flags")
        assert len(inserts) == 1
        sources_param = inserts[0][7]
        assert isinstance(sources_param, str)
        assert json.loads(sources_param) == expected


def test_services_do_not_reference_asyncpg_types_json():
    for module in (active_learning_service, conversation_log_service):
        assert "types.Json" not in inspect.getsource(module)
