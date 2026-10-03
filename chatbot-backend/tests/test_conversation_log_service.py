import asyncio
import inspect

from app.config import settings
from app.services import conversation_log_service


class _FakeJson:
    def __init__(self, value):
        self.value = value


class _FakeTypes:
    Json = _FakeJson


class _FakeAsyncpgModule:
    types = _FakeTypes


class _FakeConnection:
    def __init__(self, calls: list):
        self._calls = calls

    async def execute(self, query, *args):
        self._calls.append((" ".join(query.split()), args))


class _FakeAcquireCtx:
    def __init__(self, conn: _FakeConnection):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self):
        self.calls: list = []
        self._conn = _FakeConnection(self.calls)

    def acquire(self):
        return _FakeAcquireCtx(self._conn)


def _insert_calls(pool: _FakePool):
    return [c for c in pool.calls if c[0].startswith("INSERT INTO conversation_logs")]


def _use_fake_postgres(monkeypatch):
    fake_pool = _FakePool()

    async def fake_get_pool(dsn):
        assert dsn
        return fake_pool

    monkeypatch.setattr(conversation_log_service, "get_pool", fake_get_pool)
    monkeypatch.setattr(conversation_log_service, "get_asyncpg", lambda: _FakeAsyncpgModule)
    monkeypatch.setattr(conversation_log_service, "_schema_ready", False)
    monkeypatch.setattr(settings, "APP_DATABASE_URL", "postgresql://fake:5432/fake_db")
    return fake_pool


def test_log_turn_inserts_in_and_out_turns(monkeypatch):
    fake_pool = _use_fake_postgres(monkeypatch)

    async def scenario():
        await conversation_log_service.log_turn(
            conversation_id=1,
            inbox_id=2,
            store_key="default",
            direction="in",
            content_masked="Bonjour, mon nom est [NAME]",
            pii_types=["NAME"],
            rag_top_score=0.42,
            rag_hits_count=3,
        )
        await conversation_log_service.log_turn(
            conversation_id=1,
            inbox_id=2,
            store_key="default",
            direction="out",
            content_masked="Bonjour ! Comment puis-je vous aider ?",
            rag_top_score=0.42,
            confidence_score=0.42,
            escalated=False,
            model="gpt-4o",
            latency_ms=850,
        )

    asyncio.run(scenario())

    inserts = _insert_calls(fake_pool)
    assert len(inserts) == 2

    _, in_args = inserts[0]
    _, out_args = inserts[1]

    # Column order: conversation_id, inbox_id, store_key, direction, content_masked, ...
    assert in_args[3] == "in"
    assert in_args[4] == "Bonjour, mon nom est [NAME]"
    assert out_args[3] == "out"
    assert out_args[4] == "Bonjour ! Comment puis-je vous aider ?"
    assert out_args[11] == "gpt-4o"
    assert out_args[12] == 850


def test_log_turn_is_silent_when_app_database_url_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "APP_DATABASE_URL", "")

    async def fail_if_called(dsn):
        raise AssertionError("get_pool should never be called without APP_DATABASE_URL")

    monkeypatch.setattr(conversation_log_service, "get_pool", fail_if_called)

    async def scenario():
        await conversation_log_service.log_turn(
            conversation_id=1,
            inbox_id=None,
            store_key="default",
            direction="in",
            content_masked="hello",
        )

    # Must not raise.
    asyncio.run(scenario())


def test_log_turn_signature_has_no_raw_text_parameter():
    # Contract: log_turn only ever accepts already-masked content. There must
    # be no parameter suggesting raw/unmasked text can be logged.
    sig = inspect.signature(conversation_log_service.log_turn)
    param_names = set(sig.parameters.keys())

    assert "content_masked" in param_names
    assert not any("raw" in name.lower() for name in param_names)


def test_log_turn_stores_exactly_the_masked_value_given(monkeypatch):
    # Guards against any accidental substitution of a raw value: whatever is
    # passed as content_masked must be exactly what gets persisted.
    fake_pool = _use_fake_postgres(monkeypatch)
    masked_value = "User [PHONE] wants a refund"

    async def scenario():
        await conversation_log_service.log_turn(
            conversation_id=5,
            inbox_id=None,
            store_key="default",
            direction="in",
            content_masked=masked_value,
        )

    asyncio.run(scenario())

    inserts = _insert_calls(fake_pool)
    assert len(inserts) == 1
    _, args = inserts[0]
    assert args[4] == masked_value
