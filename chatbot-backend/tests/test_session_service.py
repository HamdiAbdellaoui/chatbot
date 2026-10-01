import asyncio

from app.config import settings
from app.services import session_service


class _FakePipeline:
    def __init__(self, store: dict[str, list[str]]):
        self._store = store
        self._ops: list[tuple[str, tuple]] = []

    def rpush(self, key, value):
        self._ops.append(("rpush", (key, value)))
        return self

    def ltrim(self, key, start, end):
        self._ops.append(("ltrim", (key, start, end)))
        return self

    def expire(self, key, ttl):
        self._ops.append(("expire", (key, ttl)))
        return self

    async def execute(self):
        for op, args in self._ops:
            if op == "rpush":
                key, value = args
                self._store.setdefault(key, []).append(value)
            elif op == "ltrim":
                key, start, _end = args
                # The service only ever calls ltrim(key, -N, -1): keep the last N elements.
                self._store[key] = self._store.get(key, [])[start:]
            # expire is a no-op for the fake store.
        self._ops = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeRedisClient:
    def __init__(self):
        self.store: dict[str, list[str]] = {}

    async def lrange(self, key, start, end):
        lst = self.store.get(key, [])
        if end == -1:
            return lst[start:]
        return lst[start:end + 1]

    def pipeline(self, transaction=True):
        return _FakePipeline(self.store)


def _use_fake_redis(monkeypatch):
    fake_client = _FakeRedisClient()
    fake_module = type("FakeRedisModule", (), {"from_url": staticmethod(lambda url, decode_responses=True: fake_client)})
    monkeypatch.setattr(session_service, "_get_redis_module", lambda: fake_module)
    monkeypatch.setattr(session_service, "_redis_client", None)
    monkeypatch.setattr(settings, "REDIS_URL", "redis://fake:6379")
    monkeypatch.setattr(settings, "ACTIVE_LEARNING_DATABASE_URL", "")
    monkeypatch.setattr(settings, "APP_DATABASE_URL", "")
    return fake_client


def test_append_then_get_history_round_trip(monkeypatch):
    _use_fake_redis(monkeypatch)
    monkeypatch.setattr(settings, "SESSION_HISTORY_TURNS", 6)

    async def scenario():
        await session_service.append_turn(1, "user", "Bonjour")
        await session_service.append_turn(1, "assistant", "Salut, comment puis-je vous aider ?")
        return await session_service.get_history(1)

    history = asyncio.run(scenario())

    assert history == [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Salut, comment puis-je vous aider ?"},
    ]


def test_history_is_truncated_to_configured_turns(monkeypatch):
    _use_fake_redis(monkeypatch)
    monkeypatch.setattr(settings, "SESSION_HISTORY_TURNS", 2)

    async def scenario():
        await session_service.append_turn(2, "user", "msg1")
        await session_service.append_turn(2, "assistant", "msg2")
        await session_service.append_turn(2, "user", "msg3")
        return await session_service.get_history(2)

    history = asyncio.run(scenario())

    assert history == [
        {"role": "assistant", "content": "msg2"},
        {"role": "user", "content": "msg3"},
    ]


def test_no_backend_configured_returns_empty_history_without_raising(monkeypatch):
    monkeypatch.setattr(settings, "REDIS_URL", "")
    monkeypatch.setattr(settings, "ACTIVE_LEARNING_DATABASE_URL", "")
    monkeypatch.setattr(settings, "APP_DATABASE_URL", "")

    async def scenario():
        await session_service.append_turn(3, "user", "hello")
        return await session_service.get_history(3)

    history = asyncio.run(scenario())

    assert history == []


class _FakeConn:
    def __init__(self, log):
        self.log = log

    async def execute(self, sql, *args):
        self.log.append(("execute", " ".join(sql.split())[:40], args))

    async def fetch(self, sql, *args):
        self.log.append(("fetch", " ".join(sql.split())[:40], args))
        return [{"role": "user", "content": "bonjour"}]


class _FakePool:
    def __init__(self):
        self.log: list = []

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return _FakeConn(pool.log)

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def test_postgres_fallback_uses_app_database_url_and_shared_pool(monkeypatch):
    fake_pool = _FakePool()
    requested_dsns: list[str] = []

    async def fake_get_pool(dsn):
        requested_dsns.append(dsn)
        return fake_pool

    monkeypatch.setattr(settings, "REDIS_URL", "")
    monkeypatch.setattr(settings, "APP_DATABASE_URL", "postgresql://app-db.test/db")
    monkeypatch.setattr(settings, "ACTIVE_LEARNING_DATABASE_URL", "")
    monkeypatch.setattr(session_service, "get_pool", fake_get_pool)
    monkeypatch.setattr(session_service, "_pg_schema_ready", False)

    async def scenario():
        await session_service.append_turn(9, "user", "bonjour")
        return await session_service.get_history(9)

    history = asyncio.run(scenario())

    assert history == [{"role": "user", "content": "bonjour"}]
    assert requested_dsns and set(requested_dsns) == {"postgresql://app-db.test/db"}
    # Schema created once, then insert/trim/select.
    ddl = [e for e in fake_pool.log if e[0] == "execute" and "CREATE" in e[1]]
    assert len(ddl) == 2
