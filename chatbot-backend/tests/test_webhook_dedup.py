import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.routes import chatwoot_webhook
from app.services import dedup_service
from app.services.chatbot_service import ChatbotResult
from tests.conftest import make_payload

URL = "/api/v1/chatwoot-webhook"


@pytest.fixture
def client(monkeypatch):
    calls = {"process": 0}

    async def fake_process(payload, *, account_id=None):
        calls["process"] += 1
        return ChatbotResult(action="reply", reply="ok")

    async def fake_send(**kwargs):
        return None

    monkeypatch.setattr(chatwoot_webhook, "process_chatwoot_message", fake_process)
    monkeypatch.setattr(chatwoot_webhook, "send_message", fake_send)
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "none")
    monkeypatch.setattr(settings, "CHATWOOT_API_TOKEN", "")
    monkeypatch.setattr(settings, "REDIS_URL", "")
    c = TestClient(app)
    c.calls = calls
    return c


def _post(client, message_id: int):
    return client.post(URL, content=json.dumps(make_payload("Bonjour", message_id=message_id)))


def test_same_message_id_is_processed_once(client):
    first = _post(client, 555)
    second = _post(client, 555)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert client.calls["process"] == 1


def test_different_message_ids_are_both_processed(client):
    _post(client, 1)
    _post(client, 2)
    assert client.calls["process"] == 2


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, tuple[str, int]] = {}
        self.set_calls: list[tuple] = []

    async def set(self, key, value, nx=False, ex=None):
        self.set_calls.append((key, value, nx, ex))
        if nx and key in self.store:
            return None
        self.store[key] = (value, ex)
        return True


def test_redis_backend_uses_set_nx_ex(client, monkeypatch):
    fake = _FakeRedis()

    async def get_client():
        return fake

    monkeypatch.setattr(settings, "REDIS_URL", "redis://fake:6379")
    monkeypatch.setattr(dedup_service, "get_redis_client", get_client)

    _post(client, 777)
    _post(client, 777)

    assert client.calls["process"] == 1
    assert fake.set_calls[0] == ("dedup:msg:777", "1", True, 600)
    assert dedup_service._memory_cache == {}


def test_redis_failure_falls_back_to_memory_without_raising(client, monkeypatch):
    async def broken_client():
        raise ConnectionError("redis down")

    monkeypatch.setattr(settings, "REDIS_URL", "redis://fake:6379")
    monkeypatch.setattr(dedup_service, "get_redis_client", broken_client)

    assert _post(client, 888).status_code == 200
    assert _post(client, 888).json()["status"] == "duplicate"
    assert client.calls["process"] == 1


def test_memory_cache_entries_expire(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(dedup_service.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(settings, "REDIS_URL", "")

    assert asyncio.run(dedup_service.is_duplicate_message(5)) is False
    assert asyncio.run(dedup_service.is_duplicate_message(5)) is True
    now[0] += dedup_service.DEDUP_TTL_S + 1
    assert asyncio.run(dedup_service.is_duplicate_message(5)) is False


def test_memory_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(dedup_service, "_MEMORY_MAX_ENTRIES", 3)
    monkeypatch.setattr(settings, "REDIS_URL", "")
    for i in range(10):
        asyncio.run(dedup_service.is_duplicate_message(i))
    assert len(dedup_service._memory_cache) == 3


def test_missing_message_id_is_never_a_duplicate():
    assert asyncio.run(dedup_service.is_duplicate_message(None)) is False
    assert asyncio.run(dedup_service.is_duplicate_message(None)) is False
