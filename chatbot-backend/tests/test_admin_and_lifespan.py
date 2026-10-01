from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.routes import admin_active_learning

URL = "/api/v1/admin/active-learning/flags"


def _client(monkeypatch):
    async def fake_list_flags(*, limit, offset):
        return []

    monkeypatch.setattr(admin_active_learning, "list_flags", fake_list_flags)
    return TestClient(app)


def test_admin_requires_correct_token(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_API_TOKEN", "admin-test-token")
    client = _client(monkeypatch)

    assert client.get(URL).status_code == 401
    assert client.get(URL, headers={"X-Admin-Token": "nope"}).status_code == 401
    assert client.get(URL, headers={"X-Admin-Token": "admin-test-token"}).status_code == 200


def test_admin_disabled_without_configured_token(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_API_TOKEN", "")
    client = _client(monkeypatch)
    assert client.get(URL, headers={"X-Admin-Token": ""}).status_code == 503


def test_lifespan_runs_startup_and_shutdown(caplog):
    with caplog.at_level("INFO"):
        with TestClient(app) as client:
            assert client.get("/").status_code == 200
    messages = [r.getMessage() for r in caplog.records]
    assert any("starting up" in m for m in messages)
    assert any("shutting down" in m for m in messages)


def test_no_deprecated_on_event_handlers():
    assert app.router.on_startup == []
    assert app.router.on_shutdown == []
