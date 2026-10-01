import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.routes import chatwoot_webhook
from app.routes.chatwoot_webhook import compute_chatwoot_signature
from app.services.chatbot_service import ChatbotResult

URL = "/api/v1/chatwoot-webhook"
TEST_TOKEN = "test-token-not-a-real-secret"
TEST_SECRET = "test-secret-not-a-real-secret"

PAYLOAD = {
    "event": "message_created",
    "message_type": "incoming",
    "id": 101,
    "content": "Bonjour",
    "conversation": {"id": 7},
    "inbox": {"id": 3, "name": "Web"},
    "account": {"id": 1},
}


@pytest.fixture
def client(monkeypatch):
    calls = {"process": 0, "send": 0}

    async def fake_process(payload, *, account_id=None):
        calls["process"] += 1
        return ChatbotResult(action="reply", reply="ok")

    async def fake_send(**kwargs):
        calls["send"] += 1

    monkeypatch.setattr(chatwoot_webhook, "process_chatwoot_message", fake_process)
    monkeypatch.setattr(chatwoot_webhook, "send_message", fake_send)
    monkeypatch.setattr(settings, "CHATWOOT_API_TOKEN", "")
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_TOKEN", TEST_TOKEN)
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_SECRET", TEST_SECRET)
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_MAX_SKEW_S", 300)
    monkeypatch.setattr(settings, "CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE", None)
    monkeypatch.setattr(settings, "REDIS_URL", "")
    c = TestClient(app)
    c.calls = calls
    return c


def _body() -> bytes:
    return json.dumps(PAYLOAD).encode("utf-8")


# --- token mode -----------------------------------------------------------

def test_token_mode_rejects_missing_token(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "token")
    resp = client.post(URL, content=_body())
    assert resp.status_code == 401
    assert client.calls["process"] == 0


def test_token_mode_rejects_wrong_token(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "token")
    resp = client.post(URL, params={"token": "wrong"}, content=_body())
    assert resp.status_code == 401
    assert client.calls["process"] == 0


def test_token_mode_accepts_correct_token(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "token")
    resp = client.post(URL, params={"token": TEST_TOKEN}, content=_body())
    assert resp.status_code == 200
    assert client.calls["process"] == 1


def test_token_mode_is_default_when_mode_unset(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "")
    assert client.post(URL, content=_body()).status_code == 401
    assert client.post(URL, params={"token": TEST_TOKEN}, content=_body()).status_code == 200


def test_token_mode_with_empty_configured_token_always_rejects(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "token")
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_TOKEN", "")
    assert client.post(URL, content=_body()).status_code == 401
    assert client.post(URL, params={"token": ""}, content=_body()).status_code == 401


def test_token_is_redacted_from_access_log_args():
    from app.utils.logger import redact_url_secrets

    redacted = redact_url_secrets(f"{URL}?a=1&token={TEST_TOKEN}&b=2")
    assert TEST_TOKEN not in redacted
    assert "token=***" in redacted


# --- signature mode -------------------------------------------------------

def _signed_headers(body: bytes, *, timestamp: int | None = None, secret: str = TEST_SECRET) -> dict:
    ts = str(timestamp if timestamp is not None else int(time.time()))
    return {
        "Content-Type": "application/json",
        "X-Chatwoot-Timestamp": ts,
        "X-Chatwoot-Signature": compute_chatwoot_signature(secret, ts, body),
    }


def test_signature_mode_accepts_valid_signature(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "signature")
    body = _body()
    resp = client.post(URL, content=body, headers=_signed_headers(body))
    assert resp.status_code == 200
    assert client.calls["process"] == 1


def test_signature_format_matches_chatwoot_spec():
    sig = compute_chatwoot_signature("k", "1700000000", b"{}")
    assert sig.startswith("sha256=")
    assert len(sig) == len("sha256=") + 64


def test_signature_mode_rejects_wrong_signature(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "signature")
    body = _body()
    resp = client.post(URL, content=body, headers=_signed_headers(body, secret="another-secret"))
    assert resp.status_code == 401
    assert client.calls["process"] == 0


def test_signature_mode_rejects_tampered_body(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "signature")
    headers = _signed_headers(_body())
    resp = client.post(URL, content=_body().replace(b"Bonjour", b"Bonsoir"), headers=headers)
    assert resp.status_code == 401


def test_signature_mode_rejects_old_timestamp(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "signature")
    body = _body()
    resp = client.post(URL, content=body, headers=_signed_headers(body, timestamp=int(time.time()) - 301))
    assert resp.status_code == 401
    assert client.calls["process"] == 0


def test_signature_mode_rejects_missing_headers(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "signature")
    resp = client.post(URL, content=_body(), params={"token": TEST_TOKEN})
    assert resp.status_code == 401


# --- none mode / deprecated flag -----------------------------------------

def test_none_mode_accepts_without_auth_and_warns(client, monkeypatch, caplog):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "none")
    with caplog.at_level("WARNING"):
        resp = client.post(URL, content=_body())
    assert resp.status_code == 200
    assert client.calls["process"] == 1
    assert any("DISABLED" in r.getMessage() for r in caplog.records)


def test_deprecated_validate_false_maps_to_none(client, monkeypatch, caplog):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "")
    monkeypatch.setattr(settings, "CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE", False)
    with caplog.at_level("WARNING"):
        resp = client.post(URL, content=_body())
    assert resp.status_code == 200
    assert any("deprecated" in r.getMessage() for r in caplog.records)


def test_explicit_mode_wins_over_deprecated_flag(client, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "token")
    monkeypatch.setattr(settings, "CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE", False)
    assert client.post(URL, content=_body()).status_code == 401


def test_secrets_never_logged(client, monkeypatch, caplog):
    monkeypatch.setattr(settings, "CHATWOOT_WEBHOOK_AUTH_MODE", "token")
    with caplog.at_level("DEBUG"):
        client.post(URL, params={"token": "wrong-value-xyz"}, content=_body())
        client.post(URL, params={"token": TEST_TOKEN}, content=_body())
    # Only server-side loggers matter (the test client's own httpx log shows its URL).
    text = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("app."))
    assert TEST_TOKEN not in text
    assert "wrong-value-xyz" not in text
