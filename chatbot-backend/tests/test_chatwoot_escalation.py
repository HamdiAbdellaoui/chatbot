import asyncio
import json

import httpx
import pytest

from app.config import settings
from app.services import chatwoot_service

_RealAsyncClient = httpx.AsyncClient


@pytest.fixture
def chatwoot_http(monkeypatch):
    """Route every Chatwoot HTTP call through an httpx.MockTransport and record it."""
    state = {"requests": [], "status_by_suffix": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        state["requests"].append((request.method, request.url.path, body))
        for suffix, status in state["status_by_suffix"].items():
            if request.url.path.endswith(suffix):
                return httpx.Response(status, json={})
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    monkeypatch.setattr(settings, "CHATWOOT_API_TOKEN", "test-api-token")
    monkeypatch.setattr(settings, "CHATWOOT_BASE_URL", "http://chatwoot.test")
    return state


def _paths(state) -> list[str]:
    return [path.rsplit("/", 1)[-1] for _, path, _ in state["requests"]]


def test_escalation_order_labels_then_open_then_assignment(chatwoot_http):
    asyncio.run(chatwoot_service.escalate_conversation(
        account_id=1, conversation_id=42, labels=["human_handoff"], assignee_id=7, team_id=None,
    ))

    assert _paths(chatwoot_http) == ["labels", "toggle_status", "assignments"]
    method, path, body = chatwoot_http["requests"][1]
    assert method == "POST"
    assert path == "/api/v1/accounts/1/conversations/42/toggle_status"
    assert body == {"status": "open"}


def test_escalation_without_assignment_still_opens(chatwoot_http):
    asyncio.run(chatwoot_service.escalate_conversation(
        account_id=1, conversation_id=42, labels=["human_handoff"], assignee_id=None, team_id=None,
    ))

    assert _paths(chatwoot_http) == ["labels", "toggle_status"]


def test_toggle_status_failure_is_logged_and_does_not_stop_escalation(chatwoot_http, caplog):
    chatwoot_http["status_by_suffix"]["/toggle_status"] = 500

    with caplog.at_level("WARNING"):
        asyncio.run(chatwoot_service.escalate_conversation(
            account_id=1, conversation_id=42, labels=["human_handoff"], assignee_id=7, team_id=None,
        ))

    assert _paths(chatwoot_http) == ["labels", "toggle_status", "assignments"]
    assert any("status=500" in r.getMessage() for r in caplog.records)


def test_send_message_private_flag(chatwoot_http):
    asyncio.run(chatwoot_service.send_message(account_id=1, conversation_id=42, content="note", private=True))
    asyncio.run(chatwoot_service.send_message(account_id=1, conversation_id=42, content="public"))

    assert chatwoot_http["requests"][0][2] == {"content": "note", "private": True}
    assert chatwoot_http["requests"][1][2] == {"content": "public"}
