"""Shared fixtures. Nothing here touches the network or needs an OpenAI key."""

import pytest

from app.config import settings
from app.services import chatbot_service


class PipelineRecorder:
    def __init__(self):
        self.retrieve_calls: list[dict] = []
        self.llm_calls: list[dict] = []
        self.grounding_calls: list[dict] = []
        self.history_appends: list[tuple] = []
        self.logged_turns: list[dict] = []
        self.reply = "Réponse de test."
        self.hits = [{"id": "n1", "score": 0.9, "payload": {"source": "faq"}}]


@pytest.fixture
def pipeline(monkeypatch):
    """Stub every external dependency of chatbot_service.process_chatwoot_message."""
    rec = PipelineRecorder()

    async def fake_retrieve(*, query, collection=None, limit=None, **kwargs):
        rec.retrieve_calls.append({"query": query, "collection": collection, "limit": limit})
        return "contexte de test", rec.hits

    async def fake_generate(**kwargs):
        rec.llm_calls.append(kwargs)
        return rec.reply

    async def fake_grounding(**kwargs):
        rec.grounding_calls.append(kwargs)
        return 0.9

    async def fake_get_history(conversation_id):
        return []

    async def fake_append(conversation_id, role, content):
        rec.history_appends.append((conversation_id, role, content))

    async def fake_log_turn(**kwargs):
        rec.logged_turns.append(kwargs)

    async def fake_flag(**kwargs):
        return None

    monkeypatch.setattr(chatbot_service, "retrieve_context", fake_retrieve)
    monkeypatch.setattr(chatbot_service, "generate_grounded_reply", fake_generate)
    monkeypatch.setattr(chatbot_service, "estimate_context_grounding", fake_grounding)
    monkeypatch.setattr(chatbot_service, "get_history", fake_get_history)
    monkeypatch.setattr(chatbot_service, "append_turn", fake_append)
    monkeypatch.setattr(chatbot_service, "log_turn", fake_log_turn)
    monkeypatch.setattr(chatbot_service, "log_low_confidence_flag", fake_flag)

    # Deterministic settings, independent of the developer's local .env.
    monkeypatch.setattr(settings, "CHATWOOT_API_TOKEN", "")
    monkeypatch.setattr(settings, "STORES_JSON", "")
    monkeypatch.setattr(settings, "ESCALATION_ENABLED", True)
    monkeypatch.setattr(settings, "ESCALATION_LOW_CONFIDENCE_ENABLED", False)
    monkeypatch.setattr(settings, "WOOCOMMERCE_COMMANDS_ENABLED", False)
    monkeypatch.setattr(settings, "PII_ALLOWED_TERMS", "")
    monkeypatch.setattr(settings, "PII_MIN_CONFIDENCE", None)
    return rec


def make_payload(content: str, *, message_id: int = 1, conversation_id: int = 42) -> dict:
    return {
        "event": "message_created",
        "message_type": "incoming",
        "id": message_id,
        "content": content,
        "conversation": {"id": conversation_id},
        "inbox": {"id": 1, "name": "Web"},
        "account": {"id": 1},
    }
