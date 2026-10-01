"""Shared fixtures. Nothing here touches the network or needs an OpenAI key."""

import json
from types import SimpleNamespace

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


def tool_call(name: str, args: dict, call_id: str = "call_1"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def chat_response(content: str | None = None, tool_calls: list | None = None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAIClient:
    """Returns the scripted responses in order and records every request."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeOpenAIClient: no scripted response left")
        return self._responses.pop(0)


class FakeWooClient:
    def __init__(self, products: dict[int, dict] | None = None):
        self.products = products if products is not None else {
            12: {"id": 12, "name": "Perceuse test", "price": "149.000", "stock_status": "instock"},
        }
        self.calls: list[tuple] = []

    async def search_products(self, *, query, limit=5):
        self.calls.append(("search_products", query))
        return []

    async def get_price_and_stock(self, *, product_id):
        self.calls.append(("get_price_and_stock", product_id))
        return dict(self.products.get(product_id) or {"id": None})

    async def create_draft_order(self, *, line_items, customer_note=None):
        self.calls.append(("create_draft_order", list(line_items), customer_note))
        return SimpleNamespace(id=999, status="pending", total="149.000", currency="TND", payment_url=None)

    async def aclose(self):
        self.calls.append(("aclose",))


@pytest.fixture
def fake_llm(monkeypatch):
    """Wire llm_service to a FakeOpenAIClient and a FakeWooClient.

    Usage: client = fake_llm([chat_response(...), ...]); client.wc is the Woo fake.
    """
    from app.services import llm_service

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key-not-real")

    def install(responses: list, *, wc: FakeWooClient | None = None):
        client = FakeOpenAIClient(responses)
        client.wc = wc if wc is not None else FakeWooClient()
        monkeypatch.setattr(llm_service, "_get_client", lambda: client)
        monkeypatch.setattr(llm_service, "get_woocommerce_client_for_store", lambda store: client.wc)
        return client

    return install


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
