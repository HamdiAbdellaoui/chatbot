import asyncio

from app.config import settings
from app.services import chatbot_service
from app.services.chatbot_service import process_chatwoot_message
from app.services.llm_service import OrderRequest, generate_grounded_reply
from app.services.store_context_service import StoreContext
from tests.conftest import chat_response, make_payload, tool_call

STORE = StoreContext(key="default", qdrant_collection="shared", woocommerce=None)


def _tool_names(request: dict) -> list[str]:
    return [t["function"]["name"] for t in (request.get("tools") or [])]


# --- llm_service: tool exposure and execution ------------------------------

def test_handoff_mode_request_order_does_not_create_order(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "WOOCOMMERCE_ORDER_MODE", "handoff")
    client = fake_llm([
        chat_response(tool_calls=[tool_call("request_order", {"product_id": 12, "quantity": 2, "customer_note": "livraison le matin"})]),
        chat_response("Votre demande a été transmise."),
    ])
    collected: list[OrderRequest] = []

    reply = asyncio.run(generate_grounded_reply(
        user_message="Je veux 2 perceuses", context="", store_context=STORE, order_requests=collected,
    ))

    assert reply == "Votre demande a été transmise."
    assert "request_order" in _tool_names(client.requests[0])
    assert "create_draft_order" not in _tool_names(client.requests[0])
    assert not any(c[0] == "create_draft_order" for c in client.wc.calls)
    assert ("get_price_and_stock", 12) in client.wc.calls
    assert collected == [OrderRequest(product_id=12, product_name="Perceuse test", price="149.000", quantity=2, customer_note="livraison le matin")]


def test_handoff_mode_ignores_create_draft_order_even_if_model_calls_it(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "WOOCOMMERCE_ORDER_MODE", "handoff")
    client = fake_llm([
        chat_response(tool_calls=[tool_call("create_draft_order", {"product_id": 12, "quantity": 1})]),
        chat_response("ok"),
    ])

    asyncio.run(generate_grounded_reply(user_message="commande", context="", store_context=STORE, order_requests=[]))

    assert not any(c[0] == "create_draft_order" for c in client.wc.calls)


def test_handoff_mode_unknown_product_is_not_forwarded(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "WOOCOMMERCE_ORDER_MODE", "handoff")
    fake_llm([
        chat_response(tool_calls=[tool_call("request_order", {"product_id": 404, "quantity": 1})]),
        chat_response("Produit introuvable."),
    ])
    collected: list[OrderRequest] = []

    asyncio.run(generate_grounded_reply(user_message="commande", context="", store_context=STORE, order_requests=collected))

    assert collected == []


def test_direct_mode_keeps_legacy_create_draft_order(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "WOOCOMMERCE_ORDER_MODE", "direct")
    client = fake_llm([
        chat_response(tool_calls=[tool_call("create_draft_order", {"product_id": 12, "quantity": 1})]),
        chat_response("Commande créée."),
    ])
    collected: list[OrderRequest] = []

    reply = asyncio.run(generate_grounded_reply(user_message="commande", context="", store_context=STORE, order_requests=collected))

    assert reply == "Commande créée."
    assert "create_draft_order" in _tool_names(client.requests[0])
    assert "request_order" not in _tool_names(client.requests[0])
    assert ("create_draft_order", [(12, 1)], None) in client.wc.calls
    assert collected == []


# --- chatbot_service: Chatwoot side effects --------------------------------

def _chatwoot_spy(monkeypatch):
    calls: list[tuple] = []

    async def fake_send(**kwargs):
        calls.append(("send_message", kwargs))

    async def fake_escalate(**kwargs):
        calls.append(("escalate", kwargs))

    async def fake_labels(**kwargs):
        return set()

    monkeypatch.setattr(chatbot_service, "send_message", fake_send)
    monkeypatch.setattr(chatbot_service, "escalate_conversation", fake_escalate)
    monkeypatch.setattr(chatbot_service, "get_conversation_labels", fake_labels)
    monkeypatch.setattr(settings, "CHATWOOT_API_TOKEN", "test-api-token")
    return calls


def test_order_request_posts_private_note_label_and_escalates(pipeline, monkeypatch):
    calls = _chatwoot_spy(monkeypatch)
    monkeypatch.setattr(settings, "ORDER_VALIDATION_LABEL", "order_validation")
    monkeypatch.setattr(settings, "ESCALATION_LABEL", "human_handoff")

    async def fake_generate(**kwargs):
        kwargs["order_requests"].append(
            OrderRequest(product_id=12, product_name="Perceuse test", price="149.000", quantity=2,
                         customer_note="appelez le +216 98 765 432")
        )
        return "transmis"

    monkeypatch.setattr(chatbot_service, "generate_grounded_reply", fake_generate)

    result = asyncio.run(process_chatwoot_message(make_payload("Je veux commander 2 perceuses")))

    assert result.escalated is True
    assert result.reason == "order_request"
    assert result.reply

    notes = [kw for name, kw in calls if name == "send_message"]
    assert len(notes) == 1
    assert notes[0]["private"] is True
    assert "#12" in notes[0]["content"] and "Perceuse test" in notes[0]["content"]
    assert "149.000" in notes[0]["content"] and "Quantité : 2" in notes[0]["content"]
    assert "98 765 432" not in notes[0]["content"]

    escalations = [kw for name, kw in calls if name == "escalate"]
    assert len(escalations) == 1
    assert "order_validation" in escalations[0]["labels"]
    assert "human_handoff" in escalations[0]["labels"]

    # Note first, then labels/escalation.
    assert [name for name, _ in calls] == ["send_message", "escalate"]


def test_no_order_request_means_no_note(pipeline, monkeypatch):
    calls = _chatwoot_spy(monkeypatch)

    result = asyncio.run(process_chatwoot_message(make_payload("Quels sont vos horaires ?")))

    assert result.escalated is False
    assert calls == []
