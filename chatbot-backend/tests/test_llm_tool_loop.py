import asyncio

from app.config import settings
from app.services.llm_service import generate_grounded_reply
from app.services.store_context_service import StoreContext
from tests.conftest import chat_response, tool_call

STORE = StoreContext(key="default", qdrant_collection="shared", woocommerce=None)


def test_two_tool_rounds_then_final_answer(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MAX_TOOL_ROUNDS", 3)
    client = fake_llm([
        chat_response(tool_calls=[tool_call("search_products", {"query": "perceuse"}, "c1")]),
        chat_response(tool_calls=[tool_call("get_price_and_stock", {"product_id": 12}, "c2")]),
        chat_response("La perceuse coûte 149 DT."),
    ])

    reply = asyncio.run(generate_grounded_reply(user_message="prix perceuse ?", context="", store_context=STORE))

    assert reply == "La perceuse coûte 149 DT."
    assert len(client.requests) == 3
    # Tools are exposed on every round.
    assert all(r.get("tools") for r in client.requests)
    assert [c[0] for c in client.wc.calls if c[0] != "aclose"] == ["search_products", "get_price_and_stock"]
    # Each tool result is sent back with its call id.
    last_messages = client.requests[2]["messages"]
    tool_ids = [m["tool_call_id"] for m in last_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert tool_ids == ["c1", "c2"]


def test_loop_is_bounded_and_ends_with_a_tool_free_call(fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MAX_TOOL_ROUNDS", 2)
    client = fake_llm([
        chat_response(tool_calls=[tool_call("search_products", {"query": "a"}, "c1")]),
        chat_response(tool_calls=[tool_call("search_products", {"query": "b"}, "c2")]),
        chat_response("Réponse finale."),
    ])

    reply = asyncio.run(generate_grounded_reply(user_message="?", context="", store_context=STORE))

    assert reply == "Réponse finale."
    assert len(client.requests) == 3
    assert client.requests[0].get("tools") and client.requests[1].get("tools")
    assert "tools" not in client.requests[2]


def test_no_tool_call_returns_first_answer(fake_llm):
    client = fake_llm([chat_response("Bonjour !")])

    reply = asyncio.run(generate_grounded_reply(user_message="salut", context="", store_context=STORE))

    assert reply == "Bonjour !"
    assert len(client.requests) == 1


def test_woocommerce_client_is_closed(fake_llm):
    client = fake_llm([chat_response("ok")])
    asyncio.run(generate_grounded_reply(user_message="salut", context="", store_context=STORE))
    assert ("aclose",) in client.wc.calls
