import asyncio
import json

import pytest

from app.services.chatbot_service import process_chatwoot_message
from app.services.llm_service import execute_tool_call, generate_grounded_reply
from app.services.store_context_service import StoreContext
from tests.conftest import FakeWooClient, chat_response, make_payload, tool_call

STORE = StoreContext(key="default", qdrant_collection="shared", woocommerce=None)

# Fictitious customer data.
ORDER = {
    "id": 4521,
    "status": "processing",
    "date_created": "2026-09-28T10:00:00",
    "total": "298.000",
    "currency": "TND",
    "billing": {
        "first_name": "Prénomtest",
        "last_name": "Nomtest",
        "email": "Client.Test@Example.com",
        "phone": "98765432",
        "address_1": "1 rue de Test",
    },
    "line_items": [{"name": "Perceuse test", "quantity": 2}],
}
SAFE_KEYS = {"status", "date_created", "total", "currency"}


def _wc() -> FakeWooClient:
    wc = FakeWooClient()
    wc.orders = {4521: ORDER}
    return wc


def _run(args: dict, contacts: list[str]) -> dict:
    result = asyncio.run(execute_tool_call("get_order_status", args, wc_client=_wc(), verification_contacts=contacts))
    return json.loads(result)


@pytest.mark.parametrize(
    "contacts",
    [
        ["client.test@example.com"],
        ["CLIENT.TEST@EXAMPLE.COM "],
        ["+216 98 765 432"],
        ["0021698765432"],
        ["autre@example.com", "98 765 432"],
    ],
)
def test_matching_contact_returns_only_safe_fields(contacts):
    data = _run({"order_id": 4521, "email_or_phone": "[EMAIL]"}, contacts)
    assert set(data) == SAFE_KEYS
    assert data["status"] == "processing"


@pytest.mark.parametrize(
    "args,contacts",
    [
        ({"order_id": 4521, "email_or_phone": "[EMAIL]"}, ["autre@example.com"]),
        ({"order_id": 4521, "email_or_phone": "[PHONE]"}, ["22 111 333"]),
        ({"order_id": 4521, "email_or_phone": "[EMAIL]"}, []),  # nothing detected in the message
        ({"order_id": 9999, "email_or_phone": "[EMAIL]"}, ["client.test@example.com"]),  # unknown order
        ({"order_id": "abc", "email_or_phone": "[EMAIL]"}, ["client.test@example.com"]),
    ],
)
def test_unverified_gives_neutral_answer(args, contacts):
    data = _run(args, contacts)
    assert data["result"] == "unable_to_verify"
    assert "status" not in data


def test_placeholder_from_model_is_never_used_as_contact():
    data = _run({"order_id": 4521, "email_or_phone": "[EMAIL]"}, [])
    assert data["result"] == "unable_to_verify"


def test_tool_result_never_leaks_customer_details():
    raw = asyncio.run(execute_tool_call(
        "get_order_status", {"order_id": 4521, "email_or_phone": "[EMAIL]"},
        wc_client=_wc(), verification_contacts=["client.test@example.com"],
    ))
    for secret in ("Prénomtest", "Nomtest", "example.com", "98765432", "rue de Test", "Perceuse test"):
        assert secret.lower() not in raw.lower()


def test_raw_contacts_reach_the_tool_but_not_the_model(fake_llm):
    wc = _wc()
    client = fake_llm([
        chat_response(tool_calls=[tool_call("get_order_status", {"order_id": 4521, "email_or_phone": "[EMAIL]"})]),
        chat_response("Votre commande est en cours de préparation."),
    ], wc=wc)

    reply = asyncio.run(generate_grounded_reply(
        user_message="Où en est ma commande 4521 ? mon email est [EMAIL]",
        context="",
        store_context=STORE,
        verification_contacts=["client.test@example.com"],
    ))

    assert reply == "Votre commande est en cours de préparation."
    sent_to_model = json.dumps([r["messages"] for r in client.requests], ensure_ascii=False, default=str)
    assert "client.test@example.com" not in sent_to_model.lower()
    assert "processing" in sent_to_model


def test_pipeline_passes_raw_contacts_server_side_only(pipeline):
    asyncio.run(process_chatwoot_message(make_payload("Commande 4521, mon email client.test@example.com")))

    kwargs = pipeline.llm_calls[0]
    assert kwargs["verification_contacts"] == ["client.test@example.com"]
    assert "client.test@example.com" not in kwargs["user_message"]
