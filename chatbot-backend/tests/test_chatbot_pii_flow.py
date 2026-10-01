import asyncio

from app.services.chatbot_service import process_chatwoot_message
from tests.conftest import make_payload

# Fictitious values only.
EMAIL = "client.test@example.com"
PHONE = "+216 98 765 432"


def _all_strings(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(_all_strings(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return " ".join(_all_strings(v) for v in obj)
    return ""


def test_raw_pii_never_reaches_retrieval_or_llm(pipeline):
    message = f"Bonjour, mon email est {EMAIL} et mon numéro {PHONE}, où est ma commande ?"

    result = asyncio.run(process_chatwoot_message(make_payload(message)))

    assert result.action == "reply"
    assert len(pipeline.retrieve_calls) == 1
    assert len(pipeline.llm_calls) == 1

    sent_outside = _all_strings(pipeline.retrieve_calls) + _all_strings(pipeline.llm_calls) + _all_strings(pipeline.grounding_calls)
    assert EMAIL not in sent_outside
    assert "98 765 432" not in sent_outside
    assert "98765432" not in sent_outside

    # Retrieval and LLM get the very same masked text.
    query = pipeline.retrieve_calls[0]["query"]
    assert query == pipeline.llm_calls[0]["user_message"]
    assert "commande" in query


def test_history_and_logs_only_store_masked_text(pipeline):
    message = f"Contactez-moi sur {EMAIL} svp"

    asyncio.run(process_chatwoot_message(make_payload(message)))

    stored = _all_strings(pipeline.history_appends) + _all_strings(pipeline.logged_turns)
    assert EMAIL not in stored
