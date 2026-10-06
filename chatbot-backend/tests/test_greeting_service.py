import asyncio

import pytest

from app.config import settings
from app.services import chatbot_service
from app.services.chatbot_service import process_chatwoot_message
from app.services.greeting_service import is_greeting_only
from app.services.messages import LOCALIZED_MESSAGES
from tests.conftest import make_payload


@pytest.mark.parametrize(
    "text",
    [
        "bonjour",
        "Bonjour !",
        "BONSOIR",
        "salut",
        "Bonjour madame",
        "bonjour à tous",
        "bonjourrr",
        "hello",
        "Hi there",
        "good morning",
        "aslema",
        "3aslema",
        "Aslema, labes ?",
        "salam 3likom",
        "Salam alaykom",
        "ahla w sahla",
        "sba7 el khir",
        "مرحبا",
        "مرحباً",
        "السلام عليكم",
        "السلام عليكم ورحمة الله وبركاته",
        "أهلا",
        "عسلامة",
        "صباح الخير",
    ],
)
def test_greetings_are_detected(text):
    assert is_greeting_only(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "👋",
        "bonjour, chneya siyessa mta3 el retour ?",
        "Bonjour, quel est le prix de la perceuse ?",
        "bonjour je veux retourner un produit",
        "salam, 9adech el livraison ?",
        "hello, do you ship to Sfax?",
        "السلام عليكم، كم سعر التوصيل؟",
        "tout le monde",  # companion words only, no greeting
        "bonjour bonjour bonjour bonjour bonjour bonjour",  # more than 5 words
        "livraison",
    ],
)
def test_questions_are_not_greetings(text):
    assert is_greeting_only(text) is False


def test_greeting_skips_rag_llm_and_escalation(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "ESCALATION_LOW_CONFIDENCE_ENABLED", True)
    # A retrieval this weak would escalate a real question for low confidence.
    pipeline.hits = [{"id": "n1", "score": 0.05, "payload": {"source": "faq"}}]

    async def fail_escalation(**kwargs):
        raise AssertionError("a greeting must not escalate")

    monkeypatch.setattr(chatbot_service, "_apply_escalation_labels", fail_escalation)

    result = asyncio.run(process_chatwoot_message(make_payload("bonsoir")))

    assert result.action == "reply"
    assert result.escalated is False
    assert result.reply == LOCALIZED_MESSAGES["greeting"]["fr"]
    assert pipeline.retrieve_calls == []
    assert pipeline.llm_calls == []
    assert pipeline.grounding_calls == []


@pytest.mark.parametrize(
    "text, language",
    [("aslema", "darija"), ("السلام عليكم", "ar"), ("bonjour", "fr")],
)
def test_greeting_reply_is_localized(pipeline, text, language):
    result = asyncio.run(process_chatwoot_message(make_payload(text)))

    assert result.reply == LOCALIZED_MESSAGES["greeting"][language]


def test_greeting_still_suppressed_after_human_handoff(pipeline, monkeypatch):
    monkeypatch.setattr(settings, "CHATWOOT_API_TOKEN", "test-token")

    async def fake_labels(**kwargs):
        return [settings.ESCALATION_LABEL]

    monkeypatch.setattr(chatbot_service, "get_conversation_labels", fake_labels)

    result = asyncio.run(process_chatwoot_message(make_payload("bonjour")))

    assert result.action == "no_reply"
    assert result.reason == "already_escalated"


def test_question_after_greeting_still_uses_rag(pipeline):
    result = asyncio.run(process_chatwoot_message(make_payload("Bonjour, quel est le délai de livraison ?")))

    assert result.reply == pipeline.reply
    assert len(pipeline.retrieve_calls) == 1
    assert len(pipeline.llm_calls) == 1
