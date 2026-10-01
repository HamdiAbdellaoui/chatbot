import asyncio

import pytest

from app.config import Settings, settings
from app.services.chatbot_service import process_chatwoot_message
from app.services.escalation_service import detect_escalation_request
from app.services.messages import LOCALIZED_MESSAGES, get_message
from tests.conftest import make_payload

DEFAULT_KEYWORDS = Settings.model_fields["ESCALATION_KEYWORDS"].default


@pytest.fixture
def default_keywords(monkeypatch):
    # Independent of the developer's .env.
    monkeypatch.setattr(settings, "ESCALATION_KEYWORDS", DEFAULT_KEYWORDS)
    monkeypatch.setattr(settings, "ESCALATION_ACK_MESSAGE", "")
    monkeypatch.setattr(settings, "ESCALATION_SEND_ACK", True)


@pytest.mark.parametrize(
    "text",
    [
        "Avez-vous un support mural pour meuleuse ?",
        "Le support de fixation est-il inclus ?",
        "Quel est l'agent de nettoyage conseillé pour ce produit ?",
        "Je cherche une perceuse sans fil",
        "Les ressources humaines sont fermées ?",
    ],
)
def test_no_false_positive(default_keywords, text):
    assert detect_escalation_request(text).should_escalate is False


@pytest.mark.parametrize(
    "text",
    [
        "Je veux parler à un conseiller",
        "je veux parler a un conseiller",  # without accent
        "n7eb nahki m3a 3abd",
        "Nheb nahki m3a wa7ed",
        "نحب نحكي مع موظف",
        "أريد التحدث مع موظف من فضلكم",
        "Can I talk to a human please?",
        "Passez-moi le service client",
        "Je veux un humain",
    ],
)
def test_escalation_phrases(default_keywords, text):
    assert detect_escalation_request(text).should_escalate is True


def test_keywords_setting_is_read_at_call_time(monkeypatch):
    monkeypatch.setattr(settings, "ESCALATION_KEYWORDS", "superviseur")
    assert detect_escalation_request("je veux le superviseur").should_escalate is True
    monkeypatch.setattr(settings, "ESCALATION_KEYWORDS", "autre")
    assert detect_escalation_request("je veux le superviseur").should_escalate is False


def test_ack_in_french_for_french_message(pipeline, default_keywords):
    result = asyncio.run(process_chatwoot_message(make_payload("Je veux parler à un conseiller")))

    assert result.escalated is True
    assert result.reply == LOCALIZED_MESSAGES["escalation_ack"]["fr"]
    assert pipeline.llm_calls == []


def test_ack_in_darija_for_darija_message(pipeline, default_keywords):
    result = asyncio.run(process_chatwoot_message(make_payload("n7eb nahki m3a 3abd")))

    assert result.escalated is True
    assert result.reply == LOCALIZED_MESSAGES["escalation_ack"]["darija"]


def test_explicit_ack_env_value_wins(pipeline, default_keywords, monkeypatch):
    monkeypatch.setattr(settings, "ESCALATION_ACK_MESSAGE", "Message personnalisé")

    result = asyncio.run(process_chatwoot_message(make_payload("n7eb nahki m3a 3abd")))

    assert result.reply == "Message personnalisé"


def test_empty_message_is_localized(pipeline):
    result = asyncio.run(process_chatwoot_message(make_payload("   ")))
    assert result.reply == LOCALIZED_MESSAGES["empty_message"]["other"]


def test_every_message_exists_in_every_language():
    for key, by_language in LOCALIZED_MESSAGES.items():
        for language in ("fr", "ar", "darija", "other"):
            assert by_language.get(language), (key, language)


def test_unknown_language_falls_back_to_other():
    assert get_message("generic_error", "xx") == LOCALIZED_MESSAGES["generic_error"]["other"]
    assert get_message("generic_error", None) == LOCALIZED_MESSAGES["generic_error"]["other"]


def test_llm_not_configured_message_is_localized(monkeypatch):
    from app.services.llm_service import generate_grounded_reply

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    reply = asyncio.run(generate_grounded_reply(user_message="Bonjour", context="", language="fr"))
    assert reply == LOCALIZED_MESSAGES["not_configured"]["fr"]
