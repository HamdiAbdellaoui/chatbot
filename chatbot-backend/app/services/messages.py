"""Fixed (non-LLM) messages sent to customers, localized by detected language.

Languages follow language_service.detect_language: "fr", "ar", "darija"
(Latin-script Tunisian Arabizi) and "other" (English, also the fallback).
"""

from __future__ import annotations

from typing import Optional

from app.config import settings

LOCALIZED_MESSAGES: dict[str, dict[str, str]] = {
    "escalation_ack": {
        "fr": "D'accord, je vous mets en relation avec un conseiller. Il vous répondra ici dès que possible.",
        "ar": "حسناً، سأحوّلك إلى أحد موظفي خدمة العملاء. سيرد عليك هنا في أقرب وقت.",
        "darija": "Behi, taw n7awlek l wa7ed mel conseillers. Bech yjawbek houni fi a9rab wa9t.",
        "other": "Okay — I'm connecting you to a human agent.",
    },
    "empty_message": {
        "fr": "Je n'ai reçu aucun texte. Pouvez-vous écrire votre question ?",
        "ar": "لم أتلقَّ أي نص. هل يمكنك كتابة سؤالك؟",
        "darija": "Ma wsolni 7atta message. Tnajem tekteb so2alek?",
        "other": "I didn't receive any text. Could you please type your question?",
    },
    "generic_error": {
        "fr": "Désolé, je rencontre un problème pour répondre. Merci de réessayer dans un instant.",
        "ar": "عذراً، أواجه مشكلة في الرد حالياً. يرجى المحاولة بعد قليل.",
        "darija": "Same7ni, famma mochkla taw. 3awed jarreb ba3d chwaya.",
        "other": "Sorry, I'm having trouble answering right now. Please try again in a moment.",
    },
    "not_configured": {
        "fr": "Désolé, l'assistant n'est pas encore configuré. Merci de réessayer plus tard.",
        "ar": "عذراً، المساعد غير مهيأ بعد. يرجى المحاولة لاحقاً.",
        "darija": "Same7ni, el assistant mazel moch configuré. 3awed jarreb ba3d.",
        "other": "Sorry, the assistant is not configured yet. Please try again later.",
    },
    "order_forwarded": {
        "fr": "Merci ! Votre demande de commande a été transmise à un conseiller, qui va la vérifier et revenir vers vous ici.",
        "ar": "شكراً! تم تحويل طلب الشراء إلى أحد المستشارين، سيتحقق منه ويعود إليك هنا.",
        "darija": "Ya3tik essa7a! Talbek wsel l conseiller, bech ychoufou w yrja3lek houni.",
        "other": "Thanks! Your order request has been forwarded to an advisor, who will check it and get back to you here.",
    },
}

# Messages that an explicitly set environment variable overrides (any language).
_ENV_OVERRIDES = {
    "escalation_ack": "ESCALATION_ACK_MESSAGE",
}


def get_message(key: str, language: Optional[str]) -> str:
    env_name = _ENV_OVERRIDES.get(key)
    if env_name:
        override = (getattr(settings, env_name, "") or "").strip()
        if override:
            return override

    by_language = LOCALIZED_MESSAGES[key]
    return by_language.get(language or "other") or by_language["other"]
