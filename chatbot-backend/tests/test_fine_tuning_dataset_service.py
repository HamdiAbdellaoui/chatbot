from app.services.fine_tuning_dataset_service import (
    anonymize_text,
    example_to_review_row,
    conversation_to_examples,
    evaluate_conversation_quality,
    detect_language_hint,
    normalize_training_text,
    parse_review_row,
)


def test_normalize_training_text_collapses_whitespace_and_quotes():
    text = "  Bonjour\r\n\u00a0\u201cSalam\u201d   world...  "
    assert normalize_training_text(text) == 'Bonjour "Salam" world.'


def test_anonymize_text_masks_pii():
    text = "My name is Ali and my phone is +216 22 333 444"
    masked = anonymize_text(text)
    assert "Ali" not in masked
    assert "+216" not in masked
    assert "[NAME]" in masked or "[PHONE]" in masked


def test_detect_language_hint():
    assert detect_language_hint("شنوة السعر؟") == "ar"
    assert detect_language_hint("Bonjour, merci pour votre aide") == "fr"
    assert detect_language_hint("Salam hello") == "mixed"


def test_conversation_to_examples_builds_training_pairs():
    conversation = {
        "id": 42,
        "status": "resolved",
        "labels": [{"title": "shipping"}],
        "messages": [
            {
                "id": 1,
                "sender_type": "contact",
                "content_type": "text",
                "content": "Salut, mon nom est Sara et mon numéro est 22333444",
                "created_at": "2026-06-02T10:00:00Z",
            },
            {
                "id": 2,
                "sender_type": "agent",
                "content_type": "text",
                "content": "Bonjour Sara, votre commande est en cours de livraison.",
                "created_at": "2026-06-02T10:01:00Z",
            },
        ],
    }

    examples = conversation_to_examples(conversation)
    assert len(examples) == 1
    example = examples[0]
    assert example.metadata["store"] == "default"
    assert example.metadata["topic"] == "shipping"
    assert example.metadata["resolution"] == "resolved"
    assert example.messages[0].role == "system"
    assert example.messages[1].role == "user"
    assert example.messages[2].role == "assistant"
    assert "Sara" not in example.messages[1].content


def test_evaluate_conversation_quality_rejects_low_signal_threads():
    conversation = {
        "id": 7,
        "status": "open",
        "messages": [
            {"id": 1, "sender_type": "contact", "content_type": "text", "content": "Hi"},
            {"id": 2, "sender_type": "agent", "content_type": "text", "content": "ok"},
        ],
    }

    quality = evaluate_conversation_quality(conversation)
    assert quality.accepted is False
    assert "unresolved_status:open" in quality.reasons
    assert "too_short" in quality.reasons


def test_evaluate_conversation_quality_accepts_resolved_threads():
    conversation = {
        "id": 8,
        "status": "resolved",
        "labels": [{"title": "shipping"}],
        "messages": [
            {"id": 1, "sender_type": "contact", "content_type": "text", "content": "Bonjour, quelle est la date de livraison pour ma commande 12345 ?", "created_at": "2026-06-02T10:00:00Z"},
            {"id": 2, "sender_type": "agent", "content_type": "text", "content": "La livraison est prévue demain. Merci pour votre patience.", "created_at": "2026-06-02T10:01:00Z"},
        ],
    }

    quality = evaluate_conversation_quality(conversation)
    assert quality.accepted is True
    assert quality.score >= 0.6


def test_review_row_round_trip():
    conversation = {
        "id": 9,
        "status": "resolved",
        "labels": [{"title": "returns"}],
        "messages": [
            {"id": 1, "sender_type": "contact", "content_type": "text", "content": "Bonjour, je veux retourner ma commande.", "created_at": "2026-06-02T10:00:00Z"},
            {"id": 2, "sender_type": "agent", "content_type": "text", "content": "Bien sûr, voici la procédure.", "created_at": "2026-06-02T10:01:00Z"},
        ],
    }

    example = conversation_to_examples(conversation)[0]
    review_row = example_to_review_row(example)
    messages, metadata = parse_review_row(
        {
            "messages_json": review_row.messages_json,
            "metadata_json": review_row.metadata_json,
        }
    )

    assert len(messages) == 3
    assert metadata["topic"] == "returns"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
