"""Fine-tuning dataset preparation utilities.

This module turns Chatwoot-style conversations into anonymized training rows.
It is intentionally dependency-light so it can be reused by CLI scripts and tests.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Iterable, Optional

from app.services.pii_service import detect_pii, mask_pii, normalize_text, select_entities_to_mask


_WHITESPACE_RE = re.compile(r"\s+")
_SPACING_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_MULTI_PUNCT_RE = re.compile(r"([!?.,:;]){2,}")


@dataclass(frozen=True)
class DatasetMessage:
    role: str
    content: str


@dataclass(frozen=True)
class TrainingExample:
    id: str
    messages: list[DatasetMessage]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ConversationQualityResult:
    accepted: bool
    score: float
    reasons: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReviewRow:
    id: str
    review_status: str
    messages_json: str
    metadata_json: str
    notes: str = ""


def normalize_training_text(text: str) -> str:
    """Normalize text while preserving the original language."""
    if not text:
        return ""

    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u00a0", " ")
    value = value.replace("“", '"').replace("”", '"').replace("„", '"')
    value = value.replace("’", "'").replace("`", "'")
    value = value.replace("…", "...")
    value = _SPACING_PUNCT_RE.sub(r"\1", value)
    value = _MULTI_PUNCT_RE.sub(lambda m: m.group(1)[0], value)
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip()


_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language_hint(text: str) -> str:
    """Lightweight language hint for metadata.

    The goal is not perfect language detection; it is to help filter and audit the
    training set.
    """
    if not text or not text.strip():
        return "unknown"

    arabic = len(_ARABIC_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))

    if arabic and latin:
        return "mixed"
    if arabic:
        return "ar"
    if latin:
        lower = f" {text.lower()} "
        french_markers = (" le ", " la ", " les ", " merci", " bonjour", " commande", " livraison", " retour")
        english_markers = (" the ", " and ", " thank ", " hello ", " order ", " shipping")
        darija_markers = (" salam ", " slm ", " chnowa ", " chnoua ", " labes ", " kifash ", " barka ")

        matches = 0
        if any(marker in lower for marker in french_markers):
            matches += 1
        if any(marker in lower for marker in english_markers):
            matches += 1
        if any(marker in lower for marker in darija_markers):
            matches += 1

        if matches >= 2:
            return "mixed"
        if any(marker in lower for marker in french_markers):
            return "fr"
        if any(marker in lower for marker in english_markers):
            return "en"
        if any(marker in lower for marker in darija_markers):
            return "darija"
        return "latin"
    return "unknown"


def anonymize_text(text: str, *, allowed_terms: Iterable[str] | None = None) -> str:
    """Normalize and mask PII in one pass."""
    normalized = normalize_training_text(text)
    if not normalized:
        return ""

    entities = detect_pii(normalized, allowed_terms=allowed_terms)
    selected = select_entities_to_mask(entities, allowed_terms=allowed_terms)
    masked = mask_pii(normalized, entities=selected, allowed_terms=allowed_terms)
    return normalize_training_text(masked)


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _message_role(message: dict[str, Any]) -> str | None:
    sender_type = str(message.get("sender_type") or message.get("message_type") or "").strip().lower()
    content_type = str(message.get("content_type") or "").strip().lower()

    if content_type and content_type != "text":
        return None

    if sender_type in {"contact", "customer", "incoming", "user"}:
        return "user"
    if sender_type in {"agent", "assignee", "outgoing", "bot"}:
        return "assistant"

    sender = message.get("sender")
    if isinstance(sender, dict):
        sender_type = str(sender.get("type") or sender.get("role") or "").strip().lower()
        if sender_type in {"contact", "customer"}:
            return "user"
        if sender_type in {"agent", "assignee", "user"}:
            return "assistant"

    return None


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content") or message.get("body") or message.get("text") or ""
    return content if isinstance(content, str) else ""


def _conversation_id(conversation: dict[str, Any]) -> str:
    value = conversation.get("id") or conversation.get("conversation_id") or conversation.get("uuid")
    return str(value or "")


def _conversation_topic(conversation: dict[str, Any]) -> str:
    labels = conversation.get("labels") or conversation.get("conversation_labels") or []
    if isinstance(labels, list):
        for item in labels:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                title = item.get("title") or item.get("name")
                if isinstance(title, str) and title.strip():
                    return title.strip()

    subject = conversation.get("subject") or conversation.get("title")
    if isinstance(subject, str) and subject.strip():
        return subject.strip()

    return "unknown"


def _conversation_store(conversation: dict[str, Any], default_store: str) -> str:
    for key in ("store_key", "store", "account", "inbox_name"):
        value = conversation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("name") or value.get("key")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return default_store


def _conversation_resolution(conversation: dict[str, Any], assistant_messages: list[str]) -> str:
    status = str(conversation.get("status") or conversation.get("state") or "").strip().lower()
    labels = conversation.get("labels") or conversation.get("conversation_labels") or []
    label_names = []
    if isinstance(labels, list):
        for item in labels:
            if isinstance(item, str):
                label_names.append(item.strip().lower())
            elif isinstance(item, dict):
                title = item.get("title") or item.get("name")
                if isinstance(title, str):
                    label_names.append(title.strip().lower())

    if any("human_handoff" in label for label in label_names):
        return "escalated_to_human"
    if status in {"resolved", "closed", "done"}:
        return "resolved"
    if assistant_messages:
        return f"agent_replied_{status or 'unknown'}"
    return f"unresolved_{status or 'unknown'}"


def _extract_messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "conversation_messages", "data"):
        value = conversation.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = value.get("messages")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    nested = conversation.get("conversation")
    if isinstance(nested, dict):
        value = nested.get("messages")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _sort_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(message: dict[str, Any]) -> tuple[int, str]:
        dt = _parse_dt(message.get("created_at") or message.get("created_on") or message.get("timestamp"))
        ts = int(dt.timestamp()) if dt else 0
        identifier = str(message.get("id") or message.get("message_id") or "")
        return ts, identifier

    return sorted(messages, key=key)


def _is_short_text(text: str, *, min_alpha_chars: int = 8) -> bool:
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    return alpha_chars < min_alpha_chars


def _conversation_status(conversation: dict[str, Any]) -> str:
    return str(conversation.get("status") or conversation.get("state") or "").strip().lower()


def _conversation_labels(conversation: dict[str, Any]) -> list[str]:
    labels = conversation.get("labels") or conversation.get("conversation_labels") or []
    out: list[str] = []
    if isinstance(labels, list):
        for item in labels:
            if isinstance(item, str):
                value = item.strip()
                if value:
                    out.append(value.lower())
            elif isinstance(item, dict):
                title = item.get("title") or item.get("name")
                if isinstance(title, str) and title.strip():
                    out.append(title.strip().lower())
    return out


def evaluate_conversation_quality(
    conversation: dict[str, Any],
    *,
    min_turns: int = 2,
    min_user_turns: int = 1,
    min_assistant_turns: int = 1,
    require_resolved: bool = True,
    require_nontrivial_length: bool = True,
    max_short_turn_ratio: float = 0.35,
) -> ConversationQualityResult:
    """Strict conversation-level filter before generating training rows.

    The goal is to reject low-signal threads early so humans only review likely
    useful examples.
    """
    messages = _sort_messages(_extract_messages(conversation))
    total_messages = len(messages)
    user_messages = 0
    assistant_messages = 0
    text_messages = 0
    short_messages = 0
    repeated_messages = 0
    seen_contents: set[str] = set()
    char_count = 0

    for message in messages:
        role = _message_role(message)
        content = normalize_training_text(_message_content(message))
        if role is None:
            continue
        if not content:
            continue

        text_messages += 1
        char_count += len(content)
        if role == "user":
            user_messages += 1
        elif role == "assistant":
            assistant_messages += 1

        if _is_short_text(content):
            short_messages += 1

        normalized_content = normalize_text(content)
        if normalized_content in seen_contents:
            repeated_messages += 1
        else:
            seen_contents.add(normalized_content)

    reasons: list[str] = []
    status = _conversation_status(conversation)
    labels = _conversation_labels(conversation)

    if total_messages < min_turns or text_messages < min_turns:
        reasons.append("too_few_messages")

    if user_messages < min_user_turns:
        reasons.append("missing_user_turn")

    if assistant_messages < min_assistant_turns:
        reasons.append("missing_assistant_turn")

    if require_resolved and status not in {"resolved", "closed", "done"} and not any("human_handoff" in label for label in labels):
        reasons.append(f"unresolved_status:{status or 'unknown'}")

    if require_nontrivial_length and char_count < 40:
        reasons.append("too_short")

    if text_messages and (short_messages / text_messages) > max_short_turn_ratio:
        reasons.append("too_many_short_turns")

    if text_messages >= 3 and repeated_messages >= max(2, text_messages // 2):
        reasons.append("too_repetitive")

    score = 1.0
    score -= 0.20 if total_messages < 4 else 0.0
    score -= 0.20 if char_count < 80 else 0.0
    score -= 0.20 if short_messages > 0 else 0.0
    score -= 0.20 if repeated_messages > 0 else 0.0
    score -= 0.20 if reasons else 0.0
    score = max(0.0, min(1.0, score))

    accepted = not reasons and score >= 0.6
    return ConversationQualityResult(
        accepted=accepted,
        score=score,
        reasons=reasons,
        metadata={
            "status": status or "unknown",
            "labels": labels,
            "message_count": total_messages,
            "text_message_count": text_messages,
            "user_message_count": user_messages,
            "assistant_message_count": assistant_messages,
            "character_count": char_count,
            "short_message_count": short_messages,
            "repeated_message_count": repeated_messages,
        },
    )


def conversation_to_examples(
    conversation: dict[str, Any],
    *,
    default_store: str = "default",
    include_system_message: bool = True,
    allowed_terms: Iterable[str] | None = None,
    quality_result: ConversationQualityResult | None = None,
) -> list[TrainingExample]:
    """Convert a single Chatwoot conversation into one or more SFT examples."""
    messages = _sort_messages(_extract_messages(conversation))
    if not messages:
        return []

    store = _conversation_store(conversation, default_store)
    topic = _conversation_topic(conversation)
    conversation_id = _conversation_id(conversation)

    user_turns: list[str] = []
    assistant_turns: list[str] = []
    examples: list[TrainingExample] = []

    for message in messages:
        role = _message_role(message)
        content = _message_content(message)
        if role is None:
            continue

        cleaned = anonymize_text(content, allowed_terms=allowed_terms)
        if not cleaned:
            continue

        if role == "user":
            user_turns.append(cleaned)
            continue

        assistant_turns.append(cleaned)
        if not user_turns:
            continue

        prior_user = user_turns[-1]
        msg_id = str(message.get("id") or len(examples) + 1)
        language = detect_language_hint(f"{prior_user} {cleaned}")
        resolution = _conversation_resolution(conversation, assistant_turns)

        items = []
        if include_system_message:
            items.append(
                DatasetMessage(
                    role="system",
                    content=(
                        "You are a customer support assistant. Mirror the user's language, stay concise, "
                        "and do not invent facts or expose personal data."
                    ),
                )
            )
        items.append(DatasetMessage(role="user", content=prior_user))
        items.append(DatasetMessage(role="assistant", content=cleaned))

        examples.append(
            TrainingExample(
                id=f"{conversation_id}:{msg_id}",
                messages=items,
                metadata={
                    "conversation_id": _safe_int(conversation_id) or conversation_id,
                    "store": store,
                    "topic": topic,
                    "resolution": resolution,
                    "language": language,
                    "quality_score": quality_result.score if quality_result else None,
                    "quality_accepted": quality_result.accepted if quality_result else None,
                    "quality_reasons": quality_result.reasons if quality_result else [],
                },
            )
        )

    return examples


def summarize_examples(examples: list[TrainingExample]) -> dict[str, Any]:
    """Build a compact metadata summary for the output dataset."""
    store_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()

    for ex in examples:
        meta = ex.metadata
        store_counts[str(meta.get("store") or "unknown")] += 1
        topic_counts[str(meta.get("topic") or "unknown")] += 1
        resolution_counts[str(meta.get("resolution") or "unknown")] += 1
        language_counts[str(meta.get("language") or "unknown")] += 1

    return {
        "total_examples": len(examples),
        "stores": dict(store_counts),
        "topics": dict(topic_counts),
        "resolutions": dict(resolution_counts),
        "languages": dict(language_counts),
    }


def example_to_review_row(example: TrainingExample, *, review_status: str = "pending", notes: str = "") -> ReviewRow:
    """Serialize a training example into a review-friendly CSV row."""
    messages_json = json.dumps(
        [{"role": message.role, "content": message.content} for message in example.messages],
        ensure_ascii=False,
    )
    metadata_json = json.dumps(example.metadata, ensure_ascii=False)
    return ReviewRow(
        id=example.id,
        review_status=review_status,
        messages_json=messages_json,
        metadata_json=metadata_json,
        notes=notes,
    )


def parse_review_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse a CSV review row back into messages and metadata.

    Expected columns:
    - messages_json
    - metadata_json
    - review_status
    """
    raw_messages = row.get("messages_json") or row.get("messages") or ""
    raw_metadata = row.get("metadata_json") or row.get("metadata") or "{}"

    messages = json.loads(raw_messages) if isinstance(raw_messages, str) and raw_messages.strip() else []
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) and raw_metadata.strip() else {}

    if not isinstance(messages, list):
        raise ValueError("messages_json must decode to a list")
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object")

    normalized_messages: list[dict[str, Any]] = []
    for item in messages:
        if isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("content"), str):
            normalized_messages.append({"role": item["role"], "content": item["content"]})

    return normalized_messages, metadata
