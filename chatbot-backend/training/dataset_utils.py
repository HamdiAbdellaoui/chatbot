"""Dataset validation, review-row handling, and splitting utilities."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.services.pii_service import detect_pii, mask_pii, normalize_text, select_entities_to_mask


_WHITESPACE_RE = re.compile(r"\s+")
_SPACING_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_MULTI_PUNCT_RE = re.compile(r"([!?.,:;]){2,}")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_RE = re.compile(r"[A-Za-z]")

ALLOWED_ROLES = {"system", "user", "assistant"}
REQUIRED_METADATA_KEYS = ("store", "topic", "resolution")


@dataclass(frozen=True)
class FineTuningRecord:
    id: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ValidationIssue:
    line_number: int
    example_id: str | None
    message: str


DatasetValidationIssue = ValidationIssue


@dataclass(frozen=True)
class DatasetValidationReport:
    total_examples: int
    valid_examples: int
    invalid_examples: int
    language_distribution: dict[str, int]
    topic_distribution: dict[str, int]
    average_conversation_length: float
    issues: list[ValidationIssue]


@dataclass(frozen=True)
class SplitResult:
    train: list[FineTuningRecord]
    validation: list[FineTuningRecord]
    test: list[FineTuningRecord]


DatasetSplit = SplitResult


@dataclass(frozen=True)
class ReviewRow:
    id: str
    review_status: str
    messages_json: str
    metadata_json: str
    notes: str = ""


@dataclass(frozen=True)
class ConversationQualityResult:
    accepted: bool
    score: float
    reasons: list[str]
    metadata: dict[str, Any]


def normalize_training_text(text: str) -> str:
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


def detect_language_hint(text: str) -> str:
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
    return sum(1 for ch in text if ch.isalpha()) < min_alpha_chars


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
        if role is None or not content:
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
) -> list[FineTuningRecord]:
    messages = _sort_messages(_extract_messages(conversation))
    if not messages:
        return []

    store = _conversation_store(conversation, default_store)
    topic = _conversation_topic(conversation)
    conversation_id = _conversation_id(conversation)
    user_turns: list[str] = []
    assistant_turns: list[str] = []
    examples: list[FineTuningRecord] = []

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
        items: list[dict[str, Any]] = []
        if include_system_message:
            items.append(
                {
                    "role": "system",
                    "content": (
                        "You are a customer support assistant. Mirror the user's language, stay concise, "
                        "and do not invent facts or expose personal data."
                    ),
                }
            )
        items.append({"role": "user", "content": prior_user})
        items.append({"role": "assistant", "content": cleaned})

        metadata = {
            "conversation_id": _safe_int(conversation_id) or conversation_id,
            "store": store,
            "topic": topic,
            "resolution": resolution,
            "language": language,
            "quality_score": quality_result.score if quality_result else None,
            "quality_accepted": quality_result.accepted if quality_result else None,
            "quality_reasons": quality_result.reasons if quality_result else [],
        }
        examples.append(FineTuningRecord(id=f"{conversation_id}:{msg_id}", messages=items, metadata=metadata))

    return examples


def summarize_examples(examples: list[FineTuningRecord]) -> dict[str, Any]:
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


def example_to_review_row(example: FineTuningRecord, *, review_status: str = "pending", notes: str = "") -> ReviewRow:
    messages_json = json.dumps([{ "role": message["role"], "content": message["content"]} for message in example.messages], ensure_ascii=False)
    metadata_json = json.dumps(example.metadata, ensure_ascii=False)
    return ReviewRow(id=example.id, review_status=review_status, messages_json=messages_json, metadata_json=metadata_json, notes=notes)


def parse_review_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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


def load_jsonl_records(path: str | Path) -> list[tuple[int, dict[str, Any] | None, str | None]]:
    records: list[tuple[int, dict[str, Any] | None, str | None]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                records.append((line_number, None, f"invalid_json: {exc.msg}"))
                continue
            if not isinstance(payload, dict):
                records.append((line_number, None, "invalid_example: root must be a JSON object"))
                continue
            records.append((line_number, payload, None))
    return records


def _validate_message(message: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(message, dict):
        return ["messages must contain objects"]
    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not role.strip():
        errors.append("message.role must be a non-empty string")
    elif role not in ALLOWED_ROLES:
        errors.append(f"message.role must be one of {sorted(ALLOWED_ROLES)}")
    if not isinstance(content, str) or not content.strip():
        errors.append("message.content must be a non-empty string")
    return errors


def _coerce_record(payload: dict[str, Any]) -> tuple[FineTuningRecord | None, list[str]]:
    errors: list[str] = []
    raw_id = payload.get("id")
    raw_messages = payload.get("messages")
    raw_metadata = payload.get("metadata")
    if not isinstance(raw_id, str) or not raw_id.strip():
        errors.append("missing_or_invalid_id")
    if not isinstance(raw_messages, list) or not raw_messages:
        errors.append("messages must be a non-empty list")
    if not isinstance(raw_metadata, dict):
        errors.append("metadata must be an object")

    messages: list[dict[str, Any]] = []
    if isinstance(raw_messages, list):
        seen_user_turn = False
        seen_assistant_turn = False
        for message in raw_messages:
            message_errors = _validate_message(message)
            if message_errors:
                errors.extend(message_errors)
            else:
                role = message["role"]
                if role == "user":
                    seen_user_turn = True
                elif role == "assistant":
                    seen_assistant_turn = True
                messages.append({"role": role, "content": message["content"]})
        if messages and (not seen_user_turn or not seen_assistant_turn):
            errors.append("messages must contain at least one user turn and one assistant turn")

    metadata: dict[str, Any] = {}
    if isinstance(raw_metadata, dict):
        metadata = dict(raw_metadata)
        for key in REQUIRED_METADATA_KEYS:
            if not isinstance(metadata.get(key), str) or not str(metadata.get(key)).strip():
                errors.append(f"metadata.{key} must be a non-empty string")

    if errors:
        return None, errors
    return FineTuningRecord(id=raw_id.strip(), messages=messages, metadata=metadata), []


def validate_dataset_records(records: Sequence[tuple[int, dict[str, Any] | None, str | None]]) -> DatasetValidationReport:
    issues: list[ValidationIssue] = []
    valid: list[FineTuningRecord] = []
    language_counter: Counter[str] = Counter()
    topic_counter: Counter[str] = Counter()
    conversation_lengths: list[int] = []

    for line_number, payload, parse_error in records:
        if parse_error is not None:
            issues.append(ValidationIssue(line_number=line_number, example_id=None, message=parse_error))
            continue
        assert payload is not None
        record, errors = _coerce_record(payload)
        if errors:
            example_id = payload.get("id") if isinstance(payload.get("id"), str) else None
            for error in errors:
                issues.append(ValidationIssue(line_number=line_number, example_id=example_id, message=error))
            continue
        assert record is not None
        valid.append(record)
        language = str(record.metadata.get("language") or "unknown").strip().lower() or "unknown"
        topic = str(record.metadata.get("topic") or "unknown").strip().lower() or "unknown"
        language_counter[language] += 1
        topic_counter[topic] += 1
        conversation_lengths.append(len(record.messages))

    total = len(records)
    valid_count = len(valid)
    invalid_count = total - valid_count
    avg_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0.0
    return DatasetValidationReport(
        total_examples=total,
        valid_examples=valid_count,
        invalid_examples=invalid_count,
        language_distribution=dict(sorted(language_counter.items())),
        topic_distribution=dict(sorted(topic_counter.items())),
        average_conversation_length=avg_length,
        issues=issues,
    )


def extract_valid_records(records: Sequence[tuple[int, dict[str, Any] | None, str | None]]) -> list[FineTuningRecord]:
    valid: list[FineTuningRecord] = []
    for _, payload, parse_error in records:
        if parse_error is not None or payload is None:
            continue
        record, errors = _coerce_record(payload)
        if errors or record is None:
            continue
        valid.append(record)
    return valid


def split_records(
    records: Sequence[FineTuningRecord],
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> SplitResult:
    total_ratio = train_ratio + validation_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    total = len(items)
    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)
    return SplitResult(train=items[:train_end], validation=items[train_end:validation_end], test=items[validation_end:])


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def records_to_jsonl_rows(records: Sequence[FineTuningRecord]) -> list[dict[str, Any]]:
    return [{"id": record.id, "messages": record.messages, "metadata": record.metadata} for record in records]
