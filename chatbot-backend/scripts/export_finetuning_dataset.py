"""Export an anonymized fine-tuning dataset from Chatwoot conversations.

This script supports two input modes:
- Live Chatwoot API export using CHATWOOT_API_TOKEN + CHATWOOT_ACCOUNT_ID
- A local JSON file containing a Chatwoot export payload or a list of conversations

Output:
- JSONL training file in chat-completions format
- metadata JSON with store/topic/resolution counts
- optional CSV manifest for easier review
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.chatwoot_service import ChatwootError, get_conversation_messages, list_conversations
from training.dataset_utils import (
    FineTuningRecord as TrainingExample,
    conversation_to_examples,
    evaluate_conversation_quality,
    example_to_review_row,
    summarize_examples,
)


async def _fetch_conversations_from_chatwoot(
    *,
    account_id: int,
    status: str | None,
    per_page: int,
    max_conversations: int | None,
) -> list[dict[str, Any]]:
    page = 1
    out: list[dict[str, Any]] = []

    while True:
        batch = await list_conversations(account_id=account_id, page=page, per_page=per_page, status=status)
        if not batch:
            break

        out.extend(batch)
        if max_conversations is not None and len(out) >= max_conversations:
            return out[:max_conversations]

        if len(batch) < per_page:
            break
        page += 1

    return out


def _load_local_export(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("conversations", "data", "items", "payload"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise SystemExit("Unsupported export JSON shape. Expected a list of conversations or a dict containing 'conversations'.")


async def _hydrate_conversations(
    conversations: list[dict[str, Any]],
    *,
    account_id: int | None,
) -> list[dict[str, Any]]:
    """Ensure each conversation includes messages.

    If the input already contains messages, it is left intact.
    Otherwise, we fetch messages from the Chatwoot API when possible.
    """
    if account_id is None:
        return conversations

    hydrated: list[dict[str, Any]] = []
    for conv in conversations:
        if not isinstance(conv, dict):
            continue
        has_messages = any(isinstance(conv.get(key), list) for key in ("messages", "conversation_messages", "data"))
        if has_messages:
            hydrated.append(conv)
            continue

        conv_id = conv.get("id") or conv.get("conversation_id")
        if not isinstance(conv_id, int) and not (isinstance(conv_id, str) and conv_id.isdigit()):
            hydrated.append(conv)
            continue

        try:
            messages = await get_conversation_messages(account_id=account_id, conversation_id=int(conv_id))
        except ChatwootError:
            hydrated.append(conv)
            continue

        updated = dict(conv)
        updated["messages"] = messages
        hydrated.append(updated)

    return hydrated


async def _run(args: argparse.Namespace) -> int:
    if args.input:
        source_conversations = _load_local_export(Path(args.input))
        source_name = str(Path(args.input))
    else:
        account_id = args.account_id or settings.CHATWOOT_ACCOUNT_ID
        if not isinstance(account_id, int):
            raise SystemExit("Missing account id. Provide --account-id or set CHATWOOT_ACCOUNT_ID.")
        source_conversations = await _fetch_conversations_from_chatwoot(
            account_id=account_id,
            status=args.status,
            per_page=args.per_page,
            max_conversations=args.max_conversations,
        )
        source_name = f"chatwoot_account_{account_id}"

    account_id = args.account_id or settings.CHATWOOT_ACCOUNT_ID
    source_conversations = await _hydrate_conversations(source_conversations, account_id=account_id if isinstance(account_id, int) else None)

    examples: list[TrainingExample] = []
    rejected_rows: list[dict[str, Any]] = []
    for conv in source_conversations:
        quality = evaluate_conversation_quality(
            conv,
            min_turns=args.min_turns,
            min_user_turns=args.min_user_turns,
            min_assistant_turns=args.min_assistant_turns,
            require_resolved=not args.allow_unresolved,
            require_nontrivial_length=not args.allow_short_conversations,
            max_short_turn_ratio=args.max_short_turn_ratio,
        )

        if not quality.accepted:
            rejected_rows.append(
                {
                    "conversation_id": conv.get("id") or conv.get("conversation_id") or conv.get("uuid") or "",
                    "store": conv.get("store_key") or conv.get("store") or args.default_store,
                    "topic": conv.get("subject") or conv.get("title") or "unknown",
                    "status": quality.metadata.get("status"),
                    "score": quality.score,
                    "reasons": json.dumps(quality.reasons, ensure_ascii=False),
                    "message_count": quality.metadata.get("message_count"),
                    "user_message_count": quality.metadata.get("user_message_count"),
                    "assistant_message_count": quality.metadata.get("assistant_message_count"),
                    "character_count": quality.metadata.get("character_count"),
                }
            )
            continue

        examples.extend(
            conversation_to_examples(
                conv,
                default_store=args.default_store,
                include_system_message=not args.no_system_message,
                allowed_terms=args.allowed_terms,
                quality_result=quality,
            )
        )

    if not examples:
        raise SystemExit("No training examples could be generated from the provided conversations.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / args.review_name
    metadata_path = output_dir / args.metadata_name
    rejected_path = output_dir / args.rejected_name

    summary = summarize_examples(examples)
    summary.update(
        {
            "source": source_name,
            "total_conversations": len(source_conversations),
            "review_file": str(review_path.name),
        }
    )
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "review_status",
                "messages_json",
                "metadata_json",
                "notes",
                "conversation_id",
                "store",
                "topic",
                "resolution",
                "language",
                "quality_score",
                "quality_accepted",
                "quality_reasons",
            ],
        )
        writer.writeheader()
        for example in examples:
            meta = example.metadata
            review_row = example_to_review_row(example, review_status="pending")
            writer.writerow(
                {
                    "id": review_row.id,
                    "review_status": review_row.review_status,
                    "messages_json": review_row.messages_json,
                    "metadata_json": review_row.metadata_json,
                    "notes": review_row.notes,
                    "conversation_id": meta.get("conversation_id"),
                    "store": meta.get("store"),
                    "topic": meta.get("topic"),
                    "resolution": meta.get("resolution"),
                    "language": meta.get("language"),
                    "quality_score": meta.get("quality_score"),
                    "quality_accepted": meta.get("quality_accepted"),
                    "quality_reasons": json.dumps(meta.get("quality_reasons") or [], ensure_ascii=False),
                }
            )

    with rejected_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conversation_id",
                "store",
                "topic",
                "status",
                "score",
                "reasons",
                "message_count",
                "user_message_count",
                "assistant_message_count",
                "character_count",
            ],
        )
        writer.writeheader()
        for row in rejected_rows:
            writer.writerow(row)

    print(f"Wrote review queue: {review_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote rejected review queue: {rejected_path}")
    print(f"Examples: {len(examples)}")
    print(f"Rejected conversations: {len(rejected_rows)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a fine-tuning dataset from Chatwoot conversations.")
    parser.add_argument("--input", help="Path to a local Chatwoot export JSON file")
    parser.add_argument("--account-id", type=int, default=None, help="Chatwoot account id (defaults to CHATWOOT_ACCOUNT_ID)")
    parser.add_argument("--status", default=None, help="Optional Chatwoot conversation status filter (open, resolved, pending, ...)")
    parser.add_argument("--per-page", type=int, default=100, help="Chatwoot page size for live export")
    parser.add_argument("--max-conversations", type=int, default=None, help="Optional limit for live export")
    parser.add_argument("--output-dir", default=str(BACKEND_ROOT / "artifacts" / "fine_tuning"), help="Output directory")
    parser.add_argument("--review-name", default="fine_tuning_review.csv", help="CSV review filename")
    parser.add_argument("--metadata-name", default="fine_tuning_metadata.json", help="Metadata JSON filename")
    parser.add_argument("--rejected-name", default="fine_tuning_rejected.csv", help="CSV of rejected conversations")
    parser.add_argument("--default-store", default="default", help="Fallback store name for metadata")
    parser.add_argument(
        "--allowed-terms",
        nargs="*",
        default=[],
        help="Extra safe terms to keep unmasked (for example product names or cities)",
    )
    parser.add_argument("--no-system-message", action="store_true", help="Omit the system message from each training example")
    parser.add_argument("--min-turns", type=int, default=2, help="Minimum total message turns required")
    parser.add_argument("--min-user-turns", type=int, default=1, help="Minimum user turns required")
    parser.add_argument("--min-assistant-turns", type=int, default=1, help="Minimum assistant turns required")
    parser.add_argument("--max-short-turn-ratio", type=float, default=0.35, help="Reject threads with too many very short turns")
    parser.add_argument("--allow-unresolved", action="store_true", help="Allow unresolved conversations through the filter")
    parser.add_argument("--allow-short-conversations", action="store_true", help="Allow very short conversations through the filter")
    args = parser.parse_args()

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
