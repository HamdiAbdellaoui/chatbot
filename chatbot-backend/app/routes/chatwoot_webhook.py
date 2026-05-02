"""Chatwoot Agent Bot webhook endpoint.

Goals:
- Correctly receive message events from Chatwoot
- Validate webhook authenticity (HMAC)
- Extract message content, conversation_id, inbox_id
- Send a reply back to Chatwoot
"""

import base64
import binascii
import hashlib
import hmac
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request

from app.config import settings
from app.services.chatbot_service import process_chatwoot_message
from app.services.chatwoot_service import ChatwootError, send_message

logger = logging.getLogger(__name__)
router = APIRouter()


def _normalize_signature_header(value: str) -> str:
    # Chatwoot header values may include whitespace or quotes.
    return value.strip().strip('"').strip("'")


def _signature_to_bytes(signature_header: str) -> Optional[bytes]:
    """Parse signature header into raw bytes.

Supports the common encodings used by webhook systems:
- hex string (64 chars for sha256)
- base64 string
"""
    sig = _normalize_signature_header(signature_header)

    # Some systems prefix values (e.g. "sha256=..."). Be tolerant.
    if sig.lower().startswith("sha256="):
        sig = sig.split("=", 1)[1].strip()

    # Try hex first
    try:
        if len(sig) == 64:
            return binascii.unhexlify(sig)
    except (binascii.Error, ValueError):
        pass

    # Try base64
    try:
        return base64.b64decode(sig, validate=True)
    except (binascii.Error, ValueError):
        return None


def verify_signature(raw_body: bytes, signature_header: Optional[str]) -> None:
    """Verify Chatwoot webhook signature.

We compute HMAC-SHA256 over the *raw request body* using the shared secret.
"""
    if not settings.CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE:
        logger.warning("Webhook signature validation is DISABLED (CHATWOOT_VALIDATE_WEBHOOK_SIGNATURE=false).")
        return

    if not signature_header:
        logger.warning("Missing X-Chatwoot-Hmac-SHA256 header.")
        raise HTTPException(status_code=401, detail="HMAC signature missing.")

    provided = _signature_to_bytes(signature_header)
    if provided is None:
        logger.warning("Unparseable webhook signature header value.")
        raise HTTPException(status_code=401, detail="Invalid HMAC signature.")

    secret = settings.CHATWOOT_WEBHOOK_SECRET.encode("utf-8")
    expected = hmac.new(secret, raw_body, hashlib.sha256).digest()

    if not hmac.compare_digest(expected, provided):
        logger.error("Invalid webhook HMAC signature.")
        raise HTTPException(status_code=401, detail="Invalid HMAC signature.")

    logger.debug("Webhook HMAC signature verified successfully.")


def _extract_message_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the message object from different Chatwoot webhook payload shapes."""
    if isinstance(payload.get("message"), dict):
        return payload["message"]

    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("message"), dict):
        return data["message"]

    # Fall back to treating top-level as the message container
    return payload


def _get_nested_id(obj: Any, *path: str) -> Optional[int]:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, int):
        return cur
    if isinstance(cur, str) and cur.isdigit():
        return int(cur)
    return None


def parse_chatwoot_event(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Chatwoot webhook payload into a normalized structure."""
    message = _extract_message_payload(payload)

    event_type = payload.get("event") or payload.get("event_name")
    message_type = message.get("message_type") or payload.get("message_type")
    is_private = bool(message.get("private") if "private" in message else payload.get("private", False))

    content = message.get("content") or payload.get("content") or ""
    content = content if isinstance(content, str) else ""

    message_id = _get_nested_id(message, "id") or _get_nested_id(payload, "id")
    conversation_id = (
        _get_nested_id(message, "conversation_id")
        or _get_nested_id(payload, "conversation", "id")
        or _get_nested_id(payload, "conversation_id")
    )
    inbox_id = (
        _get_nested_id(message, "inbox_id")
        or _get_nested_id(payload, "inbox", "id")
        or _get_nested_id(payload, "inbox_id")
    )
    account_id = (
        _get_nested_id(message, "account_id")
        or _get_nested_id(payload, "account", "id")
        or settings.CHATWOOT_ACCOUNT_ID
    )

    sender = message.get("sender") or payload.get("sender") or {}
    sender_name = sender.get("name") if isinstance(sender, dict) else None
    inbox_name = (payload.get("inbox") or {}).get("name") if isinstance(payload.get("inbox"), dict) else None

    return {
        "event_type": event_type,
        "message_type": message_type,
        "is_private": is_private,
        "content": content,
        "message_id": message_id,
        "conversation_id": conversation_id,
        "inbox_id": inbox_id,
        "account_id": account_id,
        "sender_name": sender_name,
        "inbox_name": inbox_name,
        "raw": payload,
    }


@router.post("/chatwoot-webhook", include_in_schema=False)
async def handle_chatwoot_webhook(request: Request):
    """
    This endpoint receives all events from the Chatwoot Agent Bot.
    It verifies the request signature and processes message-related events.
    """
    raw_body = await request.body()
    signature = request.headers.get("X-Chatwoot-Hmac-SHA256")
    verify_signature(raw_body, signature)

    try:
        payload: Dict[str, Any] = json.loads(raw_body.decode("utf-8"))
    except Exception:
        logger.error("Failed to decode JSON webhook payload.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    event = parse_chatwoot_event(payload)
    logger.info(
        "Chatwoot webhook received event=%s message_type=%s conversation_id=%s inbox_id=%s message_id=%s private=%s",
        event.get("event_type"),
        event.get("message_type"),
        event.get("conversation_id"),
        event.get("inbox_id"),
        event.get("message_id"),
        event.get("is_private"),
    )
    # Never log raw payloads; they may include PII.

    # Ignore non-customer messages.
    if event.get("event_type") != "message_created":
        return {"status": "ignored", "reason": "Not a message_created event."}

    if event.get("message_type") != "incoming":
        return {"status": "ignored", "reason": "Not an incoming message."}

    if event.get("is_private"):
        return {"status": "ignored", "reason": "Private message."}

    if not event.get("conversation_id") or not event.get("inbox_id"):
        logger.warning("Missing required identifiers in webhook payload.")
        raise HTTPException(status_code=400, detail="Missing conversation_id or inbox_id")

    account_id = event.get("account_id")
    conversation_id = event.get("conversation_id")

    # Generate next action (reply vs no_reply) from the chatbot pipeline.
    result = await process_chatwoot_message(payload, account_id=account_id if isinstance(account_id, int) else None)

    if result.action == "no_reply":
        return {
            "status": "suppressed",
            "reason": result.reason or "no_reply",
            "conversation_id": conversation_id,
            "inbox_id": event.get("inbox_id"),
        }

    response_message = result.reply or ""

    # Preferred: send reply via Chatwoot REST API (reliable, explicit).
    if isinstance(account_id, int) and isinstance(conversation_id, int) and settings.CHATWOOT_API_TOKEN:
        try:
            await send_message(account_id=account_id, conversation_id=conversation_id, content=response_message)
        except ChatwootError as e:
            # Do not log response bodies; they may include sensitive data.
            logger.error("Failed to send reply to Chatwoot (status=%s)", e.status_code)

            # Do NOT return a 5xx, or Chatwoot may retry webhooks and cause duplicates.
            return {
                "status": "send_failed",
                "conversation_id": conversation_id,
                "inbox_id": event.get("inbox_id"),
                "escalated": result.escalated,
                "reply": {"content": response_message} if response_message else None,
            }

        return {"status": "sent", "conversation_id": conversation_id, "inbox_id": event.get("inbox_id"), "escalated": result.escalated}

    # Fallback: return reply content in response body for local debugging.
    # (Depending on Chatwoot configuration, it may or may not use this.)
    logger.warning(
        "CHATWOOT_API_TOKEN/CHATWOOT_ACCOUNT_ID not configured; returning reply in webhook response body for debugging."
    )
    return {"status": "debug", "reply": {"content": response_message}}


@router.get("/chatwoot-webhook", include_in_schema=False)
async def verify_webhook_url():
    """
    Chatwoot sends a GET request to verify the webhook URL when you first add it.
    This endpoint handles that verification check by returning a 200 OK response.
    """
    logger.info("Received GET request for webhook URL verification from Chatwoot.")
    return {"status": "ok"}
