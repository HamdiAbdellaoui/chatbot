# Webhook endpoint for receiving events from Chatwoot.

import logging
from fastapi import APIRouter, Request, HTTPException, Header
from app.services.chatbot_service import process_chatwoot_message
import hmac
import hashlib
import json

logger = logging.getLogger(__name__)
router = APIRouter()

# This is a placeholder for the actual secret you will get from your .env file
# We will replace this with a dependency injection system later for better practice.
from app.config import settings
WEBHOOK_SECRET = settings.CHATWOOT_WEBHOOK_SECRET.encode('utf-8')


async def verify_signature(request: Request, x_chatwoot_hmac_sha256: str = Header(None, alias="X-Chatwoot-Hmac-SHA256")):
    """
    Verify the HMAC signature of the incoming webhook request from Chatwoot.
    This ensures that the request is authentic and came from your Chatwoot instance.
    """
    if not x_chatwoot_hmac_sha256:
        logger.warning("HMAC signature missing from request.")
        raise HTTPException(status_code=401, detail="HMAC signature missing.")

    raw_body = await request.body()
    
    # The signature is created using the raw request body and your webhook secret.
    h = hmac.new(WEBHOOK_SECRET, raw_body, hashlib.sha256)
    expected_signature = h.hexdigest()

    if not hmac.compare_digest(expected_signature, x_chatwoot_hmac_sha256):
        logger.error("Invalid HMAC signature.")
        raise HTTPException(status_code=401, detail="Invalid HMAC signature.")
    
    logger.debug("HMAC signature verified successfully.")


@router.post("/chatwoot-webhook", include_in_schema=False)
async def handle_chatwoot_webhook(request: Request):
    """
    This endpoint receives all events from the Chatwoot Agent Bot.
    It verifies the request signature and processes message-related events.
    """
    # First, verify the signature to ensure the request is from Chatwoot
    # The dependency will handle the exception if verification fails.
    await verify_signature(request)

    try:
        payload = await request.json()
        logger.info(f"Received webhook payload: {json.dumps(payload, indent=2)}")

        # We only care about messages created by end-users (contacts)
        # and we don't want the bot to reply to its own messages.
        event_type = payload.get("event")
        message_type = payload.get("message_type")
        is_private = payload.get("private", False)
        
        if event_type == "message_created" and message_type == "incoming" and not is_private:
            # This is a message from a customer. Process it.
            response_message = await process_chatwoot_message(payload)
            
            # The service layer returns the content for the reply.
            # The actual sending is handled by the Chatwoot client (to be built).
            # For now, we just return a JSON response for debugging.
            return {"status": "success", "reply_content": response_message}
        else:
            # Ignore other events like message_updated, agent activity, etc.
            logger.info(f"Ignoring event: {event_type}, message_type: {message_type}")
            return {"status": "ignored", "reason": "Not a customer message."}

    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from request body.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")
    except Exception as e:
        logger.exception("An unexpected error occurred while processing the webhook.")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/chatwoot-webhook", include_in_schema=False)
async def verify_webhook_url():
    """
    Chatwoot sends a GET request to verify the webhook URL when you first add it.
    This endpoint handles that verification check by returning a 200 OK response.
    """
    logger.info("Received GET request for webhook URL verification from Chatwoot.")
    return {"status": "ok"}
