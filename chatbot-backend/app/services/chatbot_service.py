# This service contains the core business logic for the chatbot.
# It processes the incoming message payload and orchestrates the response.

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, Literal

from app.config import settings
from app.services.llm_service import OrderRequest, generate_grounded_reply, estimate_context_grounding
from app.services.rag_service import retrieve_context
from app.services.store_context_service import resolve_store
from app.services.woocommerce_service import WooCommerceError, get_woocommerce_client_for_store
from app.services.escalation_service import detect_escalation_request, detect_low_confidence
from app.services.chatwoot_service import ChatwootError, escalate_conversation, get_conversation_labels, send_message
from app.services.pii_service import detect_pii, mask_pii, select_entities_to_mask
from app.services.active_learning_service import log_low_confidence_flag
from app.services.session_service import get_history, append_turn
from app.services.conversation_log_service import log_turn
from app.services.language_service import detect_language
from app.services.confidence_service import combine_confidence, is_business_decision, should_call_llm_confidence_signal


def _parse_wc_command(text: str) -> tuple[str, list[str]] | None:
    t = (text or "").strip()
    if not t.lower().startswith("/wc"):
        return None

    parts = t.split()
    if len(parts) < 2:
        return ("help", [])

    return (parts[1].lower(), parts[2:])

logger = logging.getLogger(__name__)

ORDER_FORWARDED_MESSAGE = (
    "Merci ! Votre demande de commande a été transmise à un conseiller, "
    "qui va la vérifier et revenir vers vous ici."
)


def _log_event(event: str, *, conversation_id: int | None, store_id: str | None, escalation_reason: str | None) -> None:
    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "store_id": store_id,
        "escalation_reason": escalation_reason,
    }
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _parse_allowed_terms_csv(value: str) -> list[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


@dataclass(frozen=True)
class ChatbotResult:
    action: Literal["reply", "no_reply"]
    reply: str | None = None
    escalated: bool = False
    reason: str | None = None


def _extract_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    message = payload.get("message")
    if isinstance(message, dict):
        return message

    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("message"), dict):
        return data["message"]

    return payload


def _get_id(obj: Any, *path: str) -> int | None:
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


async def _apply_escalation_labels(
    *,
    effective_account_id: int | None,
    conversation_id: int | None,
    extra_labels: list[str] | None = None,
) -> None:
    """Apply escalation labels/assignment in Chatwoot. Best-effort: never raises."""
    if not (isinstance(effective_account_id, int) and isinstance(conversation_id, int) and settings.CHATWOOT_API_TOKEN):
        return

    try:
        labels_to_apply = [settings.ESCALATION_LABEL]
        if settings.ESCALATION_SEND_ACK and settings.ESCALATION_ACK_LABEL:
            labels_to_apply.append(settings.ESCALATION_ACK_LABEL)
        # One single call: Chatwoot's labels endpoint replaces the label set.
        labels_to_apply.extend(l for l in (extra_labels or []) if l and l not in labels_to_apply)

        await asyncio.wait_for(
            escalate_conversation(
                account_id=effective_account_id,
                conversation_id=conversation_id,
                labels=labels_to_apply,
                assignee_id=settings.ESCALATION_ASSIGNEE_ID,
                team_id=settings.ESCALATION_TEAM_ID,
            ),
            timeout=settings.ESCALATION_API_BUDGET_S,
        )
    except ChatwootError as e:
        logger.warning("Escalation API call failed (status=%s)", e.status_code)
    except TimeoutError:
        logger.warning("Escalation API call timed out")


def _format_order_note(requests: list[OrderRequest]) -> str:
    """Private note for agents. Customer notes are re-masked defensively."""
    allowed_terms = _parse_allowed_terms_csv(settings.PII_ALLOWED_TERMS)
    lines = ["Demande de commande à valider (transmise par le bot, aucune commande créée dans WooCommerce) :"]
    for r in requests:
        lines.append(f"- Produit #{r.product_id} : {r.product_name or '?'}")
        lines.append(f"  Prix unitaire : {r.price or '?'} | Quantité : {r.quantity}")
        if r.customer_note:
            note = mask_pii(r.customer_note, min_confidence=settings.PII_MIN_CONFIDENCE, allowed_terms=allowed_terms)
            lines.append(f"  Note client : {note}")
    return "\n".join(lines)


async def _hand_over_order_request(
    *,
    requests: list[OrderRequest],
    effective_account_id: int | None,
    conversation_id: int | None,
) -> None:
    """Private note + order label + escalation so an agent validates the order. Never raises."""
    if not (isinstance(effective_account_id, int) and isinstance(conversation_id, int) and settings.CHATWOOT_API_TOKEN):
        logger.warning("Order request received but Chatwoot API is not configured; cannot notify an agent")
        return

    try:
        await asyncio.wait_for(
            send_message(
                account_id=effective_account_id,
                conversation_id=conversation_id,
                content=_format_order_note(requests),
                private=True,
            ),
            timeout=settings.ESCALATION_API_BUDGET_S,
        )
    except ChatwootError as e:
        logger.warning("Failed to post order request note (status=%s)", e.status_code)
    except TimeoutError:
        logger.warning("Posting order request note timed out")

    await _apply_escalation_labels(
        effective_account_id=effective_account_id,
        conversation_id=conversation_id,
        extra_labels=[settings.ORDER_VALIDATION_LABEL],
    )

async def process_chatwoot_message(payload: Dict[str, Any], *, account_id: int | None = None) -> ChatbotResult:
    """
    Processes an incoming message from Chatwoot and generates a response.

    Args:
        payload: The JSON payload from the Chatwoot webhook.

    Returns:
        ChatbotResult describing whether to reply or suppress.
    """
    try:
        # 1. Extract relevant information from the payload (robust to payload shape)
        message = _extract_message(payload)

        user_message = message.get("content") or payload.get("content") or ""
        user_message = user_message if isinstance(user_message, str) else ""

        conversation_id = (
            _get_id(message, "conversation_id")
            or _get_id(payload, "conversation", "id")
            or _get_id(payload, "conversation_id")
        )
        inbox_id = (
            _get_id(message, "inbox_id")
            or _get_id(payload, "inbox", "id")
            or _get_id(payload, "inbox_id")
        )

        effective_account_id = account_id or _get_id(message, "account_id") or _get_id(payload, "account", "id") or settings.CHATWOOT_ACCOUNT_ID

        sender = message.get("sender") or payload.get("sender") or {}
        contact_name = sender.get("name") if isinstance(sender, dict) else None
        inbox_name = (payload.get("inbox") or {}).get("name") if isinstance(payload.get("inbox"), dict) else None
        
        logger.info(
            "Processing message contact_name_present=%s inbox_name=%s inbox_id=%s conversation_id=%s",
            bool(contact_name),
            inbox_name,
            inbox_id,
            conversation_id,
        )
        logger.info("User message received (len=%s)", len(user_message))

        detected_language = detect_language(user_message)
        logger.info("Detected language=%s", detected_language)

        if not user_message.strip():
            return ChatbotResult(action="reply", reply="I didn't receive any text. Could you please type your question?")

        # 2. Resolve store context (multi-store)
        store = resolve_store(inbox_id=inbox_id, inbox_name=inbox_name)

        # 2b. Escalation guard: if already escalated, stop responding.
        if settings.ESCALATION_ENABLED and isinstance(effective_account_id, int) and isinstance(conversation_id, int) and settings.CHATWOOT_API_TOKEN:
            try:
                labels = await asyncio.wait_for(
                    get_conversation_labels(account_id=effective_account_id, conversation_id=conversation_id),
                    timeout=settings.ESCALATION_API_BUDGET_S,
                )
                if settings.ESCALATION_LABEL and settings.ESCALATION_LABEL in labels:
                    logger.info("Conversation already escalated (label=%s). Suppressing bot reply.", settings.ESCALATION_LABEL)
                    _log_event(
                        "escalation_suppressed",
                        conversation_id=conversation_id,
                        store_id=store.key,
                        escalation_reason="already_escalated",
                    )
                    return ChatbotResult(action="no_reply", escalated=True, reason="already_escalated")
            except ChatwootError as e:
                logger.warning("Failed to check conversation labels (status=%s); continuing", e.status_code)
            except TimeoutError:
                logger.warning("Timeout checking conversation labels; continuing")

        # 2c. Keyword escalation request.
        if settings.ESCALATION_ENABLED:
            decision = detect_escalation_request(user_message)
            if decision.should_escalate:
                logger.info("Escalation requested (reason=%s)", decision.reason)
                _log_event(
                    "escalation_triggered",
                    conversation_id=conversation_id if isinstance(conversation_id, int) else None,
                    store_id=store.key,
                    escalation_reason=decision.reason,
                )
                await _apply_escalation_labels(effective_account_id=effective_account_id, conversation_id=conversation_id)

                # Send one final acknowledgement if configured, then stop future replies via label.
                if settings.ESCALATION_SEND_ACK:
                    return ChatbotResult(action="reply", reply=settings.ESCALATION_ACK_MESSAGE, escalated=True, reason=decision.reason)
                return ChatbotResult(action="no_reply", escalated=True, reason=decision.reason)

        # Optional: explicit WooCommerce commands for development/testing.
        # Example:
        #   /wc search iphone
        #   /wc price 123
        #   /wc draft 123 2
        wc_cmd = _parse_wc_command(user_message) if settings.WOOCOMMERCE_COMMANDS_ENABLED else None
        if wc_cmd is not None:
            action, args = wc_cmd

            if action in {"help", "?"}:
                return ChatbotResult(action="reply", reply="WooCommerce commands: /wc search <term> | /wc price <product_id> | /wc draft <product_id> <qty>")

            client = get_woocommerce_client_for_store(store)
            if client is None:
                return ChatbotResult(action="reply", reply="WooCommerce is not configured for this store.")

            try:
                if action == "search":
                    term = " ".join(args).strip()
                    if not term:
                        return ChatbotResult(action="reply", reply="Usage: /wc search <term>")
                    products = await client.search_products(query=term, limit=5)
                    if not products:
                        return ChatbotResult(action="reply", reply=f"No products found for '{term}'.")
                    lines = ["Top products:"]
                    for p in products:
                        lines.append(f"- id={p.id} name={p.name} price={p.price} stock={p.stock_status} qty={p.stock_quantity}")
                    return ChatbotResult(action="reply", reply="\n".join(lines))

                if action == "price":
                    if not args or not args[0].isdigit():
                        return ChatbotResult(action="reply", reply="Usage: /wc price <product_id>")
                    pid = int(args[0])
                    info = await client.get_price_and_stock(product_id=pid)
                    return ChatbotResult(action="reply", reply=(
                        f"Product {info.get('id')}: {info.get('name')}\n"
                        f"Price: {info.get('price')} (regular={info.get('regular_price')} sale={info.get('sale_price')})\n"
                        f"Stock: status={info.get('stock_status')} qty={info.get('stock_quantity')} manage_stock={info.get('manage_stock')}\n"
                        f"Link: {info.get('permalink')}"
                    ))

                if action in {"draft", "order"}:
                    if len(args) < 2 or not args[0].isdigit() or not args[1].isdigit():
                        return ChatbotResult(action="reply", reply="Usage: /wc draft <product_id> <qty>")
                    pid = int(args[0])
                    qty = int(args[1])
                    if qty <= 0:
                        return ChatbotResult(action="reply", reply="Quantity must be >= 1")
                    order = await client.create_draft_order(line_items=[(pid, qty)])
                    return ChatbotResult(action="reply", reply=f"Draft order created: id={order.id} status={order.status} total={order.total} currency={order.currency} payment_url={order.payment_url}")

                return ChatbotResult(action="reply", reply="Unknown /wc command. Use /wc help")

            except WooCommerceError as e:
                logger.warning("WooCommerce error store=%s action=%s status=%s", store.key, action, e.status_code)
                return ChatbotResult(action="reply", reply="Sorry — I couldn't reach the store system right now. Please try again.")
            finally:
                await client.aclose()

        # 3. PII masking. Must happen before anything leaves the backend: the
        # retrieval query is embedded by an external provider (OpenAI), and the
        # LLM, history and logs also only ever see masked text.
        allowed_terms = _parse_allowed_terms_csv(settings.PII_ALLOWED_TERMS)
        pii_entities = detect_pii(user_message, allowed_terms=allowed_terms)
        pii_to_mask = select_entities_to_mask(
            pii_entities,
            min_confidence=settings.PII_MIN_CONFIDENCE,
            allowed_terms=allowed_terms,
        )
        masked_user_message = mask_pii(
            user_message,
            entities=pii_entities,
            min_confidence=settings.PII_MIN_CONFIDENCE,
            allowed_terms=allowed_terms,
        )

        # 3b. Retrieve context from Qdrant (basic RAG) using the masked text only.
        collection = store.qdrant_collection
        top_k = settings.RAG_TOP_K
        logger.info(
            "Resolved store=%s (inbox_id=%s inbox_name=%s) -> collection=%s",
            store.key,
            inbox_id,
            inbox_name,
            collection,
        )
        logger.info("Retrieving context from Qdrant collection=%s top_k=%s", collection, top_k)
        context, hits = await retrieve_context(query=masked_user_message, collection=collection, limit=top_k)
        if hits:
            top_score = hits[0].get("score")
            sources = []
            for h in hits[: min(len(hits), 3)]:
                payload = h.get("payload") or {}
                sources.append(payload.get("source") or payload.get("doc_id") or payload.get("title") or "unknown")
            logger.info("Retrieved %s context snippets (top_score=%s sources=%s)", len(hits), top_score, sources)
        else:
            logger.info("No context snippets retrieved")
            top_score = None
            sources = []

        # Full conversation log (masked content only). Best-effort: log_turn never raises.
        pii_types_detected = sorted({e.type for e in pii_to_mask}) if pii_to_mask else None
        await log_turn(
            conversation_id=conversation_id if isinstance(conversation_id, int) else None,
            inbox_id=inbox_id if isinstance(inbox_id, int) else None,
            store_key=store.key,
            direction="in",
            content_masked=masked_user_message,
            pii_types=pii_types_detected,
            rag_top_score=top_score if isinstance(top_score, float) else None,
            rag_hits_count=len(hits),
        )

        # 4. Generate grounded response via LLM
        logger.info("Calling LLM with retrieved context")
        if pii_entities:
            detected_types = sorted({e.type for e in pii_entities})
            confidences = [e.confidence for e in pii_entities]
            logger.info(
                "PII detected (types=%s count=%s conf_min=%.2f conf_max=%.2f)",
                detected_types,
                len(pii_entities),
                min(confidences),
                max(confidences),
            )

        if pii_to_mask:
            masked_types = sorted({e.type for e in pii_to_mask})
            logger.info("Applied PII masking before retrieval/LLM (types=%s count=%s)", masked_types, len(pii_to_mask))

        if settings.PII_DEBUG and pii_entities:
            # Safe debug: log only spans + confidence, never raw values.
            spans = [
                {
                    "type": e.type,
                    "start": e.start,
                    "end": e.end,
                    "confidence": round(e.confidence, 2),
                }
                for e in pii_entities
            ]
            logger.debug("PII spans=%s", json.dumps(spans, ensure_ascii=False, separators=(",", ":")))

        history = await get_history(conversation_id) if isinstance(conversation_id, int) else []

        llm_started_at = time.perf_counter()
        order_requests: list[OrderRequest] = []
        response_content = await generate_grounded_reply(
            user_message=masked_user_message,
            context=context,
            store_context=store,
            history=history,
            language=detected_language,
            order_requests=order_requests,
        )
        latency_ms = int((time.perf_counter() - llm_started_at) * 1000)

        # 4a. Order request (WOOCOMMERCE_ORDER_MODE=handoff): a human validates it in Chatwoot.
        if order_requests:
            logger.info("Order request forwarded to a human (count=%s)", len(order_requests))
            _log_event(
                "escalation_triggered",
                conversation_id=conversation_id if isinstance(conversation_id, int) else None,
                store_id=store.key,
                escalation_reason="order_request",
            )
            await _hand_over_order_request(
                requests=order_requests,
                effective_account_id=effective_account_id,
                conversation_id=conversation_id,
            )
            order_reply = ORDER_FORWARDED_MESSAGE
            await log_turn(
                conversation_id=conversation_id if isinstance(conversation_id, int) else None,
                inbox_id=inbox_id if isinstance(inbox_id, int) else None,
                store_key=store.key,
                direction="out",
                content_masked=order_reply,
                rag_top_score=top_score if isinstance(top_score, float) else None,
                escalated=True,
                escalation_reason="order_request",
                model=settings.OPENAI_MODEL,
                latency_ms=latency_ms,
            )
            if isinstance(conversation_id, int):
                await append_turn(conversation_id, "user", masked_user_message)
                await append_turn(conversation_id, "assistant", order_reply)
            return ChatbotResult(action="reply", reply=order_reply, escalated=True, reason="order_request")

        # 4b. Combined 3-signal confidence score (RAG score + LLM self-assessment +
        # business-decision rule). The LLM signal needs the generated answer, so
        # this can only be computed after the LLM call above, not before it.
        # Perf: the self-assessment call is a second LLM round-trip, so it's only
        # made when the RAG score alone is ambiguous (see should_call_llm_confidence_signal).
        rag_score_for_confidence = top_score if isinstance(top_score, float) else None
        if should_call_llm_confidence_signal(rag_score=rag_score_for_confidence, hits_count=len(hits)):
            llm_confidence = await estimate_context_grounding(user_message=masked_user_message, context=context, answer=response_content)
        else:
            llm_confidence = None
            logger.info("Skipping LLM confidence self-assessment (rag_score=%s hits=%s not ambiguous)", top_score, len(hits))
        business_decision = is_business_decision(masked_user_message)
        confidence_score = combine_confidence(
            rag_score=rag_score_for_confidence,
            llm_confidence=llm_confidence,
            is_business_decision=business_decision,
        )
        logger.info(
            "Confidence score=%.2f (rag_score=%s llm_confidence=%s business_decision=%s)",
            confidence_score,
            top_score,
            llm_confidence,
            business_decision,
        )

        # 4c. Optional: escalate when the combined confidence is too low.
        if settings.ESCALATION_ENABLED:
            lc = detect_low_confidence(confidence_score=confidence_score, hits_count=len(hits))
            if lc.should_escalate:
                logger.info("Escalating due to low confidence (reason=%s confidence_score=%.2f)", lc.reason, confidence_score)
                # Active learning: persist a flag for supervisor review.
                await log_low_confidence_flag(
                    store_key=store.key,
                    inbox_id=inbox_id if isinstance(inbox_id, int) else None,
                    conversation_id=conversation_id if isinstance(conversation_id, int) else None,
                    reason=lc.reason,
                    top_score=top_score if isinstance(top_score, float) else None,
                    hits_count=len(hits),
                    sources=sources,
                    masked_user_message=masked_user_message,
                )
                _log_event(
                    "escalation_triggered",
                    conversation_id=conversation_id if isinstance(conversation_id, int) else None,
                    store_id=store.key,
                    escalation_reason=lc.reason,
                )
                await _apply_escalation_labels(effective_account_id=effective_account_id, conversation_id=conversation_id)

                ack_reply = settings.ESCALATION_ACK_MESSAGE if settings.ESCALATION_SEND_ACK else None
                await log_turn(
                    conversation_id=conversation_id if isinstance(conversation_id, int) else None,
                    inbox_id=inbox_id if isinstance(inbox_id, int) else None,
                    store_key=store.key,
                    direction="out",
                    content_masked=ack_reply,
                    rag_top_score=top_score if isinstance(top_score, float) else None,
                    confidence_score=confidence_score,
                    escalated=True,
                    escalation_reason=lc.reason,
                    model=settings.OPENAI_MODEL,
                    latency_ms=latency_ms,
                )

                if settings.ESCALATION_SEND_ACK:
                    return ChatbotResult(action="reply", reply=settings.ESCALATION_ACK_MESSAGE, escalated=True, reason=lc.reason)
                return ChatbotResult(action="no_reply", escalated=True, reason=lc.reason)

        await log_turn(
            conversation_id=conversation_id if isinstance(conversation_id, int) else None,
            inbox_id=inbox_id if isinstance(inbox_id, int) else None,
            store_key=store.key,
            direction="out",
            content_masked=response_content,
            rag_top_score=top_score if isinstance(top_score, float) else None,
            confidence_score=confidence_score,
            escalated=False,
            escalation_reason=None,
            model=settings.OPENAI_MODEL,
            latency_ms=latency_ms,
        )

        if isinstance(conversation_id, int):
            await append_turn(conversation_id, "user", masked_user_message)
            await append_turn(conversation_id, "assistant", response_content)

        # 5. Return the generated content
        return ChatbotResult(action="reply", reply=response_content)

    except Exception as e:
        logger.exception("Error processing Chatwoot message.")
        # In case of an error, return a user-friendly message.
        return ChatbotResult(action="reply", reply="I'm sorry, but I encountered an error. Please try again later.")
