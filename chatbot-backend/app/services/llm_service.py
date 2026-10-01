"""LLM service for generating chatbot replies.

Phase 1 scope:
- No RAG
- Use OpenAI GPT-4o
- Enforce language mirroring via a system prompt
"""

from __future__ import annotations

import hmac
import logging
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from openai import AsyncOpenAI

from app.config import settings
from app.services.messages import get_message
from app.services.store_context_service import StoreContext
from app.services.woocommerce_service import get_woocommerce_client_for_store

logger = logging.getLogger(__name__)


_LANGUAGE_MIRRORING_SYSTEM_PROMPT = """
You are a customer support assistant for a Tunisian e-commerce platform.

Rules:
- Always respond in the exact same language and dialect/register the user used.
- If the user writes in Tunisian Darija, respond in Tunisian Darija.
- If the user mixes French and Arabic, mirror the same mix.
- Do not switch languages unless the user does first.
- Be concise and helpful.
""".strip()

_SYSTEM_PROMPT_FR = """
Tu es un assistant du service client pour une plateforme e-commerce tunisienne.

Règles :
- Réponds toujours en français, de façon claire et concise.
- Ne change pas de langue à moins que l'utilisateur ne le fasse en premier.
- Sois utile et direct.
""".strip()

_SYSTEM_PROMPT_AR = """
أنت مساعد لخدمة العملاء لدى منصة تجارة إلكترونية تونسية.

القواعد:
- أجب دائماً بالعربية الفصحى، بوضوح واختصار.
- لا تغيّر اللغة إلا إذا غيّرها العميل أولاً.
- كن مفيداً ومباشراً.
""".strip()

_SYSTEM_PROMPT_DARIJA = """
Tu es un assistant du service client pour une plateforme e-commerce tunisienne.

Règles :
- Réponds en darija tunisienne, exactement comme le ferait un agent tunisien au quotidien.
- N'hésite pas à mélanger arabe et français dans la même phrase (arabizi) si c'est naturel dans ce registre.
- Ne change pas de registre à moins que le client ne le fasse en premier.
- Reste concis et utile.
""".strip()

_SYSTEM_PROMPTS_BY_LANGUAGE: dict[str, str] = {
    "fr": _SYSTEM_PROMPT_FR,
    "ar": _SYSTEM_PROMPT_AR,
    "darija": _SYSTEM_PROMPT_DARIJA,
    # "other" (and any unrecognized value) falls back to the generic,
    # language-mirroring prompt used before language detection existed.
    "other": _LANGUAGE_MIRRORING_SYSTEM_PROMPT,
}


def _resolve_system_prompt(language: Optional[str]) -> str:
    if not language:
        return _LANGUAGE_MIRRORING_SYSTEM_PROMPT
    return _SYSTEM_PROMPTS_BY_LANGUAGE.get(language, _LANGUAGE_MIRRORING_SYSTEM_PROMPT)


_RAG_GROUNDING_RULES = """
You will receive a CONTEXT section containing snippets from the company's knowledge base.

Grounding rules:
- Use the CONTEXT to answer. If the CONTEXT does not contain the answer, say you don't know and ask a clarifying question or suggest transferring to a human agent.
- Do NOT invent policies, prices, stock, delivery times, or other facts not present in the CONTEXT.
- If you cite information, keep it paraphrased and concise.
""".strip()


_RAG_USER_PROMPT_TEMPLATE = """
CONTEXT:
{context}

USER QUESTION:
{question}
""".strip()


def build_rag_user_prompt(*, question: str, context: str) -> str:
    """Build the user-facing RAG prompt payload.

    Keeping this in one place makes it easier to debug prompt issues and to
    adjust formatting without touching orchestration code.
    """
    safe_context = context.strip() if context and context.strip() else "[no context found]"
    safe_question = question.strip()
    return _RAG_USER_PROMPT_TEMPLATE.format(context=safe_context, question=safe_question)


def _get_client() -> AsyncOpenAI:
    # The OpenAI SDK reads the API key from this argument.
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def generate_reply(*, user_message: str, system_prompt: Optional[str] = None, language: Optional[str] = None) -> str:
    """Generate a reply using OpenAI Chat Completions.

    Error strategy:
    - Log the exception details server-side
    - Return a safe, user-friendly fallback string to the caller
    """
    if not user_message.strip():
        return get_message("empty_message", language)

    # Fail fast on common misconfiguration to avoid confusing behavior during development.
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in {"your_openai_api_key", "sk-..."}:
        logger.error("OPENAI_API_KEY is not configured (or is still a placeholder).")
        return get_message("not_configured", language)

    prompt = system_prompt or _resolve_system_prompt(language)

    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=settings.OPENAI_MAX_TOKENS,
            temperature=0.2,
            timeout=settings.OPENAI_REQUEST_TIMEOUT_S,
        )

        content = (response.choices[0].message.content or "").strip()
        if not content:
            logger.warning("OpenAI returned an empty message content")
            return "Sorry — I couldn't generate a response. Please try again."

        return content

    except Exception:
        logger.exception("OpenAI request failed")
        # Keep the fallback short and neutral; do not leak internal details.
        return get_message("generic_error", language)

_SEARCH_AND_PRICE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Searches the WooCommerce store for products based on a keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search keyword (e.g. 'iphone', 'shoes')."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_and_stock",
            "description": "Gets the current price and stock status for a specific WooCommerce product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "The ID of the product."}
                },
                "required": ["product_id"],
            },
        },
    },
]

_ORDER_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "product_id": {"type": "integer", "description": "The product ID."},
        "quantity": {"type": "integer", "description": "Quantity to order."},
        "customer_note": {"type": "string", "description": "Optional notes from the customer."}
    },
    "required": ["product_id", "quantity"],
}

# WOOCOMMERCE_ORDER_MODE=direct (tests only): the model creates the order itself.
_CREATE_DRAFT_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "create_draft_order",
        "description": "Creates a draft order in WooCommerce with the given line items.",
        "parameters": _ORDER_TOOL_PARAMETERS,
    },
}

# WOOCOMMERCE_ORDER_MODE=handoff (default): the model only forwards the request
# to a human advisor, who validates and creates the order.
_REQUEST_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "request_order",
        "description": (
            "Forwards the customer's order request (product, quantity, note) to a human sales advisor, "
            "who will validate it and create the order. Does not create any order by itself."
        ),
        "parameters": _ORDER_TOOL_PARAMETERS,
    },
}


_ORDER_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": (
            "Gets the status of an existing order. The customer must have given, in their current message, "
            "the email or phone number used for the order; pass it exactly as it appears "
            "(it may appear as a placeholder such as [EMAIL] or [PHONE])."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer", "description": "The order number."},
                "email_or_phone": {"type": "string", "description": "Email or phone given by the customer for this order."},
            },
            "required": ["order_id", "email_or_phone"],
        },
    },
}

_PII_PLACEHOLDER_RE = re.compile(r"\[(?:EMAIL|PHONE|NAME|ADDRESS)\]")

_ORDER_STATUS_UNVERIFIED = json.dumps({
    "result": "unable_to_verify",
    "message": "Unable to verify this order with the information provided. Ask the customer to check the order number and the email or phone used for the order.",
})


def _normalize_email(value: str) -> str | None:
    v = (value or "").strip().lower()
    return v if "@" in v else None


def _normalize_phone(value: str) -> str | None:
    digits = re.sub(r"\D", "", value or "")
    if digits.startswith("00"):
        digits = digits[2:]
    # Compare national numbers: Tunisian numbers have 8 digits after the 216 prefix.
    if len(digits) < 8:
        return None
    return digits[-8:]


def _contact_matches_billing(candidates: Sequence[str], billing: Dict[str, Any]) -> bool:
    billing_email = _normalize_email(str(billing.get("email") or ""))
    billing_phone = _normalize_phone(str(billing.get("phone") or ""))
    for candidate in candidates:
        email = _normalize_email(candidate)
        if email and billing_email and hmac.compare_digest(email, billing_email):
            return True
        phone = _normalize_phone(candidate) if not email else None
        if phone and billing_phone and hmac.compare_digest(phone, billing_phone):
            return True
    return False


async def _get_order_status(
    args: Dict[str, Any],
    *,
    wc_client: Any,
    verification_contacts: Optional[Sequence[str]],
) -> str:
    order_id = _as_positive_int(args.get("order_id"))
    if order_id is None:
        return _ORDER_STATUS_UNVERIFIED

    # Raw contact values detected server-side in the current message (never
    # shown to the model), plus the model's argument when it is not a placeholder.
    candidates = [c for c in (verification_contacts or []) if isinstance(c, str) and c.strip()]
    from_model = args.get("email_or_phone")
    if isinstance(from_model, str) and from_model.strip() and not _PII_PLACEHOLDER_RE.search(from_model):
        candidates.append(from_model)
    if not candidates:
        logger.info("Order status lookup without contact information; not verified")
        return _ORDER_STATUS_UNVERIFIED

    try:
        order = await wc_client.get_order(order_id=order_id)
    except Exception as e:
        # Same neutral answer for "not found" and errors: no order-id enumeration.
        logger.info("Order status lookup failed (%s); not verified", type(e).__name__)
        return _ORDER_STATUS_UNVERIFIED

    billing = order.get("billing") if isinstance(order.get("billing"), dict) else {}
    if not order or not _contact_matches_billing(candidates, billing):
        logger.info("Order status lookup: contact does not match; not verified")
        return _ORDER_STATUS_UNVERIFIED

    logger.info("Order status lookup: verified")
    return json.dumps({
        "status": order.get("status"),
        "date_created": order.get("date_created"),
        "total": order.get("total"),
        "currency": order.get("currency"),
    })


def _order_mode() -> str:
    mode = (settings.WOOCOMMERCE_ORDER_MODE or "").strip().lower()
    return "direct" if mode == "direct" else "handoff"


def get_woocommerce_tools() -> list[dict]:
    order_tool = _CREATE_DRAFT_ORDER_TOOL if _order_mode() == "direct" else _REQUEST_ORDER_TOOL
    return [*_SEARCH_AND_PRICE_TOOLS, order_tool, _ORDER_STATUS_TOOL]


@dataclass(frozen=True)
class OrderRequest:
    """An order request forwarded to a human (handoff mode). Server-side only."""
    product_id: int
    product_name: str | None
    price: str | None
    quantity: int
    customer_note: str | None


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        n = int(value.strip())
        return n if n > 0 else None
    return None


async def execute_tool_call(
    function_name: str,
    args: Dict[str, Any],
    *,
    wc_client: Any,
    order_requests: Optional[List[OrderRequest]] = None,
    verification_contacts: Optional[Sequence[str]] = None,
) -> str:
    """Execute one tool requested by the model and return its JSON/text result.

    Never raises: errors are returned to the model as text.
    """
    if wc_client is None:
        return "(Error: WooCommerce client is not available for this store)"

    try:
        if function_name == "get_order_status":
            return await _get_order_status(args, wc_client=wc_client, verification_contacts=verification_contacts)

        if function_name == "search_products":
            res = await wc_client.search_products(query=args.get("query", ""))
            return json.dumps([{"id": p.id, "name": p.name, "price": p.price, "stock_status": p.stock_status} for p in res])

        if function_name == "get_price_and_stock":
            res = await wc_client.get_price_and_stock(product_id=args.get("product_id"))
            return json.dumps(res)

        if function_name == "create_draft_order" and _order_mode() == "direct":
            res = await wc_client.create_draft_order(
                line_items=[(args.get("product_id"), args.get("quantity"))],
                customer_note=args.get("customer_note")
            )
            return json.dumps({"order_id": res.id, "status": res.status, "total": res.total})

        if function_name == "request_order" and _order_mode() == "handoff":
            product_id = _as_positive_int(args.get("product_id"))
            quantity = _as_positive_int(args.get("quantity"))
            if product_id is None or quantity is None:
                return json.dumps({"status": "invalid_request", "message": "product_id and quantity must be positive integers."})

            # Read-only check that the product exists; no order is created.
            info = await wc_client.get_price_and_stock(product_id=product_id)
            if not info or info.get("id") is None:
                return json.dumps({"status": "product_not_found", "product_id": product_id})

            note = args.get("customer_note")
            request = OrderRequest(
                product_id=product_id,
                product_name=str(info.get("name")) if info.get("name") else None,
                price=str(info.get("price")) if info.get("price") not in (None, "") else None,
                quantity=quantity,
                customer_note=note.strip() if isinstance(note, str) and note.strip() else None,
            )
            if order_requests is not None:
                order_requests.append(request)
            return json.dumps({
                "status": "forwarded_to_advisor",
                "message": "The order request was forwarded to a human advisor, who will validate it and get back to the customer. No order has been created yet.",
                "product_name": request.product_name,
                "quantity": quantity,
            })

        return f"(Error: Unknown function {function_name})"
    except Exception as e:
        logger.warning("Tool %s failed (%s)", function_name, type(e).__name__)
        return f"Failed to execute {function_name}: {e}"


def _assistant_tool_call_message(message: Any) -> Dict[str, Any]:
    """Serialize the assistant's tool-call turn so it can be sent back in `messages`."""
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ],
    }


async def generate_grounded_reply(*, user_message: str, context: str, system_prompt: Optional[str] = None, store_context: Optional[StoreContext] = None, history: Optional[list[dict]] = None, language: Optional[str] = None, order_requests: Optional[List[OrderRequest]] = None, verification_contacts: Optional[Sequence[str]] = None) -> str:
    """Generate a reply grounded on retrieved context (basic RAG) and capable of calling WooCommerce tools.

    In WOOCOMMERCE_ORDER_MODE=handoff, every validated `request_order` call is
    appended to `order_requests` (when provided) so the caller can hand the
    conversation over to a human.

    `verification_contacts` are raw emails/phones detected in the current
    customer message. They are only used server-side by `get_order_status`
    and are never sent to the model.
    """
    merged_system = (system_prompt or _resolve_system_prompt(language)) + "\n\n" + _RAG_GROUNDING_RULES
    augmented_user = build_rag_user_prompt(question=user_message, context=context)

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in {"your_openai_api_key", "sk-..."}:
        logger.error("OPENAI_API_KEY is not configured.")
        return get_message("not_configured", language)

    client = _get_client()
    messages = [{"role": "system", "content": merged_system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": augmented_user})

    wc_client = get_woocommerce_client_for_store(store_context) if store_context else None
    tools = get_woocommerce_tools() if wc_client else None

    max_rounds = max(1, int(settings.LLM_MAX_TOOL_ROUNDS))

    async def complete(*, with_tools: bool):
        kwargs: Dict[str, Any] = {}
        if with_tools and tools:
            kwargs["tools"] = tools
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            temperature=0.2,
            timeout=settings.OPENAI_REQUEST_TIMEOUT_S,
            **kwargs,
        )
        return response.choices[0].message

    try:
        # Up to max_rounds model turns with tools exposed; stop as soon as the
        # model answers without requesting a tool.
        for _ in range(max_rounds):
            response_message = await complete(with_tools=True)
            if not response_message.tool_calls:
                return (response_message.content or "").strip()

            # We append the assistant's request to call a tool
            messages.append(_assistant_tool_call_message(response_message))

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}

                tool_result = await execute_tool_call(
                    function_name,
                    args,
                    wc_client=wc_client,
                    order_requests=order_requests,
                    verification_contacts=verification_contacts,
                )

                # Append tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                })

        # Tool budget exhausted: one last turn without tools forces a text answer.
        logger.info("LLM tool rounds exhausted (max=%s); requesting final answer without tools", max_rounds)
        final_message = await complete(with_tools=False)
        return (final_message.content or "").strip()

    except Exception:
        logger.exception("OpenAI grounded request failed")
        return get_message("generic_error", language)

    finally:
        if wc_client is not None:
            try:
                await wc_client.aclose()
            except Exception:
                logger.warning("Failed to close WooCommerce client")


# --- LLM self-assessment (confidence signal #2, see confidence_service.py) ---

_GROUNDING_SELF_ASSESSMENT_SYSTEM_PROMPT = """
You will be shown a CONTEXT, a USER QUESTION, and an ANSWER that was already generated.
Rate, from 0.0 to 1.0, how much the ANSWER relies on facts present in the CONTEXT
rather than on general knowledge not found in the CONTEXT.
Respond with ONLY a compact JSON object, no prose: {"grounding_confidence": <float between 0.0 and 1.0>}
""".strip()

# Keep this self-assessment call cheap and short: it must never meaningfully
# delay the reply already generated. Any failure/timeout below simply drops
# this signal (returns None) rather than raising.
_GROUNDING_SELF_ASSESSMENT_TIMEOUT_S = 6.0


async def estimate_context_grounding(*, user_message: str, context: str, answer: str) -> Optional[float]:
    """Ask the LLM to self-rate how grounded `answer` is in `context`.

    Fail-safe by design: returns None (never raises) on missing API key,
    timeout, malformed JSON, or an out-of-range value, so a missing signal
    simply degrades confidence_service.combine_confidence() gracefully.
    """
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in {"your_openai_api_key", "sk-..."}:
        return None

    try:
        client = _get_client()
        safe_context = context.strip() if context and context.strip() else "[no context found]"
        prompt = (
            f"CONTEXT:\n{safe_context}\n\n"
            f"USER QUESTION:\n{user_message.strip()}\n\n"
            f"ANSWER:\n{answer.strip()}"
        )
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _GROUNDING_SELF_ASSESSMENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=20,
            temperature=0.0,
            timeout=_GROUNDING_SELF_ASSESSMENT_TIMEOUT_S,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)
        value = float(data.get("grounding_confidence"))
        if not (0.0 <= value <= 1.0):
            logger.warning("LLM grounding self-assessment out of range (value=%s); dropping signal", value)
            return None
        return value
    except Exception:
        logger.warning("LLM grounding self-assessment failed; continuing without this signal", exc_info=True)
        return None
