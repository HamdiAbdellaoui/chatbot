"""LLM service for generating chatbot replies.

Phase 1 scope:
- No RAG
- Use OpenAI GPT-4o
- Enforce language mirroring via a system prompt
"""

from __future__ import annotations

import logging
import json
from typing import Optional

from openai import AsyncOpenAI

from app.config import settings
from app.services.store_context_service import StoreContext
from app.services.woocommerce_service import get_woocommerce_client_for_store, WooCommerceError

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


async def generate_reply(*, user_message: str, system_prompt: Optional[str] = None) -> str:
    """Generate a reply using OpenAI Chat Completions.

    Error strategy:
    - Log the exception details server-side
    - Return a safe, user-friendly fallback string to the caller
    """
    if not user_message.strip():
        return "I didn't receive any text. Could you please type your question?"

    # Fail fast on common misconfiguration to avoid confusing behavior during development.
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in {"your_openai_api_key", "sk-..."}:
        logger.error("OPENAI_API_KEY is not configured (or is still a placeholder).")
        return "Sorry, the assistant is not configured yet. Please try again later."

    prompt = system_prompt or _LANGUAGE_MIRRORING_SYSTEM_PROMPT

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
        return "Sorry, I'm having trouble answering right now. Please try again in a moment."

_WOOCOMMERCE_TOOLS = [
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
    {
        "type": "function",
        "function": {
            "name": "create_draft_order",
            "description": "Creates a draft order in WooCommerce with the given line items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "The product ID."},
                    "quantity": {"type": "integer", "description": "Quantity to order."},
                    "customer_note": {"type": "string", "description": "Optional notes from the customer."}
                },
                "required": ["product_id", "quantity"],
            },
        },
    }
]

async def generate_grounded_reply(*, user_message: str, context: str, system_prompt: Optional[str] = None, store_context: Optional[StoreContext] = None, history: Optional[list[dict]] = None) -> str:
    """Generate a reply grounded on retrieved context (basic RAG) and capable of calling WooCommerce tools."""
    merged_system = (system_prompt or _LANGUAGE_MIRRORING_SYSTEM_PROMPT) + "\n\n" + _RAG_GROUNDING_RULES
    augmented_user = build_rag_user_prompt(question=user_message, context=context)

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY in {"your_openai_api_key", "sk-..."}:
        logger.error("OPENAI_API_KEY is not configured.")
        return "Sorry, the assistant is not configured yet. Please try again later."

    client = _get_client()
    messages = [{"role": "system", "content": merged_system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": augmented_user})

    wc_client = get_woocommerce_client_for_store(store_context) if store_context else None
    tools = _WOOCOMMERCE_TOOLS if wc_client else None

    try:
        
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            temperature=0.2,
            timeout=settings.OPENAI_REQUEST_TIMEOUT_S,
            tools=tools,
        )

        response_message = response.choices[0].message
        
        if response_message.tool_calls:
            # We append the assistant's request to call a tool
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                tool_result = "(Error: Unable to parse tool arguments)"
                if wc_client:
                    try:
                        if function_name == "search_products":
                            res = await wc_client.search_products(query=args.get("query", ""))
                            # Format to JSON string
                            tool_result = json.dumps([{"id": p.id, "name": p.name, "price": p.price, "stock_status": p.stock_status} for p in res])
                        elif function_name == "get_price_and_stock":
                            res = await wc_client.get_price_and_stock(product_id=args.get("product_id"))
                            tool_result = json.dumps(res)
                        elif function_name == "create_draft_order":
                            res = await wc_client.create_draft_order(
                                line_items=[(args.get("product_id"), args.get("quantity"))],
                                customer_note=args.get("customer_note")
                            )
                            tool_result = json.dumps({"order_id": res.id, "status": res.status, "total": res.total})
                        else:
                            tool_result = f"(Error: Unknown function {function_name})"
                    except Exception as e:
                        tool_result = f"Failed to execute {function_name}: {e}"
                else:
                    tool_result = "(Error: WooCommerce client is not available for this store)"

                # Append tool response
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                })
            
            # Send second request with tool results
            second_response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=0.2,
                timeout=settings.OPENAI_REQUEST_TIMEOUT_S,
            )
            return (second_response.choices[0].message.content or "").strip()
        else:
            return (response_message.content or "").strip()

    except Exception:
        logger.exception("OpenAI grounded request failed")
        return "Sorry, I'm having trouble answering right now. Please try again in a moment."

    finally:
        if wc_client is not None:
            try:
                await wc_client.aclose()
            except Exception:
                logger.warning("Failed to close WooCommerce client")
