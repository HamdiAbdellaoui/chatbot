"""LLM service for generating chatbot replies.

Phase 1 scope:
- No RAG
- Use OpenAI GPT-4o
- Enforce language mirroring via a system prompt
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI

from app.config import settings

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


async def generate_grounded_reply(*, user_message: str, context: str, system_prompt: Optional[str] = None) -> str:
    """Generate a reply grounded on retrieved context (basic RAG).

    This is still a simple prompt-based approach (no LlamaIndex).
    """
    merged_system = (system_prompt or _LANGUAGE_MIRRORING_SYSTEM_PROMPT) + "\n\n" + _RAG_GROUNDING_RULES
    augmented_user = build_rag_user_prompt(question=user_message, context=context)
    return await generate_reply(user_message=augmented_user, system_prompt=merged_system)
