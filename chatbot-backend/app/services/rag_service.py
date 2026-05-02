"""Basic RAG retrieval service (Qdrant only).

This step focuses on:
- retrieving top-k relevant snippets from Qdrant
- formatting a context block to inject into the GPT prompt

No LlamaIndex involved.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from app.config import settings
from app.services.qdrant_service import search as qdrant_search

logger = logging.getLogger(__name__)


def _format_context(hits: List[Dict[str, Any]], *, max_chars: int) -> str:
    """Format Qdrant search hits into a compact CONTEXT block."""
    parts: List[str] = []
    used = 0

    for idx, hit in enumerate(hits, start=1):
        payload = hit.get("payload") or {}
        text = payload.get("text") or ""
        if not isinstance(text, str):
            text = ""

        source = payload.get("source") or payload.get("doc_id") or payload.get("title") or "unknown"
        score = hit.get("score")

        snippet = text.strip().replace("\r\n", "\n")
        snippet = " ".join(snippet.split())  # collapse whitespace

        block = f"[{idx}] source={source} score={score:.4f}\n{snippet}" if isinstance(score, float) else f"[{idx}] source={source}\n{snippet}"

        if not block:
            continue

        # +2 for separator newlines
        if used + len(block) + 2 > max_chars:
            remaining = max(0, max_chars - used - 2)
            if remaining <= 0:
                break
            block = block[:remaining]

        parts.append(block)
        used += len(block) + 2

        if used >= max_chars:
            break

    return "\n\n".join(parts).strip()


async def retrieve_context(*, query: str, collection: str | None = None, limit: int | None = None) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve relevant context for a query from Qdrant.

    Returns:
      (context_text, raw_hits)

    Failure strategy:
    - If Qdrant/embeddings are unavailable, log and return empty context.
    """
    collection_name = collection or settings.QDRANT_COLLECTION
    top_k = limit if limit is not None else settings.RAG_TOP_K

    try:
        hits = await qdrant_search(collection=collection_name, query=query, limit=top_k)
        context = _format_context(hits, max_chars=settings.RAG_MAX_CONTEXT_CHARS)
        return context, hits
    except Exception:
        logger.exception("RAG retrieval failed (collection=%s)", collection_name)
        return "", []
