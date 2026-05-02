"""Test RAG + GPT end-to-end without Chatwoot.

Flow:
- Retrieve context from Qdrant for a query
- Inject context into prompt
- Generate a grounded reply

Usage (PowerShell, from chatbot-backend/):
  python scripts/test_rag_llm.py --query "chneya siyessa mtaa el retour?"

Pre-req:
- Qdrant running and sample docs inserted:
    python scripts/qdrant_demo.py --insert-samples
- OPENAI_API_KEY set in .env
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running this script directly: `python scripts/test_rag_llm.py ...`
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.rag_service import retrieve_context
from app.services.llm_service import generate_grounded_reply


async def main_async(query: str) -> int:
    context, hits = await retrieve_context(query=query)
    print(f"Retrieved hits: {len(hits)}")
    if context:
        print("\n--- CONTEXT ---")
        print(context)

    print("\n--- ANSWER ---")
    answer = await generate_grounded_reply(user_message=query, context=context)
    print(answer)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test RAG + GPT")
    parser.add_argument("--query", required=True, help="User query")
    args = parser.parse_args()

    return asyncio.run(main_async(args.query))


if __name__ == "__main__":
    raise SystemExit(main())
