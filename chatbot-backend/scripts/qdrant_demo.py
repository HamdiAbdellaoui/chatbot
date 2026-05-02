"""Minimal Qdrant retrieval demo (no LlamaIndex).

What it does:
- Optionally inserts a few sample documents into Qdrant
- Runs a search query and prints the top matches

Usage (PowerShell, from chatbot-backend/):
  # 1) Insert samples into the default collection (QDRANT_COLLECTION, default: shared)
  python scripts/qdrant_demo.py --insert-samples

  # 2) Query
  python scripts/qdrant_demo.py --query "return policy" --limit 3

Requirements:
- Qdrant reachable at QDRANT_URL (default http://localhost:6333)
- Embeddings provider configured:
  - Default: EMBEDDINGS_PROVIDER=openai and OPENAI_API_KEY set
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running this script directly: `python scripts/qdrant_demo.py ...`
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.qdrant_service import Document, search, upsert_documents


SAMPLE_DOCS = [
    Document(
        id="policy_return",
        text="Return policy: Customers can return products within 14 days with the receipt. Items must be unused.",
        metadata={"source": "policy", "lang": "en"},
    ),
    Document(
        id="policy_shipping",
        text="Shipping policy: Delivery usually takes 24-72 hours in Tunisia depending on the region.",
        metadata={"source": "policy", "lang": "en"},
    ),
    Document(
        id="darija_return",
        text="Siyessa mtaa el retour: tnajem ترجع السلعة fi 14 youm ken mazel ma t7alletch w 3andek facture.",
        metadata={"source": "policy", "lang": "mix"},
    ),
]


async def main_async(args: argparse.Namespace) -> int:
    collection = args.collection or settings.QDRANT_COLLECTION

    if args.insert_samples:
        await upsert_documents(collection=collection, documents=SAMPLE_DOCS)

    if args.query:
        hits = await search(collection=collection, query=args.query, limit=args.limit)
        print(f"Collection: {collection}")
        print(f"Query: {args.query}")
        print("\nTop hits:")
        for h in hits:
            payload = h.get("payload") or {}
            text = payload.get("text", "")
            print(f"- id={h['id']} score={h['score']:.4f} source={payload.get('source')} text={text[:120]}")

    if not args.insert_samples and not args.query:
        print("Nothing to do. Use --insert-samples and/or --query")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal Qdrant demo")
    parser.add_argument("--collection", default="", help="Qdrant collection (default: QDRANT_COLLECTION)")
    parser.add_argument("--insert-samples", action="store_true", help="Insert sample documents")
    parser.add_argument("--query", default="", help="Query text")
    parser.add_argument("--limit", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
