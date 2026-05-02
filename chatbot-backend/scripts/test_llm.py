"""Quick connectivity test for the OpenAI GPT integration (no webhook).

Usage (PowerShell, from chatbot-backend/):
  python scripts/test_llm.py --message "salam, chnowa a7سن batterie INGCO 20V?"

Notes:
- Reads OPENAI_API_KEY and other settings via app.config (loads .env)
- Prints the model response to stdout
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running this script directly: `python scripts/test_llm.py ...`
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.llm_service import generate_reply


async def _run(message: str) -> None:
    reply = await generate_reply(user_message=message)
    print(reply)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test GPT connectivity via generate_reply().")
    parser.add_argument("--message", required=True, help="Message to send to the model")
    args = parser.parse_args()

    asyncio.run(_run(args.message))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
