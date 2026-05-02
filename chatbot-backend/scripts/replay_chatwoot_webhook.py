"""Replay a captured Chatwoot webhook payload deterministically.

What it does:
- Reads a JSON payload file from disk
- Sends it to your backend webhook endpoint
- Computes the HMAC signature over the *exact bytes* sent

Usage (PowerShell):
  # Uses CHATWOOT_WEBHOOK_SECRET from environment if --secret is omitted
  python scripts/replay_chatwoot_webhook.py --file .\\payload.json --url http://localhost:8000/api/v1/chatwoot-webhook

  # Provide secret explicitly
  python scripts/replay_chatwoot_webhook.py --file .\\payload.json --secret "your_secret" --url http://localhost:8000/api/v1/chatwoot-webhook

  # Send without signature (only works if backend validation disabled)
  python scripts/replay_chatwoot_webhook.py --file .\\payload.json --no-signature

Tips:
- If you captured the payload from logs, keep it as raw JSON text.
- This script signs and sends the file as-is (including whitespace/newlines).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
from typing import Any

import httpx


def _compute_signature(body: bytes, secret: str, fmt: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    if fmt == "hex":
        return digest.hex()
    if fmt == "base64":
        return base64.b64encode(digest).decode("ascii")
    raise ValueError(f"Unsupported signature format: {fmt}")


def _load_body_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _validate_json(body: bytes) -> Any:
    # Validates JSON and returns the decoded object (used only for optional prints)
    text = body.decode("utf-8")
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a Chatwoot webhook payload to the backend.")
    parser.add_argument("--file", required=True, help="Path to JSON payload file")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/v1/chatwoot-webhook",
        help="Webhook URL (default: http://localhost:8000/api/v1/chatwoot-webhook)",
    )
    parser.add_argument(
        "--secret",
        default="",
        help="Webhook secret. If omitted, reads CHATWOOT_WEBHOOK_SECRET from environment.",
    )
    parser.add_argument(
        "--signature-format",
        choices=["hex", "base64"],
        default="hex",
        help="Signature header encoding to use (default: hex)",
    )
    parser.add_argument(
        "--no-signature",
        action="store_true",
        help="Do not send the X-Chatwoot-Hmac-SHA256 header (for debugging only)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Request timeout in seconds (default: 20)",
    )
    args = parser.parse_args()

    body = _load_body_bytes(args.file)

    # Optional: validate JSON so we fail fast on malformed capture
    try:
        payload = _validate_json(body)
    except Exception as exc:
        print(f"ERROR: payload is not valid UTF-8 JSON: {exc}", file=sys.stderr)
        return 2

    # Small helpful debug print (without dumping entire payload)
    event_type = payload.get("event") if isinstance(payload, dict) else None
    print(f"Loaded payload bytes={len(body)} event={event_type!r}")

    headers = {"Content-Type": "application/json"}

    if not args.no_signature:
        secret = args.secret or os.environ.get("CHATWOOT_WEBHOOK_SECRET", "")
        if not secret:
            print(
                "ERROR: missing secret. Provide --secret or set CHATWOOT_WEBHOOK_SECRET in environment.",
                file=sys.stderr,
            )
            return 2

        signature = _compute_signature(body, secret, args.signature_format)
        headers["X-Chatwoot-Hmac-SHA256"] = signature
        print(f"Using signature_format={args.signature_format} header_len={len(signature)}")
    else:
        print("Sending without signature header (--no-signature).")

    timeout = httpx.Timeout(args.timeout)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(args.url, content=body, headers=headers)

    print(f"Response status={resp.status_code}")
    print(resp.text)

    return 0 if resp.status_code < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
