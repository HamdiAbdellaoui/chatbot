"""Replay a captured Chatwoot webhook payload deterministically.

What it does:
- Reads a JSON payload file from disk
- Sends it to your backend webhook endpoint, authenticated like Chatwoot would:
  - token mode (default, Chatwoot v4.1.0): appends ?token=<secret> to the URL
  - signature mode (Chatwoot >= v4.13): sends X-Chatwoot-Timestamp and
    X-Chatwoot-Signature = "sha256=" + hex(HMAC-SHA256(secret, "{timestamp}." + body))
    computed over the *exact bytes* sent

Usage (PowerShell):
  # Token mode; uses CHATWOOT_WEBHOOK_TOKEN from environment if --token is omitted
  python scripts/replay_chatwoot_webhook.py --file .\\payload.json --url http://localhost:8000/api/v1/chatwoot-webhook

  # Token mode with an explicit token
  python scripts/replay_chatwoot_webhook.py --file .\\payload.json --token "your_token"

  # Signature mode; uses CHATWOOT_WEBHOOK_SECRET from environment if --secret is omitted
  python scripts/replay_chatwoot_webhook.py --file .\\payload.json --signature-mode --secret "your_secret"

  # Send without any authentication (only works with CHATWOOT_WEBHOOK_AUTH_MODE=none)
  python scripts/replay_chatwoot_webhook.py --file .\\payload.json --no-auth

Tips:
- If you captured the payload from logs, keep it as raw JSON text.
- This script signs and sends the file as-is (including whitespace/newlines).
- The token and signature are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx


def _compute_signature(body: bytes, secret: str, timestamp: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), f"{timestamp}.".encode("utf-8") + body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _with_token(url: str, token: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "token"]
    query.append(("token", token))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


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
        "--token",
        default="",
        help="Token mode: shared token. If omitted, reads CHATWOOT_WEBHOOK_TOKEN from environment.",
    )
    parser.add_argument(
        "--signature-mode",
        action="store_true",
        help="Use Chatwoot >= v4.13 signature headers instead of the URL token.",
    )
    parser.add_argument(
        "--secret",
        default="",
        help="Signature mode: HMAC secret. If omitted, reads CHATWOOT_WEBHOOK_SECRET from environment.",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="Signature mode: override X-Chatwoot-Timestamp (unix seconds; default: now).",
    )
    parser.add_argument(
        "--no-auth",
        "--no-signature",
        dest="no_auth",
        action="store_true",
        help="Send without token or signature (only accepted with CHATWOOT_WEBHOOK_AUTH_MODE=none)",
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
    url = args.url

    if args.no_auth:
        print("Sending without authentication (--no-auth).")
    elif args.signature_mode:
        secret = args.secret or os.environ.get("CHATWOOT_WEBHOOK_SECRET", "")
        if not secret:
            print(
                "ERROR: missing secret. Provide --secret or set CHATWOOT_WEBHOOK_SECRET in environment.",
                file=sys.stderr,
            )
            return 2
        timestamp = args.timestamp or str(int(time.time()))
        headers["X-Chatwoot-Timestamp"] = timestamp
        headers["X-Chatwoot-Signature"] = _compute_signature(body, secret, timestamp)
        print(f"Using signature mode timestamp={timestamp}")
    else:
        token = args.token or os.environ.get("CHATWOOT_WEBHOOK_TOKEN", "")
        if not token:
            print(
                "ERROR: missing token. Provide --token or set CHATWOOT_WEBHOOK_TOKEN in environment.",
                file=sys.stderr,
            )
            return 2
        url = _with_token(url, token)
        print("Using token mode (token appended to URL)")

    timeout = httpx.Timeout(args.timeout)
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, content=body, headers=headers)

    print(f"Response status={resp.status_code}")
    print(resp.text)

    return 0 if resp.status_code < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
