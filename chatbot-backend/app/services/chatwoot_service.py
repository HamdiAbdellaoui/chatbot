"""Chatwoot API wrapper.

Used for:
- Sending bot replies
- Escalating to a human (add label, optionally assign)
- Checking whether a conversation is already escalated (labels)

We keep this in a service so both the webhook route and orchestration layer can
reuse it without duplicating HTTP logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class ChatwootError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, details: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


def _base_url() -> str:
    return settings.CHATWOOT_BASE_URL.rstrip("/")


def _headers() -> Dict[str, str]:
    if not settings.CHATWOOT_API_TOKEN:
        raise ChatwootError("CHATWOOT_API_TOKEN is not configured")
    return {
        "api_access_token": settings.CHATWOOT_API_TOKEN,
        "Content-Type": "application/json",
    }


def _timeout() -> httpx.Timeout:
    # Keep connect short to avoid hanging webhooks.
    t = float(settings.CHATWOOT_REQUEST_TIMEOUT_S)
    return httpx.Timeout(connect=min(3.0, t), read=t, write=t, pool=min(3.0, t))


async def send_message(*, account_id: int, conversation_id: int, content: str) -> None:
    url = f"{_base_url()}/api/v1/accounts/{account_id}/conversations/{conversation_id}/messages"
    payload = {"content": content}

    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            resp = await client.post(url, headers=_headers(), json=payload)
        except httpx.RequestError as e:
            raise ChatwootError("Failed to send message to Chatwoot", details=str(e)) from e

        if resp.status_code >= 400:
            raise ChatwootError("Chatwoot send message failed", status_code=resp.status_code, details=resp.text[:4000])


async def get_conversation(*, account_id: int, conversation_id: int) -> Dict[str, Any]:
    url = f"{_base_url()}/api/v1/accounts/{account_id}/conversations/{conversation_id}"
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            resp = await client.get(url, headers=_headers())
        except httpx.RequestError as e:
            raise ChatwootError("Failed to fetch conversation", details=str(e)) from e

        if resp.status_code >= 400:
            raise ChatwootError("Chatwoot get conversation failed", status_code=resp.status_code, details=resp.text[:4000])

        try:
            data = resp.json()
        except Exception as e:
            raise ChatwootError("Chatwoot get conversation returned non-JSON", status_code=resp.status_code, details=resp.text[:4000]) from e

        return data if isinstance(data, dict) else {}


def _extract_labels(conversation_payload: Dict[str, Any]) -> set[str]:
    # Chatwoot payload shape varies by endpoint/version.
    for key in ("labels", "conversation_labels"):
        val = conversation_payload.get(key)
        if isinstance(val, list):
            out = set()
            for x in val:
                if isinstance(x, str) and x.strip():
                    out.add(x.strip())
                elif isinstance(x, dict) and isinstance(x.get("title"), str):
                    out.add(x["title"].strip())
            if out:
                return out

    # Some responses nest conversation under "payload" or "data"
    for container_key in ("payload", "data", "conversation"):
        nested = conversation_payload.get(container_key)
        if isinstance(nested, dict):
            nested_labels = nested.get("labels")
            if isinstance(nested_labels, list):
                return {str(x).strip() for x in nested_labels if str(x).strip()}

    return set()


async def get_conversation_labels(*, account_id: int, conversation_id: int) -> set[str]:
    data = await get_conversation(account_id=account_id, conversation_id=conversation_id)
    return _extract_labels(data)


async def add_labels(*, account_id: int, conversation_id: int, labels: Sequence[str]) -> None:
    # Endpoint name differs across versions; this is the common one.
    url = f"{_base_url()}/api/v1/accounts/{account_id}/conversations/{conversation_id}/labels"
    payload = {"labels": [l for l in labels if isinstance(l, str) and l.strip()]}

    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            resp = await client.post(url, headers=_headers(), json=payload)
        except httpx.RequestError as e:
            raise ChatwootError("Failed to add labels", details=str(e)) from e

        if resp.status_code == 404:
            # Try PUT fallback (some versions use PUT)
            try:
                resp = await client.put(url, headers=_headers(), json=payload)
            except httpx.RequestError as e:
                raise ChatwootError("Failed to add labels (PUT fallback)", details=str(e)) from e

        if resp.status_code >= 400:
            raise ChatwootError("Chatwoot add labels failed", status_code=resp.status_code, details=resp.text[:4000])


async def assign_conversation(*, account_id: int, conversation_id: int, assignee_id: int) -> None:
    # Common endpoint
    url = f"{_base_url()}/api/v1/accounts/{account_id}/conversations/{conversation_id}/assignments"
    payload = {"assignee_id": assignee_id}

    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            resp = await client.post(url, headers=_headers(), json=payload)
        except httpx.RequestError as e:
            raise ChatwootError("Failed to assign conversation", details=str(e)) from e

        if resp.status_code >= 400:
            raise ChatwootError("Chatwoot assign conversation failed", status_code=resp.status_code, details=resp.text[:4000])


async def update_conversation(*, account_id: int, conversation_id: int, data: Dict[str, Any]) -> None:
    url = f"{_base_url()}/api/v1/accounts/{account_id}/conversations/{conversation_id}"
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            resp = await client.put(url, headers=_headers(), json=data)
        except httpx.RequestError as e:
            raise ChatwootError("Failed to update conversation", details=str(e)) from e

        if resp.status_code >= 400:
            raise ChatwootError("Chatwoot update conversation failed", status_code=resp.status_code, details=resp.text[:4000])


async def assign_team(*, account_id: int, conversation_id: int, team_id: int) -> None:
    """Best-effort team assignment.

    Chatwoot API shapes differ across versions; we try common approaches.
    """

    # Attempt #1: assignments endpoint (some installs may accept team_id)
    url = f"{_base_url()}/api/v1/accounts/{account_id}/conversations/{conversation_id}/assignments"
    payload = {"team_id": team_id}
    async with httpx.AsyncClient(timeout=_timeout()) as client:
        try:
            resp = await client.post(url, headers=_headers(), json=payload)
            if resp.status_code < 400:
                return
        except httpx.RequestError:
            # Fall through to update API.
            pass

    # Attempt #2: update conversation
    await update_conversation(account_id=account_id, conversation_id=conversation_id, data={"team_id": team_id})


async def escalate_conversation(
    *,
    account_id: int,
    conversation_id: int,
    labels: Sequence[str],
    assignee_id: int | None,
    team_id: int | None,
) -> None:
    # Label must always be applied for persistent state.
    await add_labels(account_id=account_id, conversation_id=conversation_id, labels=labels)

    # Hybrid strategy: assign to agent or team if configured.
    if assignee_id is not None:
        await assign_conversation(account_id=account_id, conversation_id=conversation_id, assignee_id=assignee_id)
        return

    if team_id is not None:
        await assign_team(account_id=account_id, conversation_id=conversation_id, team_id=team_id)
