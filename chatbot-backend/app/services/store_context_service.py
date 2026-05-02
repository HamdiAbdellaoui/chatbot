"""Multi-store context resolution.

Phase 3 goal:
- Resolve which store a Chatwoot message belongs to (based on inbox_id/inbox_name)
- Provide store-specific settings (Qdrant collection + WooCommerce credentials)

This keeps multi-store logic out of webhook and orchestration code.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WooCommerceCredentials:
    base_url: str
    consumer_key: str
    consumer_secret: str


@dataclass(frozen=True)
class StoreContext:
    key: str
    qdrant_collection: str
    woocommerce: Optional[WooCommerceCredentials]


def _norm_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _as_int_set(values: Any) -> set[int]:
    if not isinstance(values, list):
        return set()
    out: set[int] = set()
    for v in values:
        if isinstance(v, int):
            out.add(v)
        elif isinstance(v, str) and v.isdigit():
            out.add(int(v))
    return out


def _as_norm_str_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    out: set[str] = set()
    for v in values:
        if isinstance(v, str) and v.strip():
            out.add(_norm_name(v))
    return out


def _parse_woocommerce(data: Any) -> Optional[WooCommerceCredentials]:
    if not isinstance(data, dict):
        return None

    def resolve_value(v: Any) -> str:
        raw = str(v or "").strip()
        if raw.lower().startswith("env:"):
            env_name = raw.split(":", 1)[1].strip()
            return (os.getenv(env_name) or "").strip()
        return raw

    base_url = resolve_value(data.get("base_url"))
    consumer_key = resolve_value(data.get("consumer_key"))
    consumer_secret = resolve_value(data.get("consumer_secret"))

    if not base_url or not consumer_key or not consumer_secret:
        return None

    return WooCommerceCredentials(
        base_url=base_url,
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
    )


@lru_cache(maxsize=1)
def _load_store_mapping() -> Dict[str, Dict[str, Any]]:
    raw = (settings.STORES_JSON or "").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except Exception:
        logger.exception("Failed to parse STORES_JSON; ignoring multi-store mapping")
        return {}

    if not isinstance(parsed, dict):
        logger.error("STORES_JSON must be a JSON object of {store_key: store_config}")
        return {}

    return parsed


def _default_store() -> StoreContext:
    return StoreContext(
        key=settings.DEFAULT_STORE_KEY,
        qdrant_collection=settings.QDRANT_COLLECTION,
        woocommerce=None,
    )


def resolve_store(*, inbox_id: int | None, inbox_name: str | None) -> StoreContext:
    """Resolve store context from Chatwoot inbox info.

    Matching precedence:
    1) inbox_id match
    2) inbox_name match (case/whitespace-insensitive)
    3) default store

    Expected STORES_JSON shape (per store key):
      {
        "match": {"inbox_ids": [1,2], "inbox_names": ["Store A"]},
        "qdrant_collection": "store_a",
        "woocommerce": {"base_url": "...", "consumer_key": "...", "consumer_secret": "..."}
      }
    """

    mapping = _load_store_mapping()
    if not mapping:
        return _default_store()

    normalized_inbox_name = _norm_name(inbox_name) if isinstance(inbox_name, str) and inbox_name.strip() else None

    # Pre-build match tables for deterministic behavior.
    candidates: list[tuple[str, Dict[str, Any]]] = []
    for key, cfg in mapping.items():
        if not isinstance(cfg, dict):
            continue
        candidates.append((str(key), cfg))

    # 1) inbox_id match
    if isinstance(inbox_id, int):
        for key, cfg in candidates:
            match = cfg.get("match") if isinstance(cfg.get("match"), dict) else {}
            inbox_ids = _as_int_set(match.get("inbox_ids"))
            if inbox_id in inbox_ids:
                return _store_from_cfg(key, cfg)

    # 2) inbox_name match
    if normalized_inbox_name:
        for key, cfg in candidates:
            match = cfg.get("match") if isinstance(cfg.get("match"), dict) else {}
            inbox_names = _as_norm_str_set(match.get("inbox_names"))
            if normalized_inbox_name in inbox_names:
                return _store_from_cfg(key, cfg)

    return _default_store()


def _store_from_cfg(key: str, cfg: Dict[str, Any]) -> StoreContext:
    qdrant_collection = str(cfg.get("qdrant_collection") or "").strip() or settings.QDRANT_COLLECTION
    woocommerce = _parse_woocommerce(cfg.get("woocommerce"))
    return StoreContext(key=key, qdrant_collection=qdrant_collection, woocommerce=woocommerce)
