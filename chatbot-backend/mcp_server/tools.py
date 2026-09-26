"""Tool implementations backing the MCP server.

Kept separate from woocommerce_mcp_server.py on purpose: this module has no
dependency on the `mcp` SDK, so it stays importable (and unit-testable) even
when the optional POC dependency is not installed.

All WooCommerce access goes through the existing production client
(app/services/woocommerce_service.py) — nothing is reimplemented here.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List

from app.services.store_context_service import (
    StoreContext,
    _load_store_mapping,
    _store_from_cfg,
)
from app.services.woocommerce_service import get_woocommerce_client_for_store

logger = logging.getLogger(__name__)


def resolve_store_by_id(store_id: str) -> StoreContext:
    """Resolve a StoreContext from a STORES_JSON store key.

    store_context_service exposes resolve_store(inbox_id/inbox_name) because
    production resolves stores from Chatwoot inbox metadata. MCP tools address
    stores by key instead, so this composes the same module's mapping loader
    and config parser rather than duplicating them. It intentionally reuses the
    private helpers so this POC does not have to modify production code; a real
    migration would promote a public resolve_store_by_key() there instead.
    """
    mapping = _load_store_mapping()
    cfg = mapping.get(store_id)
    if not isinstance(cfg, dict):
        known = sorted(mapping) or ["(none — STORES_JSON is empty)"]
        raise ValueError(f"Unknown store_id {store_id!r}. Configured stores: {known}")
    return _store_from_cfg(store_id, cfg)


def _client_for_store(store_id: str):
    store = resolve_store_by_id(store_id)
    client = get_woocommerce_client_for_store(store)
    if client is None:
        raise ValueError(f"Store {store_id!r} has no WooCommerce credentials configured")
    return client


async def search_products(*, query: str, store_id: str) -> List[Dict[str, Any]]:
    """Search products in the WooCommerce store identified by store_id."""
    client = _client_for_store(store_id)
    try:
        products = await client.search_products(query=query)
        return [asdict(p) for p in products]
    finally:
        await client.aclose()


async def get_price_and_stock(*, product_id: int, store_id: str) -> Dict[str, Any]:
    """Get price and stock for one product in the store identified by store_id."""
    client = _client_for_store(store_id)
    try:
        return await client.get_price_and_stock(product_id=product_id)
    finally:
        await client.aclose()
