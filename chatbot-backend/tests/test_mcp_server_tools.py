"""Wiring tests for the MCP server tools (POC).

Goal: verify the MCP tools delegate to the existing WooCommerce client and
resolve stores through store_context_service — not to retest WooCommerce itself.
The `mcp` SDK is intentionally not imported here (mcp_server.tools has no
dependency on it), so these tests run without the optional POC dependency.
"""

import asyncio

import pytest

from app.services.store_context_service import StoreContext
from app.services.woocommerce_service import WooProductSummary
from mcp_server import tools

_FAKE_MAPPING = {
    "store_a": {
        "qdrant_collection": "store_a",
        "woocommerce": {
            "base_url": "https://store-a.tld",
            "consumer_key": "ck_test",
            "consumer_secret": "cs_test",
        },
    }
}


class _FakeWooClient:
    def __init__(self):
        self.search_calls: list[dict] = []
        self.price_calls: list[dict] = []
        self.closed = False

    async def search_products(self, *, query: str, limit: int = 5):
        self.search_calls.append({"query": query, "limit": limit})
        return [
            WooProductSummary(
                id=42,
                name="iPhone 15",
                permalink="https://store-a.tld/p/42",
                price="2999",
                stock_status="instock",
                stock_quantity=7,
            )
        ]

    async def get_price_and_stock(self, *, product_id: int):
        self.price_calls.append({"product_id": product_id})
        return {"id": product_id, "name": "iPhone 15", "price": "2999", "stock_status": "instock"}

    async def aclose(self):
        self.closed = True


def _use_fake_store_and_client(monkeypatch):
    """Wire a fake STORES_JSON mapping + fake WooCommerce client."""
    fake_client = _FakeWooClient()
    received_stores: list[StoreContext] = []

    def fake_get_client(store: StoreContext):
        received_stores.append(store)
        return fake_client

    monkeypatch.setattr(tools, "_load_store_mapping", lambda: _FAKE_MAPPING)
    monkeypatch.setattr(tools, "get_woocommerce_client_for_store", fake_get_client)
    return fake_client, received_stores


def test_search_products_delegates_to_woocommerce_client(monkeypatch):
    fake_client, received_stores = _use_fake_store_and_client(monkeypatch)

    result = asyncio.run(tools.search_products(query="iphone", store_id="store_a"))

    # Delegated to the existing client with the right query.
    assert fake_client.search_calls == [{"query": "iphone", "limit": 5}]
    # Dataclass results converted to plain dicts for MCP transport.
    assert result == [
        {
            "id": 42,
            "name": "iPhone 15",
            "permalink": "https://store-a.tld/p/42",
            "price": "2999",
            "stock_status": "instock",
            "stock_quantity": 7,
        }
    ]
    assert fake_client.closed is True


def test_get_price_and_stock_delegates_to_woocommerce_client(monkeypatch):
    fake_client, _ = _use_fake_store_and_client(monkeypatch)

    result = asyncio.run(tools.get_price_and_stock(product_id=42, store_id="store_a"))

    assert fake_client.price_calls == [{"product_id": 42}]
    assert result["id"] == 42
    assert fake_client.closed is True


def test_store_credentials_are_resolved_via_store_context_service(monkeypatch):
    _, received_stores = _use_fake_store_and_client(monkeypatch)

    asyncio.run(tools.search_products(query="iphone", store_id="store_a"))

    # The StoreContext handed to the client factory was built by
    # store_context_service from the STORES_JSON mapping, not hand-rolled here.
    assert len(received_stores) == 1
    store = received_stores[0]
    assert store.key == "store_a"
    assert store.qdrant_collection == "store_a"
    assert store.woocommerce is not None
    assert store.woocommerce.base_url == "https://store-a.tld"
    assert store.woocommerce.consumer_key == "ck_test"


def test_unknown_store_id_raises_clear_error(monkeypatch):
    monkeypatch.setattr(tools, "_load_store_mapping", lambda: _FAKE_MAPPING)

    with pytest.raises(ValueError, match="Unknown store_id"):
        asyncio.run(tools.search_products(query="iphone", store_id="nope"))


def test_store_without_woocommerce_credentials_raises_clear_error(monkeypatch):
    monkeypatch.setattr(tools, "_load_store_mapping", lambda: {"store_b": {"qdrant_collection": "store_b"}})
    monkeypatch.setattr(tools, "get_woocommerce_client_for_store", lambda store: None)

    with pytest.raises(ValueError, match="no WooCommerce credentials"):
        asyncio.run(tools.get_price_and_stock(product_id=1, store_id="store_b"))
