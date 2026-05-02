"""WooCommerce integration service (REST API client).

Goals (Phase 3 / Step 6):
- Search products
- Fetch product price and stock
- Create draft order (pending)

This is intentionally a thin wrapper over the WooCommerce REST API.
In later steps, you can expose these methods via an MCP tool layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from app.config import settings
from app.services.store_context_service import StoreContext

logger = logging.getLogger(__name__)


class WooCommerceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, details: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


@dataclass(frozen=True)
class WooProductSummary:
    id: int
    name: str
    permalink: str | None
    price: str | None
    stock_status: str | None
    stock_quantity: int | None


@dataclass(frozen=True)
class WooDraftOrderResult:
    id: int
    status: str
    total: str | None
    currency: str | None
    payment_url: str | None


class WooCommerceClient:
    def __init__(self, *, base_url: str, consumer_key: str, consumer_secret: str):
        if not base_url.lower().startswith("http"):
            raise ValueError("WooCommerce base_url must be http(s)")
        self._base_url = base_url.rstrip("/")
        self._auth = (consumer_key, consumer_secret)

        timeout = httpx.Timeout(connect=5.0, read=settings.WOOCOMMERCE_REQUEST_TIMEOUT_S, write=10.0, pool=5.0)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _url(self, path: str) -> str:
        # WooCommerce REST v3
        return f"{self._base_url}/wp-json/wc/v3{path}"

    async def _request(self, method: str, path: str, *, params: Dict[str, Any] | None = None, json_body: Any = None) -> Any:
        url = self._url(path)
        try:
            resp = await self._client.request(method, url, params=params, json=json_body, auth=self._auth)
        except httpx.RequestError as e:
            raise WooCommerceError("WooCommerce request failed", details=str(e)) from e

        if resp.status_code >= 400:
            # Avoid logging secrets; response text is safe to log.
            raise WooCommerceError(
                f"WooCommerce API error ({resp.status_code})",
                status_code=resp.status_code,
                details=resp.text[:4000],
            )

        # WooCommerce usually returns JSON
        if not resp.content:
            return None

        try:
            return resp.json()
        except Exception as e:
            raise WooCommerceError("WooCommerce returned non-JSON response", status_code=resp.status_code, details=resp.text[:4000]) from e

    async def search_products(self, *, query: str, limit: int = 5) -> List[WooProductSummary]:
        data = await self._request(
            "GET",
            "/products",
            params={"search": query, "per_page": max(1, min(limit, 20))},
        )

        out: List[WooProductSummary] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                pid = item.get("id")
                if not isinstance(pid, int):
                    continue
                out.append(
                    WooProductSummary(
                        id=pid,
                        name=str(item.get("name") or "").strip(),
                        permalink=str(item.get("permalink") or "").strip() or None,
                        price=str(item.get("price") or "").strip() or None,
                        stock_status=str(item.get("stock_status") or "").strip() or None,
                        stock_quantity=item.get("stock_quantity") if isinstance(item.get("stock_quantity"), int) else None,
                    )
                )
        return out

    async def get_product(self, *, product_id: int) -> Dict[str, Any]:
        data = await self._request("GET", f"/products/{product_id}")
        return data if isinstance(data, dict) else {}

    async def get_price_and_stock(self, *, product_id: int) -> Dict[str, Any]:
        p = await self.get_product(product_id=product_id)
        return {
            "id": p.get("id"),
            "name": p.get("name"),
            "permalink": p.get("permalink"),
            "price": p.get("price"),
            "regular_price": p.get("regular_price"),
            "sale_price": p.get("sale_price"),
            "currency": None,  # Woo product endpoint doesn’t reliably include currency
            "stock_status": p.get("stock_status"),
            "manage_stock": p.get("manage_stock"),
            "stock_quantity": p.get("stock_quantity"),
        }

    async def create_draft_order(
        self,
        *,
        line_items: Sequence[Tuple[int, int]],
        customer_note: str | None = None,
    ) -> WooDraftOrderResult:
        payload: Dict[str, Any] = {
            "status": "pending",
            "set_paid": False,
            "line_items": [{"product_id": pid, "quantity": qty} for pid, qty in line_items],
        }
        if customer_note:
            payload["customer_note"] = customer_note

        data = await self._request("POST", "/orders", json_body=payload)
        if not isinstance(data, dict) or not isinstance(data.get("id"), int):
            raise WooCommerceError("WooCommerce create order returned unexpected response")

        return WooDraftOrderResult(
            id=int(data["id"]),
            status=str(data.get("status") or "").strip(),
            total=str(data.get("total") or "").strip() or None,
            currency=str(data.get("currency") or "").strip() or None,
            payment_url=str(data.get("payment_url") or "").strip() or None,
        )


def get_woocommerce_client_for_store(store: StoreContext) -> Optional[WooCommerceClient]:
    """Factory for a WooCommerce client from a resolved StoreContext.

    Returns None if the store has no WooCommerce credentials configured.
    """

    creds = store.woocommerce
    if not creds:
        return None

    try:
        return WooCommerceClient(
            base_url=creds.base_url,
            consumer_key=creds.consumer_key,
            consumer_secret=creds.consumer_secret,
        )
    except Exception:
        logger.exception("Failed to initialize WooCommerce client for store=%s", store.key)
        return None
