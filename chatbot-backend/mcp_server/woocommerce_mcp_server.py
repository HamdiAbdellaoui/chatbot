"""WooCommerce MCP server — PROTOTYPE / POC (see README.md).

Exposes exactly 2 tools over the standard MCP stdio transport. This is NOT the
production path: in production the chatbot reaches WooCommerce through OpenAI
function calling (app/services/llm_service.py), which this file does not touch.

Run standalone (from chatbot-backend/):
    python mcp_server/woocommerce_mcp_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure `app` / `mcp_server` imports work when run as a script
# (same bootstrap pattern as eval/run_batch_eval.py).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_server import tools  # noqa: E402

mcp = FastMCP("woocommerce-poc")


@mcp.tool()
async def search_products(query: str, store_id: str) -> List[Dict[str, Any]]:
    """Search products by keyword in a given WooCommerce store.

    Args:
        query: Search keyword (e.g. "iphone", "chaussures").
        store_id: Store key as defined in the STORES_JSON config.
    """
    return await tools.search_products(query=query, store_id=store_id)


@mcp.tool()
async def get_price_and_stock(product_id: int, store_id: str) -> Dict[str, Any]:
    """Get current price and stock status for one product.

    Args:
        product_id: WooCommerce product id.
        store_id: Store key as defined in the STORES_JSON config.
    """
    return await tools.get_price_and_stock(product_id=product_id, store_id=store_id)


if __name__ == "__main__":
    # Default transport is stdio.
    mcp.run()
