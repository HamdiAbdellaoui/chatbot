"""Minimal MCP client demo — spawns the server, lists tools, calls one.

Produces a capturable trace (proof of execution) for the report.

Usage (from chatbot-backend/):
    python mcp_server/demo_client.py
    python mcp_server/demo_client.py --query iphone --store-id store_a
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SERVER_SCRIPT = Path(__file__).resolve().parent / "woocommerce_mcp_server.py"


async def run_demo(*, query: str, store_id: str) -> None:
    # cwd=REPO_ROOT so the server loads chatbot-backend/.env (pydantic-settings
    # resolves env_file=".env" relative to the working directory).
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(REPO_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            print(f"=== Connected to MCP server: {init_result.serverInfo.name} ===\n")

            listed = await session.list_tools()
            print(f"--- Tools exposed ({len(listed.tools)}) ---")
            for tool in listed.tools:
                print(f"  * {tool.name}: {(tool.description or '').splitlines()[0]}")
                print(f"    input schema: {tool.inputSchema}")
            print()

            print(f"--- Calling search_products(query={query!r}, store_id={store_id!r}) ---")
            result = await session.call_tool(
                "search_products",
                {"query": query, "store_id": store_id},
            )

            print(f"isError: {result.isError}")
            for block in result.content:
                print(f"content: {block}")

            structured = getattr(result, "structuredContent", None)
            if structured is not None:
                print(f"structuredContent: {structured}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP demo client for the WooCommerce POC server.")
    parser.add_argument("--query", default="iphone", help="Product search keyword.")
    parser.add_argument("--store-id", default="store_a", help="Store key from STORES_JSON.")
    args = parser.parse_args()

    asyncio.run(run_demo(query=args.query, store_id=args.store_id))


if __name__ == "__main__":
    main()
