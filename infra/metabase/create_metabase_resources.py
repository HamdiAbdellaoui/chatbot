"""Bootstrap Metabase resources: add Postgres DB, create SQL cards, and dashboard.

Usage (after Metabase is running at http://localhost:3001):

    python create_metabase_resources.py --url http://localhost:3001 --email admin@example.com --password metabase_admin_password

This script uses the Metabase REST API to:
- create an admin session (or use existing)
- register the Postgres database (if not already present)
- create three SQL cards (response latency, escalation rate, low-confidence flags)
- create a dashboard and add the cards

Note: This is best-effort automation. You can also import the SQL queries manually in Metabase UI.
"""

from __future__ import annotations

import argparse
import time
import json
import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

CARD_SQL_DIR = Path(__file__).parent / "sql"
DASHBOARD_JSON = Path(__file__).parent / "dashboard.json"


def login(base_url: str, email: str, password: str) -> str:
    r = requests.post(f"{base_url}/api/session", json={"username": email, "password": password})
    r.raise_for_status()
    return r.json()["id"]


def ensure_db(base_url: str, session: str, pg_config: dict[str, Any]) -> int:
    headers = {"X-Metabase-Session": session}
    # list databases
    r = requests.get(f"{base_url}/api/database", headers=headers)
    r.raise_for_status()
    for db in r.json():
        if db.get("details", {}).get("dbname") == pg_config["dbname"]:
            logger.info("Found existing Metabase DB id=%s", db["id"])
            return db["id"]

    payload = {
        "name": pg_config["name"],
        "engine": "postgres",
        "details": {
            "host": pg_config["host"],
            "port": pg_config["port"],
            "dbname": pg_config["dbname"],
            "user": pg_config["user"],
            "password": pg_config["password"],
        },
    }
    r = requests.post(f"{base_url}/api/database", headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["id"]


def create_card(base_url: str, session: str, db_id: int, name: str, query: str) -> int:
    headers = {"X-Metabase-Session": session}
    payload = {
        "name": name,
        "dataset_query": {
            "database": db_id,
            "native": {"query": query},
            "type": "native",
        },
        "display": "table",
    }
    r = requests.post(f"{base_url}/api/card", headers=headers, json=payload)
    r.raise_for_status()
    return r.json()["id"]


def create_dashboard(base_url: str, session: str, name: str, card_ids: list[int]) -> int:
    headers = {"X-Metabase-Session": session}
    payload = {"name": name}
    r = requests.post(f"{base_url}/api/dashboard", headers=headers, json=payload)
    r.raise_for_status()
    dash = r.json()
    # add cards to dashboard
    for idx, card_id in enumerate(card_ids):
        placement = {
            "cardId": card_id,
            "sizeX": 6,
            "sizeY": 4,
            "row": (idx // 2) * 4,
            "col": (idx % 2) * 6,
        }
        add_resp = requests.post(f"{base_url}/api/dashboard/{dash['id']}/cards", headers=headers, json=placement)
        add_resp.raise_for_status()
    return dash["id"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Metabase resources")
    parser.add_argument("--url", default="http://localhost:3001", help="Metabase base URL")
    parser.add_argument("--email", required=True, help="Metabase admin email")
    parser.add_argument("--password", required=True, help="Metabase admin password")
    parser.add_argument("--pg-host", default="postgres", help="Postgres host (compose service name)")
    parser.add_argument("--pg-port", type=int, default=5432, help="Postgres port")
    parser.add_argument("--pg-db", default="chatwoot_db", help="Postgres database name")
    parser.add_argument("--pg-user", default="chatwoot_user", help="Postgres username")
    parser.add_argument("--pg-pass", default=None, help="Postgres password")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    session = login(base_url, args.email, args.password)
    logger.info("Logged into Metabase session=%s", session)

    db_id = ensure_db(base_url, session, {"name": "chatwoot_postgres", "host": args.pg_host, "port": args.pg_port, "dbname": args.pg_db, "user": args.pg_user, "password": args.pg_pass or ""})
    logger.info("Using Metabase DB id=%s", db_id)

    # create cards from SQL files
    card_ids = []
    for sql_file in ["response_latency.sql", "escalation_rate.sql", "low_confidence_flags.sql"]:
        q = (CARD_SQL_DIR / sql_file).read_text(encoding="utf-8")
        cid = create_card(base_url, session, db_id, sql_file.replace(".sql", ""), q)
        logger.info("Created card id=%s for %s", cid, sql_file)
        card_ids.append(cid)

    dash_id = create_dashboard(base_url, session, "AI Operations Overview", card_ids)
    logger.info("Created dashboard id=%s", dash_id)

    print(json.dumps({"dashboard_id": dash_id, "card_ids": card_ids}))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
