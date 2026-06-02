"""Admin endpoints for Active Learning flags.

Protected by X-Admin-Token header.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query

from app.config import settings
from app.services.active_learning_service import get_flag, list_flags

logger = logging.getLogger(__name__)
router = APIRouter()


def _require_admin_token(x_admin_token: Optional[str]) -> None:
    expected = (settings.ADMIN_API_TOKEN or "").strip()
    if not expected:
        # Fail closed: if token is not configured, admin endpoints are disabled.
        raise HTTPException(status_code=503, detail="Admin endpoints are not configured")

    provided = (x_admin_token or "").strip()
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/active-learning/flags")
async def admin_list_flags(
    *,
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Dict[str, Any]:
    _require_admin_token(x_admin_token)
    items = await list_flags(limit=limit, offset=offset)
    return {"items": items, "limit": limit, "offset": offset}


@router.get("/active-learning/flags/{flag_id}")
async def admin_get_flag(
    *,
    flag_id: int,
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> Dict[str, Any]:
    _require_admin_token(x_admin_token)
    item = await get_flag(flag_id=flag_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Not found")
    return item
