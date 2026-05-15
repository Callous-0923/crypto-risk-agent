"""Notification dispatcher — WebSocket broadcast + optional webhook."""
from __future__ import annotations

import asyncio
import json
from typing import Set

import httpx

from src.core.config import settings
from src.core.logging import get_logger
from src.core.proxy import get_httpx_client_kwargs
from src.domain.models import RiskAlert

logger = get_logger(__name__)

# Active WebSocket connections
_ws_clients: Set[asyncio.Queue] = set()


def register_ws_client() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _ws_clients.add(q)
    return q


def unregister_ws_client(q: asyncio.Queue) -> None:
    _ws_clients.discard(q)


async def _broadcast_ws(payload: dict) -> None:
    dead = set()
    for q in list(_ws_clients):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.add(q)
    for q in dead:
        _ws_clients.discard(q)


async def _send_webhook(alert: RiskAlert) -> None:
    if not settings.webhook_url:
        return
    payload = alert.model_dump(mode="json")
    try:
        async with httpx.AsyncClient(**get_httpx_client_kwargs(service="webhook", timeout=10)) as client:
            r = await client.post(settings.webhook_url, json=payload)
            r.raise_for_status()
            logger.info("Webhook sent for alert %s", alert.alert_id)
    except Exception as e:
        logger.error("Webhook failed for alert %s: %s", alert.alert_id, e)


async def dispatch_alert(alert: RiskAlert) -> None:
    payload = {
        "type": "alert",
        "alert_id": alert.alert_id,
        "case_id": alert.case_id,
        "severity": alert.severity.value,
        "title": alert.title,
        "body_zh": alert.body_zh,
        "idempotency_key": alert.idempotency_key,
        "created_at": alert.created_at.isoformat(),
    }
    await asyncio.gather(
        _broadcast_ws(payload),
        _send_webhook(alert),
        return_exceptions=True,
    )
    logger.info("Alert dispatched: %s [%s]", alert.title, alert.severity.value)
