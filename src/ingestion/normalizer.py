"""Normalizer: assign dedupe_key and publish to in-process event bus."""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone

from src.core.logging import get_logger
from src.domain.models import RawEvent
from src.persistence.repositories import save_raw_event

logger = get_logger(__name__)

# In-process async queue — consumers subscribe to this
_event_bus: asyncio.Queue[RawEvent] = asyncio.Queue(maxsize=10_000)


def get_event_bus() -> asyncio.Queue[RawEvent]:
    return _event_bus


def _make_dedupe_key(event: RawEvent) -> str:
    window = int(event.event_ts.timestamp() // 30)
    raw = f"{event.asset}:{event.source}:{event.event_type}:{window}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def normalize_and_publish(event: RawEvent) -> None:
    from src.observability.metrics import (
        ingest_event_total, ingest_lag_seconds, event_bus_queue_size,
    )

    event.dedupe_key = _make_dedupe_key(event)

    # 指标：事件入库数 + 采集延迟
    ingest_event_total.labels(
        asset=event.asset.value,
        source=event.source,
        event_type=event.event_type,
    ).inc()

    now_ts = datetime.now(tz=timezone.utc).timestamp()
    event_ts = event.event_ts.timestamp() if event.event_ts.tzinfo else event.event_ts.timestamp()
    lag = max(0.0, now_ts - event_ts)
    ingest_lag_seconds.labels(source=event.source).observe(lag)

    try:
        await save_raw_event(event)
    except Exception as e:
        logger.warning("Failed to persist raw event %s: %s", event.event_id, e)

    try:
        _event_bus.put_nowait(event)
        event_bus_queue_size.set(_event_bus.qsize())
    except asyncio.QueueFull:
        logger.warning("Event bus full, dropping event %s", event.event_id)
        event_bus_queue_size.set(_event_bus.maxsize)
