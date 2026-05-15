"""Utilities for building candle data from persisted 30s snapshots."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.domain.models import Asset, FeatureSnapshot, MarketCandle
from src.persistence.repositories import get_recent_snapshots


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def aggregate_snapshots_to_candles(
    asset: Asset,
    snapshots: list[FeatureSnapshot],
    *,
    interval: str = "1m",
    limit: int = 60,
    now: datetime | None = None,
) -> list[MarketCandle]:
    if interval != "1m":
        raise ValueError(f"Unsupported interval: {interval}")

    ordered = sorted(
        (snap for snap in snapshots if snap.price > 0),
        key=lambda snap: snap.window_end,
    )
    buckets: dict[datetime, MarketCandle] = {}
    now = _as_utc(now or datetime.now(tz=timezone.utc))

    for snap in ordered:
        window_end = _as_utc(snap.window_end)
        bucket_start = window_end.replace(second=0, microsecond=0)
        bucket_end = bucket_start + timedelta(minutes=1)
        candle = buckets.get(bucket_start)
        if candle is None:
            candle = MarketCandle(
                asset=asset,
                interval=interval,
                open_time=bucket_start,
                close_time=bucket_end,
                open=snap.price,
                high=snap.price,
                low=snap.price,
                close=snap.price,
                snapshot_count=1,
                is_closed=now >= bucket_end,
            )
            buckets[bucket_start] = candle
            continue

        candle.high = max(candle.high, snap.price)
        candle.low = min(candle.low, snap.price)
        candle.close = snap.price
        candle.snapshot_count += 1
        candle.is_closed = now >= candle.close_time

    return list(buckets.values())[-limit:]


async def load_market_candles(
    asset: Asset,
    *,
    interval: str = "1m",
    limit: int = 60,
) -> list[MarketCandle]:
    snapshots_per_candle = 2 if interval == "1m" else 1
    snapshots = await get_recent_snapshots(asset, n=max(limit * snapshots_per_candle * 3, 60))
    return aggregate_snapshots_to_candles(asset, snapshots, interval=interval, limit=limit)
