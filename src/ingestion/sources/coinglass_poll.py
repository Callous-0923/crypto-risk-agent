"""Binance Futures REST polling — OI + funding rate (replaces CoinGlass, no API key needed)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from src.core.logging import get_logger
from src.core.proxy import get_httpx_client_kwargs
from src.domain.models import Asset, RawEvent
from src.ingestion.normalizer import normalize_and_publish

logger = get_logger(__name__)

BASE = "https://fapi.binance.com/fapi/v1"
SYMBOL_MAP = {Asset.BTC: "BTCUSDT", Asset.ETH: "ETHUSDT", Asset.SOL: "SOLUSDT"}


async def _fetch_oi(client: httpx.AsyncClient, asset: Asset) -> RawEvent | None:
    try:
        r = await client.get(
            f"{BASE}/openInterest",
            params={"symbol": SYMBOL_MAP[asset]},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return RawEvent(
            asset=asset,
            source="binance_futures_rest",
            event_type="open_interest",
            event_ts=datetime.fromtimestamp(data["time"] / 1000, tz=timezone.utc),
            payload={"oi_usd": float(data["openInterest"])},
        )
    except Exception as e:
        logger.warning("Binance OI fetch error for %s: %s", asset, e)
        return None


async def _fetch_funding(client: httpx.AsyncClient, asset: Asset) -> RawEvent | None:
    try:
        r = await client.get(
            f"{BASE}/fundingRate",
            params={"symbol": SYMBOL_MAP[asset], "limit": 1},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        latest = data[0] if data else {}
        return RawEvent(
            asset=asset,
            source="binance_futures_rest",
            event_type="funding_rate",
            event_ts=datetime.fromtimestamp(latest.get("fundingTime", 0) / 1000, tz=timezone.utc),
            payload={"funding_rate": float(latest.get("fundingRate", 0))},
        )
    except Exception as e:
        logger.warning("Binance funding fetch error for %s: %s", asset, e)
        return None


async def run_coinglass_poll(interval: int = 60) -> None:
    async with httpx.AsyncClient(**get_httpx_client_kwargs(service="Binance futures REST")) as client:
        while True:
            for asset in Asset:
                oi = await _fetch_oi(client, asset)
                if oi:
                    await normalize_and_publish(oi)
                funding = await _fetch_funding(client, asset)
                if funding:
                    await normalize_and_publish(funding)
            await asyncio.sleep(interval)
