"""OKX REST polling — OI + funding rate, every 60s."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from src.core.logging import get_logger
from src.domain.models import Asset, RawEvent
from src.ingestion.normalizer import normalize_and_publish

logger = get_logger(__name__)

BASE = "https://www.okx.com/api/v5/public"
INST_SWAP = {Asset.BTC: "BTC-USDT-SWAP", Asset.ETH: "ETH-USDT-SWAP", Asset.SOL: "SOL-USDT-SWAP"}


async def _fetch_oi(client: httpx.AsyncClient, asset: Asset) -> RawEvent | None:
    try:
        r = await client.get(
            f"{BASE}/open-interest",
            params={"instType": "SWAP", "instId": INST_SWAP[asset]},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json().get("data", [{}])[0]
        return RawEvent(
            asset=asset,
            source="okx_rest",
            event_type="open_interest",
            event_ts=datetime.fromtimestamp(int(data.get("ts", 0)) / 1000, tz=timezone.utc),
            payload={"oi_usd": float(data.get("oiUsd", 0))},
        )
    except Exception as e:
        logger.warning("OKX OI fetch error for %s: %s", asset, e)
        return None


async def _fetch_funding(client: httpx.AsyncClient, asset: Asset) -> RawEvent | None:
    try:
        r = await client.get(
            f"{BASE}/funding-rate",
            params={"instId": INST_SWAP[asset]},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json().get("data", [{}])[0]
        return RawEvent(
            asset=asset,
            source="okx_rest",
            event_type="funding_rate",
            event_ts=datetime.fromtimestamp(int(data.get("fundingTime", 0)) / 1000, tz=timezone.utc),
            payload={"funding_rate": float(data.get("fundingRate", 0))},
        )
    except Exception as e:
        logger.warning("OKX funding fetch error for %s: %s", asset, e)
        return None


async def run_okx_rest_poll(interval: int = 60) -> None:
    async with httpx.AsyncClient() as client:
        while True:
            for asset in Asset:
                oi = await _fetch_oi(client, asset)
                if oi:
                    await normalize_and_publish(oi)
                funding = await _fetch_funding(client, asset)
                if funding:
                    await normalize_and_publish(funding)
            await asyncio.sleep(interval)
