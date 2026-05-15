"""Binance spot + futures WebSocket ingestion."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import websockets
from websockets.exceptions import ConnectionClosed

from src.core.logging import get_logger
from src.core.proxy import get_ws_proxy
from src.domain.models import Asset, RawEvent
from src.ingestion.normalizer import normalize_and_publish

logger = get_logger(__name__)

ASSETS = [Asset.BTC, Asset.ETH, Asset.SOL]

SPOT_STREAMS = [
    f"{a.value.lower()}usdt@aggTrade"
    for a in ASSETS
] + [
    f"{a.value.lower()}usdt@bookTicker"
    for a in ASSETS
]

FUTURES_STREAMS = [
    f"{a.value.lower()}usdt@markPrice"
    for a in ASSETS
] + [
    f"{a.value.lower()}usdt@forceOrder"
    for a in ASSETS
]

SPOT_WS_BASE = "wss://data-stream.binance.vision/stream?streams="
FUTURES_WS_BASE = "wss://fstream.binance.com/stream?streams="


def _asset_from_symbol(symbol: str) -> Asset | None:
    for a in ASSETS:
        if symbol.upper().startswith(a.value):
            return a
    return None


def _parse_spot(data: dict) -> RawEvent | None:
    stream: str = data.get("stream", "")
    msg: dict = data.get("data", {})
    event_type = msg.get("e", "")
    symbol: str = msg.get("s", "")
    asset = _asset_from_symbol(symbol)
    if asset is None:
        return None

    if event_type == "aggTrade":
        return RawEvent(
            asset=asset,
            source="binance_spot",
            event_type="price",
            event_ts=datetime.fromtimestamp(msg["T"] / 1000, tz=timezone.utc),
            payload={"price": float(msg["p"]), "qty": float(msg["q"])},
        )
    if event_type == "bookTicker":
        return RawEvent(
            asset=asset,
            source="binance_spot",
            event_type="order_book",
            event_ts=datetime.utcnow(),
            payload={
                "bid": float(msg["b"]),
                "ask": float(msg["a"]),
                "spread_bps": (float(msg["a"]) - float(msg["b"])) / float(msg["b"]) * 10000,
            },
        )
    return None


def _parse_futures(data: dict) -> RawEvent | None:
    stream: str = data.get("stream", "")
    msg: dict = data.get("data", {})
    event_type = msg.get("e", "")

    if event_type == "markPriceUpdate":
        symbol = msg.get("s", "")
        asset = _asset_from_symbol(symbol)
        if asset is None:
            return None
        return RawEvent(
            asset=asset,
            source="binance_futures",
            event_type="mark_price",
            event_ts=datetime.fromtimestamp(msg["T"] / 1000, tz=timezone.utc),
            payload={
                "mark_price": float(msg["p"]),
                "funding_rate": float(msg.get("r", 0)),
                "next_funding_ts": msg.get("T", 0),
            },
        )

    if event_type == "forceOrder":
        order = msg.get("o", {})
        symbol = order.get("s", "")
        asset = _asset_from_symbol(symbol)
        if asset is None:
            return None
        qty = float(order.get("q", 0))
        price = float(order.get("p", 0))
        return RawEvent(
            asset=asset,
            source="binance_futures",
            event_type="liquidation",
            event_ts=datetime.fromtimestamp(order.get("T", 0) / 1000, tz=timezone.utc),
            payload={
                "side": order.get("S"),
                "qty": qty,
                "price": price,
                "usd_value": qty * price,
            },
        )
    return None


async def _run_stream(url: str, parser, label: str) -> None:
    proxy = get_ws_proxy()
    connect_kwargs: dict = {"ping_interval": 20}
    if proxy:
        connect_kwargs["proxy"] = proxy
        logger.info("Using proxy %s for %s", proxy, label)

    while True:
        try:
            async with websockets.connect(url, **connect_kwargs) as ws:
                logger.info("Connected to %s", label)
                async for raw in ws:
                    data = json.loads(raw)
                    event = parser(data)
                    if event:
                        await normalize_and_publish(event)
        except ConnectionClosed as e:
            logger.warning("%s closed (%s), reconnecting in 3s", label, e)
        except Exception as e:
            logger.error("%s error: %s, reconnecting in 5s", label, e)
            await asyncio.sleep(2)
        await asyncio.sleep(3)


async def run_binance_spot() -> None:
    url = SPOT_WS_BASE + "/".join(SPOT_STREAMS)
    await _run_stream(url, _parse_spot, "binance_spot")


async def run_binance_futures() -> None:
    url = FUTURES_WS_BASE + "/".join(FUTURES_STREAMS)
    await _run_stream(url, _parse_futures, "binance_futures")
