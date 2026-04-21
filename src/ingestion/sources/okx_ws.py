"""OKX WebSocket ingestion — spot + swap price, liquidations."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import websockets
from websockets.exceptions import ConnectionClosed

from src.core.logging import get_logger
from src.domain.models import Asset, RawEvent
from src.ingestion.normalizer import normalize_and_publish

logger = get_logger(__name__)

OKX_WS = "wss://ws.okx.com:8443/ws/v5/public"

ASSETS = [Asset.BTC, Asset.ETH, Asset.SOL]
INST_SPOT = {Asset.BTC: "BTC-USDT", Asset.ETH: "ETH-USDT", Asset.SOL: "SOL-USDT"}
INST_SWAP = {Asset.BTC: "BTC-USDT-SWAP", Asset.ETH: "ETH-USDT-SWAP", Asset.SOL: "SOL-USDT-SWAP"}

SYMBOL_TO_ASSET: dict[str, Asset] = {
    **{v: k for k, v in INST_SPOT.items()},
    **{v: k for k, v in INST_SWAP.items()},
}

SUBSCRIBE_MSG = {
    "op": "subscribe",
    "args": (
        [{"channel": "trades", "instId": v} for v in INST_SPOT.values()] +
        [{"channel": "mark-price", "instId": v} for v in INST_SWAP.values()] +
        [{"channel": "liquidation-orders", "instType": "SWAP"}]
    ),
}


def _parse(msg: dict) -> list[RawEvent]:
    events: list[RawEvent] = []
    channel = msg.get("arg", {}).get("channel", "")
    data_list = msg.get("data", [])

    if channel == "trades":
        inst_id = msg["arg"].get("instId", "")
        asset = SYMBOL_TO_ASSET.get(inst_id)
        if not asset:
            return []
        for d in data_list:
            events.append(RawEvent(
                asset=asset,
                source="okx_spot",
                event_type="price",
                event_ts=datetime.fromtimestamp(int(d["ts"]) / 1000, tz=timezone.utc),
                payload={"price": float(d["px"]), "qty": float(d["sz"])},
            ))

    elif channel == "mark-price":
        inst_id = msg["arg"].get("instId", "")
        asset = SYMBOL_TO_ASSET.get(inst_id)
        if not asset:
            return []
        for d in data_list:
            events.append(RawEvent(
                asset=asset,
                source="okx_swap",
                event_type="mark_price",
                event_ts=datetime.fromtimestamp(int(d["ts"]) / 1000, tz=timezone.utc),
                payload={"mark_price": float(d["markPx"])},
            ))

    elif channel == "liquidation-orders":
        for d in data_list:
            inst_id = d.get("instId", "")
            asset = SYMBOL_TO_ASSET.get(inst_id)
            if not asset:
                continue
            for detail in d.get("details", []):
                qty = float(detail.get("sz", 0))
                price = float(detail.get("bkPx", 0))
                events.append(RawEvent(
                    asset=asset,
                    source="okx_swap",
                    event_type="liquidation",
                    event_ts=datetime.fromtimestamp(int(detail.get("ts", 0)) / 1000, tz=timezone.utc),
                    payload={
                        "side": detail.get("side"),
                        "qty": qty,
                        "price": price,
                        "usd_value": qty * price,
                    },
                ))

    return events


async def run_okx_ws() -> None:
    from src.observability.metrics import ws_reconnect_total
    while True:
        try:
            async with websockets.connect(OKX_WS, ping_interval=20) as ws:
                await ws.send(json.dumps(SUBSCRIBE_MSG))
                logger.info("OKX WebSocket connected and subscribed")
                async for raw in ws:
                    msg = json.loads(raw)
                    if "data" not in msg:
                        continue
                    for event in _parse(msg):
                        await normalize_and_publish(event)
        except ConnectionClosed as e:
            logger.warning("OKX WS closed (%s), reconnecting in 3s", e)
            ws_reconnect_total.labels(source="okx").inc()
        except Exception as e:
            logger.error("OKX WS error: %s, reconnecting in 5s", e)
            ws_reconnect_total.labels(source="okx").inc()
            await asyncio.sleep(2)
        await asyncio.sleep(3)
