from __future__ import annotations

import asyncio
import csv
import io
import urllib.error
import urllib.request
import zipfile
from datetime import datetime

from src.domain.models import Asset, HistoricalMarketBar
from src.persistence.repositories import save_historical_market_bars

BINANCE_PUBLIC_BASE_URL = "https://data.binance.vision/data"
ASSET_SYMBOLS = {
    Asset.BTC: "BTCUSDT",
    Asset.ETH: "ETHUSDT",
    Asset.SOL: "SOLUSDT",
}


def _month_start(value: datetime) -> datetime:
    return datetime(value.year, value.month, 1)


def _next_month(value: datetime) -> datetime:
    if value.month == 12:
        return datetime(value.year + 1, 1, 1)
    return datetime(value.year, value.month + 1, 1)


def _iter_months(start: datetime, end: datetime):
    current = _month_start(start)
    while current < end:
        yield current
        current = _next_month(current)


def _timestamp_to_datetime(value: str) -> datetime:
    raw = int(float(value))
    if raw > 10**15:
        return datetime.utcfromtimestamp(raw / 1_000_000)
    return datetime.utcfromtimestamp(raw / 1_000)


def _market_path(market_type: str) -> str:
    if market_type == "spot":
        return "spot"
    if market_type in {"futures_um", "um"}:
        return "futures/um"
    raise ValueError(f"unsupported market_type: {market_type}")


def binance_monthly_kline_url(
    *,
    asset: Asset,
    month: datetime,
    interval: str = "1m",
    market_type: str = "spot",
) -> str:
    symbol = ASSET_SYMBOLS[asset]
    return (
        f"{BINANCE_PUBLIC_BASE_URL}/{_market_path(market_type)}/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{month:%Y-%m}.zip"
    )


import time as _time


def _download_csv_rows(url: str, *, retries: int = 3, backoff: float = 5.0) -> list[list[str]]:
    """下载 Binance 月度 K 线 zip，带自动重试。"""
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
                if not csv_names:
                    return []
                with archive.open(csv_names[0]) as file:
                    text = io.TextIOWrapper(file, encoding="utf-8")
                    return [row for row in csv.reader(text) if row]
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []
            last_err = exc
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as exc:
            last_err = exc
        if attempt < retries - 1:
            _time.sleep(backoff * (attempt + 1))

    if last_err:
        raise last_err
    return []


def _parse_kline_rows(
    rows: list[list[str]],
    *,
    asset: Asset,
    market_type: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> list[HistoricalMarketBar]:
    symbol = ASSET_SYMBOLS[asset]
    bars: list[HistoricalMarketBar] = []
    for row in rows:
        if len(row) < 11 or not row[0].replace(".", "", 1).isdigit():
            continue
        open_time = _timestamp_to_datetime(row[0])
        if open_time < start or open_time >= end:
            continue
        bars.append(HistoricalMarketBar(
            source="binance_public",
            market_type=market_type,
            asset=asset,
            symbol=symbol,
            interval=interval,
            open_time=open_time,
            close_time=_timestamp_to_datetime(row[6]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[7]),
            trade_count=int(float(row[8])),
            taker_buy_base_volume=float(row[9]),
            taker_buy_quote_volume=float(row[10]),
        ))
    return bars


async def backfill_binance_public_klines(
    *,
    assets: list[Asset],
    start: datetime,
    end: datetime,
    interval: str = "1m",
    market_types: list[str] | None = None,
) -> dict:
    market_types = market_types or ["spot", "futures_um"]
    results: list[dict] = []
    total_saved = 0
    total_months = len(assets) * len(market_types) * len(list(_iter_months(start, end)))
    month_idx = 0
    for asset in assets:
        for market_type in market_types:
            for month in _iter_months(start, end):
                month_idx += 1
                url = binance_monthly_kline_url(
                    asset=asset, month=month, interval=interval, market_type=market_type,
                )
                status = f"[{month_idx}/{total_months}] {asset.value}/{market_type} {month:%Y-%m}"
                try:
                    rows = await asyncio.to_thread(_download_csv_rows, url)
                    bars = _parse_kline_rows(
                        rows, asset=asset, market_type=market_type,
                        interval=interval, start=start, end=end,
                    )
                    saved = await save_historical_market_bars(bars)
                    total_saved += saved
                    print(f"  {status} : {saved:>8,} bars saved ({len(rows):>8,} raw)")
                except Exception as exc:
                    print(f"  {status} : FAILED ({exc})")
                    rows, saved = 0, 0
                results.append({
                    "asset": asset.value,
                    "market_type": market_type,
                    "month": f"{month:%Y-%m}",
                    "rows": len(rows),
                    "saved": saved,
                    "url": url,
                })
    return {
        "source": "binance_public",
        "interval": interval,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_saved": total_saved,
        "files": results,
    }
