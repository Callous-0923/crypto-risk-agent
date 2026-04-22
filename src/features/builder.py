"""Feature builder — maintains rolling state, emits FeatureSnapshot every 30s."""
from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque

from src.core.logging import get_logger
from src.domain.models import Asset, FeatureSnapshot, RawEvent
from src.ingestion.normalizer import get_event_bus
from src.persistence.repositories import save_feature_snapshot

logger = get_logger(__name__)

SNAPSHOT_INTERVAL = 30  # seconds


class _AssetState:
    """Rolling buffers for one asset."""

    def __init__(self) -> None:
        self.trusted_prices: Deque[tuple[datetime, float]] = deque(maxlen=600)  # validated snapshot series
        self.oi_samples: Deque[tuple[datetime, float]] = deque(maxlen=30)
        self.liq_usd: Deque[tuple[datetime, float]] = deque(maxlen=600)
        self.funding_rates: Deque[float] = deque(maxlen=100)
        self.last_mark_price: float = 0.0
        self.last_spot_price: float = 0.0
        self.last_trusted_price: float = 0.0
        self.binance_price: float = 0.0
        self.okx_price: float = 0.0
        self.last_update: datetime = datetime.now(tz=timezone.utc)

    def push_price(self, ts: datetime, price: float, *, source: str = "") -> None:
        self.last_spot_price = price
        if source.startswith("binance"):
            self.binance_price = price
        elif source.startswith("okx"):
            self.okx_price = price
        self.last_update = ts

    def record_trusted_price(self, ts: datetime, price: float) -> None:
        self.trusted_prices.append((ts, price))
        self.last_trusted_price = price

    def push_oi(self, ts: datetime, oi_usd: float) -> None:
        self.oi_samples.append((ts, oi_usd))

    def push_liq(self, ts: datetime, usd_value: float) -> None:
        self.liq_usd.append((ts, usd_value))

    def push_funding(self, rate: float) -> None:
        self.funding_rates.append(rate)

    def _trusted_prices_in_window(self, seconds: int, *, now: datetime | None = None) -> list[float]:
        now = now or datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - seconds
        return [price for ts, price in self.trusted_prices if ts.timestamp() >= cutoff]

    def _ret(self, window_s: int, *, now: datetime | None = None) -> float:
        prices = self._trusted_prices_in_window(window_s, now=now)
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0]

    def _vol_z(self, window_s: int = 60, baseline_s: int = 300, *, now: datetime | None = None) -> float:
        recent = self._trusted_prices_in_window(window_s, now=now)
        if len(recent) < 3:
            return 0.0
        baseline = self._trusted_prices_in_window(baseline_s, now=now)
        if len(baseline) < 10:
            return 0.0

        def stdev_ret(prices: list[float]) -> float:
            rets = [(b - a) / a for a, b in zip(prices, prices[1:]) if a != 0]
            return statistics.stdev(rets) if len(rets) >= 2 else 0.0

        recent_vol = stdev_ret(recent)
        baseline_vol = stdev_ret(baseline)
        if baseline_vol == 0:
            return 0.0
        baseline_mean = 0.0
        # simplified z-score: (recent - mean) / stdev using sample stdev of baseline windows
        return (recent_vol - baseline_vol) / (baseline_vol + 1e-9) * 2

    def _oi_delta_15m_pct(self) -> float:
        now = datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - 900  # 15min
        relevant = [(ts, v) for ts, v in self.oi_samples if ts.timestamp() >= cutoff]
        if len(relevant) < 2:
            return 0.0
        old = relevant[0][1]
        new = relevant[-1][1]
        return (new - old) / old if old != 0 else 0.0

    def _liq_5m_usd(self) -> float:
        now = datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - 300
        return sum(v for ts, v in self.liq_usd if ts.timestamp() >= cutoff)

    def _funding_z(self) -> float:
        rates = list(self.funding_rates)
        if len(rates) < 10:
            return 0.0
        mean = statistics.mean(rates)
        stdev = statistics.stdev(rates)
        if stdev == 0:
            return 0.0
        return (rates[-1] - mean) / stdev

    def _is_stale(self) -> bool:
        age = (datetime.now(tz=timezone.utc) - self.last_update).total_seconds()
        return age > 120  # stale after 2 min

    def build_snapshot(
        self,
        asset: Asset,
        *,
        trusted_price: float | None = None,
        conflict_detected: bool | None = None,
        as_of: datetime | None = None,
    ) -> FeatureSnapshot:
        as_of = as_of or datetime.now(tz=timezone.utc)
        effective_price = (
            self.last_trusted_price
            if trusted_price is None and self.last_trusted_price > 0
            else self.last_spot_price if trusted_price is None
            else trusted_price
        )
        return FeatureSnapshot(
            asset=asset,
            window_end=as_of,
            price=effective_price,
            ret_1m=self._ret(60, now=as_of),
            ret_5m=self._ret(300, now=as_of),
            vol_z_1m=self._vol_z(60, 300, now=as_of),
            oi_delta_15m_pct=self._oi_delta_15m_pct(),
            liq_5m_usd=self._liq_5m_usd(),
            funding_z=self._funding_z(),
            source_stale=self._is_stale(),
            cross_source_conflict=False if conflict_detected is None else conflict_detected,
        )


class FeatureBuilder:
    def __init__(self) -> None:
        self._states: dict[Asset, _AssetState] = defaultdict(_AssetState)

    def ingest(self, event: RawEvent) -> None:
        state = self._states[event.asset]
        p = event.payload

        if event.event_type == "price":
            state.push_price(event.event_ts, float(p.get("price", 0)), source=event.source)
        elif event.event_type == "mark_price":
            state.last_mark_price = float(p.get("mark_price", 0))
            fr = float(p.get("funding_rate", 0))
            if fr != 0:
                state.push_funding(fr)
        elif event.event_type == "open_interest":
            state.push_oi(event.event_ts, float(p.get("oi_usd", 0)))
        elif event.event_type == "liquidation":
            state.push_liq(event.event_ts, float(p.get("usd_value", 0)))
        elif event.event_type == "funding_rate":
            state.push_funding(float(p.get("funding_rate", 0)))

    def get_snapshot(self, asset: Asset) -> FeatureSnapshot:
        return self._states[asset].build_snapshot(asset)

    async def build_snapshot_validated(self, asset: Asset) -> FeatureSnapshot:
        from src.ingestion.validator import validate_sources

        state = self._states[asset]
        as_of = datetime.now(tz=timezone.utc)
        validation = await validate_sources(asset, state.binance_price, state.okx_price)
        if validation.conflict_detected:
            logger.warning(
                "Cross-source conflict detected for %s: %s",
                asset.value,
                validation.resolution_reason,
            )
        if validation.trusted_price > 0:
            state.record_trusted_price(as_of, validation.trusted_price)
        return state.build_snapshot(
            asset,
            trusted_price=validation.trusted_price,
            conflict_detected=validation.conflict_detected,
            as_of=as_of,
        )


# Singleton
_builder = FeatureBuilder()


def get_feature_builder() -> FeatureBuilder:
    return _builder


async def run_snapshot_cycle(builder: FeatureBuilder | None = None) -> dict[Asset, FeatureSnapshot]:
    builder = builder or get_feature_builder()
    if hasattr(builder, "build_snapshot_validated"):
        built = await asyncio.gather(*(builder.build_snapshot_validated(asset) for asset in Asset))
    else:
        built = [builder.get_snapshot(asset) for asset in Asset]
    snaps = {asset: snap for asset, snap in zip(Asset, built)}

    persist_results = await asyncio.gather(
        *(save_feature_snapshot(snap) for snap in snaps.values()),
        return_exceptions=True,
    )
    persisted_assets: dict[Asset, FeatureSnapshot] = {}
    for asset, result in zip(snaps.keys(), persist_results):
        if isinstance(result, Exception):
            logger.warning("Failed to save snapshot for %s: %s", asset.value, result)
            continue
        persisted_assets[asset] = snaps[asset]

    if not persisted_assets:
        return {}

    from src.graph.coordinator import coordinate_cross_asset
    from src.graph.orchestrator import process_snapshot

    await asyncio.gather(*(process_snapshot(snap) for snap in persisted_assets.values()))
    await coordinate_cross_asset(persisted_assets)
    return persisted_assets


async def run_feature_builder() -> None:
    """Consume event bus, build features, emit snapshots periodically."""
    bus = get_event_bus()
    builder = get_feature_builder()

    async def _snapshot_loop() -> None:
        while True:
            await asyncio.sleep(SNAPSHOT_INTERVAL)
            await asyncio.shield(run_snapshot_cycle(builder))

    snapshot_task = asyncio.create_task(_snapshot_loop())

    try:
        while True:
            event: RawEvent = await bus.get()
            builder.ingest(event)
            bus.task_done()
    finally:
        snapshot_task.cancel()
        await asyncio.gather(snapshot_task, return_exceptions=True)
