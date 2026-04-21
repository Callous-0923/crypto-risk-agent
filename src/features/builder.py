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
        self.prices: Deque[tuple[datetime, float]] = deque(maxlen=600)  # ~10min
        self.oi_samples: Deque[tuple[datetime, float]] = deque(maxlen=30)
        self.liq_usd: Deque[tuple[datetime, float]] = deque(maxlen=600)
        self.funding_rates: Deque[float] = deque(maxlen=100)
        self.last_mark_price: float = 0.0
        self.last_spot_price: float = 0.0
        self.last_update: datetime = datetime.now(tz=timezone.utc)

    def push_price(self, ts: datetime, price: float) -> None:
        self.prices.append((ts, price))
        self.last_spot_price = price
        self.last_update = ts

    def push_oi(self, ts: datetime, oi_usd: float) -> None:
        self.oi_samples.append((ts, oi_usd))

    def push_liq(self, ts: datetime, usd_value: float) -> None:
        self.liq_usd.append((ts, usd_value))

    def push_funding(self, rate: float) -> None:
        self.funding_rates.append(rate)

    def _prices_in_window(self, seconds: int) -> list[float]:
        now = datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - seconds
        return [p for ts, p in self.prices if ts.timestamp() >= cutoff]

    def _ret(self, window_s: int) -> float:
        prices = self._prices_in_window(window_s)
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0]

    def _vol_z(self, window_s: int = 60, baseline_s: int = 300) -> float:
        recent = self._prices_in_window(window_s)
        if len(recent) < 3:
            return 0.0
        baseline = self._prices_in_window(baseline_s)
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

    def _cross_source_conflict(self) -> bool:
        if self.last_spot_price == 0 or self.last_mark_price == 0:
            return False
        diff = abs(self.last_spot_price - self.last_mark_price) / self.last_spot_price
        return diff > 0.005  # 0.5% divergence

    def build_snapshot(self, asset: Asset) -> FeatureSnapshot:
        return FeatureSnapshot(
            asset=asset,
            price=self.last_spot_price,
            ret_1m=self._ret(60),
            ret_5m=self._ret(300),
            vol_z_1m=self._vol_z(60, 300),
            oi_delta_15m_pct=self._oi_delta_15m_pct(),
            liq_5m_usd=self._liq_5m_usd(),
            funding_z=self._funding_z(),
            source_stale=self._is_stale(),
            cross_source_conflict=self._cross_source_conflict(),
        )


class FeatureBuilder:
    def __init__(self) -> None:
        self._states: dict[Asset, _AssetState] = defaultdict(_AssetState)

    def ingest(self, event: RawEvent) -> None:
        state = self._states[event.asset]
        p = event.payload

        if event.event_type == "price":
            state.push_price(event.event_ts, float(p.get("price", 0)))
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


# Singleton
_builder = FeatureBuilder()


def get_feature_builder() -> FeatureBuilder:
    return _builder


async def run_feature_builder() -> None:
    """Consume event bus, build features, emit snapshots periodically."""
    bus = get_event_bus()
    builder = get_feature_builder()

    async def _snapshot_loop() -> None:
        while True:
            await asyncio.sleep(SNAPSHOT_INTERVAL)
            for asset in Asset:
                snap = builder.get_snapshot(asset)
                try:
                    await save_feature_snapshot(snap)
                except Exception as e:
                    logger.warning("Failed to save snapshot for %s: %s", asset, e)

    asyncio.create_task(_snapshot_loop())

    while True:
        event: RawEvent = await bus.get()
        builder.ingest(event)
        bus.task_done()
