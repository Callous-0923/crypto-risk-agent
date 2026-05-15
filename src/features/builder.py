"""Feature builder — maintains rolling state, emits FeatureSnapshot every 30s.

各特征计算方法均基于滚动缓存的时间序列数据。
以下特征需要 trade/volume 数据源接入后才可填充：
  - volume_1m ~ volume_15m, quote_volume_* 系列
  - trade_count_1m, trade_count_z_15m
  - taker_buy_ratio_1m, taker_buy_ratio_5m
  - volume_z_15m, volume_z_60m
"""
from __future__ import annotations

import asyncio
import math
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
    """滚动缓存池，为每个资产维护价格/持仓/爆仓/费率时间序列。"""

    def __init__(self) -> None:
        self.trusted_prices: Deque[tuple[datetime, float]] = deque(maxlen=3600)
        self.oi_samples: Deque[tuple[datetime, float]] = deque(maxlen=120)
        self.liq_usd: Deque[tuple[datetime, float]] = deque(maxlen=600)
        self.funding_rates: Deque[float] = deque(maxlen=200)
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

    # ------------------------------------------------------------------
    # 基础工具方法
    # ------------------------------------------------------------------

    def _trusted_prices_in_window(self, seconds: int, *, now: datetime | None = None) -> list[float]:
        """获取指定时间窗口内的可信价格序列。"""
        now = now or datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - seconds
        return [price for ts, price in self.trusted_prices if ts.timestamp() >= cutoff]

    @staticmethod
    def _price_returns(prices: list[float]) -> list[float]:
        """从价格序列计算对数收益率序列。"""
        return [
            math.log(b / a)
            for a, b in zip(prices, prices[:-1])
            if a > 0 and b > 0
        ]

    @staticmethod
    def _stdev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return statistics.stdev(values)

    # ------------------------------------------------------------------
    # 收益率计算
    # ------------------------------------------------------------------

    def _ret(self, window_s: int, *, now: datetime | None = None) -> float:
        """滚动窗口简单收益率。"""
        prices = self._trusted_prices_in_window(window_s, now=now)
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0.0

    # ------------------------------------------------------------------
    # 已实现波动率
    # ------------------------------------------------------------------

    def _realized_vol(self, window_s: int, *, now: datetime | None = None) -> float:
        """基于对数收益率的年化已实现波动率近似值（非年化缩放）。
        单位：窗口内对数收益率的标准差。
        """
        prices = self._trusted_prices_in_window(window_s, now=now)
        if len(prices) < 5:
            return 0.0
        log_rets = self._price_returns(prices)
        return self._stdev(log_rets)

    # ------------------------------------------------------------------
    # 波动率 z-score
    # ------------------------------------------------------------------

    def _vol_z(self, window_s: int = 60, baseline_s: int = 300, *, now: datetime | None = None) -> float:
        """近期波动率相较基线波动率的标准化偏离程度。"""
        recent = self._trusted_prices_in_window(window_s, now=now)
        if len(recent) < 3:
            return 0.0
        baseline = self._trusted_prices_in_window(baseline_s, now=now)
        if len(baseline) < 10:
            return 0.0

        def _stdev_log_ret(prices: list[float]) -> float:
            rets = [
                math.log(b / a)
                for a, b in zip(prices, prices[1:])
                if a > 0 and b > 0
            ]
            return statistics.stdev(rets) if len(rets) >= 2 else 0.0

        recent_vol = _stdev_log_ret(recent)
        baseline_vol = _stdev_log_ret(baseline)
        if baseline_vol == 0:
            return 0.0
        return (recent_vol - baseline_vol) / (baseline_vol + 1e-9) * 2

    # ------------------------------------------------------------------
    # 价格形态特征
    # ------------------------------------------------------------------

    def _price_range_pct(self, window_s: int, *, now: datetime | None = None) -> float:
        """窗口内价格振幅 (high-low)/first。"""
        prices = self._trusted_prices_in_window(window_s, now=now)
        if len(prices) < 2 or prices[0] <= 0:
            return 0.0
        return (max(prices) - min(prices)) / prices[0]

    def _close_position(self, window_s: int, *, now: datetime | None = None) -> float:
        """收盘价在窗口内价格区间的相对位置 0~1。"""
        prices = self._trusted_prices_in_window(window_s, now=now)
        if len(prices) < 2:
            return 0.0
        low, high, close = min(prices), max(prices), prices[-1]
        if high == low:
            return 0.5
        return (close - low) / (high - low)

    # ------------------------------------------------------------------
    # 最大回撤 / 最大反弹
    # ------------------------------------------------------------------

    def _max_drawdown(self, window_s: int, *, now: datetime | None = None) -> float:
        """窗口内从峰值到谷底的最大回撤比例（正值）。"""
        prices = self._trusted_prices_in_window(window_s, now=now)
        if len(prices) < 3:
            return 0.0
        peak = prices[0]
        max_dd = 0.0
        for price in prices[1:]:
            if price > peak:
                peak = price
            dd = (peak - price) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _max_runup(self, window_s: int, *, now: datetime | None = None) -> float:
        """窗口内从谷底到峰值的最大反弹比例（正值）。"""
        prices = self._trusted_prices_in_window(window_s, now=now)
        if len(prices) < 3:
            return 0.0
        trough = prices[0]
        max_ru = 0.0
        for price in prices[1:]:
            if price < trough:
                trough = price
            ru = (price - trough) / trough if trough > 0 else 0.0
            if ru > max_ru:
                max_ru = ru
        return max_ru

    # ------------------------------------------------------------------
    # ATR_14 和波动率状态
    # ------------------------------------------------------------------

    def _atr_14(self, *, now: datetime | None = None) -> float:
        """基于价格快照序列估算的 14 周期平均真实波幅。
        使用相邻快照价格差的绝对值作为 true range 的近似。
        """
        prices = self._trusted_prices_in_window(600, now=now)
        if len(prices) < 15:
            return 0.0
        tr_values: list[float] = []
        for a, b in zip(prices, prices[1:]):
            if a > 0:
                tr_values.append(abs(b - a) / a)
        if not tr_values:
            return 0.0
        return sum(tr_values[-14:]) / min(14, len(tr_values))

    def _volatility_regime(self, *, now: datetime | None = None) -> float:
        """60 分钟内波动率对长期均值的比值，>1 表示高波。"""
        rv_short = self._realized_vol(300, now=now)
        rv_long = self._realized_vol(1800, now=now)
        if rv_long < 1e-9:
            return 1.0
        return rv_short / rv_long

    # ------------------------------------------------------------------
    # OI 变动
    # ------------------------------------------------------------------

    def _oi_delta_pct(self, window_s: int, *, now: datetime | None = None) -> float:
        """指定窗口内的 OI 变化百分比。"""
        now = now or datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - window_s
        relevant = [(ts, v) for ts, v in self.oi_samples if ts.timestamp() >= cutoff]
        if len(relevant) < 2:
            return 0.0
        old = relevant[0][1]
        new = relevant[-1][1]
        return (new - old) / old if old != 0 else 0.0

    def _oi_delta_15m_pct(self, *, now: datetime | None = None) -> float:
        return self._oi_delta_pct(900, now=now)

    # ------------------------------------------------------------------
    # 爆仓
    # ------------------------------------------------------------------

    def _liq_5m_usd(self, *, now: datetime | None = None) -> float:
        now = now or datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - 300
        return sum(v for ts, v in self.liq_usd if ts.timestamp() >= cutoff)

    # ------------------------------------------------------------------
    # 资金费率 & 期现基差
    # ------------------------------------------------------------------

    def _funding_z(self) -> float:
        """资金费率 z-score。"""
        rates = list(self.funding_rates)
        if len(rates) < 10:
            return 0.0
        mean = statistics.mean(rates)
        stdev_val = statistics.stdev(rates)
        if stdev_val == 0:
            return 0.0
        return (rates[-1] - mean) / stdev_val

    def _futures_basis_pct(self) -> float:
        """标记价格相对现货价格的基差百分比。"""
        if self.last_spot_price <= 0:
            return 0.0
        return (self.last_mark_price - self.last_spot_price) / self.last_spot_price

    def _basis_z(self) -> float:
        """基差的滚动 z-score。"""
        rates = list(self.funding_rates)
        if len(rates) < 10:
            return 0.0
        basis_series: list[float] = []
        # 以资金费率作为基差的代理变量
        for rate in rates:
            basis_series.append(rate)
        if len(basis_series) < 10:
            return 0.0
        mean = statistics.mean(basis_series)
        stdev_val = statistics.stdev(basis_series)
        if stdev_val == 0:
            return 0.0
        return (basis_series[-1] - mean) / stdev_val

    # ------------------------------------------------------------------
    # 数据质量
    # ------------------------------------------------------------------

    def _is_stale(self, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(tz=timezone.utc)
        age = (now - self.last_update).total_seconds()
        return age > 120

    # ------------------------------------------------------------------
    # 快照构建
    # ------------------------------------------------------------------

    def _snap_ret(self, window_s: int, now: datetime) -> float:
        return self._ret(window_s, now=now)

    def _snap_vol(self, window_s: int, now: datetime) -> float:
        return self._realized_vol(window_s, now=now)

    def build_snapshot(
        self,
        asset: Asset,
        *,
        trusted_price: float | None = None,
        conflict_detected: bool | None = None,
        as_of: datetime | None = None,
    ) -> FeatureSnapshot:
        """从滚动缓存构建当前时刻的特征快照。
        注：volume/trade_count/taker_buy 系列字段暂未实现数据源接入，
        值恒为 0.0，待接入后填充。
        """
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
            ret_1m=self._snap_ret(60, as_of),
            ret_5m=self._snap_ret(300, as_of),
            ret_15m=self._snap_ret(900, as_of),
            ret_30m=self._snap_ret(1800, as_of),
            ret_60m=self._snap_ret(3600, as_of),
            vol_z_1m=self._vol_z(60, 300, now=as_of),
            realized_vol_5m=self._snap_vol(300, as_of),
            realized_vol_15m=self._snap_vol(900, as_of),
            realized_vol_60m=self._snap_vol(3600, as_of),
            price_range_pct_1m=self._price_range_pct(60, now=as_of),
            close_position_1m=self._close_position(60, now=as_of),
            max_drawdown_15m=self._max_drawdown(900, now=as_of),
            max_drawdown_60m=self._max_drawdown(3600, now=as_of),
            max_runup_15m=self._max_runup(900, now=as_of),
            max_runup_60m=self._max_runup(3600, now=as_of),
            atr_14=self._atr_14(now=as_of),
            volatility_regime_60m=self._volatility_regime(now=as_of),
            oi_delta_15m_pct=self._oi_delta_15m_pct(now=as_of),
            oi_delta_5m_pct=self._oi_delta_pct(300, now=as_of),
            oi_delta_60m_pct=self._oi_delta_pct(3600, now=as_of),
            liq_5m_usd=self._liq_5m_usd(now=as_of),
            funding_z=self._funding_z(),
            futures_basis_pct=self._futures_basis_pct(),
            basis_z_60m=self._basis_z(),
            source_stale=self._is_stale(now=as_of),
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

    async def build_snapshot_validated(self, asset: Asset, *, as_of: datetime | None = None) -> FeatureSnapshot:
        from src.ingestion.validator import validate_sources

        state = self._states[asset]
        as_of = as_of or datetime.now(tz=timezone.utc)
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

    async def _predict_and_process(asset: Asset, snap: FeatureSnapshot) -> None:
        ml_prediction = None
        try:
            from src.persistence.repositories import get_recent_snapshots
            from src.ml.risk_model import predict_snapshot

            recent = await get_recent_snapshots(asset, n=12)
            history = [item for item in reversed(recent) if item.snapshot_id != snap.snapshot_id]
            ml_prediction = await predict_snapshot(snap, history=history, persist=True)
        except Exception as exc:
            logger.warning("Risk model prediction failed for %s: %s", asset.value, exc)
        await process_snapshot(snap, ml_prediction=ml_prediction)

    await asyncio.gather(*(_predict_and_process(asset, snap) for asset, snap in persisted_assets.items()))
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
