from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from statistics import mean, pstdev

from src.domain.models import Asset, FeatureSnapshot, HistoricalMarketBar
from src.persistence.repositories import list_historical_market_bars


def _safe_ret(current: float, previous: float) -> float:
    return current / previous - 1.0 if previous > 0 else 0.0


def _sum(values: list[float]) -> float:
    return float(sum(values)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _zscore(value: float, baseline: list[float]) -> float:
    sigma = _std(baseline)
    if sigma <= 1e-12:
        return 0.0
    return (value - _mean(baseline)) / sigma


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _drawdown_from_peak(current: float, peak: float) -> float:
    return max(0.0, (peak - current) / peak) if peak > 0 else 0.0


def _stable_snapshot_id(asset: Asset, window_end: datetime) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"historical-snapshot:{asset.value}:{window_end.isoformat()}"))


def _true_range(bar: HistoricalMarketBar, previous_close: float) -> float:
    return max(
        bar.high - bar.low,
        abs(bar.high - previous_close),
        abs(bar.low - previous_close),
    )


def _window(items: list[HistoricalMarketBar], end_index: int, size: int) -> list[HistoricalMarketBar]:
    return items[max(0, end_index - size + 1):end_index + 1]


def _percentile_rank(value: float, baseline: list[float]) -> float:
    if not baseline:
        return 0.0
    return sum(1 for item in baseline if item <= value) / len(baseline)


def build_snapshots_from_bars(
    *,
    asset: Asset,
    spot_bars: list[HistoricalMarketBar],
    futures_bars: list[HistoricalMarketBar] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[FeatureSnapshot]:
    ordered = sorted(spot_bars, key=lambda item: item.open_time)
    futures_by_time = {bar.open_time: bar for bar in sorted(futures_bars or [], key=lambda item: item.open_time)}
    closes = [bar.close for bar in ordered]
    one_min_returns = [
        0.0 if index == 0 else _safe_ret(ordered[index].close, ordered[index - 1].close)
        for index in range(len(ordered))
    ]
    realized_vol_15_history: list[float] = []
    basis_history: list[float] = []
    snapshots: list[FeatureSnapshot] = []
    total_bars = len(ordered)

    for index, bar in enumerate(ordered):
        if index % 50000 == 0 and index > 0:
            print(f"    [{asset.value}] feature build: {index:,}/{total_bars:,} bars...")
        if index < 60:
            continue
        window_end = bar.close_time
        if start is not None and window_end < start:
            continue
        if end is not None and window_end >= end:
            continue

        def ret(minutes: int) -> float:
            if index < minutes:
                return 0.0
            return _safe_ret(bar.close, closes[index - minutes])

        def returns(minutes: int) -> list[float]:
            return one_min_returns[max(1, index - minutes + 1):index + 1]

        bars_5 = _window(ordered, index, 5)
        bars_15 = _window(ordered, index, 15)
        bars_60 = _window(ordered, index, 60)
        prev_60_volumes = [item.quote_volume for item in ordered[max(0, index - 60):index]]
        prev_15m_quote_sums = [
            _sum([item.quote_volume for item in _window(ordered, offset, 15)])
            for offset in range(max(14, index - 60), index)
        ]
        prev_15m_trade_sums = [
            _sum([float(item.trade_count) for item in _window(ordered, offset, 15)])
            for offset in range(max(14, index - 60), index)
        ]
        prev_60_abs_returns = [abs(value) for value in one_min_returns[max(1, index - 60):index]]
        current_abs_return = abs(one_min_returns[index])

        realized_vol_5m = _std(returns(5))
        realized_vol_15m = _std(returns(15))
        realized_vol_60m = _std(returns(60))
        realized_vol_15_history.append(realized_vol_15m)

        high_low_range = bar.high - bar.low
        close_position = _ratio(bar.close - bar.low, high_low_range)
        range_pct = _ratio(high_low_range, bar.close)
        max_close_15 = max(item.close for item in bars_15)
        min_close_15 = min(item.close for item in bars_15)
        max_close_60 = max(item.close for item in bars_60)
        min_close_60 = min(item.close for item in bars_60)
        true_ranges = [
            _true_range(ordered[offset], ordered[offset - 1].close)
            for offset in range(max(1, index - 13), index + 1)
        ]

        futures_bar = futures_by_time.get(bar.open_time)
        basis_pct = _safe_ret(futures_bar.close, bar.close) if futures_bar else 0.0
        basis_history.append(basis_pct)
        prior_basis = basis_history[-61:-1]

        quote_volume_5m = _sum([item.quote_volume for item in bars_5])
        quote_volume_15m = _sum([item.quote_volume for item in bars_15])
        taker_quote_5m = _sum([item.taker_buy_quote_volume for item in bars_5])

        snap = FeatureSnapshot(
            snapshot_id=_stable_snapshot_id(asset, window_end),
            asset=asset,
            window_end=window_end,
            price=bar.close,
            ret_1m=one_min_returns[index],
            ret_5m=ret(5),
            ret_15m=ret(15),
            ret_30m=ret(30),
            ret_60m=ret(60),
            vol_z_1m=_zscore(current_abs_return, prev_60_abs_returns),
            realized_vol_5m=realized_vol_5m,
            realized_vol_15m=realized_vol_15m,
            realized_vol_60m=realized_vol_60m,
            price_range_pct_1m=range_pct,
            close_position_1m=close_position,
            max_drawdown_15m=_drawdown_from_peak(bar.close, max_close_15),
            max_drawdown_60m=_drawdown_from_peak(bar.close, max_close_60),
            max_runup_15m=max(0.0, _safe_ret(bar.close, min_close_15)),
            max_runup_60m=max(0.0, _safe_ret(bar.close, min_close_60)),
            atr_14=_ratio(_mean(true_ranges), bar.close),
            volatility_regime_60m=_percentile_rank(realized_vol_15m, realized_vol_15_history[-61:-1]),
            volume_1m=bar.volume,
            quote_volume_1m=bar.quote_volume,
            volume_5m=_sum([item.volume for item in bars_5]),
            quote_volume_5m=quote_volume_5m,
            volume_15m=_sum([item.volume for item in bars_15]),
            quote_volume_15m=quote_volume_15m,
            volume_z_15m=_zscore(quote_volume_15m, prev_15m_quote_sums),
            volume_z_60m=_zscore(bar.quote_volume, prev_60_volumes),
            trade_count_1m=float(bar.trade_count),
            trade_count_z_15m=_zscore(
                _sum([float(item.trade_count) for item in bars_15]),
                prev_15m_trade_sums,
            ),
            taker_buy_ratio_1m=_ratio(bar.taker_buy_quote_volume, bar.quote_volume),
            taker_buy_ratio_5m=_ratio(taker_quote_5m, quote_volume_5m),
            futures_basis_pct=basis_pct,
            basis_z_60m=_zscore(basis_pct, prior_basis),
            source_stale=False,
            cross_source_conflict=False,
            ingest_lag_ms=0.0,
        )
        snapshots.append(snap)
    return snapshots


async def load_historical_snapshot_series(
    *,
    assets: list[Asset],
    start: datetime,
    end: datetime,
    max_snapshots_per_asset: int | None = None,
) -> dict[Asset, list[FeatureSnapshot]]:
    series: dict[Asset, list[FeatureSnapshot]] = {}
    query_start = start - timedelta(minutes=90)
    for asset in assets:
        spot_bars = await list_historical_market_bars(
            asset=asset,
            market_type="spot",
            source="binance_public",
            interval="1m",
            start=query_start,
            end=end,
        )
        futures_bars = await list_historical_market_bars(
            asset=asset,
            market_type="futures_um",
            source="binance_public",
            interval="1m",
            start=query_start,
            end=end,
        )
        snapshots = build_snapshots_from_bars(
            asset=asset,
            spot_bars=spot_bars,
            futures_bars=futures_bars,
            start=start,
            end=end,
        )
        if max_snapshots_per_asset is not None:
            snapshots = snapshots[:max_snapshots_per_asset]
        series[asset] = snapshots
    return series
