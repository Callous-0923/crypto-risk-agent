from __future__ import annotations

import math
from statistics import mean, pstdev

from src.domain.models import Asset, FeatureSnapshot


FEATURE_COLUMNS = [
    "asset_BTC",
    "asset_ETH",
    "asset_SOL",
    "price_log",
    "ret_1m",
    "ret_1m_abs",
    "ret_5m",
    "ret_5m_abs",
    "ret_15m",
    "ret_15m_abs",
    "ret_30m",
    "ret_30m_abs",
    "ret_60m",
    "ret_60m_abs",
    "vol_z_1m",
    "vol_z_1m_abs",
    "realized_vol_5m",
    "realized_vol_15m",
    "realized_vol_60m",
    "price_range_pct_1m",
    "close_position_1m",
    "max_drawdown_15m",
    "max_drawdown_60m",
    "max_runup_15m",
    "max_runup_60m",
    "atr_14",
    "volatility_regime_60m",
    "volume_1m_log",
    "quote_volume_1m_log",
    "volume_5m_log",
    "quote_volume_5m_log",
    "volume_15m_log",
    "quote_volume_15m_log",
    "volume_z_15m",
    "volume_z_60m",
    "trade_count_1m_log",
    "trade_count_z_15m",
    "taker_buy_ratio_1m",
    "taker_buy_ratio_5m",
    "oi_delta_15m_pct",
    "oi_delta_15m_pct_abs",
    "oi_delta_5m_pct",
    "oi_delta_5m_pct_abs",
    "oi_delta_60m_pct",
    "oi_delta_60m_pct_abs",
    "liq_5m_usd_log",
    "funding_z",
    "funding_z_abs",
    "futures_basis_pct",
    "basis_z_60m",
    "source_stale",
    "cross_source_conflict",
    "ret_1m_mean_5",
    "ret_1m_std_5",
    "ret_5m_mean_5",
    "ret_5m_std_5",
    "vol_z_mean_5",
    "oi_delta_mean_5",
    "funding_z_mean_5",
    "ret_1m_mean_10",
    "ret_1m_std_10",
    "ret_5m_mean_10",
    "ret_5m_std_10",
    "vol_z_mean_10",
    "oi_delta_mean_10",
    "funding_z_mean_10",
    "price_direction_consistency_5",
    "leverage_pressure",
    "risk_signal_count",
]


def _safe_log(value: float) -> float:
    return math.log(max(0.0, value) + 1.0)


def _avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _direction_consistency(values: list[float]) -> float:
    non_zero = [1 if value > 0 else -1 for value in values if abs(value) > 1e-12]
    if not non_zero:
        return 0.0
    positive = sum(1 for value in non_zero if value > 0)
    negative = len(non_zero) - positive
    return max(positive, negative) / len(non_zero)


def build_feature_dict(
    snap: FeatureSnapshot,
    *,
    history: list[FeatureSnapshot] | None = None,
) -> dict[str, float]:
    history = list(history or [])
    last_5 = history[-5:]
    last_10 = history[-10:]

    def values(items: list[FeatureSnapshot], field: str) -> list[float]:
        return [float(getattr(item, field, 0.0) or 0.0) for item in items]

    risk_signal_count = sum([
        abs(snap.ret_1m) >= 0.015,
        abs(snap.ret_5m) >= 0.008,
        abs(getattr(snap, "ret_15m", 0.0)) >= 0.012,
        abs(getattr(snap, "ret_60m", 0.0)) >= 0.02,
        abs(snap.vol_z_1m) >= 1.5,
        getattr(snap, "realized_vol_15m", 0.0) >= 0.004,
        abs(getattr(snap, "volume_z_15m", 0.0)) >= 1.5,
        abs(snap.oi_delta_15m_pct) >= 0.04,
        snap.liq_5m_usd >= 1_000_000,
        abs(snap.funding_z) >= 1.5,
    ])

    row = {
        "asset_BTC": 1.0 if snap.asset == Asset.BTC else 0.0,
        "asset_ETH": 1.0 if snap.asset == Asset.ETH else 0.0,
        "asset_SOL": 1.0 if snap.asset == Asset.SOL else 0.0,
        "price_log": _safe_log(snap.price),
        "ret_1m": snap.ret_1m,
        "ret_1m_abs": abs(snap.ret_1m),
        "ret_5m": snap.ret_5m,
        "ret_5m_abs": abs(snap.ret_5m),
        "ret_15m": getattr(snap, "ret_15m", 0.0),
        "ret_15m_abs": abs(getattr(snap, "ret_15m", 0.0)),
        "ret_30m": getattr(snap, "ret_30m", 0.0),
        "ret_30m_abs": abs(getattr(snap, "ret_30m", 0.0)),
        "ret_60m": getattr(snap, "ret_60m", 0.0),
        "ret_60m_abs": abs(getattr(snap, "ret_60m", 0.0)),
        "vol_z_1m": snap.vol_z_1m,
        "vol_z_1m_abs": abs(snap.vol_z_1m),
        "realized_vol_5m": getattr(snap, "realized_vol_5m", 0.0),
        "realized_vol_15m": getattr(snap, "realized_vol_15m", 0.0),
        "realized_vol_60m": getattr(snap, "realized_vol_60m", 0.0),
        "price_range_pct_1m": getattr(snap, "price_range_pct_1m", 0.0),
        "close_position_1m": getattr(snap, "close_position_1m", 0.0),
        "max_drawdown_15m": getattr(snap, "max_drawdown_15m", 0.0),
        "max_drawdown_60m": getattr(snap, "max_drawdown_60m", 0.0),
        "max_runup_15m": getattr(snap, "max_runup_15m", 0.0),
        "max_runup_60m": getattr(snap, "max_runup_60m", 0.0),
        "atr_14": getattr(snap, "atr_14", 0.0),
        "volatility_regime_60m": getattr(snap, "volatility_regime_60m", 0.0),
        "volume_1m_log": _safe_log(getattr(snap, "volume_1m", 0.0)),
        "quote_volume_1m_log": _safe_log(getattr(snap, "quote_volume_1m", 0.0)),
        "volume_5m_log": _safe_log(getattr(snap, "volume_5m", 0.0)),
        "quote_volume_5m_log": _safe_log(getattr(snap, "quote_volume_5m", 0.0)),
        "volume_15m_log": _safe_log(getattr(snap, "volume_15m", 0.0)),
        "quote_volume_15m_log": _safe_log(getattr(snap, "quote_volume_15m", 0.0)),
        "volume_z_15m": getattr(snap, "volume_z_15m", 0.0),
        "volume_z_60m": getattr(snap, "volume_z_60m", 0.0),
        "trade_count_1m_log": _safe_log(getattr(snap, "trade_count_1m", 0.0)),
        "trade_count_z_15m": getattr(snap, "trade_count_z_15m", 0.0),
        "taker_buy_ratio_1m": getattr(snap, "taker_buy_ratio_1m", 0.0),
        "taker_buy_ratio_5m": getattr(snap, "taker_buy_ratio_5m", 0.0),
        "oi_delta_15m_pct": snap.oi_delta_15m_pct,
        "oi_delta_15m_pct_abs": abs(snap.oi_delta_15m_pct),
        "oi_delta_5m_pct": getattr(snap, "oi_delta_5m_pct", 0.0),
        "oi_delta_5m_pct_abs": abs(getattr(snap, "oi_delta_5m_pct", 0.0)),
        "oi_delta_60m_pct": getattr(snap, "oi_delta_60m_pct", 0.0),
        "oi_delta_60m_pct_abs": abs(getattr(snap, "oi_delta_60m_pct", 0.0)),
        "liq_5m_usd_log": _safe_log(snap.liq_5m_usd),
        "funding_z": snap.funding_z,
        "funding_z_abs": abs(snap.funding_z),
        "futures_basis_pct": getattr(snap, "futures_basis_pct", 0.0),
        "basis_z_60m": getattr(snap, "basis_z_60m", 0.0),
        "source_stale": 1.0 if snap.source_stale else 0.0,
        "cross_source_conflict": 1.0 if snap.cross_source_conflict else 0.0,
        "ret_1m_mean_5": _avg(values(last_5, "ret_1m")),
        "ret_1m_std_5": _std(values(last_5, "ret_1m")),
        "ret_5m_mean_5": _avg(values(last_5, "ret_5m")),
        "ret_5m_std_5": _std(values(last_5, "ret_5m")),
        "vol_z_mean_5": _avg(values(last_5, "vol_z_1m")),
        "oi_delta_mean_5": _avg(values(last_5, "oi_delta_15m_pct")),
        "funding_z_mean_5": _avg(values(last_5, "funding_z")),
        "ret_1m_mean_10": _avg(values(last_10, "ret_1m")),
        "ret_1m_std_10": _std(values(last_10, "ret_1m")),
        "ret_5m_mean_10": _avg(values(last_10, "ret_5m")),
        "ret_5m_std_10": _std(values(last_10, "ret_5m")),
        "vol_z_mean_10": _avg(values(last_10, "vol_z_1m")),
        "oi_delta_mean_10": _avg(values(last_10, "oi_delta_15m_pct")),
        "funding_z_mean_10": _avg(values(last_10, "funding_z")),
        "price_direction_consistency_5": _direction_consistency(values(last_5, "ret_1m")),
        "leverage_pressure": abs(snap.oi_delta_15m_pct) * max(0.0, abs(snap.funding_z)),
        "risk_signal_count": float(risk_signal_count),
    }
    return {key: float(row.get(key, 0.0) or 0.0) for key in FEATURE_COLUMNS}


def build_matrix_rows(
    ordered_snapshots: list[FeatureSnapshot],
    *,
    history_size: int = 10,
) -> list[dict[str, float]]:
    rows = []
    for index, snap in enumerate(ordered_snapshots):
        rows.append(build_feature_dict(
            snap,
            history=ordered_snapshots[max(0, index - history_size):index],
        ))
    return rows


def rows_to_matrix(
    rows: list[dict[str, float]],
    *,
    columns: list[str] | None = None,
) -> list[list[float]]:
    active_columns = columns or FEATURE_COLUMNS
    return [[float(row.get(column, 0.0) or 0.0) for column in active_columns] for row in rows]
