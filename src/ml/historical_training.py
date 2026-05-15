from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from src.domain.models import Asset, FeatureSnapshot
from src.ml.historical_features import load_historical_snapshot_series
from src.ml.labeling import future_summary
from src.ml.risk_model import train_risk_model
from src.persistence.repositories import save_model_label


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def _take_evenly(items: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / (limit - 1)
    return [items[round(index * step)] for index in range(limit)]


def _current_signal_score(snap: FeatureSnapshot) -> float:
    return sum([
        min(abs(snap.ret_5m) / 0.01, 1.0),
        min(abs(snap.ret_15m) / 0.015, 1.0),
        min(snap.realized_vol_15m / 0.006, 1.0),
        min(abs(snap.volume_z_15m) / 2.0, 1.0),
        min(abs(snap.futures_basis_pct) / 0.003, 1.0),
        min(abs(snap.basis_z_60m) / 2.0, 1.0),
    ])


def _candidate_records(
    series_by_asset: dict[Asset, list[FeatureSnapshot]],
    *,
    horizon_seconds: int,
    p2_quantile: float,
    p1_quantile: float,
) -> tuple[list[dict], dict[str, Any]]:
    records: list[dict] = []
    thresholds_by_asset: dict[str, dict[str, float]] = {}
    for asset, series in series_by_asset.items():
        ordered = sorted(series, key=lambda item: item.window_end)
        summaries: list[tuple[FeatureSnapshot, dict[str, Any]]] = []
        realized_moves: list[float] = []
        for index, snap in enumerate(ordered[:-1]):
            summary = future_summary(snap, ordered[index + 1:], horizon_seconds=horizon_seconds)
            if summary.get("future_count", 0) <= 0:
                continue
            summaries.append((snap, summary))
            realized_moves.append(float(summary.get("max_abs_return", 0.0) or 0.0))
        p2_threshold = max(0.004, _quantile(realized_moves, p2_quantile))
        p1_threshold = max(p2_threshold * 1.8, _quantile(realized_moves, p1_quantile))
        thresholds_by_asset[asset.value] = {
            "p2_abs_return": p2_threshold,
            "p1_abs_return": p1_threshold,
        }
        for snap, summary in summaries:
            max_abs_return = float(summary.get("max_abs_return", 0.0) or 0.0)
            if max_abs_return >= p1_threshold:
                label = "p1"
                probability = min(1.0, max_abs_return / max(p1_threshold, 1e-12))
            elif max_abs_return >= p2_threshold:
                label = "p2"
                probability = min(0.85, max_abs_return / max(p1_threshold, 1e-12))
            else:
                label = "none"
                probability = min(0.25, max_abs_return / max(p2_threshold, 1e-12))
            records.append({
                "asset": asset,
                "snapshot": snap,
                "label": label,
                "risk_probability": probability,
                "confidence": 0.72 if label != "none" else 0.65,
                "realized_move": max_abs_return,
                "current_signal_score": _current_signal_score(snap),
                "summary": summary,
            })
    return records, {"thresholds_by_asset": thresholds_by_asset}


def _select_training_records(records: list[dict], max_items: int | None) -> list[dict]:
    """从候选记录中采样训练集——平衡正负样本。

    策略:
      1. 正类 (p1/p2) 全部保留
      2. 负类分 hard_negatives 和 safe_negatives 两层:
         - hard_negatives: signal_score >= 1.0, 模型难以区分的边界样本
         - safe_negatives: 明显安全的负样本, 随机采样控制比例
      3. 最终正负比约 1:1.2, 远优于原始 1:9
    """
    if max_items is None or len(records) <= max_items:
        return records
    positives = [item for item in records if item["label"] != "none"]
    negatives = [item for item in records if item["label"] == "none"]
    hard_negatives = sorted(
        [item for item in negatives if item["current_signal_score"] >= 1.0],
        key=lambda item: item["current_signal_score"],
        reverse=True,
    )
    safe_negatives = [item for item in negatives if item["current_signal_score"] < 1.0]

    # 正类全部保留, 若不足则不做负类截断
    if len(positives) >= max_items * 0.9:
        return sorted(positives[:max_items], key=lambda item: item["snapshot"].window_end)

    # 按照正类数量 * 1.2 配负类, 不超过 max_items
    target_negatives = min(
        len(hard_negatives) + len(safe_negatives),
        max(1000, min(int(len(positives) * 1.2), max_items - len(positives)))
    )
    hard_allocate = min(len(hard_negatives), int(target_negatives * 0.6))
    safe_allocate = target_negatives - hard_allocate

    selected = (
        positives
        + hard_negatives[:hard_allocate]
        + _take_evenly(safe_negatives, safe_allocate)
    )
    return sorted(selected, key=lambda item: item["snapshot"].window_end)


async def label_historical_snapshots(
    series_by_asset: dict[Asset, list[FeatureSnapshot]],
    *,
    horizon_seconds: int,
    max_items: int | None,
    p2_quantile: float = 0.85,
    p1_quantile: float = 0.95,
) -> dict[str, Any]:
    records, metadata = _candidate_records(
        series_by_asset,
        horizon_seconds=horizon_seconds,
        p2_quantile=p2_quantile,
        p1_quantile=p1_quantile,
    )
    selected = _select_training_records(records, max_items)
    breakdown = Counter()
    for item in selected:
        snap = item["snapshot"]
        breakdown[item["label"]] += 1
        await save_model_label(
            snapshot_id=snap.snapshot_id,
            asset=snap.asset,
            window_end=snap.window_end,
            horizon_seconds=horizon_seconds,
            label=item["label"],
            risk_probability=item["risk_probability"],
            confidence=item["confidence"],
            labeling_method="historical_dynamic_weak_label",
            rationale="future-window dynamic quantile weak label",
            judge_payload={
                "source": "historical_public_market_data",
                "realized_move": item["realized_move"],
                "current_signal_score": item["current_signal_score"],
                "summary": item["summary"],
                **metadata["thresholds_by_asset"][snap.asset.value],
            },
        )
    return {
        "candidates": len(records),
        "labeled": len(selected),
        "label_breakdown": dict(breakdown),
        **metadata,
    }


async def train_historical_risk_model(
    *,
    assets: list[Asset],
    start: datetime,
    end: datetime,
    horizon_seconds: int,
    max_snapshots_per_asset: int | None,
    max_label_items: int | None,
    min_samples: int,
    p2_quantile: float = 0.85,
    p1_quantile: float = 0.95,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    series_by_asset = await load_historical_snapshot_series(
        assets=assets,
        start=start,
        end=end,
        max_snapshots_per_asset=max_snapshots_per_asset,
    )
    labeling_summary = await label_historical_snapshots(
        series_by_asset,
        horizon_seconds=horizon_seconds,
        max_items=max_label_items,
        p2_quantile=p2_quantile,
        p1_quantile=p1_quantile,
    )
    result = await train_risk_model(
        series_by_asset,
        horizon_seconds=horizon_seconds,
        max_label_items=0,
        use_llm_judge=False,
        min_samples=min_samples,
        model_params=model_params,
        training_metadata={
            "source": "binance_public_historical",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "assets": [asset.value for asset in assets],
            "labeling": labeling_summary,
        },
    )
    result["historical_labeling_summary"] = labeling_summary
    return result
