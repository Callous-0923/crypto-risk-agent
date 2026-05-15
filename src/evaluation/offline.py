"""Offline weak-label evaluation over persisted feature snapshots.

The runtime proxy metrics are useful for operations, but they cannot measure
true recall without a labeled denominator. This module builds a conservative
weak-label dataset from real historical snapshots by asking: "from this
snapshot, did the market realize a large move within the future horizon?"
"""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Callable

from sqlalchemy import select

from src.domain.models import Asset, FeatureSnapshot, Severity
from src.persistence.database import AsyncSessionLocal, FeatureSnapshotRow
from src.rules.config import RuleThresholds, registry
from src.rules.engine import (
    early_warning_profile_from_hits,
    evaluate_with_thresholds,
    is_early_warning_reviewable,
)

SeverityLabel = str

NONE_LABEL = "none"
PREDICTION_KEYS = (
    "rules_baseline_positive",
    "early_warning_positive",
    "risk_detection_positive",
    "agent_alert_positive",
)


def _normalize_dt(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo else value


def _snapshot_from_row(row: FeatureSnapshotRow) -> FeatureSnapshot:
    return FeatureSnapshot(**{
        column.name: getattr(row, column.name)
        for column in FeatureSnapshotRow.__table__.columns
    })


def _highest_severity(snap: FeatureSnapshot, thresholds: RuleThresholds) -> Severity | None:
    hits = evaluate_with_thresholds(snap, thresholds)
    if not hits:
        return None
    order = {Severity.P1: 0, Severity.P2: 1, Severity.P3: 2}
    return min(hits, key=lambda hit: order[hit.severity]).severity


def _prediction_payload(
    snap: FeatureSnapshot,
    thresholds: RuleThresholds,
    recent_snapshots: list[FeatureSnapshot] | None = None,
) -> dict:
    recent_snapshots = recent_snapshots or []
    hits = evaluate_with_thresholds(snap, thresholds, recent_snapshots=recent_snapshots)
    severity = None
    if hits:
        order = {Severity.P1: 0, Severity.P2: 1, Severity.P3: 2}
        severity = min(hits, key=lambda hit: order[hit.severity]).severity
    early_warning_profile = early_warning_profile_from_hits(hits)
    prior_candidate_count = 0
    persistence_window = max(0, thresholds.early_warning_persistence_window - 1)
    persistence_candidates = recent_snapshots[-persistence_window:] if persistence_window else []
    for offset, previous in enumerate(persistence_candidates):
        previous_index = len(recent_snapshots) - len(persistence_candidates) + offset
        previous_history = recent_snapshots[
            max(0, previous_index - thresholds.early_warning_dynamic_history):previous_index
        ]
        previous_profile = early_warning_profile_from_hits(
            evaluate_with_thresholds(previous, thresholds, recent_snapshots=previous_history)
        )
        if previous_profile["candidate"]:
            prior_candidate_count += 1
    early_warning_recent_candidate_count = (
        prior_candidate_count + (1 if early_warning_profile["candidate"] else 0)
    )
    early_warning_positive = is_early_warning_reviewable(
        hits,
        recent_candidate_count=early_warning_recent_candidate_count,
        thresholds=thresholds,
    )
    formal_risk_positive = severity in {Severity.P1, Severity.P2}
    return {
        "severity": severity.value if severity else NONE_LABEL,
        "rule_ids": [hit.rule_id for hit in hits],
        "early_warning_score": early_warning_profile["score"],
        "early_warning_signal_count": early_warning_profile["signal_count"],
        "early_warning_trend_hits": early_warning_profile.get("trend_hits", 0),
        "early_warning_trend_confirmed": early_warning_profile.get("trend_confirmed", False),
        "early_warning_candidate": early_warning_profile["candidate"],
        "early_warning_recent_candidate_count": early_warning_recent_candidate_count,
        "rules_baseline_positive": formal_risk_positive,
        "early_warning_positive": early_warning_positive,
        "risk_detection_positive": formal_risk_positive or early_warning_positive,
        "agent_alert_positive": severity == Severity.P1,
    }


def _label_from_future_move(
    anchor: FeatureSnapshot,
    future: list[FeatureSnapshot],
    thresholds: RuleThresholds,
    *,
    label_price_change_p1: float | None = None,
    label_price_change_p2: float | None = None,
) -> dict | None:
    if anchor.price <= 0:
        return None

    p1_threshold = label_price_change_p1 or thresholds.price_change_p1
    p2_threshold = label_price_change_p2 or thresholds.price_change_p2
    anchor_time = _normalize_dt(anchor.window_end)
    if anchor.liq_5m_usd >= thresholds.liq_usd_p1:
        return {
            "label": Severity.P1.value,
            "label_time": anchor_time,
            "reason": "current_liquidation_cascade",
            "realized_return": 0.0,
        }

    first_p2: dict | None = None
    for snap in future:
        if snap.price <= 0:
            continue
        realized_return = (snap.price - anchor.price) / anchor.price
        abs_return = abs(realized_return)
        payload = {
            "label_time": _normalize_dt(snap.window_end),
            "realized_return": realized_return,
        }
        if abs_return >= p1_threshold:
            return {
                **payload,
                "label": Severity.P1.value,
                "reason": "future_price_move_p1",
            }
        if first_p2 is None and abs_return >= p2_threshold:
            first_p2 = {
                **payload,
                "label": Severity.P2.value,
                "reason": "future_price_move_p2",
            }

    if first_p2 is not None:
        return first_p2
    return {
        "label": NONE_LABEL,
        "label_time": anchor_time,
        "reason": "no_future_large_move",
        "realized_return": 0.0,
    }


def _take_evenly(items: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / (limit - 1)
    return [items[round(index * step)] for index in range(limit)]


def build_weak_label_episodes(
    series_by_asset: dict[Asset, list[FeatureSnapshot]],
    *,
    thresholds: RuleThresholds | None = None,
    horizon_seconds: int = 900,
    min_gap_seconds: int = 300,
    max_episodes: int = 600,
    label_price_change_p1: float | None = None,
    label_price_change_p2: float | None = None,
) -> list[dict]:
    thresholds = thresholds or registry.thresholds
    episodes: list[dict] = []
    min_coverage_seconds = min(120, horizon_seconds * 0.5)

    for asset, series in series_by_asset.items():
        ordered = sorted(series, key=lambda item: _normalize_dt(item.window_end))
        last_episode_at: datetime | None = None
        for index, anchor in enumerate(ordered[:-1]):
            anchor_time = _normalize_dt(anchor.window_end)
            if last_episode_at is not None:
                if (anchor_time - last_episode_at).total_seconds() < min_gap_seconds:
                    continue

            horizon_end = anchor_time + timedelta(seconds=horizon_seconds)
            future: list[FeatureSnapshot] = []
            for candidate in ordered[index + 1:]:
                candidate_time = _normalize_dt(candidate.window_end)
                if candidate_time > horizon_end:
                    break
                future.append(candidate)

            if not future:
                continue
            coverage_seconds = (_normalize_dt(future[-1].window_end) - anchor_time).total_seconds()
            if coverage_seconds < min_coverage_seconds:
                continue

            label_payload = _label_from_future_move(
                anchor,
                future,
                thresholds,
                label_price_change_p1=label_price_change_p1,
                label_price_change_p2=label_price_change_p2,
            )
            if label_payload is None:
                continue

            history_window = max(
                thresholds.early_warning_dynamic_history,
                thresholds.early_warning_trend_window,
                thresholds.early_warning_persistence_window,
            )
            previous_window = ordered[max(0, index - history_window):index]
            prediction = _prediction_payload(anchor, thresholds, previous_window)
            episodes.append({
                "asset": asset.value,
                "snapshot_id": anchor.snapshot_id,
                "snapshot_time": anchor_time,
                "price": anchor.price,
                "label": label_payload["label"],
                "label_time": label_payload["label_time"],
                "label_reason": label_payload["reason"],
                "realized_return": label_payload["realized_return"],
                "prediction": prediction,
            })
            last_episode_at = anchor_time

    positives = [item for item in episodes if item["label"] != NONE_LABEL]
    negatives = [item for item in episodes if item["label"] == NONE_LABEL]
    if not episodes:
        return []

    if positives and negatives:
        positive_limit = min(len(positives), max(1, int(max_episodes * 0.7)))
        negative_limit = max_episodes - positive_limit
        selected = _take_evenly(positives, positive_limit) + _take_evenly(negatives, negative_limit)
    else:
        selected = _take_evenly(episodes, max_episodes)

    return sorted(selected, key=lambda item: (item["snapshot_time"], item["asset"]))


def _binary_metrics(
    episodes: list[dict],
    *,
    prediction_key: str,
    truth_selector: Callable[[dict], bool],
) -> dict:
    tp = fp = tn = fn = 0
    lead_times: list[float] = []
    rule_counter: Counter[str] = Counter()

    for episode in episodes:
        truth_positive = truth_selector(episode)
        predicted_positive = bool(episode["prediction"].get(prediction_key))
        if predicted_positive:
            rule_counter.update(episode["prediction"].get("rule_ids", []))
        if truth_positive and predicted_positive:
            tp += 1
            lead_times.append(
                max(
                    0.0,
                    (episode["label_time"] - episode["snapshot_time"]).total_seconds(),
                )
            )
        elif truth_positive and not predicted_positive:
            fn += 1
        elif not truth_positive and predicted_positive:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    false_positive_rate = fp / (fp + tn) if (fp + tn) else None
    miss_rate = fn / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    def _percentile(values: list[float], ratio: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int((len(ordered) - 1) * ratio))
        return ordered[index]

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "miss_rate": miss_rate,
        "f1": f1,
        "predicted_positive": tp + fp,
        "truth_positive": tp + fn,
        "avg_lead_time_seconds": sum(lead_times) / len(lead_times) if lead_times else None,
        "p50_lead_time_seconds": _percentile(lead_times, 0.50),
        "p95_lead_time_seconds": _percentile(lead_times, 0.95),
        "top_rules": [
            {"rule_id": rule_id, "count": count}
            for rule_id, count in rule_counter.most_common(5)
        ],
    }


def evaluate_weak_label_episodes(
    episodes: list[dict],
    *,
    horizon_seconds: int,
    min_gap_seconds: int,
    label_price_change_p1: float | None = None,
    label_price_change_p2: float | None = None,
) -> dict:
    label_breakdown = Counter(item["label"] for item in episodes)
    prediction_breakdown = Counter(item["prediction"]["severity"] for item in episodes)
    reason_breakdown = Counter(item["label_reason"] for item in episodes)

    p1_truth = lambda item: item["label"] == Severity.P1.value
    p1p2_truth = lambda item: item["label"] in {Severity.P1.value, Severity.P2.value}

    rules_baseline = _binary_metrics(
        episodes,
        prediction_key="rules_baseline_positive",
        truth_selector=p1p2_truth,
    )
    early_warning = _binary_metrics(
        episodes,
        prediction_key="early_warning_positive",
        truth_selector=p1p2_truth,
    )
    risk_detection = _binary_metrics(
        episodes,
        prediction_key="risk_detection_positive",
        truth_selector=p1p2_truth,
    )
    agent_alert = _binary_metrics(
        episodes,
        prediction_key="agent_alert_positive",
        truth_selector=p1_truth,
    )
    agent_alert_vs_all_risk = _binary_metrics(
        episodes,
        prediction_key="agent_alert_positive",
        truth_selector=p1p2_truth,
    )

    baseline_candidates = rules_baseline["predicted_positive"]
    agent_candidates = agent_alert_vs_all_risk["predicted_positive"]
    alert_reduction_rate = (
        max(0, baseline_candidates - agent_candidates) / baseline_candidates
        if baseline_candidates else 0
    )

    return {
        "labeling_mode": "weak_label_future_price_move",
        "labeling_note": (
            "Labels are derived from future realized price moves in persisted real snapshots; "
            "they are suitable for recall/precision testing but are not a substitute for human ground truth."
        ),
        "horizon_seconds": horizon_seconds,
        "sample_gap_seconds": min_gap_seconds,
        "label_thresholds": {
            "price_change_p1": label_price_change_p1,
            "price_change_p2": label_price_change_p2,
        },
        "episodes": len(episodes),
        "label_breakdown": dict(label_breakdown),
        "prediction_severity_breakdown": dict(prediction_breakdown),
        "label_reason_breakdown": dict(reason_breakdown),
        "rules_baseline": rules_baseline,
        "early_warning_policy": early_warning,
        "risk_detection_policy": risk_detection,
        "agent_alert_policy": agent_alert,
        "agent_alert_vs_all_risk": agent_alert_vs_all_risk,
        "core_metrics": {
            "early_warning_recall": early_warning["recall"],
            "early_warning_precision": early_warning["precision"],
            "early_warning_avg_lead_time_seconds": early_warning["avg_lead_time_seconds"],
            "early_warning_conversion_rate": early_warning["precision"],
            "formal_alert_recall": agent_alert["recall"],
            "formal_alert_false_positive_rate": agent_alert["false_positive_rate"],
        },
        "baseline_comparison": {
            "baseline": "pure_rules_alert_all_p1_p2",
            "rules_baseline_alert_candidates": baseline_candidates,
            "agent_alert_candidates": agent_candidates,
            "alert_reduction_rate": alert_reduction_rate,
            "risk_recall_delta": (
                agent_alert_vs_all_risk["recall"] - rules_baseline["recall"]
                if agent_alert_vs_all_risk["recall"] is not None and rules_baseline["recall"] is not None
                else None
            ),
            "precision_delta": (
                agent_alert_vs_all_risk["precision"] - rules_baseline["precision"]
                if agent_alert_vs_all_risk["precision"] is not None and rules_baseline["precision"] is not None
                else None
            ),
        },
        "sample_episodes": [
            {
                "asset": item["asset"],
                "snapshot_id": item["snapshot_id"],
                "snapshot_time": item["snapshot_time"].isoformat(),
                "label": item["label"],
                "label_time": item["label_time"].isoformat(),
                "label_reason": item["label_reason"],
                "realized_return": item["realized_return"],
                "predicted_severity": item["prediction"]["severity"],
                "early_warning_score": item["prediction"].get("early_warning_score"),
                "early_warning_signal_count": item["prediction"].get("early_warning_signal_count"),
                "early_warning_trend_hits": item["prediction"].get("early_warning_trend_hits"),
                "early_warning_trend_confirmed": item["prediction"].get("early_warning_trend_confirmed"),
                "early_warning_recent_candidate_count": item["prediction"].get("early_warning_recent_candidate_count"),
                "rule_ids": item["prediction"]["rule_ids"][:5],
            }
            for item in episodes[:10]
        ],
    }


def _safe_metric(value: float | None) -> float:
    return float(value) if value is not None else 0.0


def _tuning_score(metrics: dict, horizon_seconds: int) -> float:
    precision = _safe_metric(metrics.get("precision"))
    recall = _safe_metric(metrics.get("recall"))
    f1 = _safe_metric(metrics.get("f1"))
    lead = _safe_metric(metrics.get("avg_lead_time_seconds"))
    predicted = _safe_metric(metrics.get("predicted_positive"))
    total = _safe_metric(metrics.get("tp")) + _safe_metric(metrics.get("fp")) + _safe_metric(metrics.get("tn")) + _safe_metric(metrics.get("fn"))
    lead_quality = min(1.0, lead / horizon_seconds) if horizon_seconds > 0 else 0.0
    noise_penalty = (predicted / total) if total else 0.0
    return 0.50 * f1 + 0.20 * precision + 0.25 * recall + 0.03 * lead_quality - 0.03 * noise_penalty


def tune_early_warning_thresholds(
    series_by_asset: dict[Asset, list[FeatureSnapshot]],
    *,
    base_thresholds: RuleThresholds | None = None,
    horizon_seconds: int = 900,
    min_gap_seconds: int = 300,
    max_episodes: int = 600,
    label_price_change_p1: float | None = None,
    label_price_change_p2: float | None = None,
    top_n: int = 5,
) -> dict:
    base_thresholds = base_thresholds or registry.thresholds
    candidates: list[RuleThresholds] = []
    for min_score in (0.35, 0.50, 0.65):
        for min_signals in (1, 2):
            for persistence_hits in (1, 2):
                for min_trend_hits in (1, 2):
                    for dynamic_baseline in (False, True):
                        quantiles = (base_thresholds.early_warning_dynamic_quantile,) if not dynamic_baseline else (0.70, 0.85)
                        for dynamic_quantile in quantiles:
                            candidates.append(base_thresholds.model_copy(update={
                                "early_warning_min_score": min_score,
                                "early_warning_min_signals": min_signals,
                                "early_warning_single_signal_min_score": max(0.65, min_score + 0.20),
                                "early_warning_persistence_hits": persistence_hits,
                                "early_warning_min_trend_hits": min_trend_hits,
                                "early_warning_dynamic_quantile": dynamic_quantile,
                                "early_warning_dynamic_baseline": dynamic_baseline,
                            }))

    ranked: list[dict] = []
    for thresholds in candidates:
        episodes = build_weak_label_episodes(
            series_by_asset,
            thresholds=thresholds,
            horizon_seconds=horizon_seconds,
            min_gap_seconds=min_gap_seconds,
            max_episodes=max_episodes,
            label_price_change_p1=label_price_change_p1,
            label_price_change_p2=label_price_change_p2,
        )
        if not episodes:
            continue
        result = evaluate_weak_label_episodes(
            episodes,
            horizon_seconds=horizon_seconds,
            min_gap_seconds=min_gap_seconds,
            label_price_change_p1=label_price_change_p1,
            label_price_change_p2=label_price_change_p2,
        )
        metrics = result["early_warning_policy"]
        ranked.append({
            "score": _tuning_score(metrics, horizon_seconds),
            "episodes": result["episodes"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "miss_rate": metrics["miss_rate"],
            "false_positive_rate": metrics["false_positive_rate"],
            "avg_lead_time_seconds": metrics["avg_lead_time_seconds"],
            "predicted_positive": metrics["predicted_positive"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "thresholds": {
                "early_warning_min_score": thresholds.early_warning_min_score,
                "early_warning_min_signals": thresholds.early_warning_min_signals,
                "early_warning_single_signal_min_score": thresholds.early_warning_single_signal_min_score,
                "early_warning_persistence_hits": thresholds.early_warning_persistence_hits,
                "early_warning_min_trend_hits": thresholds.early_warning_min_trend_hits,
                "early_warning_dynamic_quantile": thresholds.early_warning_dynamic_quantile,
                "early_warning_dynamic_baseline": thresholds.early_warning_dynamic_baseline,
            },
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {
        "objective": "balanced_f1_precision_recall_lead_time_with_noise_penalty",
        "candidates_evaluated": len(candidates),
        "top_configs": ranked[:max(1, top_n)],
    }


async def load_snapshot_series(
    *,
    days: int | None = None,
    max_snapshots_per_asset: int = 1200,
) -> dict[Asset, list[FeatureSnapshot]]:
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days) if days else None
    max_snapshots_per_asset = max(100, min(max_snapshots_per_asset, 10000))
    series_by_asset: dict[Asset, list[FeatureSnapshot]] = {}

    async with AsyncSessionLocal() as session:
        for asset in Asset:
            query = (
                select(FeatureSnapshotRow)
                .where(FeatureSnapshotRow.asset == asset.value)
                .order_by(FeatureSnapshotRow.window_end.desc())
                .limit(max_snapshots_per_asset)
            )
            if cutoff:
                query = query.where(FeatureSnapshotRow.window_end >= cutoff)
            rows = (await session.execute(query)).scalars().all()
            snapshots = [_snapshot_from_row(row) for row in rows]
            series_by_asset[asset] = sorted(snapshots, key=lambda item: _normalize_dt(item.window_end))

    return series_by_asset


async def run_offline_weak_label_evaluation(
    *,
    days: int | None = None,
    horizon_seconds: int = 900,
    min_gap_seconds: int = 300,
    max_snapshots_per_asset: int = 1200,
    max_episodes: int = 600,
    label_price_change_p1: float | None = None,
    label_price_change_p2: float | None = None,
    include_tuning: bool = False,
    tuning_max_episodes: int | None = None,
) -> dict:
    started = time.perf_counter()
    series_by_asset = await load_snapshot_series(
        days=days,
        max_snapshots_per_asset=max_snapshots_per_asset,
    )
    snapshots_loaded = sum(len(items) for items in series_by_asset.values())
    episodes = build_weak_label_episodes(
        series_by_asset,
        thresholds=registry.thresholds,
        horizon_seconds=horizon_seconds,
        min_gap_seconds=min_gap_seconds,
        max_episodes=max_episodes,
        label_price_change_p1=label_price_change_p1,
        label_price_change_p2=label_price_change_p2,
    )
    result = evaluate_weak_label_episodes(
        episodes,
        horizon_seconds=horizon_seconds,
        min_gap_seconds=min_gap_seconds,
        label_price_change_p1=label_price_change_p1,
        label_price_change_p2=label_price_change_p2,
    )
    result["label_thresholds"] = {
        "price_change_p1": label_price_change_p1 or registry.thresholds.price_change_p1,
        "price_change_p2": label_price_change_p2 or registry.thresholds.price_change_p2,
    }
    elapsed = time.perf_counter() - started
    result["sample_sizes"] = {
        "snapshots_loaded": snapshots_loaded,
        "max_snapshots_per_asset": max_snapshots_per_asset,
        "max_episodes": max_episodes,
    }
    result["throughput"] = {
        "evaluation_seconds": elapsed,
        "episodes_per_second": len(episodes) / elapsed if elapsed > 0 else 0,
    }
    if include_tuning and snapshots_loaded:
        result["early_warning_tuning"] = tune_early_warning_thresholds(
            series_by_asset,
            base_thresholds=registry.thresholds,
            horizon_seconds=horizon_seconds,
            min_gap_seconds=min_gap_seconds,
            max_episodes=tuning_max_episodes or max_episodes,
            label_price_change_p1=label_price_change_p1,
            label_price_change_p2=label_price_change_p2,
        )
    return result
