"""Rule engine — deterministic, no LLM involvement.

阈值从 RuleRegistry 读取，支持热更新和版本回放。
evaluate() 默认用当前 active 版本；
evaluate_with_version() 用指定 RuleThresholds 实例（供回放使用）。
"""
from __future__ import annotations

from src.domain.models import Asset, FeatureSnapshot, RuleHit, Severity
from src.rules.config import RuleThresholds, registry


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _activation_score(value: float, warning_threshold: float, formal_threshold: float) -> float:
    if value < warning_threshold:
        return 0.0
    if formal_threshold <= warning_threshold:
        return 1.0
    normalized = _clamp((value - warning_threshold) / (formal_threshold - warning_threshold))
    return 0.35 + 0.65 * normalized


def _quantile(values: list[float], ratio: float) -> float | None:
    cleaned = sorted(abs(value) for value in values if value is not None)
    if not cleaned:
        return None
    index = min(len(cleaned) - 1, max(0, round((len(cleaned) - 1) * ratio)))
    return cleaned[index]


def _adaptive_threshold(
    static_threshold: float,
    formal_threshold: float,
    history_values: list[float],
    t: RuleThresholds,
) -> float:
    if not t.early_warning_dynamic_baseline or len(history_values) < 4:
        return static_threshold
    baseline = _quantile(history_values[-t.early_warning_dynamic_history:], t.early_warning_dynamic_quantile)
    if baseline is None or baseline <= 0:
        return static_threshold
    lower_bound = static_threshold * 0.75
    upper_bound = formal_threshold * 0.85 if formal_threshold > static_threshold else static_threshold
    return _clamp(baseline * 1.15, lower_bound, upper_bound)


def _signal_is_active(snap: FeatureSnapshot, signal: str, thresholds: dict[str, float]) -> bool:
    if signal == "price_drift":
        return abs(snap.ret_5m) >= thresholds["ret_5m"]
    if signal == "vol_expansion":
        return thresholds["vol_z"] <= abs(snap.vol_z_1m) < thresholds["vol_z_formal"]
    if signal == "oi_buildup":
        return thresholds["oi_delta"] <= abs(snap.oi_delta_15m_pct) < thresholds["oi_delta_formal"]
    if signal == "funding_bias":
        return thresholds["funding_z"] <= abs(snap.funding_z) < thresholds["funding_z_formal"]
    return False


def _signal_direction(snap: FeatureSnapshot, signal: str) -> int:
    if signal == "price_drift":
        return 1 if snap.ret_5m > 0 else -1 if snap.ret_5m < 0 else 0
    if signal == "oi_buildup":
        return 1 if snap.oi_delta_15m_pct > 0 else -1 if snap.oi_delta_15m_pct < 0 else 0
    if signal == "funding_bias":
        return 1 if snap.funding_z > 0 else -1 if snap.funding_z < 0 else 0
    return 0


def build_early_warning_profile(
    snap: FeatureSnapshot,
    t: RuleThresholds,
    recent_snapshots: list[FeatureSnapshot] | None = None,
) -> dict:
    """Score weak early-warning signals before they are allowed into review."""
    history = list(recent_snapshots or [])[-t.early_warning_dynamic_history:]
    adaptive_thresholds = {
        "ret_5m": _adaptive_threshold(
            t.early_warning_ret_5m,
            t.price_change_p2,
            [item.ret_5m for item in history],
            t,
        ),
        "vol_z": _adaptive_threshold(
            t.early_warning_vol_z,
            t.vol_z_spike,
            [item.vol_z_1m for item in history],
            t,
        ),
        "oi_delta": _adaptive_threshold(
            t.early_warning_oi_delta,
            t.oi_delta_p2,
            [item.oi_delta_15m_pct for item in history],
            t,
        ),
        "funding_z": _adaptive_threshold(
            t.early_warning_funding_z,
            t.funding_z_p2,
            [item.funding_z for item in history],
            t,
        ),
        "vol_z_formal": t.vol_z_spike,
        "oi_delta_formal": t.oi_delta_p2,
        "funding_z_formal": t.funding_z_p2,
    }
    signals: dict[str, float] = {}
    abs_ret_5m = abs(snap.ret_5m)
    abs_vol_z = abs(snap.vol_z_1m)
    abs_oi_delta = abs(snap.oi_delta_15m_pct)
    abs_funding_z = abs(snap.funding_z)

    if abs_ret_5m >= adaptive_thresholds["ret_5m"]:
        signals["price_drift"] = _activation_score(
            abs_ret_5m,
            adaptive_thresholds["ret_5m"],
            t.price_change_p2,
        )
    if adaptive_thresholds["vol_z"] <= abs_vol_z < t.vol_z_spike:
        signals["vol_expansion"] = _activation_score(
            abs_vol_z,
            adaptive_thresholds["vol_z"],
            t.vol_z_spike,
        )
    if adaptive_thresholds["oi_delta"] <= abs_oi_delta < t.oi_delta_p2:
        signals["oi_buildup"] = _activation_score(
            abs_oi_delta,
            adaptive_thresholds["oi_delta"],
            t.oi_delta_p2,
        )
    if adaptive_thresholds["funding_z"] <= abs_funding_z < t.funding_z_p2:
        signals["funding_bias"] = _activation_score(
            abs_funding_z,
            adaptive_thresholds["funding_z"],
            t.funding_z_p2,
        )

    signal_count = len(signals)
    trend_series = history[-max(0, t.early_warning_trend_window - 1):] + [snap]
    persistent_signals: dict[str, int] = {}
    for signal in signals:
        current_direction = _signal_direction(snap, signal)
        hits = 0
        for item in trend_series:
            if not _signal_is_active(item, signal, adaptive_thresholds):
                continue
            item_direction = _signal_direction(item, signal)
            if current_direction and item_direction and item_direction != current_direction:
                continue
            hits += 1
        persistent_signals[signal] = hits
    trend_hits = sum(1 for count in persistent_signals.values() if count >= t.early_warning_min_trend_hits)
    trend_confirmed = (
        not history
        or t.early_warning_min_trend_hits <= 1
        or trend_hits > 0
    )
    direction_aligned = bool(
        snap.ret_5m
        and snap.funding_z
        and (snap.ret_5m > 0) == (snap.funding_z > 0)
    )
    if not signals:
        score = 0.0
    else:
        signal_strength = sum(signals.values()) / signal_count
        coverage_bonus = min(0.25, 0.08 * signal_count)
        direction_bonus = 0.08 if direction_aligned and signal_count >= 2 else 0.0
        trend_bonus = min(0.12, 0.04 * trend_hits)
        score = _clamp(signal_strength * 0.70 + coverage_bonus + direction_bonus + trend_bonus)

    multi_signal_candidate = (
        signal_count >= t.early_warning_min_signals
        and score >= t.early_warning_min_score
    )
    strong_single_signal_candidate = (
        signal_count == 1
        and score >= t.early_warning_single_signal_min_score
    )
    candidate = (multi_signal_candidate or strong_single_signal_candidate) and trend_confirmed
    return {
        "score": score,
        "signal_count": signal_count,
        "signals": signals,
        "adaptive_thresholds": adaptive_thresholds,
        "persistent_signals": persistent_signals,
        "trend_hits": trend_hits,
        "trend_confirmed": trend_confirmed,
        "direction_aligned": direction_aligned,
        "candidate": candidate,
        "multi_signal_candidate": multi_signal_candidate,
        "strong_single_signal_candidate": strong_single_signal_candidate,
        "min_score": t.early_warning_min_score,
        "min_signals": t.early_warning_min_signals,
        "single_signal_min_score": t.early_warning_single_signal_min_score,
    }


def early_warning_profile_from_hits(hits: list[RuleHit]) -> dict:
    ew_hits = [hit for hit in hits if hit.rule_id.startswith("EW_")]
    if not ew_hits:
        return {
            "score": 0.0,
            "signal_count": 0,
            "signals": {},
            "trend_hits": 0,
            "trend_confirmed": False,
            "candidate": False,
        }
    best = max(
        ew_hits,
        key=lambda hit: float(hit.evidence.get("early_warning_score", hit.confidence) or 0.0),
    )
    evidence = best.evidence
    return {
        "score": float(evidence.get("early_warning_score", best.confidence) or 0.0),
        "signal_count": int(evidence.get("early_warning_signal_count", len(ew_hits)) or 0),
        "signals": evidence.get("early_warning_signals", {}),
        "trend_hits": int(evidence.get("early_warning_trend_hits", 0) or 0),
        "trend_confirmed": bool(evidence.get("early_warning_trend_confirmed", False)),
        "candidate": bool(evidence.get("early_warning_candidate", False)),
    }


def is_early_warning_reviewable(
    hits: list[RuleHit],
    *,
    recent_candidate_count: int = 1,
    thresholds: RuleThresholds | None = None,
) -> bool:
    thresholds = thresholds or registry.thresholds
    profile = early_warning_profile_from_hits(hits)
    return bool(
        profile["candidate"]
        and recent_candidate_count >= thresholds.early_warning_persistence_hits
    )


def evaluate(snap: FeatureSnapshot) -> list[RuleHit]:
    """使用当前 active 规则版本评估快照，并记录 Prometheus 指标。"""
    import time
    from src.observability.metrics import (
        rule_evaluate_total, rule_hit_total, rule_evaluate_duration_seconds,
        data_quality_event_total,
    )
    version = registry.active_version
    t0 = time.perf_counter()
    hits = evaluate_with_thresholds(snap, registry.thresholds)
    elapsed = time.perf_counter() - t0

    rule_evaluate_total.labels(asset=snap.asset.value, rule_version=version).inc()
    rule_evaluate_duration_seconds.labels(asset=snap.asset.value).observe(elapsed)

    for hit in hits:
        rule_hit_total.labels(
            asset=snap.asset.value,
            rule_id=hit.rule_id,
            severity=hit.severity.value,
            rule_version=version,
        ).inc()
        # 数据质量事件单独计数
        if hit.rule_id.startswith("QA_"):
            issue = "stale" if "STALE" in hit.rule_id else "conflict"
            data_quality_event_total.labels(
                asset=snap.asset.value, issue_type=issue,
            ).inc()

    return hits


def evaluate_with_thresholds(
    snap: FeatureSnapshot,
    t: RuleThresholds,
    recent_snapshots: list[FeatureSnapshot] | None = None,
) -> list[RuleHit]:
    """
    用指定阈值评估快照——规则版本回放的核心入口。
    规则逻辑与阈值完全解耦，t 可以是任意历史版本。
    """
    hits: list[RuleHit] = []
    asset = snap.asset

    # -----------------------------------------------------------------------
    # Market rules
    # -----------------------------------------------------------------------
    abs_ret_1m = abs(snap.ret_1m)
    if abs_ret_1m >= t.price_change_p1:
        hits.append(RuleHit(
            rule_id="MKT_EXTREME_VOL_P1",
            asset=asset,
            severity=Severity.P1,
            confidence=min(1.0, abs_ret_1m / t.price_change_p1),
            description=f"{asset.value} 1m价格变动 {snap.ret_1m:.2%}，触发P1极端波动",
            evidence={"ret_1m": snap.ret_1m, "price": snap.price,
                      "threshold": t.price_change_p1},
            dedupe_key=f"MKT_EXTREME_VOL_P1:{asset.value}",
        ))
    elif abs_ret_1m >= t.price_change_p2:
        hits.append(RuleHit(
            rule_id="MKT_EXTREME_VOL_P2",
            asset=asset,
            severity=Severity.P2,
            confidence=min(1.0, abs_ret_1m / t.price_change_p1),
            description=f"{asset.value} 1m价格变动 {snap.ret_1m:.2%}，触发P2波动预警",
            evidence={"ret_1m": snap.ret_1m, "price": snap.price,
                      "threshold": t.price_change_p2},
            dedupe_key=f"MKT_EXTREME_VOL_P2:{asset.value}",
        ))

    if abs(snap.vol_z_1m) > t.vol_z_spike:
        hits.append(RuleHit(
            rule_id="MKT_VOL_Z_SPIKE",
            asset=asset,
            severity=Severity.P2,
            confidence=min(1.0, abs(snap.vol_z_1m) / 5.0),
            description=f"{asset.value} 1m波动率 z-score={snap.vol_z_1m:.1f}，显著偏离均值",
            evidence={"vol_z_1m": snap.vol_z_1m, "threshold": t.vol_z_spike},
            dedupe_key=f"MKT_VOL_Z:{asset.value}",
        ))

    # -----------------------------------------------------------------------
    # Derivatives rules
    # -----------------------------------------------------------------------
    if snap.liq_5m_usd >= t.liq_usd_p1:
        hits.append(RuleHit(
            rule_id="DERIV_CASCADE_LIQ_P1",
            asset=asset,
            severity=Severity.P1,
            confidence=1.0,
            description=f"{asset.value} 5分钟爆仓金额 ${snap.liq_5m_usd:,.0f}，触发级联清算P1预警",
            evidence={"liq_5m_usd": snap.liq_5m_usd, "threshold": t.liq_usd_p1},
            dedupe_key=f"DERIV_CASCADE_LIQ:{asset.value}",
        ))

    abs_oi_delta = abs(snap.oi_delta_15m_pct)
    if abs_oi_delta >= t.oi_delta_p2:
        hits.append(RuleHit(
            rule_id="DERIV_OI_BUILDUP",
            asset=asset,
            severity=Severity.P2,
            confidence=min(1.0, abs_oi_delta / 0.20),
            description=f"{asset.value} 15m OI变动 {snap.oi_delta_15m_pct:.1%}，持仓异常积累",
            evidence={"oi_delta_15m_pct": snap.oi_delta_15m_pct, "threshold": t.oi_delta_p2},
            dedupe_key=f"DERIV_OI:{asset.value}",
        ))

    if abs(snap.funding_z) >= t.funding_z_p2:
        hits.append(RuleHit(
            rule_id="DERIV_FUNDING_SQUEEZE",
            asset=asset,
            severity=Severity.P2,
            confidence=min(1.0, abs(snap.funding_z) / 4.0),
            description=f"{asset.value} 资金费率 z-score={snap.funding_z:.1f}，多空极端拥挤",
            evidence={"funding_z": snap.funding_z, "threshold": t.funding_z_p2},
            dedupe_key=f"DERIV_FUNDING:{asset.value}",
        ))

    # -----------------------------------------------------------------------
    # Early-warning rules
    # -----------------------------------------------------------------------
    has_actionable_hit = any(
        hit.severity in {Severity.P1, Severity.P2}
        and not hit.rule_id.startswith("QA_")
        for hit in hits
    )
    if not has_actionable_hit:
        profile = build_early_warning_profile(snap, t, recent_snapshots=recent_snapshots)
        signals = profile["signals"]
        common_evidence = {
            "early_warning_score": profile["score"],
            "early_warning_signal_count": profile["signal_count"],
            "early_warning_signals": signals,
            "early_warning_adaptive_thresholds": profile["adaptive_thresholds"],
            "early_warning_persistent_signals": profile["persistent_signals"],
            "early_warning_trend_hits": profile["trend_hits"],
            "early_warning_trend_confirmed": profile["trend_confirmed"],
            "early_warning_candidate": profile["candidate"],
            "early_warning_min_score": profile["min_score"],
            "early_warning_min_signals": profile["min_signals"],
            "early_warning_single_signal_min_score": profile["single_signal_min_score"],
            "multi_signal_candidate": profile["multi_signal_candidate"],
            "strong_single_signal_candidate": profile["strong_single_signal_candidate"],
            "direction_aligned": profile["direction_aligned"],
        }

        if "price_drift" in signals:
            hits.append(RuleHit(
                rule_id="EW_PRICE_DRIFT_5M",
                asset=asset,
                severity=Severity.P3,
                confidence=profile["score"],
                description=(
                    f"{asset.value} 5m price drift {snap.ret_5m:.2%}; "
                    "possible early trend buildup"
                ),
                evidence={
                    **common_evidence,
                    "signal_score": signals["price_drift"],
                    "ret_5m": snap.ret_5m,
                    "threshold": t.early_warning_ret_5m,
                    "formal_p2_threshold": t.price_change_p2,
                },
                dedupe_key=f"EW_PRICE_DRIFT_5M:{asset.value}",
            ))

        if "vol_expansion" in signals:
            hits.append(RuleHit(
                rule_id="EW_VOL_EXPANSION",
                asset=asset,
                severity=Severity.P3,
                confidence=profile["score"],
                description=(
                    f"{asset.value} volatility z-score {snap.vol_z_1m:.1f}; "
                    "volatility is expanding before formal P2 threshold"
                ),
                evidence={
                    **common_evidence,
                    "signal_score": signals["vol_expansion"],
                    "vol_z_1m": snap.vol_z_1m,
                    "threshold": t.early_warning_vol_z,
                    "formal_p2_threshold": t.vol_z_spike,
                },
                dedupe_key=f"EW_VOL_EXPANSION:{asset.value}",
            ))

        if "oi_buildup" in signals:
            hits.append(RuleHit(
                rule_id="EW_OI_BUILDUP",
                asset=asset,
                severity=Severity.P3,
                confidence=profile["score"],
                description=(
                    f"{asset.value} 15m OI changed {snap.oi_delta_15m_pct:.1%}; "
                    "leverage is building before formal P2 threshold"
                ),
                evidence={
                    **common_evidence,
                    "signal_score": signals["oi_buildup"],
                    "oi_delta_15m_pct": snap.oi_delta_15m_pct,
                    "threshold": t.early_warning_oi_delta,
                    "formal_p2_threshold": t.oi_delta_p2,
                },
                dedupe_key=f"EW_OI_BUILDUP:{asset.value}",
            ))

        if "funding_bias" in signals:
            hits.append(RuleHit(
                rule_id="EW_FUNDING_BIAS",
                asset=asset,
                severity=Severity.P3,
                confidence=profile["score"],
                description=(
                    f"{asset.value} funding z-score {snap.funding_z:.1f}; "
                    "positioning bias is forming before formal P2 threshold"
                ),
                evidence={
                    **common_evidence,
                    "signal_score": signals["funding_bias"],
                    "funding_z": snap.funding_z,
                    "threshold": t.early_warning_funding_z,
                    "formal_p2_threshold": t.funding_z_p2,
                },
                dedupe_key=f"EW_FUNDING_BIAS:{asset.value}",
            ))

    # -----------------------------------------------------------------------
    # Data quality rules
    # -----------------------------------------------------------------------
    if snap.source_stale:
        hits.append(RuleHit(
            rule_id="QA_SOURCE_STALE",
            asset=asset,
            severity=Severity.P3,
            confidence=0.5,
            description=f"{asset.value} 数据源超过2分钟无更新，置信度降级",
            evidence={"source_stale": True},
            dedupe_key=f"QA_STALE:{asset.value}",
        ))

    if snap.cross_source_conflict:
        hits.append(RuleHit(
            rule_id="QA_CROSS_SOURCE_CONFLICT",
            asset=asset,
            severity=Severity.P2,
            confidence=0.8,
            description=f"{asset.value} 现货/标记价格偏差超过{t.cross_source_conflict_pct:.1%}，可能存在数据异常",
            evidence={"cross_source_conflict": True,
                      "threshold": t.cross_source_conflict_pct},
            dedupe_key=f"QA_CONFLICT:{asset.value}",
        ))

    return hits
