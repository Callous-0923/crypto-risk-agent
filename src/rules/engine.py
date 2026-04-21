"""Rule engine — deterministic, no LLM involvement.

阈值从 RuleRegistry 读取，支持热更新和版本回放。
evaluate() 默认用当前 active 版本；
evaluate_with_version() 用指定 RuleThresholds 实例（供回放使用）。
"""
from __future__ import annotations

from src.domain.models import Asset, FeatureSnapshot, RuleHit, Severity
from src.rules.config import RuleThresholds, registry


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
