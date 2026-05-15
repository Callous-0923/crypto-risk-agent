"""Demo 种子数据脚本 —— 预填充数据库以展示评测面板。

运行方式:
    python -m scripts.seed_demo_data

向 SQLite 数据库写入模拟的 FeatureSnapshot、RiskCase、RiskAlert、
LLMCall、HumanReviewAction、QualityMetricEvent 等数据，
确保前端评测汇总页有可视内容。
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta

from sqlalchemy import delete

from src.domain.models import Asset, CaseStatus, Decision, Severity
from src.persistence.database import (
    AsyncSessionLocal, Base, FeatureSnapshotRow, HumanReviewRow,
    LLMCallRow, QualityMetricEventRow, RawEventRow, RiskAlertRow,
    RiskCaseRow, RiskModelLabelRow, RiskModelPredictionRow,
)


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


def _minutes_ago(minutes: int) -> datetime:
    return _now() - timedelta(minutes=minutes)


async def _clear_all() -> None:
    """清空所有种子数据相关的表。"""
    async with AsyncSessionLocal() as session:
        tables = [
            RiskCaseRow, RiskAlertRow, FeatureSnapshotRow, RawEventRow,
            LLMCallRow, HumanReviewRow, QualityMetricEventRow,
            RiskModelLabelRow, RiskModelPredictionRow,
        ]
        for table in tables:
            await session.execute(delete(table))
        await session.commit()
    print("Cleared existing demo data.")


async def _seed_feature_snapshots() -> None:
    """写入过去 2 小时每资产 30s 间隔的模拟特征快照。"""
    assets = [Asset.BTC, Asset.ETH, Asset.SOL]
    base_prices = {Asset.BTC: 62000.0, Asset.ETH: 3000.0, Asset.SOL: 145.0}
    snapshots = []
    now = _now()

    for offset_min in range(120, 0, -1):
        for asset in assets:
            ts = now - timedelta(minutes=offset_min)
            base = base_prices[asset]
            # 添加随机波动以模拟真实行情
            import random
            noise = random.gauss(0, base * 0.001)
            price = base + noise
            ret_1m = random.gauss(0, 0.002)
            ret_5m = ret_1m * random.uniform(0.8, 2.5)
            ret_15m = ret_5m * random.uniform(0.7, 2.0)
            ret_30m = ret_15m * random.uniform(0.5, 1.8)
            ret_60m = ret_30m * random.uniform(0.5, 1.5)
            vol_z = random.gauss(0, 0.8)
            oi_delta = random.gauss(0, 0.02)
            liq = max(0, random.gauss(500000, 1000000))
            funding_z = random.gauss(0, 0.5)

            snapshots.append(FeatureSnapshotRow(
                snapshot_id=_uid(),
                asset=asset.value,
                window_end=ts,
                price=max(0.01, price),
                ret_1m=ret_1m,
                ret_5m=ret_5m,
                ret_15m=ret_15m,
                ret_30m=ret_30m,
                ret_60m=ret_60m,
                vol_z_1m=vol_z,
                realized_vol_5m=abs(random.gauss(0.003, 0.001)),
                realized_vol_15m=abs(random.gauss(0.005, 0.002)),
                realized_vol_60m=abs(random.gauss(0.008, 0.003)),
                price_range_pct_1m=abs(random.gauss(0.001, 0.0005)),
                close_position_1m=random.uniform(0.2, 0.8),
                max_drawdown_15m=abs(random.gauss(0.005, 0.003)),
                max_drawdown_60m=abs(random.gauss(0.01, 0.005)),
                max_runup_15m=abs(random.gauss(0.004, 0.002)),
                max_runup_60m=abs(random.gauss(0.008, 0.004)),
                atr_14=abs(random.gauss(0.006, 0.002)),
                volatility_regime_60m=random.uniform(0.8, 1.2),
                oi_delta_15m_pct=oi_delta,
                oi_delta_5m_pct=oi_delta * 0.4,
                oi_delta_60m_pct=oi_delta * 2.5,
                liq_5m_usd=liq,
                funding_z=funding_z,
                futures_basis_pct=random.gauss(0, 0.0005),
                basis_z_60m=random.gauss(0, 0.6),
                source_stale=False,
                cross_source_conflict=random.random() < 0.02,
                ingest_lag_ms=random.uniform(50, 500),
            ))

    async with AsyncSessionLocal() as session:
        for snap in snapshots:
            session.add(snap)
        await session.commit()
    print(f"Inserted {len(snapshots)} feature snapshots.")


async def _seed_risk_cases_and_alerts() -> None:
    """写入模拟的 RiskCase 和 RiskAlert 数据。"""
    assets = [Asset.BTC, Asset.ETH, Asset.SOL]
    cases = []

    rule_templates = [
        ("MKT_EXTREME_VOL_P1", Severity.P1, "价格1分钟内剧烈波动，触发P1极端风险"),
        ("MKT_EXTREME_VOL_P2", Severity.P2, "价格1分钟内显著波动，触发P2预警"),
        ("DERIV_CASCADE_LIQ_P1", Severity.P1, "5分钟内大额爆仓，触发级联清算P1"),
        ("DERIV_OI_BUILDUP", Severity.P2, "持仓量15分钟异常积累"),
        ("DERIV_FUNDING_SQUEEZE", Severity.P2, "资金费率极端偏离"),
        ("EW_PRICE_DRIFT_5M", Severity.P3, "5分钟价格漂移达到提前预警阈值"),
        ("EW_OI_BUILDUP", Severity.P3, "持仓量增长达到提前预警阈值"),
        ("QA_CROSS_SOURCE_CONFLICT", Severity.P2, "跨数据源价格偏差触发质量告警"),
    ]

    for i in range(60):
        asset = assets[i % 3]
        rule_idx = i % len(rule_templates)
        rule_id, severity, desc = rule_templates[rule_idx]

        if severity == Severity.P1:
            decision = Decision.EMIT
        elif severity == Severity.P2:
            decision = (
                Decision.MANUAL_REVIEW if i % 3 != 0
                else Decision.SUPPRESS
            )
        else:
            decision = Decision.SUPPRESS

        created_at = _minutes_ago(i * 3 + 5)
        case_id = _uid()
        rule_hits = [{
            "rule_id": rule_id,
            "asset": asset.value,
            "severity": severity.value,
            "confidence": 0.75 + (i % 5) * 0.05,
            "description": desc,
            "evidence": {"snapshot_price": 62000 + i * 100},
            "dedupe_key": f"{rule_id}:{asset.value}",
            "fired_at": created_at.isoformat(),
        }]

        cases.append(RiskCaseRow(
            case_id=case_id,
            asset=asset.value,
            created_at=created_at,
            updated_at=created_at,
            status=(
                CaseStatus.PENDING_REVIEW.value if decision == Decision.MANUAL_REVIEW
                else CaseStatus.CLOSED.value
            ),
            rule_hits_json=json.dumps(rule_hits),
            decision=decision.value,
            summary_zh=f"{asset.value} {desc}，置信度{0.75 + (i % 5) * 0.05:.0%}",
            severity=severity.value,
            is_coordinator_case=(i == 29),
            historical_context_zh=f"{asset.value} 近期已有相似告警，近5条case中3条涉及同类规则。",
            risk_quantification_zh=f"当前波动率偏离均值{abs(i % 3 - 1) + 1}个标准差，爆仓${1000000 + i * 50000:,.0f}。",
        ))

        # 为 EMIT 的 case 补 Alert
        if decision == Decision.EMIT:
            async with AsyncSessionLocal() as session:
                # 先存一份 (批量逻辑放一起)
                pass

    async with AsyncSessionLocal() as session:
        for case in cases:
            session.add(case)
        await session.commit()

    # 写入告警
    alerts = []
    for case_idx, case_row in enumerate(cases):
        if case_row.decision != Decision.EMIT.value:
            continue
        alert_id = _uid()
        alerts.append(RiskAlertRow(
            alert_id=alert_id,
            case_id=case_row.case_id,
            revision=1,
            severity=case_row.severity or Severity.P2.value,
            title=f"{case_row.asset} 风险告警 #{case_idx + 1}",
            body_zh=case_row.summary_zh or "风险告警触发，请关注。",
            idempotency_key=f"{case_row.case_id}:1:ws",
            created_at=case_row.created_at,
            channels_sent=json.dumps(["ws", "webhook"]),
        ))

    async with AsyncSessionLocal() as session:
        for alert in alerts:
            session.add(alert)
        await session.commit()
    print(f"Inserted {len(cases)} risk cases and {len(alerts)} alerts.")


async def _seed_llm_calls() -> None:
    """写入模拟的 LLM 调用记录。"""
    operations = [
        "technical_analysis", "macro_context", "summarizer",
        "review_assistance", "coordinator_summary", "experience_distill",
        "llm_judge_label",
    ]
    rows = []
    for i in range(120):
        op = operations[i % len(operations)]
        status = "success" if i % 10 != 0 else "timeout"
        prompt_tokens = 300 + i * 10
        completion_tokens = 100 + i * 3
        rows.append(LLMCallRow(
            call_id=_uid(),
            created_at=_minutes_ago(i * 2),
            model="doubao-seed-2-0-mini-260215",
            operation=op,
            status=status,
            duration_ms=800 + i * 5,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=0.0,
        ))
    async with AsyncSessionLocal() as session:
        for row in rows:
            session.add(row)
        await session.commit()
    print(f"Inserted {len(rows)} LLM call records.")


async def _seed_review_actions() -> None:
    """写入模拟的人工审核操作记录。"""
    rows = []
    for i in range(30):
        action = "approve" if i % 3 != 0 else "reject"
        rows.append(HumanReviewRow(
            action_id=_uid(),
            case_id=_uid(),
            reviewer="demo-reviewer",
            action=action,
            comment=f"模拟审核操作 #{i + 1}" if action == "reject" else "",
            created_at=_minutes_ago(i * 4 + 2),
        ))
    async with AsyncSessionLocal() as session:
        for row in rows:
            session.add(row)
        await session.commit()
    print(f"Inserted {len(rows)} human review actions.")


async def _seed_quality_metrics() -> None:
    """写入模拟的质量度量事件记录。"""
    events = [
        ("alert_sent", "BTC", "P1", 1.0),
        ("alert_sent", "ETH", "P1", 1.0),
        ("case_created", "BTC", "P2", 1.0),
        ("case_created", "ETH", "P2", 1.0),
        ("case_created", "SOL", "P2", 1.0),
        ("dedupe_suppressed", "BTC", "P2", 1.0),
        ("early_warning_fired", "BTC", "P3", 1.0),
        ("early_warning_fired", "ETH", "P3", 1.0),
        ("fatigue_suppressed", "BTC", "P2", 1.0),
        ("cross_asset_coordinated", "BTC", "P1", 1.0),
    ]
    rows = []
    for i, (event_type, asset, severity, value) in enumerate(events * 6):
        rows.append(QualityMetricEventRow(
            event_id=_uid(),
            created_at=_minutes_ago(i * 3),
            event_type=event_type,
            asset=asset,
            severity=severity,
            case_id=_uid(),
            dedupe_key=f"{event_type}:{asset}:{i}",
            value=value,
            details_json=json.dumps({"index": i}),
        ))
    async with AsyncSessionLocal() as session:
        for row in rows:
            session.add(row)
        await session.commit()
    print(f"Inserted {len(rows)} quality metric events.")


async def _seed_model_data() -> None:
    """写入模拟的 LightGBM 标签和预测记录。"""
    labels = []
    predictions = []
    for i in range(80):
        asset_val = ["BTC", "ETH", "SOL"][i % 3]
        window_end = _minutes_ago(i * 3)
        label = "p1" if i % 8 == 0 else "p2" if i % 5 == 0 else "none"
        labels.append(RiskModelLabelRow(
            label_id=_uid(),
            snapshot_id=_uid(),
            asset=asset_val,
            window_end=window_end,
            horizon_seconds=3600,
            label=label,
            risk_probability=0.8 if label == "p1" else 0.55 if label == "p2" else 0.15,
            confidence=0.7,
            labeling_method="llm_judge" if i % 3 == 0 else "deterministic",
            rationale=f"模拟弱标签：未来60分钟内价格变动达到{'P1' if label == 'p1' else 'P2' if label == 'p2' else 'none'} 阈值",
            judge_payload_json=json.dumps({"future_return": 0.05 if label == "p1" else 0.015}),
            created_at=window_end,
        ))
        predictions.append(RiskModelPredictionRow(
            prediction_id=_uid(),
            snapshot_id=labels[-1].snapshot_id,
            asset=asset_val,
            model_version="lgbm-20250101000000",
            raw_probability=0.7 if label == "p1" else 0.5 if label == "p2" else 0.1,
            calibrated_probability=0.75 if label == "p1" else 0.52 if label == "p2" else 0.12,
            risk_level=label.upper(),
            top_features_json=json.dumps([
                {"feature": "ret_5m", "importance": 0.15},
                {"feature": "oi_delta_15m_pct", "importance": 0.12},
                {"feature": "vol_z_1m", "importance": 0.10},
            ]),
            created_at=window_end,
        ))
    async with AsyncSessionLocal() as session:
        for row in labels:
            session.add(row)
        for row in predictions:
            session.add(row)
        await session.commit()
    print(f"Inserted {len(labels)} model labels and {len(predictions)} predictions.")


async def main() -> None:
    """执行全部种子数据填充。"""
    print("=" * 50)
    print("ETC Agent Demo Seed Data Generator")
    print("=" * 50)

    await _clear_all()
    await _seed_feature_snapshots()
    await _seed_risk_cases_and_alerts()
    await _seed_llm_calls()
    await _seed_review_actions()
    await _seed_quality_metrics()
    await _seed_model_data()

    print("=" * 50)
    print("Demo data seeding complete!")
    print("Restart backend and open http://localhost:5173 to view results.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
