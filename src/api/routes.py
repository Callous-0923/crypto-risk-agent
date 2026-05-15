"""FastAPI routes."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.domain.models import Asset, CaseStatus, Decision, HumanReviewAction, ReviewAction
from src.graph.orchestrator import resume_case
from src.notification.dispatcher import register_ws_client, unregister_ws_client
from src.persistence.repositories import (
    count_risk_cases, get_risk_case, list_alerts, list_risk_cases,
    save_review_action, save_risk_case,
)

router = APIRouter()


class RiskModelLabelRequest(BaseModel):
    days: int | None = None
    horizon_minutes: int = 60
    max_snapshots_per_asset: int = 10000
    max_items: int | None = 500
    use_llm_judge: bool = True
    force: bool = False


class RiskModelTrainRequest(BaseModel):
    days: int | None = None
    horizon_minutes: int = 60
    max_snapshots_per_asset: int = 10000
    max_label_items: int | None = 1000
    use_llm_judge: bool = True
    min_samples: int = 100


class HistoricalBackfillRequest(BaseModel):
    start: datetime
    end: datetime
    assets: list[str] | None = None
    interval: str = "1m"
    market_types: list[str] | None = None


class HistoricalRiskModelTrainRequest(BaseModel):
    start: datetime
    end: datetime
    assets: list[str] | None = None
    horizon_minutes: int = 60
    max_snapshots_per_asset: int | None = None
    max_label_items: int | None = 50000
    min_samples: int = 1000
    p2_quantile: float = 0.95
    p1_quantile: float = 0.995


# ---------------------------------------------------------------------------
# Observability: /metrics + /health
# ---------------------------------------------------------------------------

@router.get("/agent/status")
async def agent_status():
    from src.api.app import get_agent_status

    return get_agent_status()


@router.post("/agent/start")
async def agent_start():
    from src.api.app import start_ingestion

    try:
        await start_ingestion()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "started"}


@router.post("/agent/stop")
async def agent_stop():
    from src.api.app import stop_ingestion

    try:
        await stop_ingestion()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"status": "stopped"}


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint — Grafana 从这里拉指标。"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health/detailed")
async def health_check_detailed():
    """
    详细健康检查：包含 LLM 可用性、数据源状态、队列积压等综合指标。
    供 Demo 评审时快速判断系统各组件状态。
    """
    from datetime import datetime, timezone
    from src.api.app import get_agent_status
    from src.features.builder import get_feature_builder
    from src.domain.models import Asset
    from src.observability.metrics import (
        event_bus_queue_size, pending_review_gauge,
        ws_reconnect_total, ingest_event_total,
    )
    from src.persistence.database import AsyncSessionLocal

    # 1. Agent 运行状态
    agent_info = get_agent_status()

    # 2. 各资产数据新鲜度
    builder = get_feature_builder()
    asset_status = {}
    for asset in Asset:
        snap = builder.get_snapshot(asset)
        age = (datetime.now(timezone.utc) - snap.window_end.replace(tzinfo=timezone.utc)).total_seconds() \
              if snap.window_end and snap.price > 0 else 9999
        asset_status[asset.value] = {
            "price": snap.price,
            "age_seconds": round(age),
            "fresh": snap.price > 0 and age < 120,
            "oi_delta_15m_pct": snap.oi_delta_15m_pct,
            "funding_z": snap.funding_z,
        }

    # 3. LLM 可用性快速检查
    llm_status = "unknown"
    try:
        from src.core.proxy import build_openai_client
        from src.core.config import settings as app_settings
        client = build_openai_client(service="health_check")
        llm_status = "configured" if app_settings.ark_api_key else "missing_api_key"
    except Exception:
        llm_status = "unavailable"

    # 4. 队列积压
    queue_size = event_bus_queue_size._value.get()

    # 5. 待审核积压 (Prometheus gauge)
    pending_reviews = pending_review_gauge._value.get()

    # 6. 数据库行统计
    db_stats = {}
    try:
        from sqlalchemy import func, select
        from src.persistence.database import (
            FeatureSnapshotRow, RiskCaseRow, RiskAlertRow,
            LLMCallRow, HumanReviewRow, QualityMetricEventRow,
        )
        async with AsyncSessionLocal() as s:
            db_stats["feature_snapshots"] = (await s.execute(
                select(func.count(FeatureSnapshotRow.snapshot_id))
            )).scalar() or 0
            db_stats["risk_cases"] = (await s.execute(
                select(func.count(RiskCaseRow.case_id))
            )).scalar() or 0
            db_stats["risk_alerts"] = (await s.execute(
                select(func.count(RiskAlertRow.alert_id))
            )).scalar() or 0
            db_stats["llm_calls"] = (await s.execute(
                select(func.count(LLMCallRow.call_id))
            )).scalar() or 0
            db_stats["review_actions"] = (await s.execute(
                select(func.count(HumanReviewRow.action_id))
            )).scalar() or 0
            db_stats["quality_events"] = (await s.execute(
                select(func.count(QualityMetricEventRow.event_id))
            )).scalar() or 0
    except Exception as exc:
        db_stats = {"error": str(exc)}

    # 7. ML 模型状态
    ml_status = "unavailable"
    try:
        from src.ml.risk_model import model_status
        ml_info = model_status()
        ml_status = "loaded" if ml_info.get("available") else "not_trained"
    except Exception:
        ml_status = "error"

    return {
        "status": "ok",
        "agent": agent_info,
        "llm": {"status": llm_status},
        "ml_model": {"status": ml_status},
        "queue": {
            "event_bus_queue_size": queue_size,
            "pending_reviews": pending_reviews,
        },
        "assets": asset_status,
        "database": db_stats,
    }


@router.get("/health")
async def health_check():
    """
    心跳接口：检查数据流是否存活。
    - 最近 2 分钟有事件入库 → ok
    - 否则 → stale（可触发外部告警）
    """
    from datetime import datetime, timezone
    from src.api.app import get_agent_status
    from src.features.builder import get_feature_builder
    from src.domain.models import Asset
    from src.observability.metrics import event_bus_queue_size

    builder = get_feature_builder()
    asset_status = {}
    all_ok = True
    for asset in Asset:
        snap = builder.get_snapshot(asset)
        age = (datetime.now(timezone.utc) - snap.window_end.replace(tzinfo=timezone.utc)).total_seconds() \
              if snap.window_end else 9999
        ok = snap.price > 0 and age < 120
        asset_status[asset.value] = {"price": snap.price, "age_seconds": round(age), "ok": ok}
        if not ok:
            all_ok = False

    return {
        "status": "ok" if all_ok else "stale",
        "agent": get_agent_status(),
        "assets": asset_status,
        "event_bus_queue": event_bus_queue_size._value.get(),
    }


@router.get("/market/candles")
async def get_market_candles(asset: str, interval: str = "1m", limit: int = 60):
    from src.market.candles import load_market_candles

    try:
        asset_enum = Asset(asset.upper())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported asset: {asset}") from exc

    if interval != "1m":
        raise HTTPException(status_code=400, detail="Only 1m interval is supported")

    bounded_limit = max(1, min(limit, 180))
    candles = await load_market_candles(asset_enum, interval=interval, limit=bounded_limit)
    return [candle.model_dump(mode="json") for candle in candles]


@router.get("/simulation/scenarios")
async def get_simulation_scenarios():
    from src.simulation.runner import list_simulation_scenarios

    return [scenario.model_dump(mode="json") for scenario in list_simulation_scenarios()]


@router.get("/simulation/runs/latest")
async def get_latest_simulation():
    from src.simulation.runner import get_latest_simulation_run

    latest = get_latest_simulation_run()
    if latest is None:
        return None
    return latest.model_dump(mode="json")


@router.get("/simulation/runs")
async def list_simulation_runs(limit: int = 10):
    from src.simulation.runner import get_recent_simulation_runs

    bounded_limit = max(1, min(limit, 20))
    runs = get_recent_simulation_runs()[:bounded_limit]
    return [run.model_dump(mode="json") for run in runs]


@router.post("/simulation/runs")
async def run_simulation(req: SimulationRunRequest):
    from src.simulation.runner import run_simulation_scenario

    try:
        run = await run_simulation_scenario(req.scenario_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return run.model_dump(mode="json")


@router.post("/ml/risk-model/label")
async def label_risk_model_snapshots(req: RiskModelLabelRequest):
    from src.evaluation.offline import load_snapshot_series
    from src.ml.risk_model import ensure_llm_judge_labels

    horizon_seconds = max(60, min(req.horizon_minutes, 240) * 60)
    series_by_asset = await load_snapshot_series(
        days=req.days,
        max_snapshots_per_asset=max(100, min(req.max_snapshots_per_asset, 100000)),
    )
    return await ensure_llm_judge_labels(
        series_by_asset,
        horizon_seconds=horizon_seconds,
        max_items=req.max_items,
        use_llm=req.use_llm_judge,
        force=req.force,
    )


@router.post("/ml/risk-model/train")
async def train_risk_model_endpoint(req: RiskModelTrainRequest):
    from src.evaluation.offline import load_snapshot_series
    from src.ml.risk_model import train_risk_model

    horizon_seconds = max(60, min(req.horizon_minutes, 240) * 60)
    series_by_asset = await load_snapshot_series(
        days=req.days,
        max_snapshots_per_asset=max(100, min(req.max_snapshots_per_asset, 100000)),
    )
    try:
        return await train_risk_model(
            series_by_asset,
            horizon_seconds=horizon_seconds,
            max_label_items=req.max_label_items,
            use_llm_judge=req.use_llm_judge,
            min_samples=req.min_samples,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/ml/risk-model/status")
async def get_risk_model_status():
    from src.ml.risk_model import model_status
    from src.persistence.repositories import count_model_labels

    status = model_status(force_reload=True)
    status["label_count"] = await count_model_labels()
    return status


@router.get("/ml/risk-model/predictions")
async def get_risk_model_predictions(asset: Optional[str] = None, limit: int = 100):
    from src.persistence.repositories import list_model_predictions

    try:
        asset_enum = Asset(asset.upper()) if asset else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported asset: {asset}") from exc
    return await list_model_predictions(limit=max(1, min(limit, 500)), asset=asset_enum)


def _parse_assets(values: list[str] | None) -> list[Asset]:
    if not values:
        return list(Asset)
    try:
        return [Asset(value.upper()) for value in values]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported asset in {values}") from exc


@router.post("/ml/historical-data/backfill")
async def backfill_historical_market_data(req: HistoricalBackfillRequest):
    from src.ml.historical_data import backfill_binance_public_klines

    if req.end <= req.start:
        raise HTTPException(status_code=400, detail="end must be after start")
    if req.interval != "1m":
        raise HTTPException(status_code=400, detail="only 1m interval is supported")
    return await backfill_binance_public_klines(
        assets=_parse_assets(req.assets),
        start=req.start.replace(tzinfo=None),
        end=req.end.replace(tzinfo=None),
        interval=req.interval,
        market_types=req.market_types,
    )


@router.get("/ml/historical-data/status")
async def historical_market_data_status():
    from src.persistence.repositories import count_historical_market_bars

    rows = []
    for asset in Asset:
        for market_type in ["spot", "futures_um"]:
            rows.append({
                "asset": asset.value,
                "market_type": market_type,
                "rows": await count_historical_market_bars(asset=asset, market_type=market_type),
            })
    return {"source": "binance_public", "rows": rows}


@router.post("/ml/risk-model/train-historical")
async def train_historical_risk_model_endpoint(req: HistoricalRiskModelTrainRequest):
    from src.ml.historical_training import train_historical_risk_model

    if req.end <= req.start:
        raise HTTPException(status_code=400, detail="end must be after start")
    horizon_seconds = max(60, min(req.horizon_minutes, 240) * 60)
    try:
        return await train_historical_risk_model(
            assets=_parse_assets(req.assets),
            start=req.start.replace(tzinfo=None),
            end=req.end.replace(tzinfo=None),
            horizon_seconds=horizon_seconds,
            max_snapshots_per_asset=req.max_snapshots_per_asset,
            max_label_items=req.max_label_items,
            min_samples=req.min_samples,
            p2_quantile=max(0.5, min(req.p2_quantile, 0.995)),
            p1_quantile=max(0.8, min(req.p1_quantile, 0.999)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/evaluation/offline")
async def get_offline_evaluation(
    days: int | None = None,
    horizon_minutes: int = 15,
    max_snapshots_per_asset: int = 1200,
    max_episodes: int = 600,
    label_p1_pct: float | None = None,
    label_p2_pct: float | None = None,
):
    from src.evaluation.offline import run_offline_weak_label_evaluation

    bounded_horizon_seconds = max(60, min(horizon_minutes, 120) * 60)
    bounded_snapshots = max(100, min(max_snapshots_per_asset, 10000))
    bounded_episodes = max(50, min(max_episodes, 5000))

    def _bounded_pct(value: float | None) -> float | None:
        if value is None:
            return None
        return max(0.001, min(float(value), 0.5))

    return await run_offline_weak_label_evaluation(
        days=days,
        horizon_seconds=bounded_horizon_seconds,
        max_snapshots_per_asset=bounded_snapshots,
        max_episodes=bounded_episodes,
        label_price_change_p1=_bounded_pct(label_p1_pct),
        label_price_change_p2=_bounded_pct(label_p2_pct),
    )


@router.get("/evaluation/early-warning/tune")
async def tune_early_warning(
    days: int | None = None,
    horizon_minutes: int = 60,
    max_snapshots_per_asset: int = 10000,
    max_episodes: int = 1000,
    label_p1_pct: float | None = 0.03,
    label_p2_pct: float | None = 0.01,
):
    from src.evaluation.offline import load_snapshot_series, tune_early_warning_thresholds
    from src.rules.config import registry

    bounded_horizon_seconds = max(60, min(horizon_minutes, 120) * 60)
    bounded_snapshots = max(100, min(max_snapshots_per_asset, 10000))
    bounded_episodes = max(50, min(max_episodes, 5000))

    series_by_asset = await load_snapshot_series(
        days=days,
        max_snapshots_per_asset=bounded_snapshots,
    )
    return tune_early_warning_thresholds(
        series_by_asset,
        base_thresholds=registry.thresholds,
        horizon_seconds=bounded_horizon_seconds,
        max_episodes=bounded_episodes,
        label_price_change_p1=label_p1_pct,
        label_price_change_p2=label_p2_pct,
    )


@router.get("/evaluation/summary")
async def get_evaluation_summary(days: int | None = None):
    from collections import Counter, defaultdict
    from datetime import datetime, timedelta

    from sqlalchemy import func, select

    from src.core.config import settings
    from src.persistence.database import (
        AsyncSessionLocal,
        FeatureSnapshotRow,
        HumanReviewRow,
        LLMCallRow,
        QualityMetricEventRow,
        RawEventRow,
        RiskAlertRow,
        RiskCaseRow,
    )

    now = datetime.utcnow()
    cutoff = now - timedelta(days=days) if days else None

    def _since(column):
        return column >= cutoff if cutoff else True

    async with AsyncSessionLocal() as s:
        raw_events = await s.scalar(select(func.count()).select_from(RawEventRow).where(_since(RawEventRow.ingest_ts))) or 0
        feature_snapshots = await s.scalar(select(func.count()).select_from(FeatureSnapshotRow).where(_since(FeatureSnapshotRow.window_end))) or 0
        risk_cases = await s.scalar(select(func.count()).select_from(RiskCaseRow).where(_since(RiskCaseRow.created_at))) or 0
        risk_alerts = await s.scalar(select(func.count()).select_from(RiskAlertRow).where(_since(RiskAlertRow.created_at))) or 0
        human_reviews = await s.scalar(select(func.count()).select_from(HumanReviewRow).where(_since(HumanReviewRow.created_at))) or 0

        first_event_at = await s.scalar(select(func.min(RawEventRow.ingest_ts)).where(_since(RawEventRow.ingest_ts)))
        last_event_at = await s.scalar(select(func.max(RawEventRow.ingest_ts)).where(_since(RawEventRow.ingest_ts)))
        avg_ingest_lag = await s.scalar(select(func.avg(FeatureSnapshotRow.ingest_lag_ms)).where(_since(FeatureSnapshotRow.window_end)))
        ingest_lag_rows = (
            await s.execute(select(FeatureSnapshotRow.ingest_lag_ms).where(_since(FeatureSnapshotRow.window_end)))
        ).scalars().all()

        case_rows = (
            await s.execute(select(RiskCaseRow).where(_since(RiskCaseRow.created_at)))
        ).scalars().all()
        alert_rows = (
            await s.execute(select(RiskAlertRow).where(_since(RiskAlertRow.created_at)))
        ).scalars().all()
        review_rows = (
            await s.execute(select(HumanReviewRow).where(_since(HumanReviewRow.created_at)))
        ).scalars().all()
        quality_event_rows = (
            await s.execute(select(QualityMetricEventRow).where(_since(QualityMetricEventRow.created_at)))
        ).scalars().all()
        llm_rows = (
            await s.execute(select(LLMCallRow).where(_since(LLMCallRow.created_at)))
        ).scalars().all()

        asset_rows = (
            await s.execute(select(RawEventRow.asset, func.count()).where(_since(RawEventRow.ingest_ts)).group_by(RawEventRow.asset))
        ).all()
        snapshot_asset_rows = (
            await s.execute(select(FeatureSnapshotRow.asset, func.count()).where(_since(FeatureSnapshotRow.window_end)).group_by(FeatureSnapshotRow.asset))
        ).all()

    cases_by_id = {row.case_id: row for row in case_rows}
    severity_breakdown = Counter(row.severity or "unknown" for row in case_rows)
    decision_breakdown = Counter(row.decision or "none" for row in case_rows)
    status_breakdown = Counter(row.status or "unknown" for row in case_rows)
    action_breakdown = Counter(row.action for row in review_rows)

    manual_review_cases = decision_breakdown.get(Decision.MANUAL_REVIEW.value, 0)
    suppressed_cases = status_breakdown.get("suppressed", 0)
    emitted_cases = len(alert_rows)
    pending_review_cases = status_breakdown.get("pending_review", 0)
    coordinator_cases = sum(1 for row in case_rows if row.is_coordinator_case)
    memory_enriched_cases = sum(1 for row in case_rows if row.historical_context_zh or row.risk_quantification_zh)
    alert_case_ids = {row.case_id for row in alert_rows}

    case_to_alert_seconds = []
    alert_detection_latency_seconds = []

    def _parse_dt(value):
        if not value:
            return None
        if isinstance(value, datetime):
            return value.replace(tzinfo=None) if value.tzinfo else value
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            return None

    def _case_rule_hits(row):
        try:
            return json.loads(row.rule_hits_json or "[]")
        except json.JSONDecodeError:
            return []

    def _case_has_rule_prefix(row, prefix: str) -> bool:
        return any((hit.get("rule_id") or "").startswith(prefix) for hit in _case_rule_hits(row))

    for alert in alert_rows:
        case = cases_by_id.get(alert.case_id)
        if case:
            case_to_alert_seconds.append(max(0.0, (alert.created_at - case.created_at).total_seconds()))
            fired_times = [
                parsed
                for parsed in (_parse_dt(hit.get("fired_at")) for hit in _case_rule_hits(case))
                if parsed is not None
            ]
            if fired_times:
                alert_detection_latency_seconds.append(
                    max(0.0, (alert.created_at - min(fired_times)).total_seconds())
                )

    review_turnaround_seconds = []
    for review in review_rows:
        case = cases_by_id.get(review.case_id)
        if case:
            review_turnaround_seconds.append(max(0.0, (review.created_at - case.created_at).total_seconds()))

    def _avg(values):
        return sum(values) / len(values) if values else None

    def _p95(values):
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))
        return ordered[index]

    top_rules_counter: Counter[str] = Counter()
    top_rules_severity: dict[str, Counter[str]] = defaultdict(Counter)
    for row in case_rows:
        hits = _case_rule_hits(row)
        for hit in hits:
            rule_id = hit.get("rule_id")
            severity = hit.get("severity") or row.severity or "unknown"
            if rule_id:
                top_rules_counter[rule_id] += 1
                top_rules_severity[rule_id][severity] += 1

    early_warning_case_rows = [row for row in case_rows if _case_has_rule_prefix(row, "EW_")]
    early_warning_case_ids = {row.case_id for row in early_warning_case_rows}
    early_warning_pending_cases = sum(1 for row in early_warning_case_rows if row.status == CaseStatus.PENDING_REVIEW.value)
    early_warning_review_rows = [row for row in review_rows if row.case_id in early_warning_case_ids]
    latest_early_warning_review_by_case = {}
    for row in sorted(early_warning_review_rows, key=lambda item: item.created_at):
        latest_early_warning_review_by_case[row.case_id] = row
    early_warning_reviewed_cases = len(latest_early_warning_review_by_case)
    early_warning_approved_cases = sum(
        1 for row in latest_early_warning_review_by_case.values()
        if row.action in {"approve", "escalate"}
    )
    early_warning_rejected_cases = sum(
        1 for row in latest_early_warning_review_by_case.values()
        if row.action == "reject"
    )

    raw_by_asset = {asset: count for asset, count in asset_rows}
    snapshots_by_asset = {asset: count for asset, count in snapshot_asset_rows}
    case_asset_counter = Counter(row.asset for row in case_rows)
    alert_asset_counter = Counter(cases_by_id.get(alert.case_id).asset for alert in alert_rows if cases_by_id.get(alert.case_id))
    review_asset_counter = Counter(cases_by_id.get(review.case_id).asset for review in review_rows if cases_by_id.get(review.case_id))
    assets = sorted(set(raw_by_asset) | set(snapshots_by_asset) | set(case_asset_counter))

    try:
        from src.evaluation.offline import run_offline_weak_label_evaluation

        offline_evaluation = await run_offline_weak_label_evaluation(
            days=days,
            horizon_seconds=3600,
            min_gap_seconds=300,
            max_snapshots_per_asset=10000,
            max_episodes=1000,
            label_price_change_p1=0.03,
            label_price_change_p2=0.01,
            include_tuning=True,
            tuning_max_episodes=600,
        )
        offline_evaluation["available"] = True
    except Exception as exc:
        offline_evaluation = {
            "available": False,
            "labeling_mode": "weak_label_future_price_move",
            "error": str(exc),
        }

    coverage_hours = None
    if first_event_at and last_event_at:
        coverage_hours = max(0.0, (last_event_at - first_event_at).total_seconds() / 3600)

    source_stale = await _count_feature_quality(_since, "source_stale")
    cross_source_conflicts = await _count_feature_quality(_since, "cross_source_conflict")
    coverage_seconds = (coverage_hours or 0) * 3600
    events_per_second = raw_events / coverage_seconds if coverage_seconds else 0
    snapshots_per_second = feature_snapshots / coverage_seconds if coverage_seconds else 0
    cases_per_hour = risk_cases / (coverage_hours or 0) if coverage_hours else 0
    alerts_per_hour = risk_alerts / (coverage_hours or 0) if coverage_hours else 0

    latest_review_by_case = {}
    for row in sorted(review_rows, key=lambda item: item.created_at):
        latest_review_by_case[row.case_id] = row
    reviewed_case_count = len(latest_review_by_case)
    approved_review_cases = sum(1 for row in latest_review_by_case.values() if row.action in {"approve", "escalate"})
    rejected_review_cases = sum(1 for row in latest_review_by_case.values() if row.action == "reject")
    precision_proxy = approved_review_cases / reviewed_case_count if reviewed_case_count else None
    false_positive_proxy_rate = rejected_review_cases / reviewed_case_count if reviewed_case_count else None

    p1_alerts_by_asset: dict[str, list] = defaultdict(list)
    for row in alert_rows:
        if row.severity == "P1":
            p1_alerts_by_asset[row.case_id and cases_by_id.get(row.case_id).asset if cases_by_id.get(row.case_id) else ""].append(row)
    miss_horizon_seconds = 900
    eligible_non_alert_cases = [
        row for row in case_rows
        if row.severity in {"P1", "P2"}
        and row.case_id not in alert_case_ids
        and row.status != CaseStatus.PENDING_REVIEW.value
    ]
    potential_missed_cases = []
    for case in eligible_non_alert_cases:
        horizon = case.created_at + timedelta(seconds=miss_horizon_seconds)
        if any(case.created_at < alert.created_at <= horizon for alert in p1_alerts_by_asset.get(case.asset, [])):
            potential_missed_cases.append(case)
    missed_alert_proxy_rate = (
        len(potential_missed_cases) / len(eligible_non_alert_cases)
        if eligible_non_alert_cases else None
    )
    recall_proxy = 1 - missed_alert_proxy_rate if missed_alert_proxy_rate is not None else None

    quality_event_counter = Counter(row.event_type for row in quality_event_rows)
    alert_duplicate_suppressed = int(sum(
        row.value or 1
        for row in quality_event_rows
        if row.event_type == "alert_duplicate_suppressed"
    ))
    p2_case_aggregated = int(sum(
        row.value or 1
        for row in quality_event_rows
        if row.event_type == "p2_case_aggregated"
    ))
    duplicate_suppressed = alert_duplicate_suppressed + p2_case_aggregated
    alert_candidates = risk_alerts + alert_duplicate_suppressed
    case_candidates = risk_cases + duplicate_suppressed
    alert_dedupe_rate = alert_duplicate_suppressed / alert_candidates if alert_candidates else 0
    dedupe_rate = duplicate_suppressed / case_candidates if case_candidates else 0

    llm_operation_counter = Counter(row.operation for row in llm_rows)
    llm_total_cost = sum(row.estimated_cost_usd or 0 for row in llm_rows)
    llm_prompt_tokens = sum(row.prompt_tokens or 0 for row in llm_rows)
    llm_completion_tokens = sum(row.completion_tokens or 0 for row in llm_rows)
    llm_total_tokens = sum(row.total_tokens or 0 for row in llm_rows)
    llm_success_calls = sum(1 for row in llm_rows if row.status == "success")
    llm_error_calls = len(llm_rows) - llm_success_calls
    llm_operation_breakdown = []
    for operation, count in llm_operation_counter.most_common(5):
        rows = [row for row in llm_rows if row.operation == operation]
        llm_operation_breakdown.append({
            "operation": operation,
            "calls": count,
            "tokens": sum(row.total_tokens or 0 for row in rows),
            "cost_usd": sum(row.estimated_cost_usd or 0 for row in rows),
            "avg_duration_ms": _avg([row.duration_ms or 0 for row in rows]),
        })

    rules_baseline_alert_candidates = sum(
        1 for row in case_rows
        if row.severity in {"P1", "P2"}
    ) + duplicate_suppressed
    baseline_alert_reduction_rate = (
        max(0, rules_baseline_alert_candidates - risk_alerts) / rules_baseline_alert_candidates
        if rules_baseline_alert_candidates else 0
    )
    review_gate_precision_lift = (
        1.0 - precision_proxy
        if precision_proxy is not None and approved_review_cases > 0 else None
    )
    offline_core_metrics = (
        offline_evaluation.get("core_metrics", {})
        if offline_evaluation.get("available") else {}
    )
    early_warning_precision_runtime = (
        early_warning_approved_cases / early_warning_reviewed_cases
        if early_warning_reviewed_cases else None
    )
    early_warning_false_positive_runtime = (
        early_warning_rejected_cases / early_warning_reviewed_cases
        if early_warning_reviewed_cases else None
    )
    core_quality_metrics = {
        "early_warning_recall": offline_core_metrics.get("early_warning_recall"),
        "early_warning_precision": (
            offline_core_metrics.get("early_warning_precision")
            if offline_core_metrics.get("early_warning_precision") is not None
            else early_warning_precision_runtime
        ),
        "early_warning_avg_lead_time_seconds": offline_core_metrics.get("early_warning_avg_lead_time_seconds"),
        "early_warning_conversion_rate": (
            offline_core_metrics.get("early_warning_conversion_rate")
            if offline_core_metrics.get("early_warning_conversion_rate") is not None
            else early_warning_precision_runtime
        ),
        "formal_alert_recall": offline_core_metrics.get("formal_alert_recall"),
        "formal_alert_false_positive_rate": (
            false_positive_proxy_rate
            if false_positive_proxy_rate is not None
            else offline_core_metrics.get("formal_alert_false_positive_rate")
        ),
        "p95_alert_latency_seconds": _p95(alert_detection_latency_seconds),
        "pending_review_cases": pending_review_cases,
    }
    early_warning_tuning = offline_evaluation.get("early_warning_tuning", {}) if offline_evaluation.get("available") else {}
    early_warning_best_config = (
        early_warning_tuning.get("top_configs", [None])[0]
        if early_warning_tuning.get("top_configs") else None
    )

    summary_en = (
        f"Processed {raw_events} live market events across {', '.join(assets) or 'no assets'}, "
        f"producing {feature_snapshots} trusted feature snapshots. Generated {risk_cases} risk cases "
        f"({severity_breakdown.get('P1', 0)} P1 and {severity_breakdown.get('P2', 0)} P2) and emitted {risk_alerts} alerts."
    )
    summary_zh = (
        f"项目真实运行期间累计处理 {raw_events} 条市场事件，覆盖 {', '.join(assets) or '暂无资产'}，"
        f"沉淀 {feature_snapshots} 个可信特征快照；共生成 {risk_cases} 个风险案例，输出 {risk_alerts} 条告警。"
    )

    return {
        "generated_at": f"{now.isoformat()}Z",
        "window": {
            "days": days,
            "label": "all_time" if days is None else f"last_{days}d",
            "first_event_at": first_event_at.isoformat() if first_event_at else None,
            "last_event_at": last_event_at.isoformat() if last_event_at else None,
            "coverage_hours": coverage_hours or 0,
        },
        "sample_sizes": {
            "raw_events": raw_events,
            "feature_snapshots": feature_snapshots,
            "risk_cases": risk_cases,
            "risk_alerts": risk_alerts,
            "human_reviews": human_reviews,
            "llm_calls": len(llm_rows),
            "quality_events": len(quality_event_rows),
            "active_assets": len(assets),
        },
        "coverage": {
            "assets": assets,
            "asset_breakdown": [
                {
                    "asset": asset,
                    "raw_events": raw_by_asset.get(asset, 0),
                    "snapshots": snapshots_by_asset.get(asset, 0),
                    "cases": case_asset_counter.get(asset, 0),
                    "alerts": alert_asset_counter.get(asset, 0),
                    "reviews": review_asset_counter.get(asset, 0),
                }
                for asset in assets
            ],
        },
        "case_metrics": {
            "severity_breakdown": dict(severity_breakdown),
            "decision_breakdown": dict(decision_breakdown),
            "status_breakdown": dict(status_breakdown),
            "manual_review_cases": manual_review_cases,
            "manual_review_rate": manual_review_cases / risk_cases if risk_cases else 0,
            "suppressed_cases": suppressed_cases,
            "suppressed_rate": suppressed_cases / risk_cases if risk_cases else 0,
            "emitted_cases": emitted_cases,
            "emitted_alert_rate": emitted_cases / risk_cases if risk_cases else 0,
            "pending_review_cases": pending_review_cases,
            "coordinator_cases": coordinator_cases,
            "early_warning_cases": len(early_warning_case_rows),
            "early_warning_pending_cases": early_warning_pending_cases,
            "memory_enriched_cases": memory_enriched_cases,
            "memory_enrichment_rate": memory_enriched_cases / risk_cases if risk_cases else 0,
        },
        "latency_metrics": {
            "avg_snapshot_ingest_lag_ms": avg_ingest_lag,
            "p95_snapshot_ingest_lag_ms": _p95(ingest_lag_rows),
            "avg_case_to_alert_seconds": _avg(case_to_alert_seconds),
            "p95_case_to_alert_seconds": _p95(case_to_alert_seconds),
            "avg_alert_detection_seconds": _avg(alert_detection_latency_seconds),
            "p95_alert_detection_seconds": _p95(alert_detection_latency_seconds),
            "avg_review_turnaround_seconds": _avg(review_turnaround_seconds),
            "p95_review_turnaround_seconds": _p95(review_turnaround_seconds),
        },
        "data_quality": {
            "source_stale_snapshots": source_stale,
            "source_stale_rate": source_stale / feature_snapshots if feature_snapshots else 0,
            "cross_source_conflicts": cross_source_conflicts,
            "cross_source_conflict_rate": cross_source_conflicts / feature_snapshots if feature_snapshots else 0,
        },
        "human_review": {
            "actions_total": human_reviews,
            "action_breakdown": dict(action_breakdown),
            "approve_rate": action_breakdown.get("approve", 0) / human_reviews if human_reviews else 0,
            "reject_rate": action_breakdown.get("reject", 0) / human_reviews if human_reviews else 0,
            "escalate_rate": action_breakdown.get("escalate", 0) / human_reviews if human_reviews else 0,
        },
        "top_rules": [
            {
                "rule_id": rule_id,
                "count": count,
                "severity_breakdown": dict(top_rules_severity[rule_id]),
            }
            for rule_id, count in top_rules_counter.most_common(5)
        ],
        "resume_ready": {
            "summary_zh": summary_zh,
            "summary_en": summary_en,
            "highlights_zh": [
                summary_zh,
                f"案例指标：P1 {severity_breakdown.get('P1', 0)} 个、P2 {severity_breakdown.get('P2', 0)} 个，告警转化率 {risk_alerts / risk_cases:.1%}。" if risk_cases else "暂无风险案例。",
                f"人工审核 {human_reviews} 次，批准率 {action_breakdown.get('approve', 0) / human_reviews:.1%}。" if human_reviews else "暂无人工审核记录。",
                f"离线弱标注提前预警召回率 {offline_evaluation.get('risk_detection_policy', {}).get('recall'):.1%}，样本 {offline_evaluation.get('episodes', 0)} 个。" if offline_evaluation.get("available") and offline_evaluation.get("risk_detection_policy", {}).get("recall") is not None else "离线弱标注评测需要更多历史快照。",
            ],
            "highlights_en": [
                summary_en,
                f"Case mix: {severity_breakdown.get('P1', 0)} P1 and {severity_breakdown.get('P2', 0)} P2, with {risk_alerts / risk_cases:.1%} alert conversion." if risk_cases else "No risk cases yet.",
                f"Captured {human_reviews} review actions with {action_breakdown.get('approve', 0) / human_reviews:.1%} approval rate." if human_reviews else "No human review actions yet.",
                f"Runtime quality proxies: false-positive proxy {false_positive_proxy_rate:.1%}, missed-alert proxy {missed_alert_proxy_rate:.1%}, dedupe rate {dedupe_rate:.1%}." if false_positive_proxy_rate is not None and missed_alert_proxy_rate is not None else "Runtime quality proxies need more reviewed or closed non-alert cases.",
                f"Offline weak-label early-warning recall: {offline_evaluation.get('risk_detection_policy', {}).get('recall'):.1%} on {offline_evaluation.get('episodes', 0)} episodes." if offline_evaluation.get("available") and offline_evaluation.get("risk_detection_policy", {}).get("recall") is not None else "Offline weak-label evaluation needs more historical snapshots.",
            ],
        },
        "quality_metrics": {
            "labeling_mode": "runtime_proxy",
            "reviewed_cases": reviewed_case_count,
            "approved_review_cases": approved_review_cases,
            "rejected_review_cases": rejected_review_cases,
            "precision_proxy": precision_proxy,
            "false_positive_proxy_rate": false_positive_proxy_rate,
            "eligible_non_alert_cases": len(eligible_non_alert_cases),
            "potential_missed_cases": len(potential_missed_cases),
            "miss_horizon_seconds": miss_horizon_seconds,
            "missed_alert_proxy_rate": missed_alert_proxy_rate,
            "recall_proxy": recall_proxy,
            "alert_candidates": alert_candidates,
            "case_candidates": case_candidates,
            "alert_duplicate_suppressed": alert_duplicate_suppressed,
            "p2_case_aggregated": p2_case_aggregated,
            "duplicate_suppressed": duplicate_suppressed,
            "alert_dedupe_rate": alert_dedupe_rate,
            "dedupe_rate": dedupe_rate,
            "quality_event_breakdown": dict(quality_event_counter),
        },
        "core_quality_metrics": core_quality_metrics,
        "early_warning_metrics": {
            "cases": len(early_warning_case_rows),
            "pending_cases": early_warning_pending_cases,
            "reviewed_cases": early_warning_reviewed_cases,
            "approved_cases": early_warning_approved_cases,
            "rejected_cases": early_warning_rejected_cases,
            "runtime_policy": "p3_suppressed_no_alert",
            "runtime_precision": early_warning_precision_runtime,
            "runtime_false_positive_rate": early_warning_false_positive_runtime,
            "created_events": quality_event_counter.get("early_warning_created", 0),
            "aggregated_events": quality_event_counter.get("early_warning_aggregated", 0),
            "best_offline_config": early_warning_best_config,
        },
        "offline_evaluation": offline_evaluation,
        "throughput_metrics": {
            "events_per_second": events_per_second,
            "snapshots_per_second": snapshots_per_second,
            "cases_per_hour": cases_per_hour,
            "alerts_per_hour": alerts_per_hour,
        },
        "llm_cost_metrics": {
            "calls": len(llm_rows),
            "success_calls": llm_success_calls,
            "error_calls": llm_error_calls,
            "prompt_tokens": llm_prompt_tokens,
            "completion_tokens": llm_completion_tokens,
            "total_tokens": llm_total_tokens,
            "estimated_cost_usd": llm_total_cost,
            "cost_per_case_usd": llm_total_cost / risk_cases if risk_cases else None,
            "cost_per_alert_usd": llm_total_cost / risk_alerts if risk_alerts else None,
            "usage_available": any((row.total_tokens or 0) > 0 for row in llm_rows),
            "cost_configured": (
                settings.llm_input_cost_per_million_tokens > 0
                or settings.llm_output_cost_per_million_tokens > 0
            ),
            "operation_breakdown": llm_operation_breakdown,
        },
        "baseline_comparison": {
            "baseline": "rules_only_emit_p1_p2_cases",
            "rules_baseline_alert_candidates": rules_baseline_alert_candidates,
            "actual_emitted_alerts": risk_alerts,
            "alert_reduction_rate": baseline_alert_reduction_rate,
            "prevented_false_positive_proxy": rejected_review_cases,
            "review_gate_precision_lift": review_gate_precision_lift,
        },
    }


@router.post("/maintenance/compact-p2-pending")
async def compact_p2_pending_cases():
    from src.observability.metrics import pending_review_gauge
    from src.persistence.repositories import compact_pending_p2_cases, list_risk_cases

    result = await compact_pending_p2_cases()
    pending = await list_risk_cases(limit=10000)
    pending_review_gauge.set(sum(1 for c in pending if c.status == CaseStatus.PENDING_REVIEW))
    return result


async def _count_feature_quality(_since, field: str) -> int:
    from sqlalchemy import func, select

    from src.persistence.database import AsyncSessionLocal, FeatureSnapshotRow

    column = getattr(FeatureSnapshotRow, field)
    async with AsyncSessionLocal() as s:
        return await s.scalar(
            select(func.count()).select_from(FeatureSnapshotRow).where(_since(FeatureSnapshotRow.window_end)).where(column.is_(True))
        ) or 0



# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

@router.get("/cases")
async def get_cases(
    asset: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    include_suppressed: bool = False,
    status: Optional[str] = None,
    paginated: bool = False,
):
    try:
        a = Asset(asset.upper()) if asset else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported asset: {asset}") from exc

    try:
        status_filter = CaseStatus(status) if status else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported case status: {status}") from exc

    bounded_limit = max(1, min(limit, 100))
    bounded_offset = max(0, offset)
    cases = await list_risk_cases(
        a,
        bounded_limit,
        include_suppressed=include_suppressed,
        status=status_filter,
        offset=bounded_offset,
    )
    items = [c.model_dump(mode="json") for c in cases]
    if not paginated:
        return items

    total = await count_risk_cases(
        a,
        include_suppressed=include_suppressed,
        status=status_filter,
    )
    return {
        "items": items,
        "total": total,
        "limit": bounded_limit,
        "offset": bounded_offset,
        "has_more": bounded_offset + bounded_limit < total,
    }


@router.get("/cases/{case_id}")
async def get_case(case_id: str):
    case = await get_risk_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Human review resume
# ---------------------------------------------------------------------------

class ResumeRequest(BaseModel):
    reviewer: str
    action: str  # approve / reject
    comment: str = ""


class SimulationRunRequest(BaseModel):
    scenario_id: str


@router.post("/cases/{case_id}/resume")
async def resume_human_review(case_id: str, req: ResumeRequest):
    from datetime import datetime

    case = await get_risk_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    normalized_action = req.action.lower().strip()
    if normalized_action not in {"approve", "reject"}:
        raise HTTPException(status_code=400, detail="action must be approve or reject")

    approved = normalized_action == "approve"
    action = HumanReviewAction(
        case_id=case_id,
        reviewer=req.reviewer,
        action=ReviewAction.APPROVE if approved else ReviewAction.REJECT,
        comment=req.comment,
    )
    await save_review_action(action)

    resume_error = None
    try:
        await resume_case(case_id, approved=approved, comment=req.comment)
    except Exception as exc:
        if approved:
            raise
        resume_error = str(exc)

    if not approved:
        case.status = CaseStatus.CLOSED
        case.updated_at = datetime.utcnow()
        await save_risk_case(case)

    from src.observability.metrics import human_review_total, pending_review_gauge
    from src.persistence.repositories import list_risk_cases
    human_review_total.labels(action=normalized_action).inc()
    pending = await list_risk_cases(limit=10000)
    pending_review_gauge.set(sum(1 for c in pending if c.status == CaseStatus.PENDING_REVIEW))

    return {
        "status": "resumed" if resume_error is None else "closed_without_graph_resume",
        "approved": approved,
        "resume_error": resume_error,
    }


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@router.get("/alerts")
async def get_alerts(case_id: Optional[str] = None, limit: int = 50):
    alerts = await list_alerts(case_id, limit)
    return [a.model_dump(mode="json") for a in alerts]


# ---------------------------------------------------------------------------
# WebSocket push
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rule version management
# ---------------------------------------------------------------------------

class PublishVersionRequest(BaseModel):
    version_tag: str          # e.g. "v2"
    operator: str             # 操作人
    reason: str = ""          # 变更原因
    # 阈值字段（只需传要修改的）
    price_change_p1: float | None = None
    price_change_p2: float | None = None
    oi_delta_p2: float | None = None
    liq_usd_p1: float | None = None
    funding_z_p2: float | None = None
    early_warning_ret_5m: float | None = None
    early_warning_oi_delta: float | None = None
    early_warning_funding_z: float | None = None
    early_warning_vol_z: float | None = None
    early_warning_min_score: float | None = None
    early_warning_min_signals: int | None = None
    early_warning_single_signal_min_score: float | None = None
    early_warning_persistence_window: int | None = None
    early_warning_persistence_hits: int | None = None
    early_warning_dynamic_baseline: bool | None = None
    early_warning_dynamic_history: int | None = None
    early_warning_dynamic_quantile: float | None = None
    early_warning_trend_window: int | None = None
    early_warning_min_trend_hits: int | None = None
    vol_z_spike: float | None = None
    cross_source_conflict_pct: float | None = None


@router.get("/rules/versions")
async def list_rule_versions():
    """查询所有规则版本列表。"""
    from sqlalchemy import select
    from src.persistence.database import AsyncSessionLocal, RuleVersionRow
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(RuleVersionRow).order_by(RuleVersionRow.version_id.desc())
        )
        rows = result.scalars().all()
    return [
        {
            "version_id": r.version_id,
            "version_tag": r.version_tag,
            "is_active": r.is_active,
            "created_by": r.created_by,
            "created_at": r.created_at.isoformat(),
            "description": r.description,
            "thresholds": r.thresholds,
        }
        for r in rows
    ]


@router.get("/rules/active")
async def get_active_rule_version():
    """查询当前生效的规则版本和阈值。"""
    from src.rules.config import registry
    return {
        "version_tag": registry.active_version,
        "thresholds": registry.thresholds.model_dump(),
    }


@router.post("/rules/publish")
async def publish_rule_version(req: PublishVersionRequest):
    """
    发布新规则版本（热更新，无需重启）。

    只需传入要修改的字段，其余字段继承当前 active 版本。
    变更会记录到 rule_change_log 审计表。
    """
    from src.rules.config import registry, RuleThresholds

    # 基于当前版本 + 本次修改字段合并出新阈值
    current = registry.thresholds.model_dump()
    updates = req.model_dump(
        exclude={"version_tag", "operator", "reason"},
        exclude_none=True,
    )
    merged = {**current, **updates}
    new_thresholds = RuleThresholds(**merged)

    try:
        await registry.publish_version(
            new_thresholds=new_thresholds,
            version_tag=req.version_tag,
            created_by=req.operator,
            reason=req.reason,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "published",
        "version_tag": req.version_tag,
        "active_thresholds": new_thresholds.model_dump(),
    }


@router.get("/rules/changelog")
async def get_rule_changelog(limit: int = 20):
    """查询规则变更审计日志。"""
    from sqlalchemy import select
    from src.persistence.database import AsyncSessionLocal, RuleChangeLogRow
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(RuleChangeLogRow).order_by(RuleChangeLogRow.changed_at.desc()).limit(limit)
        )
        rows = result.scalars().all()
    return [
        {
            "log_id": r.log_id,
            "from_version": r.from_version,
            "to_version": r.to_version,
            "changed_by": r.changed_by,
            "diff": r.diff,
            "changed_at": r.changed_at.isoformat(),
            "reason": r.reason,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Memory layer endpoints
# ---------------------------------------------------------------------------

@router.get("/memory/status")
async def memory_status():
    from src.core.config import settings
    if not settings.memory_enabled:
        return {"enabled": False}
    from src.memory.store import get_vector_store
    store = get_vector_store()
    return {
        "enabled": True,
        "embedding_model": settings.embedding_model,
        "vector_store_size": store.size,
        "similarity_threshold": settings.memory_similarity_threshold,
        "top_k": settings.memory_top_k,
    }


@router.post("/memory/distill")
async def memory_distill(lookback_days: int = 30):
    from src.core.config import settings
    if not settings.memory_enabled:
        raise HTTPException(400, "Memory layer is disabled")
    from src.memory.experience import distill_experiences
    insights = await distill_experiences(lookback_days=lookback_days)
    return {"new_insights": len(insights), "insights": insights}


@router.get("/memory/insights")
async def memory_insights(asset: Optional[str] = None, limit: int = 20):
    from src.memory.experience import get_insights
    results = await get_insights(asset=asset, limit=limit)
    return {"insights": results}


@router.post("/memory/learn-preferences")
async def memory_learn_preferences(lookback_days: int = 60):
    from src.core.config import settings
    if not settings.memory_enabled:
        raise HTTPException(400, "Memory layer is disabled")
    from src.memory.reviewer_preference import learn_preferences
    prefs = await learn_preferences(lookback_days=lookback_days)
    return {"preferences_learned": len(prefs), "preferences": prefs}


@router.get("/memory/preferences")
async def memory_preferences(asset: Optional[str] = None, limit: int = 20):
    from src.memory.reviewer_preference import get_reviewer_preferences
    results = await get_reviewer_preferences(asset=asset, limit=limit)
    return {"preferences": results}


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    q = register_ws_client()
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=30)
                await websocket.send_text(json.dumps(payload))
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        pass
    finally:
        unregister_ws_client(q)
