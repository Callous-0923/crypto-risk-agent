"""FastAPI routes."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.domain.models import Asset, HumanReviewAction, ReviewAction
from src.graph.orchestrator import resume_case
from src.notification.dispatcher import register_ws_client, unregister_ws_client
from src.persistence.repositories import (
    get_risk_case, list_alerts, list_risk_cases, save_review_action,
)

router = APIRouter()


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



# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

@router.get("/cases")
async def get_cases(asset: Optional[str] = None, limit: int = 50, include_suppressed: bool = False):
    a = Asset(asset) if asset else None
    cases = await list_risk_cases(a, limit, include_suppressed=include_suppressed)
    return [c.model_dump(mode="json") for c in cases]


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


@router.post("/cases/{case_id}/resume")
async def resume_human_review(case_id: str, req: ResumeRequest):
    case = await get_risk_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    approved = req.action.lower() == "approve"
    action = HumanReviewAction(
        case_id=case_id,
        reviewer=req.reviewer,
        action=ReviewAction.APPROVE if approved else ReviewAction.REJECT,
        comment=req.comment,
    )
    await save_review_action(action)
    await resume_case(case_id, approved=approved, comment=req.comment)

    from src.observability.metrics import human_review_total, pending_review_gauge
    from src.persistence.repositories import list_risk_cases
    from src.domain.models import CaseStatus
    human_review_total.labels(action=req.action.lower()).inc()
    pending = await list_risk_cases(limit=10000)
    pending_review_gauge.set(sum(1 for c in pending if c.status == CaseStatus.PENDING_REVIEW))

    return {"status": "resumed", "approved": approved}


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
