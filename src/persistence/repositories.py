"""Repository pattern — thin async wrappers over SQLAlchemy."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.models import (
    Asset, CaseStatus, Decision, HumanReviewAction, RawEvent,
    FeatureSnapshot, RiskAlert, RiskCase, RuleHit, Severity,
)
from .database import (
    AsyncSessionLocal, FeatureSnapshotRow, HumanReviewRow,
    RawEventRow, RiskAlertRow, RiskCaseRow,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _case_from_row(row: RiskCaseRow) -> RiskCase:
    hits_data = json.loads(row.rule_hits_json or "[]")
    hits = [RuleHit(**h) for h in hits_data]
    return RiskCase(
        case_id=row.case_id,
        asset=Asset(row.asset),
        created_at=row.created_at,
        updated_at=row.updated_at,
        status=CaseStatus(row.status),
        rule_hits=hits,
        decision=Decision(row.decision) if row.decision else None,
        summary_zh=row.summary_zh or "",
        severity=Severity(row.severity) if row.severity else None,
        is_coordinator_case=bool(getattr(row, "is_coordinator_case", False)),
        historical_context_zh=getattr(row, "historical_context_zh", "") or "",
        risk_quantification_zh=getattr(row, "risk_quantification_zh", "") or "",
        suppression_reason=getattr(row, "suppression_reason", None),
    )


# ---------------------------------------------------------------------------
# RawEvent
# ---------------------------------------------------------------------------

async def save_raw_event(event: RawEvent) -> None:
    async with AsyncSessionLocal() as s:
        row = RawEventRow(
            event_id=event.event_id,
            trace_id=event.trace_id,
            asset=event.asset.value,
            source=event.source,
            event_type=event.event_type,
            event_ts=event.event_ts,
            ingest_ts=event.ingest_ts,
            payload=event.payload,
            dedupe_key=event.dedupe_key,
        )
        s.add(row)
        await s.commit()


# ---------------------------------------------------------------------------
# FeatureSnapshot
# ---------------------------------------------------------------------------

async def save_feature_snapshot(snap: FeatureSnapshot) -> None:
    async with AsyncSessionLocal() as s:
        row = FeatureSnapshotRow(**snap.model_dump())
        s.add(row)
        await s.commit()


async def get_latest_snapshot(asset: Asset) -> FeatureSnapshot | None:
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(FeatureSnapshotRow)
            .where(FeatureSnapshotRow.asset == asset.value)
            .order_by(FeatureSnapshotRow.window_end.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return FeatureSnapshot(**{
            c.name: getattr(row, c.name)
            for c in FeatureSnapshotRow.__table__.columns
        })


async def get_recent_snapshots(asset: Asset, n: int = 10) -> list[FeatureSnapshot]:
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(FeatureSnapshotRow)
            .where(FeatureSnapshotRow.asset == asset.value)
            .order_by(FeatureSnapshotRow.window_end.desc())
            .limit(n)
        )
        rows = result.scalars().all()
        return [
            FeatureSnapshot(**{c.name: getattr(row, c.name) for c in FeatureSnapshotRow.__table__.columns})
            for row in rows
        ]


# ---------------------------------------------------------------------------
# RiskCase
# ---------------------------------------------------------------------------

async def save_risk_case(case: RiskCase) -> None:
    async with AsyncSessionLocal() as s:
        existing = await s.get(RiskCaseRow, case.case_id)
        hits_json = json.dumps([h.model_dump(mode="json") for h in case.rule_hits])
        if existing:
            existing.updated_at = case.updated_at
            existing.status = case.status.value
            existing.rule_hits_json = hits_json
            existing.decision = case.decision.value if case.decision else None
            existing.summary_zh = case.summary_zh
            existing.severity = case.severity.value if case.severity else None
            existing.is_coordinator_case = case.is_coordinator_case
            existing.historical_context_zh = case.historical_context_zh
            existing.risk_quantification_zh = case.risk_quantification_zh
            existing.suppression_reason = case.suppression_reason
        else:
            s.add(RiskCaseRow(
                case_id=case.case_id,
                asset=case.asset.value,
                created_at=case.created_at,
                updated_at=case.updated_at,
                status=case.status.value,
                rule_hits_json=hits_json,
                decision=case.decision.value if case.decision else None,
                summary_zh=case.summary_zh,
                severity=case.severity.value if case.severity else None,
                is_coordinator_case=case.is_coordinator_case,
                historical_context_zh=case.historical_context_zh,
                risk_quantification_zh=case.risk_quantification_zh,
                suppression_reason=case.suppression_reason,
            ))
        await s.commit()


async def get_risk_case(case_id: str) -> RiskCase | None:
    async with AsyncSessionLocal() as s:
        row = await s.get(RiskCaseRow, case_id)
        return _case_from_row(row) if row else None


async def list_risk_cases(
    asset: Asset | None = None,
    limit: int = 50,
    *,
    include_suppressed: bool = False,
) -> list[RiskCase]:
    async with AsyncSessionLocal() as s:
        q = select(RiskCaseRow).order_by(RiskCaseRow.created_at.desc()).limit(limit)
        if asset:
            q = q.where(RiskCaseRow.asset == asset.value)
        if not include_suppressed:
            q = q.where(RiskCaseRow.status != CaseStatus.SUPPRESSED.value)
        result = await s.execute(q)
        return [_case_from_row(r) for r in result.scalars()]


async def find_active_case_by_dedupe_key(
    asset: Asset,
    dedupe_key: str,
    *,
    within_seconds: int = 300,
) -> RiskCase | None:
    cutoff = datetime.utcnow() - timedelta(seconds=within_seconds)
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(RiskCaseRow)
            .where(RiskCaseRow.asset == asset.value)
            .where(RiskCaseRow.created_at >= cutoff)
            .where(RiskCaseRow.status != CaseStatus.CLOSED.value)
            .order_by(RiskCaseRow.created_at.desc())
        )
        for row in result.scalars():
            hits_data = json.loads(row.rule_hits_json or "[]")
            if any(hit.get("dedupe_key") == dedupe_key for hit in hits_data):
                return _case_from_row(row)
    return None


async def list_recent_risk_cases(
    within_seconds: int,
    *,
    severities: list[Severity] | None = None,
    include_coordinator: bool = True,
    limit: int = 100,
) -> list[RiskCase]:
    cutoff = datetime.utcnow() - timedelta(seconds=within_seconds)
    async with AsyncSessionLocal() as s:
        q = (
            select(RiskCaseRow)
            .where(RiskCaseRow.created_at >= cutoff)
            .order_by(RiskCaseRow.created_at.desc())
            .limit(limit)
        )
        if severities:
            q = q.where(RiskCaseRow.severity.in_([severity.value for severity in severities]))
        if not include_coordinator:
            q = q.where(RiskCaseRow.is_coordinator_case.is_(False))
        result = await s.execute(q)
        return [_case_from_row(r) for r in result.scalars()]


async def find_recent_coordinator_case(
    assets: list[Asset],
    *,
    within_seconds: int = 300,
) -> RiskCase | None:
    if len(assets) < 2:
        return None
    recent_cases = await list_recent_risk_cases(
        within_seconds,
        include_coordinator=True,
        limit=50,
    )
    expected_assets = {asset.value for asset in assets}
    for case in recent_cases:
        if not case.is_coordinator_case:
            continue
        hit_assets = {hit.asset.value for hit in case.rule_hits}
        if expected_assets.issubset(hit_assets):
            return case
    return None


# ---------------------------------------------------------------------------
# RiskAlert
# ---------------------------------------------------------------------------

async def save_risk_alert(alert: RiskAlert) -> None:
    async with AsyncSessionLocal() as s:
        existing = await s.execute(
            select(RiskAlertRow).where(RiskAlertRow.idempotency_key == alert.idempotency_key)
        )
        if existing.scalar_one_or_none():
            return  # idempotent — already sent
        s.add(RiskAlertRow(
            alert_id=alert.alert_id,
            case_id=alert.case_id,
            revision=alert.revision,
            severity=alert.severity.value,
            title=alert.title,
            body_zh=alert.body_zh,
            idempotency_key=alert.idempotency_key,
            created_at=alert.created_at,
            channels_sent=json.dumps(alert.channels_sent),
        ))
        await s.commit()


async def list_alerts(case_id: str | None = None, limit: int = 50) -> list[RiskAlert]:
    async with AsyncSessionLocal() as s:
        q = select(RiskAlertRow).order_by(RiskAlertRow.created_at.desc()).limit(limit)
        if case_id:
            q = q.where(RiskAlertRow.case_id == case_id)
        result = await s.execute(q)
        rows = result.scalars().all()
        return [
            RiskAlert(
                alert_id=r.alert_id,
                case_id=r.case_id,
                revision=r.revision,
                severity=Severity(r.severity),
                title=r.title,
                body_zh=r.body_zh,
                idempotency_key=r.idempotency_key,
                created_at=r.created_at,
                channels_sent=json.loads(r.channels_sent or "[]"),
            )
            for r in rows
        ]


async def get_recent_alerts(asset: Asset, limit: int = 5) -> list[RiskAlert]:
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(RiskAlertRow)
            .join(RiskCaseRow, RiskCaseRow.case_id == RiskAlertRow.case_id)
            .where(RiskCaseRow.asset == asset.value)
            .order_by(RiskAlertRow.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        return [
            RiskAlert(
                alert_id=r.alert_id,
                case_id=r.case_id,
                revision=r.revision,
                severity=Severity(r.severity),
                title=r.title,
                body_zh=r.body_zh,
                idempotency_key=r.idempotency_key,
                created_at=r.created_at,
                channels_sent=json.loads(r.channels_sent or "[]"),
            )
            for r in rows
        ]


async def get_rule_version_info() -> dict:
    from src.rules.config import registry

    return {
        "version_tag": registry.active_version,
        "thresholds": registry.thresholds.model_dump(),
    }


# ---------------------------------------------------------------------------
# HumanReviewAction
# ---------------------------------------------------------------------------

async def save_review_action(action: HumanReviewAction) -> None:
    async with AsyncSessionLocal() as s:
        s.add(HumanReviewRow(
            action_id=action.action_id,
            case_id=action.case_id,
            reviewer=action.reviewer,
            action=action.action.value,
            comment=action.comment,
            created_at=action.created_at,
        ))
        await s.commit()
