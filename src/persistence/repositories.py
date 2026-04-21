"""Repository pattern — thin async wrappers over SQLAlchemy."""
from __future__ import annotations

import json
from datetime import datetime

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
            ))
        await s.commit()


async def get_risk_case(case_id: str) -> RiskCase | None:
    async with AsyncSessionLocal() as s:
        row = await s.get(RiskCaseRow, case_id)
        return _case_from_row(row) if row else None


async def list_risk_cases(asset: Asset | None = None, limit: int = 50) -> list[RiskCase]:
    async with AsyncSessionLocal() as s:
        q = select(RiskCaseRow).order_by(RiskCaseRow.created_at.desc()).limit(limit)
        if asset:
            q = q.where(RiskCaseRow.asset == asset.value)
        result = await s.execute(q)
        return [_case_from_row(r) for r in result.scalars()]


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
