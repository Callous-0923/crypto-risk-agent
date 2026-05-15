"""Repository pattern — thin async wrappers over SQLAlchemy."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.models import (
    Asset, CaseStatus, Decision, HumanReviewAction, RawEvent,
    FeatureSnapshot, HistoricalMarketBar, RiskAlert, RiskCase, RuleHit, Severity,
)
from .database import (
    AsyncSessionLocal, FeatureSnapshotRow, HistoricalMarketBarRow, HumanReviewRow,
    QualityMetricEventRow, RawEventRow, RiskAlertRow, RiskCaseRow,
    RiskModelLabelRow, RiskModelPredictionRow,
    sqlite_database_path,
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


def _dedupe_key_from_hits_json(hits_json: str, severity: str | None = None) -> str:
    try:
        hits_data = json.loads(hits_json or "[]")
    except json.JSONDecodeError:
        return ""
    if severity:
        for hit in hits_data:
            if hit.get("severity") == severity and hit.get("dedupe_key"):
                return hit["dedupe_key"]
    for hit in hits_data:
        if hit.get("dedupe_key"):
            return hit["dedupe_key"]
    return ""


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


async def list_feature_snapshots(
    *,
    asset: Asset | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> list[FeatureSnapshot]:
    async with AsyncSessionLocal() as s:
        q = select(FeatureSnapshotRow).order_by(FeatureSnapshotRow.window_end.asc()).limit(limit).offset(offset)
        if asset:
            q = q.where(FeatureSnapshotRow.asset == asset.value)
        rows = (await s.execute(q)).scalars().all()
        return [
            FeatureSnapshot(**{c.name: getattr(row, c.name) for c in FeatureSnapshotRow.__table__.columns})
            for row in rows
        ]


async def save_historical_market_bars(bars: list[HistoricalMarketBar], *, chunk_size: int = 1000) -> int:
    if not bars:
        return 0
    rows = [bar.model_dump() for bar in bars]
    async with AsyncSessionLocal() as s:
        saved = 0
        for index in range(0, len(rows), chunk_size):
            chunk = rows[index:index + chunk_size]
            stmt = sqlite_insert(HistoricalMarketBarRow).values(chunk)
            update_columns = {
                key: getattr(stmt.excluded, key)
                for key in [
                    "close_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                    "trade_count",
                    "taker_buy_base_volume",
                    "taker_buy_quote_volume",
                ]
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["source", "market_type", "symbol", "interval", "open_time"],
                set_=update_columns,
            )
            await s.execute(stmt)
            saved += len(chunk)
        await s.commit()
    return saved


async def list_historical_market_bars(
    *,
    asset: Asset,
    market_type: str = "spot",
    source: str = "binance_public",
    interval: str = "1m",
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
) -> list[HistoricalMarketBar]:
    async with AsyncSessionLocal() as s:
        q = (
            select(HistoricalMarketBarRow)
            .where(HistoricalMarketBarRow.asset == asset.value)
            .where(HistoricalMarketBarRow.market_type == market_type)
            .where(HistoricalMarketBarRow.source == source)
            .where(HistoricalMarketBarRow.interval == interval)
            .order_by(HistoricalMarketBarRow.open_time.asc())
        )
        if start is not None:
            q = q.where(HistoricalMarketBarRow.open_time >= start)
        if end is not None:
            q = q.where(HistoricalMarketBarRow.open_time < end)
        if limit is not None:
            q = q.limit(limit)
        rows = (await s.execute(q)).scalars().all()
        return [
            HistoricalMarketBar(**{c.name: getattr(row, c.name) for c in HistoricalMarketBarRow.__table__.columns})
            for row in rows
        ]


async def count_historical_market_bars(
    *,
    asset: Asset | None = None,
    market_type: str | None = None,
    source: str = "binance_public",
) -> int:
    async with AsyncSessionLocal() as s:
        q = select(func.count()).select_from(HistoricalMarketBarRow).where(HistoricalMarketBarRow.source == source)
        if asset is not None:
            q = q.where(HistoricalMarketBarRow.asset == asset.value)
        if market_type is not None:
            q = q.where(HistoricalMarketBarRow.market_type == market_type)
        return await s.scalar(q) or 0


async def count_model_labels() -> int:
    async with AsyncSessionLocal() as s:
        return await s.scalar(select(func.count()).select_from(RiskModelLabelRow)) or 0


async def get_model_label(snapshot_id: str) -> RiskModelLabelRow | None:
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(RiskModelLabelRow).where(RiskModelLabelRow.snapshot_id == snapshot_id)
        )
        return result.scalar_one_or_none()


async def save_model_label(
    *,
    snapshot_id: str,
    asset: Asset,
    window_end: datetime,
    horizon_seconds: int,
    label: str,
    risk_probability: float,
    confidence: float,
    labeling_method: str,
    rationale: str,
    judge_payload: dict,
) -> None:
    async with AsyncSessionLocal() as s:
        existing = (
            await s.execute(
                select(RiskModelLabelRow).where(RiskModelLabelRow.snapshot_id == snapshot_id)
            )
        ).scalar_one_or_none()
        payload_json = json.dumps(judge_payload, ensure_ascii=False)
        if existing:
            existing.horizon_seconds = horizon_seconds
            existing.label = label
            existing.risk_probability = risk_probability
            existing.confidence = confidence
            existing.labeling_method = labeling_method
            existing.rationale = rationale
            existing.judge_payload_json = payload_json
            existing.created_at = datetime.utcnow()
        else:
            s.add(RiskModelLabelRow(
                label_id=str(uuid.uuid4()),
                snapshot_id=snapshot_id,
                asset=asset.value,
                window_end=window_end,
                horizon_seconds=horizon_seconds,
                label=label,
                risk_probability=risk_probability,
                confidence=confidence,
                labeling_method=labeling_method,
                rationale=rationale,
                judge_payload_json=payload_json,
                created_at=datetime.utcnow(),
            ))
        await s.commit()


async def list_model_labels(limit: int = 100000) -> list[RiskModelLabelRow]:
    async with AsyncSessionLocal() as s:
        rows = (
            await s.execute(
                select(RiskModelLabelRow)
                .order_by(RiskModelLabelRow.window_end.asc())
                .limit(limit)
            )
        ).scalars().all()
        return list(rows)


async def save_model_prediction(
    *,
    snapshot_id: str,
    asset: Asset,
    model_version: str,
    raw_probability: float,
    calibrated_probability: float,
    risk_level: str,
    top_features: list[dict],
) -> None:
    async with AsyncSessionLocal() as s:
        s.add(RiskModelPredictionRow(
            prediction_id=str(uuid.uuid4()),
            snapshot_id=snapshot_id,
            asset=asset.value,
            model_version=model_version,
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            risk_level=risk_level,
            top_features_json=json.dumps(top_features, ensure_ascii=False),
            created_at=datetime.utcnow(),
        ))
        await s.commit()


async def list_model_predictions(limit: int = 100, asset: Asset | None = None) -> list[dict]:
    async with AsyncSessionLocal() as s:
        q = select(RiskModelPredictionRow).order_by(RiskModelPredictionRow.created_at.desc()).limit(limit)
        if asset:
            q = q.where(RiskModelPredictionRow.asset == asset.value)
        rows = (await s.execute(q)).scalars().all()
        return [
            {
                "prediction_id": row.prediction_id,
                "snapshot_id": row.snapshot_id,
                "asset": row.asset,
                "model_version": row.model_version,
                "raw_probability": row.raw_probability,
                "calibrated_probability": row.calibrated_probability,
                "risk_level": row.risk_level,
                "top_features": json.loads(row.top_features_json or "[]"),
                "created_at": row.created_at.isoformat(),
            }
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
    status: CaseStatus | None = None,
    offset: int = 0,
) -> list[RiskCase]:
    async with AsyncSessionLocal() as s:
        q = select(RiskCaseRow).order_by(RiskCaseRow.created_at.desc()).limit(limit).offset(offset)
        if asset:
            q = q.where(RiskCaseRow.asset == asset.value)
        if status:
            q = q.where(RiskCaseRow.status == status.value)
        if not include_suppressed:
            q = q.where(RiskCaseRow.status != CaseStatus.SUPPRESSED.value)
        result = await s.execute(q)
        return [_case_from_row(r) for r in result.scalars()]


async def count_risk_cases(
    asset: Asset | None = None,
    *,
    include_suppressed: bool = False,
    status: CaseStatus | None = None,
) -> int:
    async with AsyncSessionLocal() as s:
        q = select(func.count()).select_from(RiskCaseRow)
        if asset:
            q = q.where(RiskCaseRow.asset == asset.value)
        if status:
            q = q.where(RiskCaseRow.status == status.value)
        if not include_suppressed:
            q = q.where(RiskCaseRow.status != CaseStatus.SUPPRESSED.value)
        return await s.scalar(q) or 0


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
            .where(RiskCaseRow.updated_at >= cutoff)
            .where(RiskCaseRow.status != CaseStatus.CLOSED.value)
            .order_by(RiskCaseRow.updated_at.desc())
        )
        for row in result.scalars():
            hits_data = json.loads(row.rule_hits_json or "[]")
            if any(hit.get("dedupe_key") == dedupe_key for hit in hits_data):
                return _case_from_row(row)
    return None


async def compact_pending_p2_cases() -> dict:
    """Suppress duplicate pending P2 cases, keeping the latest per asset/dedupe key."""
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(RiskCaseRow)
            .where(RiskCaseRow.status == CaseStatus.PENDING_REVIEW.value)
            .where(RiskCaseRow.severity == Severity.P2.value)
            .order_by(RiskCaseRow.updated_at.desc())
        )
        rows = list(result.scalars().all())
        groups: dict[tuple[str, str], list[RiskCaseRow]] = {}
        for row in rows:
            dedupe_key = _dedupe_key_from_hits_json(row.rule_hits_json, Severity.P2.value)
            if not dedupe_key:
                dedupe_key = f"case:{row.case_id}"
            groups.setdefault((row.asset, dedupe_key), []).append(row)

        suppressed_cases = 0
        compacted_groups = 0
        now = datetime.utcnow()
        for (asset, dedupe_key), group_rows in groups.items():
            if len(group_rows) <= 1:
                continue
            keep = group_rows[0]
            compacted_groups += 1
            for duplicate in group_rows[1:]:
                duplicate.status = CaseStatus.SUPPRESSED.value
                duplicate.updated_at = now
                duplicate.suppression_reason = f"p2_compacted_into:{keep.case_id}"
                s.add(QualityMetricEventRow(
                    event_id=str(uuid.uuid4()),
                    created_at=now,
                    event_type="p2_case_aggregated",
                    asset=asset,
                    severity=Severity.P2.value,
                    case_id=keep.case_id,
                    dedupe_key=dedupe_key,
                    value=1.0,
                    details_json=json.dumps(
                        {
                            "compacted_case_id": duplicate.case_id,
                            "mode": "maintenance_compaction",
                        },
                        ensure_ascii=False,
                    ),
                ))
                suppressed_cases += 1

        await s.commit()

    return {
        "pending_p2_before": len(rows),
        "compacted_groups": compacted_groups,
        "suppressed_cases": suppressed_cases,
        "pending_p2_after": len(rows) - suppressed_cases,
    }


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


async def save_quality_metric_event(
    event_type: str,
    *,
    asset: Asset | None = None,
    severity: Severity | None = None,
    case_id: str = "",
    dedupe_key: str = "",
    value: float = 1.0,
    details: dict | None = None,
) -> None:
    async with AsyncSessionLocal() as s:
        s.add(QualityMetricEventRow(
            event_id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            event_type=event_type,
            asset=asset.value if asset else None,
            severity=severity.value if severity else None,
            case_id=case_id,
            dedupe_key=dedupe_key,
            value=value,
            details_json=json.dumps(details or {}, ensure_ascii=False),
        ))
        await s.commit()


def save_quality_metric_event_sync(
    event_type: str,
    *,
    asset: Asset | None = None,
    severity: Severity | None = None,
    case_id: str = "",
    dedupe_key: str = "",
    value: float = 1.0,
    details: dict | None = None,
) -> None:
    db_path = sqlite_database_path()
    if not db_path:
        return
    try:
        with sqlite3.connect(db_path, timeout=0.2) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quality_metric_event (
                    event_id VARCHAR(36) PRIMARY KEY,
                    created_at DATETIME,
                    event_type VARCHAR(80),
                    asset VARCHAR(10),
                    severity VARCHAR(5),
                    case_id VARCHAR(36) DEFAULT '',
                    dedupe_key VARCHAR(200) DEFAULT '',
                    value FLOAT DEFAULT 1.0,
                    details_json TEXT DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                INSERT INTO quality_metric_event (
                    event_id, created_at, event_type, asset, severity,
                    case_id, dedupe_key, value, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    datetime.utcnow(),
                    event_type,
                    asset.value if asset else None,
                    severity.value if severity else None,
                    case_id,
                    dedupe_key,
                    value,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )
    except sqlite3.Error:
        return


def save_llm_call_record_sync(
    *,
    model: str,
    operation: str,
    status: str,
    duration_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
) -> None:
    db_path = sqlite_database_path()
    if not db_path:
        return
    try:
        with sqlite3.connect(db_path, timeout=2.0) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_call (
                    call_id VARCHAR(36) PRIMARY KEY,
                    created_at DATETIME,
                    model VARCHAR(100),
                    operation VARCHAR(100),
                    status VARCHAR(30),
                    duration_ms FLOAT DEFAULT 0.0,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    estimated_cost_usd FLOAT DEFAULT 0.0
                )
                """
            )
            conn.execute(
                """
                INSERT INTO llm_call (
                    call_id, created_at, model, operation, status, duration_ms,
                    prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    datetime.utcnow(),
                    model,
                    operation,
                    status,
                    duration_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated_cost_usd,
                ),
            )
    except sqlite3.Error:
        return


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

async def get_recent_review_actions(
    asset: Asset, limit: int = 5
) -> list[HumanReviewAction]:
    """Return the most recent human review actions for a given asset, newest first."""
    async with AsyncSessionLocal() as s:
        result = await s.execute(
            select(HumanReviewRow)
            .join(RiskCaseRow, RiskCaseRow.case_id == HumanReviewRow.case_id)
            .where(RiskCaseRow.asset == asset.value)
            .order_by(HumanReviewRow.created_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        from src.domain.models import ReviewAction
        return [
            HumanReviewAction(
                action_id=r.action_id,
                case_id=r.case_id,
                reviewer=r.reviewer,
                action=ReviewAction(r.action),
                comment=r.comment or "",
                created_at=r.created_at,
            )
            for r in rows
        ]


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
