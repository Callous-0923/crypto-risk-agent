"""SQLAlchemy async engine + table definitions."""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.core.config import settings


class Base(DeclarativeBase):
    pass


class RawEventRow(Base):
    __tablename__ = "raw_event"
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(36), index=True)
    asset: Mapped[str] = mapped_column(String(10), index=True)
    source: Mapped[str] = mapped_column(String(50))
    event_type: Mapped[str] = mapped_column(String(50))
    event_ts: Mapped[datetime] = mapped_column(DateTime)
    ingest_ts: Mapped[datetime] = mapped_column(DateTime)
    payload: Mapped[dict] = mapped_column(JSON)
    dedupe_key: Mapped[str] = mapped_column(String(200), index=True)


class FeatureSnapshotRow(Base):
    __tablename__ = "feature_snapshot"
    snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset: Mapped[str] = mapped_column(String(10), index=True)
    window_end: Mapped[datetime] = mapped_column(DateTime, index=True)
    price: Mapped[float] = mapped_column(Float, default=0.0)
    ret_1m: Mapped[float] = mapped_column(Float, default=0.0)
    ret_5m: Mapped[float] = mapped_column(Float, default=0.0)
    vol_z_1m: Mapped[float] = mapped_column(Float, default=0.0)
    oi_delta_15m_pct: Mapped[float] = mapped_column(Float, default=0.0)
    liq_5m_usd: Mapped[float] = mapped_column(Float, default=0.0)
    funding_z: Mapped[float] = mapped_column(Float, default=0.0)
    source_stale: Mapped[bool] = mapped_column(Boolean, default=False)
    cross_source_conflict: Mapped[bool] = mapped_column(Boolean, default=False)
    ingest_lag_ms: Mapped[float] = mapped_column(Float, default=0.0)


class RiskCaseRow(Base):
    __tablename__ = "risk_case"
    case_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset: Mapped[str] = mapped_column(String(10), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="open")
    rule_hits_json: Mapped[str] = mapped_column(Text, default="[]")
    decision: Mapped[str | None] = mapped_column(String(20), nullable=True)
    summary_zh: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str | None] = mapped_column(String(5), nullable=True)
    is_coordinator_case: Mapped[bool] = mapped_column(Boolean, default=False)
    historical_context_zh: Mapped[str] = mapped_column(Text, default="")
    risk_quantification_zh: Mapped[str] = mapped_column(Text, default="")
    suppression_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class RiskAlertRow(Base):
    __tablename__ = "risk_alert"
    alert_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(36), index=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    severity: Mapped[str] = mapped_column(String(5))
    title: Mapped[str] = mapped_column(String(200))
    body_zh: Mapped[str] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    channels_sent: Mapped[str] = mapped_column(Text, default="[]")


class HumanReviewRow(Base):
    __tablename__ = "human_review_action"
    action_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(36), index=True)
    reviewer: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(20))
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)


# ---------------------------------------------------------------------------
# Rule versioning tables
# ---------------------------------------------------------------------------

class RuleVersionRow(Base):
    """每一行是一个完整的规则配置版本快照。"""
    __tablename__ = "rule_version"
    version_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_tag: Mapped[str] = mapped_column(String(50), unique=True)  # e.g. "v1", "v2"
    thresholds: Mapped[dict] = mapped_column(JSON)                      # 完整阈值 dict
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)     # 当前生效版本
    created_by: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    description: Mapped[str] = mapped_column(Text, default="")


class RuleChangeLogRow(Base):
    """每次阈值变更的审计日志，包含变更前后 diff。"""
    __tablename__ = "rule_change_log"
    log_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    from_version: Mapped[str] = mapped_column(String(50))
    to_version: Mapped[str] = mapped_column(String(50))
    changed_by: Mapped[str] = mapped_column(String(100))
    diff: Mapped[dict] = mapped_column(JSON)           # {"field": {"old": x, "new": y}}
    changed_at: Mapped[datetime] = mapped_column(DateTime)
    reason: Mapped[str] = mapped_column(Text, default="")


# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def _ensure_sqlite_column(table_name: str, column_name: str, ddl: str) -> None:
    async with engine.begin() as conn:
        result = await conn.execute(text(f"PRAGMA table_info({table_name})"))
        columns = {row[1] for row in result.fetchall()}
        if column_name in columns:
            return
        await conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {ddl}"))


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _ensure_sqlite_column(
        "risk_case",
        "is_coordinator_case",
        "is_coordinator_case BOOLEAN DEFAULT 0",
    )
    await _ensure_sqlite_column(
        "risk_case",
        "historical_context_zh",
        "historical_context_zh TEXT DEFAULT ''",
    )
    await _ensure_sqlite_column(
        "risk_case",
        "risk_quantification_zh",
        "risk_quantification_zh TEXT DEFAULT ''",
    )
    await _ensure_sqlite_column(
        "risk_case",
        "suppression_reason",
        "suppression_reason TEXT",
    )
