"""SQLAlchemy async engine + table definitions."""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint, text
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
    ret_15m: Mapped[float] = mapped_column(Float, default=0.0)
    ret_30m: Mapped[float] = mapped_column(Float, default=0.0)
    ret_60m: Mapped[float] = mapped_column(Float, default=0.0)
    vol_z_1m: Mapped[float] = mapped_column(Float, default=0.0)
    realized_vol_5m: Mapped[float] = mapped_column(Float, default=0.0)
    realized_vol_15m: Mapped[float] = mapped_column(Float, default=0.0)
    realized_vol_60m: Mapped[float] = mapped_column(Float, default=0.0)
    price_range_pct_1m: Mapped[float] = mapped_column(Float, default=0.0)
    close_position_1m: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown_15m: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown_60m: Mapped[float] = mapped_column(Float, default=0.0)
    max_runup_15m: Mapped[float] = mapped_column(Float, default=0.0)
    max_runup_60m: Mapped[float] = mapped_column(Float, default=0.0)
    atr_14: Mapped[float] = mapped_column(Float, default=0.0)
    volatility_regime_60m: Mapped[float] = mapped_column(Float, default=0.0)
    volume_1m: Mapped[float] = mapped_column(Float, default=0.0)
    quote_volume_1m: Mapped[float] = mapped_column(Float, default=0.0)
    volume_5m: Mapped[float] = mapped_column(Float, default=0.0)
    quote_volume_5m: Mapped[float] = mapped_column(Float, default=0.0)
    volume_15m: Mapped[float] = mapped_column(Float, default=0.0)
    quote_volume_15m: Mapped[float] = mapped_column(Float, default=0.0)
    volume_z_15m: Mapped[float] = mapped_column(Float, default=0.0)
    volume_z_60m: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count_1m: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count_z_15m: Mapped[float] = mapped_column(Float, default=0.0)
    taker_buy_ratio_1m: Mapped[float] = mapped_column(Float, default=0.0)
    taker_buy_ratio_5m: Mapped[float] = mapped_column(Float, default=0.0)
    oi_delta_15m_pct: Mapped[float] = mapped_column(Float, default=0.0)
    oi_delta_5m_pct: Mapped[float] = mapped_column(Float, default=0.0)
    oi_delta_60m_pct: Mapped[float] = mapped_column(Float, default=0.0)
    liq_5m_usd: Mapped[float] = mapped_column(Float, default=0.0)
    funding_z: Mapped[float] = mapped_column(Float, default=0.0)
    futures_basis_pct: Mapped[float] = mapped_column(Float, default=0.0)
    basis_z_60m: Mapped[float] = mapped_column(Float, default=0.0)
    source_stale: Mapped[bool] = mapped_column(Boolean, default=False)
    cross_source_conflict: Mapped[bool] = mapped_column(Boolean, default=False)
    ingest_lag_ms: Mapped[float] = mapped_column(Float, default=0.0)


class HistoricalMarketBarRow(Base):
    __tablename__ = "historical_market_bar"
    __table_args__ = (
        UniqueConstraint("source", "market_type", "symbol", "interval", "open_time", name="uq_historical_bar"),
    )
    bar_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source: Mapped[str] = mapped_column(String(50), index=True)
    market_type: Mapped[str] = mapped_column(String(20), index=True)
    asset: Mapped[str] = mapped_column(String(10), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    interval: Mapped[str] = mapped_column(String(10), index=True)
    open_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    close_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    open: Mapped[float] = mapped_column(Float, default=0.0)
    high: Mapped[float] = mapped_column(Float, default=0.0)
    low: Mapped[float] = mapped_column(Float, default=0.0)
    close: Mapped[float] = mapped_column(Float, default=0.0)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    quote_volume: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    taker_buy_base_volume: Mapped[float] = mapped_column(Float, default=0.0)
    taker_buy_quote_volume: Mapped[float] = mapped_column(Float, default=0.0)


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


class LLMCallRow(Base):
    __tablename__ = "llm_call"
    call_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    model: Mapped[str] = mapped_column(String(100), index=True)
    operation: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


class QualityMetricEventRow(Base):
    __tablename__ = "quality_metric_event"
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    event_type: Mapped[str] = mapped_column(String(80), index=True)
    asset: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    severity: Mapped[str | None] = mapped_column(String(5), nullable=True)
    case_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    dedupe_key: Mapped[str] = mapped_column(String(200), default="", index=True)
    value: Mapped[float] = mapped_column(Float, default=1.0)
    details_json: Mapped[str] = mapped_column(Text, default="{}")


class HumanReviewRow(Base):
    __tablename__ = "human_review_action"
    action_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(36), index=True)
    reviewer: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(20))
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)


class RiskModelLabelRow(Base):
    __tablename__ = "risk_model_label"
    label_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    asset: Mapped[str] = mapped_column(String(10), index=True)
    window_end: Mapped[datetime] = mapped_column(DateTime, index=True)
    horizon_seconds: Mapped[int] = mapped_column(Integer, default=3600)
    label: Mapped[str] = mapped_column(String(10), index=True)
    risk_probability: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    labeling_method: Mapped[str] = mapped_column(String(50), index=True)
    rationale: Mapped[str] = mapped_column(Text, default="")
    judge_payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class RiskModelPredictionRow(Base):
    __tablename__ = "risk_model_prediction"
    prediction_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(36), index=True)
    asset: Mapped[str] = mapped_column(String(10), index=True)
    model_version: Mapped[str] = mapped_column(String(80), index=True)
    raw_probability: Mapped[float] = mapped_column(Float, default=0.0)
    calibrated_probability: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(10), index=True)
    top_features_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)


# ---------------------------------------------------------------------------
# Rule versioning tables
# ---------------------------------------------------------------------------

class CaseEmbeddingRow(Base):
    """Stores vector embeddings for risk cases for semantic similarity search."""
    __tablename__ = "case_embedding"
    case_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset: Mapped[str] = mapped_column(String(10), index=True)
    severity: Mapped[str | None] = mapped_column(String(5), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(20), nullable=True)
    summary_zh: Mapped[str] = mapped_column(Text, default="")
    embedding_blob: Mapped[bytes] = mapped_column(Text)     # raw float32 bytes
    embedding_dim: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(16), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class ExperienceInsightRow(Base):
    """Distilled experience insights from historical case patterns."""
    __tablename__ = "experience_insight"
    insight_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    pattern_type: Mapped[str] = mapped_column(String(200), index=True)
    insight_zh: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    supporting_case_count: Mapped[int] = mapped_column(Integer, default=0)
    supporting_case_ids: Mapped[str] = mapped_column(Text, default="[]")
    statistics_json: Mapped[str] = mapped_column(Text, default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class ReviewerPreferenceRow(Base):
    """Learned reviewer approve/reject preferences per rule pattern."""
    __tablename__ = "reviewer_preference"
    __table_args__ = (
        UniqueConstraint("asset", "rule_pattern", name="uq_reviewer_pref"),
    )
    preference_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset: Mapped[str] = mapped_column(String(10), index=True)
    rule_pattern: Mapped[str] = mapped_column(String(500), index=True)
    approve_rate: Mapped[float] = mapped_column(Float, default=0.0)
    reject_rate: Mapped[float] = mapped_column(Float, default=0.0)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    pattern_description_zh: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


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


def sqlite_database_path() -> str | None:
    prefixes = ("sqlite+aiosqlite:///", "sqlite:///")
    for prefix in prefixes:
        if settings.database_url.startswith(prefix):
            return settings.database_url.removeprefix(prefix)
    return None


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
    feature_snapshot_columns = {
        "ret_15m": "ret_15m FLOAT DEFAULT 0.0",
        "ret_30m": "ret_30m FLOAT DEFAULT 0.0",
        "ret_60m": "ret_60m FLOAT DEFAULT 0.0",
        "realized_vol_5m": "realized_vol_5m FLOAT DEFAULT 0.0",
        "realized_vol_15m": "realized_vol_15m FLOAT DEFAULT 0.0",
        "realized_vol_60m": "realized_vol_60m FLOAT DEFAULT 0.0",
        "price_range_pct_1m": "price_range_pct_1m FLOAT DEFAULT 0.0",
        "close_position_1m": "close_position_1m FLOAT DEFAULT 0.0",
        "max_drawdown_15m": "max_drawdown_15m FLOAT DEFAULT 0.0",
        "max_drawdown_60m": "max_drawdown_60m FLOAT DEFAULT 0.0",
        "max_runup_15m": "max_runup_15m FLOAT DEFAULT 0.0",
        "max_runup_60m": "max_runup_60m FLOAT DEFAULT 0.0",
        "atr_14": "atr_14 FLOAT DEFAULT 0.0",
        "volatility_regime_60m": "volatility_regime_60m FLOAT DEFAULT 0.0",
        "volume_1m": "volume_1m FLOAT DEFAULT 0.0",
        "quote_volume_1m": "quote_volume_1m FLOAT DEFAULT 0.0",
        "volume_5m": "volume_5m FLOAT DEFAULT 0.0",
        "quote_volume_5m": "quote_volume_5m FLOAT DEFAULT 0.0",
        "volume_15m": "volume_15m FLOAT DEFAULT 0.0",
        "quote_volume_15m": "quote_volume_15m FLOAT DEFAULT 0.0",
        "volume_z_15m": "volume_z_15m FLOAT DEFAULT 0.0",
        "volume_z_60m": "volume_z_60m FLOAT DEFAULT 0.0",
        "trade_count_1m": "trade_count_1m FLOAT DEFAULT 0.0",
        "trade_count_z_15m": "trade_count_z_15m FLOAT DEFAULT 0.0",
        "taker_buy_ratio_1m": "taker_buy_ratio_1m FLOAT DEFAULT 0.0",
        "taker_buy_ratio_5m": "taker_buy_ratio_5m FLOAT DEFAULT 0.0",
        "oi_delta_5m_pct": "oi_delta_5m_pct FLOAT DEFAULT 0.0",
        "oi_delta_60m_pct": "oi_delta_60m_pct FLOAT DEFAULT 0.0",
        "futures_basis_pct": "futures_basis_pct FLOAT DEFAULT 0.0",
        "basis_z_60m": "basis_z_60m FLOAT DEFAULT 0.0",
    }
    for column_name, ddl in feature_snapshot_columns.items():
        await _ensure_sqlite_column("feature_snapshot", column_name, ddl)
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
