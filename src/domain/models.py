"""Core domain models — all Pydantic v2."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Asset(str, Enum):
    BTC = "BTC"
    ETH = "ETH"
    SOL = "SOL"


class Severity(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Decision(str, Enum):
    EMIT = "emit"
    SUPPRESS = "suppress"
    MANUAL_REVIEW = "manual_review"


class ReviewAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Raw events from data sources
# ---------------------------------------------------------------------------

class RawEvent(BaseModel):
    event_id: str = Field(default_factory=_uid)
    trace_id: str = Field(default_factory=_uid)
    asset: Asset
    source: str                    # binance_spot / binance_futures / coinglass / coingecko
    event_type: str                # price / liquidation / oi / funding / mark_price
    event_ts: datetime             # when it happened at source
    ingest_ts: datetime = Field(default_factory=_now)
    payload: dict[str, Any]
    dedupe_key: str = ""           # populated by normalizer


# ---------------------------------------------------------------------------
# Feature snapshot (windowed aggregation)
# ---------------------------------------------------------------------------

class FeatureSnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=_uid)
    asset: Asset
    window_end: datetime = Field(default_factory=_now)

    # Market features
    price: float = 0.0
    ret_1m: float = 0.0
    ret_5m: float = 0.0
    vol_z_1m: float = 0.0         # z-score of 1m volatility vs rolling baseline

    # Derivatives features
    oi_delta_15m_pct: float = 0.0
    liq_5m_usd: float = 0.0
    funding_z: float = 0.0        # z-score of funding rate

    # Quality flags
    source_stale: bool = False
    cross_source_conflict: bool = False
    ingest_lag_ms: float = 0.0


# ---------------------------------------------------------------------------
# Rule hit — one fired rule
# ---------------------------------------------------------------------------

class RuleHit(BaseModel):
    rule_id: str
    asset: Asset
    severity: Severity
    confidence: float = 1.0        # 0-1
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    dedupe_key: str = ""
    fired_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Risk case — the persistent thread unit
# ---------------------------------------------------------------------------

class CaseStatus(str, Enum):
    OPEN = "open"
    PENDING_REVIEW = "pending_review"
    CLOSED = "closed"
    SUPPRESSED = "suppressed"


class RiskCase(BaseModel):
    case_id: str = Field(default_factory=_uid)  # = LangGraph thread_id
    asset: Asset
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    status: CaseStatus = CaseStatus.OPEN
    rule_hits: list[RuleHit] = Field(default_factory=list)
    decision: Decision | None = None
    summary_zh: str = ""           # LLM-generated Chinese summary
    severity: Severity | None = None
    is_coordinator_case: bool = False
    historical_context_zh: str = ""
    risk_quantification_zh: str = ""
    suppression_reason: str | None = None


# ---------------------------------------------------------------------------
# Risk alert — what gets sent to users
# ---------------------------------------------------------------------------

class RiskAlert(BaseModel):
    alert_id: str = Field(default_factory=_uid)
    case_id: str
    revision: int = 1
    severity: Severity
    title: str
    body_zh: str
    idempotency_key: str = ""      # case_id:revision:channel
    created_at: datetime = Field(default_factory=_now)
    channels_sent: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Human review action
# ---------------------------------------------------------------------------

class HumanReviewAction(BaseModel):
    action_id: str = Field(default_factory=_uid)
    case_id: str
    reviewer: str
    action: ReviewAction
    comment: str = ""
    created_at: datetime = Field(default_factory=_now)
