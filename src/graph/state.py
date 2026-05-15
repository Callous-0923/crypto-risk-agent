"""LangGraph state definition."""
from __future__ import annotations

from typing_extensions import TypedDict

from src.domain.models import (
    Asset, Decision, FeatureSnapshot, RiskAlert, RiskCase, RuleHit, Severity,
)


class RiskState(TypedDict):
    # Input
    thread_id: str
    asset: Asset
    snapshot: FeatureSnapshot | None  # resume 路径下可能为 None
    is_coordinator_case: bool
    recent_alert_history: list[dict]
    fatigue_suppressed: bool
    memory_context: dict | None
    ml_prediction: dict | None

    # Rule results
    rule_hits: list[RuleHit]
    highest_severity: Severity | None

    # LLM output
    technical_analysis: dict | None
    macro_context: dict | None
    technical_analysis_zh: str
    macro_context_zh: str
    summary_zh: str
    review_guidance: str
    historical_context_zh: str
    risk_quantification_zh: str

    # Decision
    decision: Decision | None
    case: RiskCase | None
    case_reused: bool
    alert: RiskAlert | None

    # Human review
    human_approved: bool | None
    human_comment: str

    # Long-term memory (enriched context from MemoryManager)
    enriched_memory: dict | None
