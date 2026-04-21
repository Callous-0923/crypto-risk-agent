"""LangGraph state definition."""
from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.domain.models import (
    Asset, Decision, FeatureSnapshot, RiskAlert, RiskCase, RuleHit, Severity,
)


class RiskState(TypedDict):
    # Input
    asset: Asset
    snapshot: FeatureSnapshot | None  # resume 路径下可能为 None

    # Rule results
    rule_hits: list[RuleHit]
    highest_severity: Severity | None

    # LLM output
    summary_zh: str
    review_guidance: str

    # Decision
    decision: Decision | None
    case: RiskCase | None
    alert: RiskAlert | None

    # Human review
    human_approved: bool | None
    human_comment: str
