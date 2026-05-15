"""Reviewer preference learning — learn patterns from human approve/reject decisions."""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger
from src.domain.models import Asset, Decision, ReviewAction, RiskCase, Severity
from src.persistence.database import AsyncSessionLocal, ReviewerPreferenceRow

logger = get_logger(__name__)


async def get_reviewer_preferences(
    asset: str | None = None,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve learned reviewer preferences."""
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        q = (
            select(ReviewerPreferenceRow)
            .order_by(ReviewerPreferenceRow.sample_count.desc())
            .limit(limit)
        )
        if asset:
            q = q.where(
                (ReviewerPreferenceRow.asset == asset)
                | (ReviewerPreferenceRow.asset.is_(None))
            )
        rows = (await s.execute(q)).scalars().all()
        return [
            {
                "preference_id": row.preference_id,
                "asset": row.asset,
                "rule_pattern": row.rule_pattern,
                "approve_rate": row.approve_rate,
                "reject_rate": row.reject_rate,
                "sample_count": row.sample_count,
                "pattern_description_zh": row.pattern_description_zh,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]


async def learn_preferences(
    lookback_days: int = 60,
) -> list[dict[str, Any]]:
    """Analyze human review actions to learn reviewer preferences.

    Groups reviewed cases by rule pattern and asset, then computes:
    - Approve rate: how often this pattern gets approved
    - Reject rate: how often it gets rejected
    - Confidence: based on sample size

    Stores results as ReviewerPreference rows for use during decision-making.
    """
    from sqlalchemy import select
    from src.persistence.database import HumanReviewRow, RiskCaseRow

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    async with AsyncSessionLocal() as s:
        # Join review actions with their cases
        result = await s.execute(
            select(HumanReviewRow, RiskCaseRow)
            .join(RiskCaseRow, RiskCaseRow.case_id == HumanReviewRow.case_id)
            .where(HumanReviewRow.created_at >= cutoff)
            .order_by(HumanReviewRow.created_at.desc())
        )
        rows = result.all()

    if not rows:
        logger.info("No review actions found for preference learning")
        return []

    # Group by (asset, rule_pattern)
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"approve": 0, "reject": 0, "escalate": 0}
    )
    for review_row, case_row in rows:
        hits_data = json.loads(case_row.rule_hits_json or "[]")
        rule_ids = sorted({h.get("rule_id", "") for h in hits_data if h.get("rule_id")})
        pattern = "|".join(rule_ids) if rule_ids else "no_rules"

        key = (case_row.asset, pattern)
        action = review_row.action
        if action == ReviewAction.APPROVE.value:
            groups[key]["approve"] += 1
        elif action == ReviewAction.REJECT.value:
            groups[key]["reject"] += 1
        elif action == ReviewAction.ESCALATE.value:
            groups[key]["escalate"] += 1

    # Build preferences
    new_preferences = []
    now = datetime.utcnow()

    for (asset, pattern), counts in groups.items():
        total = counts["approve"] + counts["reject"] + counts["escalate"]
        if total < 2:
            continue

        approve_rate = counts["approve"] / total
        reject_rate = counts["reject"] / total

        # Generate description
        if reject_rate > 0.7:
            desc = f"审核者倾向于拒绝此类告警（拒绝率 {reject_rate:.0%}），建议在 review_guidance 中提示历史拒绝倾向"
        elif approve_rate > 0.8:
            desc = f"审核者通常批准此类告警（批准率 {approve_rate:.0%}），可考虑提高自动化程度"
        else:
            desc = f"审核者对此类告警态度分歧（批准 {approve_rate:.0%} / 拒绝 {reject_rate:.0%}），需谨慎处理"

        pref = ReviewerPreferenceRow(
            preference_id=str(uuid.uuid4()),
            asset=asset,
            rule_pattern=pattern,
            approve_rate=round(approve_rate, 4),
            reject_rate=round(reject_rate, 4),
            sample_count=total,
            pattern_description_zh=desc,
            created_at=now,
            updated_at=now,
        )

        async with AsyncSessionLocal() as s:
            # Upsert: check if pattern already exists
            from sqlalchemy import select as sel
            existing = (await s.execute(
                sel(ReviewerPreferenceRow).where(
                    ReviewerPreferenceRow.asset == asset,
                    ReviewerPreferenceRow.rule_pattern == pattern,
                )
            )).scalar_one_or_none()

            if existing:
                existing.approve_rate = pref.approve_rate
                existing.reject_rate = pref.reject_rate
                existing.sample_count = pref.sample_count
                existing.pattern_description_zh = pref.pattern_description_zh
                existing.updated_at = now
            else:
                s.add(pref)
            await s.commit()

        new_preferences.append({
            "asset": asset,
            "rule_pattern": pattern,
            "approve_rate": pref.approve_rate,
            "reject_rate": pref.reject_rate,
            "sample_count": total,
            "pattern_description_zh": desc,
        })

    logger.info("Learned %d reviewer preferences", len(new_preferences))
    return new_preferences


async def get_preference_for_hits(
    asset: str,
    rule_hits_json: str,
) -> dict[str, Any] | None:
    """Look up the reviewer preference for a specific set of rule hits."""
    from sqlalchemy import select

    try:
        hits_data = json.loads(rule_hits_json or "[]")
    except json.JSONDecodeError:
        return None

    rule_ids = sorted({h.get("rule_id", "") for h in hits_data if h.get("rule_id")})
    pattern = "|".join(rule_ids) if rule_ids else "no_rules"

    async with AsyncSessionLocal() as s:
        row = (await s.execute(
            select(ReviewerPreferenceRow).where(
                ReviewerPreferenceRow.asset == asset,
                ReviewerPreferenceRow.rule_pattern == pattern,
            )
        )).scalar_one_or_none()

        if row is None:
            return None

        return {
            "approve_rate": row.approve_rate,
            "reject_rate": row.reject_rate,
            "sample_count": row.sample_count,
            "pattern_description_zh": row.pattern_description_zh,
        }
