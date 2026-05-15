"""Experience distillation — summarize patterns from historical cases into reusable insights."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger
from src.domain.models import Asset, CaseStatus, Decision, Severity
from src.observability.llm_trace import record_llm_call, run_in_executor_with_context
from src.persistence.database import AsyncSessionLocal, ExperienceInsightRow

logger = get_logger(__name__)


async def get_insights(
    asset: str | None = None,
    *,
    active_only: bool = True,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve stored experience insights from the database."""
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        q = select(ExperienceInsightRow).order_by(
            ExperienceInsightRow.confidence.desc()
        ).limit(limit)
        if asset:
            q = q.where(
                (ExperienceInsightRow.asset == asset)
                | (ExperienceInsightRow.asset.is_(None))
            )
        if active_only:
            q = q.where(ExperienceInsightRow.is_active == True)  # noqa: E712
        rows = (await s.execute(q)).scalars().all()
        return [
            {
                "insight_id": row.insight_id,
                "asset": row.asset,
                "pattern_type": row.pattern_type,
                "insight_zh": row.insight_zh,
                "confidence": row.confidence,
                "supporting_case_count": row.supporting_case_count,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]


async def distill_experiences(
    asset: Asset | None = None,
    *,
    lookback_days: int = 30,
    min_cases: int = 5,
) -> list[dict[str, Any]]:
    """Analyze historical cases and distill recurring patterns into experience insights.

    This function:
    1. Loads closed/reviewed cases from the last N days
    2. Groups them by rule pattern and outcome
    3. Uses LLM to synthesize insights from recurring patterns
    4. Stores insights in the database for future use
    """
    from src.persistence.repositories import list_risk_cases, get_recent_review_actions

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    cases = await list_risk_cases(
        asset=asset,
        limit=500,
        include_suppressed=True,
    )

    # Filter to cases in the lookback window with decisions
    relevant_cases = [
        c for c in cases
        if c.created_at >= cutoff and c.decision is not None
    ]
    if len(relevant_cases) < min_cases:
        logger.info(
            "Not enough cases for distillation: %d < %d",
            len(relevant_cases), min_cases,
        )
        return []

    # Group by primary rule pattern
    pattern_groups: dict[str, list] = {}
    for case in relevant_cases:
        primary_rules = sorted({h.rule_id for h in case.rule_hits})
        pattern_key = "|".join(primary_rules) if primary_rules else "no_rules"
        pattern_groups.setdefault(pattern_key, []).append(case)

    new_insights = []
    for pattern_key, group_cases in pattern_groups.items():
        if len(group_cases) < 3:
            continue

        # Build statistics for this pattern
        total = len(group_cases)
        emitted = sum(1 for c in group_cases if c.decision == Decision.EMIT)
        suppressed = sum(1 for c in group_cases if c.decision == Decision.SUPPRESS)
        reviewed = sum(1 for c in group_cases if c.decision == Decision.MANUAL_REVIEW)
        assets_involved = list({c.asset.value for c in group_cases})
        severities = [c.severity.value for c in group_cases if c.severity]
        avg_severity_score = sum(
            {"P1": 3, "P2": 2, "P3": 1}.get(s, 0) for s in severities
        ) / max(len(severities), 1)

        sample_summaries = [
            c.summary_zh[:80] for c in group_cases[:5] if c.summary_zh
        ]

        # Ask LLM to synthesize an insight
        insight = await _synthesize_insight(
            pattern_key=pattern_key,
            total_cases=total,
            emitted=emitted,
            suppressed=suppressed,
            reviewed=reviewed,
            assets=assets_involved,
            avg_severity=avg_severity_score,
            sample_summaries=sample_summaries,
        )

        if insight:
            # Store the insight
            insight_row = ExperienceInsightRow(
                insight_id=str(uuid.uuid4()),
                asset=assets_involved[0] if len(assets_involved) == 1 else None,
                pattern_type=pattern_key,
                insight_zh=insight["insight_zh"],
                confidence=insight["confidence"],
                supporting_case_count=total,
                supporting_case_ids=json.dumps([c.case_id for c in group_cases[:20]]),
                statistics_json=json.dumps({
                    "total": total,
                    "emitted": emitted,
                    "suppressed": suppressed,
                    "reviewed": reviewed,
                    "avg_severity": avg_severity_score,
                }, ensure_ascii=False),
                is_active=True,
                created_at=datetime.utcnow(),
            )
            async with AsyncSessionLocal() as s:
                s.add(insight_row)
                await s.commit()

            new_insights.append({
                "insight_id": insight_row.insight_id,
                "pattern_type": pattern_key,
                "insight_zh": insight["insight_zh"],
                "confidence": insight["confidence"],
                "supporting_case_count": total,
            })

    logger.info("Distilled %d new experience insights", len(new_insights))
    return new_insights


async def _synthesize_insight(
    *,
    pattern_key: str,
    total_cases: int,
    emitted: int,
    suppressed: int,
    reviewed: int,
    assets: list[str],
    avg_severity: float,
    sample_summaries: list[str],
) -> dict[str, Any] | None:
    """Use LLM to synthesize an experience insight from case statistics."""
    from src.graph.nodes import _call_llm_sync

    prompt = f"""你是风控经验总结助手。根据以下历史案例统计数据，提炼一条可复用的经验规律。

规则模式: {pattern_key}
总案例数: {total_cases}
决策分布: 发出告警 {emitted}, 抑制 {suppressed}, 人工审核 {reviewed}
涉及资产: {', '.join(assets)}
平均严重程度: {avg_severity:.1f}/3.0

代表性案例摘要:
{chr(10).join(f'- {s}' for s in sample_summaries)}

请输出 JSON:
{{"insight_zh": "一句话经验总结（最多80字）", "confidence": 0.0到1.0的置信度}}"""

    try:
        text = await run_in_executor_with_context(
            lambda: _call_llm_sync(prompt, max_tokens=120, operation="experience_distill")
        )
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
        return {
            "insight_zh": data.get("insight_zh", "").strip()[:80],
            "confidence": confidence,
        }
    except Exception as exc:
        logger.warning("Experience synthesis failed: %s", exc)
        return None
