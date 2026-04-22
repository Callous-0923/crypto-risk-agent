"""Review helper agents for manual-review cases."""
from __future__ import annotations

import asyncio
import json

from src.core.logging import get_logger
from src.domain.models import FeatureSnapshot, RiskCase
from src.graph.nodes import _call_llm_sync
from src.persistence.repositories import list_risk_cases

logger = get_logger(__name__)


async def build_historical_context(case: RiskCase) -> str:
    recent_cases = await list_risk_cases(asset=case.asset, limit=10)
    comparable_cases = [
        {
            "case_id": item.case_id,
            "created_at": item.created_at.isoformat(),
            "severity": item.severity.value if item.severity else None,
            "decision": item.decision.value if item.decision else None,
            "summary_zh": item.summary_zh,
        }
        for item in recent_cases
        if item.case_id != case.case_id
    ]
    if not comparable_cases:
        return f"{case.asset.value} 近10条历史 case 中暂无可直接对比样本。"

    prompt = f"""你是人工审核助手。请基于同资产历史案例，总结值得审核员关注的模式。

当前 case:
{json.dumps(case.model_dump(mode="json"), ensure_ascii=False)}

历史 case:
{json.dumps(comparable_cases, ensure_ascii=False)}

请输出 120 字以内中文总结。"""

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _call_llm_sync(prompt, 300))
    except Exception as exc:
        logger.warning("Historical context generation failed: %s", exc)
        latest = comparable_cases[0]
        return (
            f"{case.asset.value} 近期已有相似告警，最近一次为 {latest['created_at']}，"
            f"严重级别 {latest['severity'] or 'unknown'}，建议对比是否为同一轮风险演化。"
        )


async def quantify_risk(case: RiskCase, snap: FeatureSnapshot) -> str:
    prompt = f"""你是人工审核助手。请基于当前 snapshot 和 rule_hits，给审核员一段风险量化说明。

snapshot:
{json.dumps(snap.model_dump(mode="json"), ensure_ascii=False)}

rule_hits:
{json.dumps([hit.model_dump(mode="json") for hit in case.rule_hits], ensure_ascii=False)}

请输出 120 字以内中文总结，覆盖潜在敞口、扩散速度和置信度。"""

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _call_llm_sync(prompt, 300))
    except Exception as exc:
        logger.warning("Risk quantification generation failed: %s", exc)
        return (
            f"当前 1m 涨跌 {snap.ret_1m:.2%}，5m 爆仓 ${snap.liq_5m_usd:,.0f}，"
            f"15m OI 变动 {snap.oi_delta_15m_pct:.1%}，建议按中高风险对待并优先核对流动性冲击。"
        )
