"""Review helper agents for manual-review cases."""
from __future__ import annotations

import json

from src.core.logging import get_logger
from src.domain.models import FeatureSnapshot, RiskCase
from src.graph.nodes import _call_llm_sync, _extract_json_block
from src.observability.llm_trace import run_in_executor_with_context
from src.persistence.repositories import list_risk_cases

logger = get_logger(__name__)

_HISTORICAL_LIMIT = 90
_RISK_LIMIT = 90
_REVIEW_HISTORY_LIMIT = 5


def _short_text(value: str | None, limit: int) -> str:
    text = " ".join((value or "").split()).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _build_fallback_review_assistance(
    case: RiskCase,
    snap: FeatureSnapshot,
    comparable_cases: list[dict[str, str | None]],
) -> tuple[str, str]:
    if comparable_cases:
        latest = comparable_cases[0]
        historical = (
            f"{case.asset.value} 近期已有相似告警，最近一次为 {latest['created_at']}，"
            f"级别 {latest['severity'] or 'unknown'}，建议比对是否为同一风险延续。"
        )
    else:
        historical = f"{case.asset.value} 最近 {_REVIEW_HISTORY_LIMIT} 条历史 case 中暂无可直接对比样本。"

    risk = (
        f"当前 1m 涨跌 {snap.ret_1m:.2%}，5m 爆仓 ${snap.liq_5m_usd:,.0f}，"
        f"15m OI 变动 {snap.oi_delta_15m_pct:.1%}，建议按中高风险优先核实流动性与持仓变化。"
    )
    return (
        _short_text(historical, _HISTORICAL_LIMIT),
        _short_text(risk, _RISK_LIMIT),
    )


def _compact_case_payload(case: RiskCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "asset": case.asset.value,
        "severity": case.severity.value if case.severity else None,
        "decision": case.decision.value if case.decision else None,
        "summary_zh": _short_text(case.summary_zh, 80),
        "rule_ids": [hit.rule_id for hit in case.rule_hits[:3]],
    }


def _compact_snapshot_payload(snap: FeatureSnapshot) -> dict[str, object]:
    return {
        "asset": snap.asset.value,
        "price": snap.price,
        "ret_1m": snap.ret_1m,
        "ret_5m": snap.ret_5m,
        "liq_5m_usd": snap.liq_5m_usd,
        "oi_delta_15m_pct": snap.oi_delta_15m_pct,
        "funding_z": snap.funding_z,
        "cross_source_conflict": snap.cross_source_conflict,
    }


async def build_review_assistance(case: RiskCase, snap: FeatureSnapshot) -> tuple[str, str]:
    recent_cases = await list_risk_cases(asset=case.asset, limit=_REVIEW_HISTORY_LIMIT)
    comparable_cases = [
        {
            "case_id": item.case_id,
            "created_at": item.created_at.isoformat(),
            "severity": item.severity.value if item.severity else None,
            "decision": item.decision.value if item.decision else None,
            "summary_zh": _short_text(item.summary_zh, 60),
        }
        for item in recent_cases
        if item.case_id != case.case_id
    ]
    if not comparable_cases:
        return _build_fallback_review_assistance(case, snap, comparable_cases)

    prompt = f"""你是人工审核助手。请基于当前风险案例与同资产历史案例，严格输出单行 JSON。

约束：
1. 只能输出 JSON，不得输出 markdown、标题、解释或代码块。
2. historical_context_zh 必须是中文短句，最多 90 个汉字。
3. risk_quantification_zh 必须是中文短句，最多 90 个汉字。
4. 只保留对审核员最关键的信息，不要复述全部原始字段。

当前 case:
{json.dumps(_compact_case_payload(case), ensure_ascii=False)}

当前 snapshot:
{json.dumps(_compact_snapshot_payload(snap), ensure_ascii=False)}

历史 case:
{json.dumps(comparable_cases, ensure_ascii=False)}

输出 JSON:
{{"historical_context_zh":"...","risk_quantification_zh":"..."}}"""

    try:
        text = await run_in_executor_with_context(
            lambda: _call_llm_sync(prompt, 220, operation="review_assistance")
        )
        data = _extract_json_block(text)
        historical = _short_text(data.get("historical_context_zh", ""), _HISTORICAL_LIMIT)
        risk = _short_text(data.get("risk_quantification_zh", ""), _RISK_LIMIT)
        if historical and risk:
            return historical, risk
    except Exception as exc:
        logger.warning("Review assistance generation failed: %s", exc)

    return _build_fallback_review_assistance(case, snap, comparable_cases)


async def build_historical_context(case: RiskCase) -> str:
    logger.warning("build_historical_context is deprecated; use build_review_assistance instead")
    recent_cases = await list_risk_cases(asset=case.asset, limit=_REVIEW_HISTORY_LIMIT)
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
    if comparable_cases:
        latest = comparable_cases[0]
        return _short_text(
            (
                f"{case.asset.value} 近期已有相似告警，最近一次为 {latest['created_at']}，"
                f"级别 {latest['severity'] or 'unknown'}，建议比对是否为同一风险延续。"
            ),
            _HISTORICAL_LIMIT,
        )
    return _short_text(
        f"{case.asset.value} 最近 {_REVIEW_HISTORY_LIMIT} 条历史 case 中暂无可直接对比样本。",
        _HISTORICAL_LIMIT,
    )


async def quantify_risk(case: RiskCase, snap: FeatureSnapshot) -> str:
    logger.warning("quantify_risk is deprecated; use build_review_assistance instead")
    _, risk = _build_fallback_review_assistance(case, snap, [])
    return risk
