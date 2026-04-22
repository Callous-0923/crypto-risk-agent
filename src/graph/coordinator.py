"""Cross-asset coordinator for synthetic multi-asset cases."""
from __future__ import annotations

import asyncio
import json

from src.core.config import settings
from src.core.logging import get_logger
from src.domain.models import Asset, CaseStatus, Decision, FeatureSnapshot, RiskCase, RuleHit, Severity
from src.graph.nodes import _call_llm_sync
from src.graph.orchestrator import process_coordinator_case, resume_case
from src.persistence.repositories import (
    find_recent_coordinator_case,
    get_risk_case,
    list_recent_risk_cases,
)

logger = get_logger(__name__)


def _group_latest_by_asset(cases: list[RiskCase]) -> dict[Asset, RiskCase]:
    latest: dict[Asset, RiskCase] = {}
    for case in cases:
        current = latest.get(case.asset)
        if current is None or case.created_at > current.created_at:
            latest[case.asset] = case
    return latest


async def _build_cross_asset_summary(asset_cases: dict[Asset, RiskCase]) -> tuple[str, str]:
    asset_lines = []
    for asset, case in sorted(asset_cases.items(), key=lambda item: item[0].value):
        summary = case.summary_zh or "该资产近期触发高等级风险规则。"
        asset_lines.append(f"- {asset.value}: {summary}")

    prompt = f"""你是加密货币风控协调助手。以下资产在60秒内同时触发高等级风险，请生成跨资产联动摘要。

{chr(10).join(asset_lines)}

请直接输出 JSON：
{{"summary_zh":"...","review_guidance":"..."}}"""

    loop = asyncio.get_running_loop()
    try:
        raw = await loop.run_in_executor(None, _call_llm_sync, prompt)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return (
            data.get("summary_zh", "").strip(),
            data.get("review_guidance", "").strip(),
        )
    except Exception as exc:
        logger.warning("Coordinator summary generation failed: %s", exc)
        assets_text = "/".join(asset.value for asset in sorted(asset_cases))
        return (
            f"{assets_text} 在60秒内同时触发高等级风险，疑似市场出现跨资产联动异常。",
            "请优先核对各资产同步性与市场范围，再决定是否发送聚合告警。",
        )


async def _auto_approve_if_timeout(case_id: str, timeout_s: int) -> None:
    await asyncio.sleep(timeout_s)
    case = await get_risk_case(case_id)
    if case is None or case.status != CaseStatus.PENDING_REVIEW:
        return
    await resume_case(case_id, approved=True, comment="auto-approved by coordinator timeout")


def _build_coordinator_hit(asset_cases: dict[Asset, RiskCase], severity: Severity) -> RuleHit:
    assets = sorted(asset.value for asset in asset_cases)
    return RuleHit(
        rule_id="COORD_MULTI_ASSET_EVENT",
        asset=Asset(assets[0]),
        severity=severity,
        confidence=1.0,
        description=f"{'/'.join(assets)} 在60秒内同步触发 {severity.value} 风险，命中跨资产协调规则",
        evidence={
            "assets": assets,
            "source_case_ids": [case.case_id for case in asset_cases.values()],
            "window_seconds": 60,
        },
        dedupe_key=f"COORD_MULTI_ASSET:{','.join(assets)}",
    )


async def coordinate_cross_asset(snaps: dict[Asset, FeatureSnapshot]) -> None:
    if len(snaps) < 2:
        return

    recent_cases = await list_recent_risk_cases(
        60,
        severities=[Severity.P1, Severity.P2],
        include_coordinator=False,
        limit=100,
    )
    if not recent_cases:
        return

    latest_by_asset = _group_latest_by_asset(recent_cases)
    p1_cases = {
        asset: case
        for asset, case in latest_by_asset.items()
        if case.severity == Severity.P1 and asset in snaps
    }
    if len(p1_cases) < 2:
        return

    triggered_assets = sorted(p1_cases)
    existing = await find_recent_coordinator_case(triggered_assets, within_seconds=300)
    if existing is not None:
        return

    highest = Severity.P1
    summary_zh, review_guidance = await _build_cross_asset_summary(p1_cases)
    coordinator_hit = _build_coordinator_hit(p1_cases, highest)
    case_id = await process_coordinator_case(
        asset=triggered_assets[0],
        rule_hits=[coordinator_hit],
        severity=highest,
        summary_zh=summary_zh,
        review_guidance=review_guidance,
    )
    asyncio.create_task(_auto_approve_if_timeout(case_id, settings.coordinator_auto_approve_seconds))
