"""LangGraph node implementations."""
from __future__ import annotations

from datetime import datetime

import asyncio
from openai import OpenAI

from src.core.config import settings
from src.core.logging import get_logger
from src.domain.models import (
    CaseStatus, Decision, FeatureSnapshot, RiskAlert, RiskCase, RuleHit, Severity,
)
from src.graph.state import RiskState
from src.notification.dispatcher import dispatch_alert
from src.persistence.repositories import save_risk_case, save_risk_alert

logger = get_logger(__name__)

_llm = OpenAI(api_key=settings.ark_api_key, base_url=settings.ark_base_url)


def _call_llm_sync(prompt: str) -> str:
    """同步调用 LLM，在线程池中执行避免事件循环冲突。记录耗时和错误指标。"""
    import time
    from src.observability.metrics import llm_call_total, llm_call_duration_seconds, llm_error_total
    t0 = time.perf_counter()
    try:
        response = _llm.chat.completions.create(
            model=settings.llm_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        llm_call_duration_seconds.observe(elapsed)
        llm_call_total.labels(status="success").inc()
        return response.choices[0].message.content.strip()
    except Exception as e:
        elapsed = time.perf_counter() - t0
        llm_call_duration_seconds.observe(elapsed)
        err_type = "timeout" if "timeout" in str(e).lower() else \
                   "auth" if "401" in str(e) else \
                   "network" if "connect" in str(e).lower() else "other"
        llm_error_total.labels(error_type=err_type).inc()
        llm_call_total.labels(status=err_type).inc()
        raise


# ---------------------------------------------------------------------------
# Node: run rules
# ---------------------------------------------------------------------------

def node_run_rules(state: RiskState) -> dict:
    from src.rules.engine import evaluate
    snap = state.get("snapshot")
    # resume 路径：snapshot 已在首次执行时处理，rule_hits 已存于 checkpoint
    if snap is None or not isinstance(snap, FeatureSnapshot):
        return {}
    hits = evaluate(snap)
    if not hits:
        return {"rule_hits": [], "highest_severity": None}

    order = {Severity.P1: 0, Severity.P2: 1, Severity.P3: 2}
    highest = min(hits, key=lambda h: order[h.severity]).severity
    return {"rule_hits": hits, "highest_severity": highest}


# ---------------------------------------------------------------------------
# Node: LLM explainer (only for explanation, never for decisions)
# ---------------------------------------------------------------------------

async def node_llm_explain(state: RiskState) -> dict:
    hits: list[RuleHit] = state.get("rule_hits", [])
    snap = state.get("snapshot")
    # resume 路径：摘要已在首次执行时生成，直接跳过
    if snap is None or not isinstance(snap, FeatureSnapshot):
        return {}

    hits_text = "\n".join(
        f"- [{h.severity.value}] {h.rule_id}: {h.description} (置信度={h.confidence:.0%})"
        for h in hits
    )
    prompt = f"""你是一个加密货币市场风控分析助手。请根据以下规则触发信息，生成一段简洁的中文风险告警说明（100字以内），以及给人工审核人员的简短指引（50字以内）。

资产: {snap.asset.value}
当前价格: ${snap.price:,.2f}
1分钟涨跌: {snap.ret_1m:.2%}
5分钟涨跌: {snap.ret_5m:.2%}
5分钟爆仓金额: ${snap.liq_5m_usd:,.0f}
15分钟OI变动: {snap.oi_delta_15m_pct:.1%}

触发规则:
{hits_text}

请输出JSON格式：
{{"summary_zh": "...", "review_guidance": "..."}}"""

    try:
        import json
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _call_llm_sync, prompt)
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        return {
            "summary_zh": data.get("summary_zh", ""),
            "review_guidance": data.get("review_guidance", ""),
        }
    except Exception as e:
        logger.error("LLM explain failed: %s", e)
        hits_desc = "；".join(h.description for h in hits)
        return {
            "summary_zh": f"{snap.asset.value} 风险预警：{hits_desc}",
            "review_guidance": "请检查市场数据并决定是否发送告警。",
        }


# ---------------------------------------------------------------------------
# Node: decision
# ---------------------------------------------------------------------------

def node_decide(state: RiskState) -> dict:
    hits = state.get("rule_hits", [])
    if not hits:
        return {"decision": Decision.SUPPRESS}

    severity = state.get("highest_severity")
    if severity == Severity.P1:
        return {"decision": Decision.EMIT}
    if severity == Severity.P2:
        return {"decision": Decision.MANUAL_REVIEW}
    # P3 only → suppress
    return {"decision": Decision.SUPPRESS}


# ---------------------------------------------------------------------------
# Node: build case + persist
# ---------------------------------------------------------------------------

async def node_build_case(state: RiskState) -> dict:
    from src.observability.metrics import case_created_total, pending_review_gauge
    from src.persistence.repositories import list_risk_cases
    from src.domain.models import CaseStatus as CS

    decision = state.get("decision")
    severity = state.get("highest_severity")
    case = RiskCase(
        asset=state["asset"],
        rule_hits=state.get("rule_hits", []),
        decision=decision,
        summary_zh=state.get("summary_zh", ""),
        severity=severity,
        status=CS.PENDING_REVIEW if decision == Decision.MANUAL_REVIEW else CS.OPEN,
    )
    await save_risk_case(case)

    # 指标
    case_created_total.labels(
        asset=case.asset.value,
        severity=severity.value if severity else "none",
        decision=decision.value if decision else "none",
    ).inc()

    # 更新 pending_review 积压数
    pending = await list_risk_cases(limit=10000)
    pending_count = sum(1 for c in pending if c.status == CS.PENDING_REVIEW)
    pending_review_gauge.set(pending_count)

    return {"case": case}


# ---------------------------------------------------------------------------
# Node: human review interrupt
# Callers use langgraph interrupt() mechanism — this node just sets status
# ---------------------------------------------------------------------------

def node_human_review(state: RiskState) -> dict:
    """
    此节点使用 interrupt_before 机制：
    - 图在本节点「之前」暂停，不执行节点本身
    - resume 时外部通过 update_state 把 human_approved/human_comment 写入 state
    - 然后图从本节点继续执行，直接读 state 里的值路由到 send_alert 或 END
    """
    return {
        "human_approved": state.get("human_approved"),
        "human_comment": state.get("human_comment", ""),
    }


# ---------------------------------------------------------------------------
# Node: send alert
# ---------------------------------------------------------------------------

async def node_send_alert(state: RiskState) -> dict:
    from src.persistence.repositories import get_risk_case as _get_case
    case = state.get("case")
    if not case:
        return {}
    # checkpoint 反序列化后 Pydantic 对象变成 dict，重新从数据库加载
    if isinstance(case, dict):
        case_id = case.get("case_id", "")
        case = await _get_case(case_id)
        if not case:
            return {}

    decision = state.get("decision")
    if decision == Decision.MANUAL_REVIEW:
        approved = state.get("human_approved")
        if not approved:
            logger.info("Case %s rejected by human reviewer", case.case_id)
            case.status = CaseStatus.CLOSED
            await save_risk_case(case)
            return {}

    severity = state.get("highest_severity", Severity.P3)
    revision = 1
    channel = "webhook"
    idempotency_key = f"{case.case_id}:{revision}:{channel}"

    alert = RiskAlert(
        case_id=case.case_id,
        revision=revision,
        severity=severity,
        title=f"[{severity.value}] {case.asset.value} 风险预警",
        body_zh=state.get("summary_zh", ""),
        idempotency_key=idempotency_key,
        channels_sent=[channel],
    )

    await save_risk_alert(alert)
    await dispatch_alert(alert)

    from src.observability.metrics import alert_sent_total
    alert_sent_total.labels(
        asset=case.asset.value,
        severity=severity.value,
        channel=channel,
    ).inc()

    case.status = CaseStatus.CLOSED
    await save_risk_case(case)
    return {"alert": alert}
