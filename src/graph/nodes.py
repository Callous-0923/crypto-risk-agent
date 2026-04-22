"""LangGraph node implementations."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import OpenAI

from src.core.config import settings
from src.core.logging import get_logger
from src.domain.models import CaseStatus, Decision, FeatureSnapshot, RiskAlert, RiskCase, RuleHit, Severity
from src.graph.agent_tools import GRAPH_TOOLS, call_tool
from src.graph.state import RiskState
from src.notification.dispatcher import dispatch_alert
from src.persistence.repositories import save_risk_alert, save_risk_case

logger = get_logger(__name__)

_llm = OpenAI(api_key=settings.ark_api_key, base_url=settings.ark_base_url)


def _call_llm_sync(prompt: str, max_tokens: int = 300) -> str:
    return _call_chat_completion_sync(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    ).content.strip()


def _call_chat_completion_sync(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
):
    """同步调用 LLM，在线程池中执行避免事件循环冲突。记录耗时和错误指标。"""
    import time
    from src.observability.metrics import llm_call_duration_seconds, llm_call_total, llm_error_total

    t0 = time.perf_counter()
    try:
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "max_tokens": max_tokens,
            "messages": messages,
            "timeout": 30,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False
        response = _llm.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0
        llm_call_duration_seconds.observe(elapsed)
        llm_call_total.labels(status="success").inc()
        return response.choices[0].message
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        llm_call_duration_seconds.observe(elapsed)
        err_type = (
            "timeout" if "timeout" in str(exc).lower()
            else "auth" if "401" in str(exc)
            else "network" if "connect" in str(exc).lower()
            else "other"
        )
        llm_error_total.labels(error_type=err_type).inc()
        llm_call_total.labels(status=err_type).inc()
        raise


def _extract_json_block(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("LLM did not return JSON")
    return json.loads(text[start:end])


async def _call_llm_with_tools(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    max_tokens: int,
    max_rounds: int,
) -> str:
    loop = asyncio.get_running_loop()
    working_messages = list(messages)

    for _ in range(max_rounds + 1):
        message = await loop.run_in_executor(
            None,
            lambda: _call_chat_completion_sync(working_messages, max_tokens=max_tokens, tools=tools),
        )
        tool_calls = getattr(message, "tool_calls", None) or []
        content = (message.content or "").strip()
        if not tool_calls:
            return content

        assistant_message = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ],
        }
        working_messages.append(assistant_message)

        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            tool_output = await call_tool(tool_call.function.name, args)
            working_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )

    raise RuntimeError("Tool loop exceeded max_rounds")


def _format_rule_hits(hits: list[RuleHit]) -> str:
    return "\n".join(
        f"- [{hit.severity.value}] {hit.rule_id}: {hit.description} (置信度={hit.confidence:.0%})"
        for hit in hits
    )


def _technical_fallback(snap: FeatureSnapshot) -> dict[str, Any]:
    return {
        "key_metrics": {
            "ret_1m": snap.ret_1m,
            "ret_5m": snap.ret_5m,
            "liq_5m_usd": snap.liq_5m_usd,
            "oi_delta_15m_pct": snap.oi_delta_15m_pct,
            "funding_z": snap.funding_z,
        },
        "confidence": 0.6,
        "narrative_zh": f"{snap.asset.value} 出现显著价格或衍生品异常，需结合兄弟资产走势确认是否为系统性风险。",
    }


def _macro_fallback(snap: FeatureSnapshot, hits: list[RuleHit]) -> dict[str, Any]:
    return {
        "key_metrics": {
            "oi_delta_15m_pct": snap.oi_delta_15m_pct,
            "funding_z": snap.funding_z,
            "rule_hit_count": len(hits),
        },
        "confidence": 0.6,
        "narrative_zh": f"{snap.asset.value} 当前规则命中 {len(hits)} 条，需结合历史告警与规则阈值判断是否持续扩散。",
    }


def _summary_fallback(snap: FeatureSnapshot, hits: list[RuleHit], mismatch_flags: list[str]) -> dict[str, str]:
    hit_desc = "；".join(hit.description for hit in hits) or f"{snap.asset.value} 无新增规则命中"
    review_guidance = "请核对快照与规则证据，确认是否需要发送风险告警。"
    if mismatch_flags:
        review_guidance += " 检测到专家输出与原始快照不一致，请以 snapshot 为准。"
    return {
        "summary_zh": f"{snap.asset.value} 风险摘要：{hit_desc}",
        "review_guidance": review_guidance,
    }


def _collect_metric_mismatches(snap: FeatureSnapshot, payloads: list[dict[str, Any] | None]) -> list[str]:
    expectations = {
        "price": (snap.price, 1e-6),
        "ret_1m": (snap.ret_1m, 1e-3),
        "ret_5m": (snap.ret_5m, 1e-3),
        "liq_5m_usd": (snap.liq_5m_usd, 1.0),
        "oi_delta_15m_pct": (snap.oi_delta_15m_pct, 1e-3),
        "funding_z": (snap.funding_z, 1e-3),
    }
    mismatches: list[str] = []
    for payload in payloads:
        if not payload:
            continue
        metrics = payload.get("key_metrics", {})
        for key, (expected, tolerance) in expectations.items():
            if key not in metrics:
                continue
            value = metrics[key]
            if not isinstance(value, (int, float)):
                continue
            if abs(float(value) - expected) > tolerance:
                mismatches.append(f"{key}={value} != snapshot({expected})")
    return mismatches


def _fatigue_window_hit(recent_alert_history: list[dict]) -> bool:
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(minutes=5)
    recent_count = 0
    for item in recent_alert_history:
        created_at = item.get("created_at")
        if not created_at:
            continue
        try:
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if created_dt.tzinfo is not None:
            created_dt = created_dt.replace(tzinfo=None)
        if created_dt >= cutoff:
            recent_count += 1
    return recent_count >= 3


def _select_case_dedupe_key(hits: list[RuleHit], severity: Severity | None) -> str:
    for hit in hits:
        if hit.severity == severity and hit.dedupe_key:
            return hit.dedupe_key
    for hit in hits:
        if hit.dedupe_key:
            return hit.dedupe_key
    return ""


# ---------------------------------------------------------------------------
# Node: load memory
# ---------------------------------------------------------------------------

async def node_load_memory(state: RiskState) -> dict:
    from src.persistence.repositories import get_recent_alerts

    asset = state.get("asset")
    if asset is None:
        return {"recent_alert_history": [], "fatigue_suppressed": False}

    alerts = await get_recent_alerts(asset, limit=5)
    return {
        "recent_alert_history": [
            {
                "alert_id": alert.alert_id,
                "case_id": alert.case_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "created_at": alert.created_at.isoformat(),
            }
            for alert in alerts
        ],
        "fatigue_suppressed": False,
    }


# ---------------------------------------------------------------------------
# Node: run rules
# ---------------------------------------------------------------------------

def node_run_rules(state: RiskState) -> dict:
    from src.rules.engine import evaluate

    snap = state.get("snapshot")
    if snap is None or not isinstance(snap, FeatureSnapshot):
        return {}
    hits = evaluate(snap)
    if not hits:
        return {"rule_hits": [], "highest_severity": None}

    order = {Severity.P1: 0, Severity.P2: 1, Severity.P3: 2}
    highest = min(hits, key=lambda hit: order[hit.severity]).severity
    return {"rule_hits": hits, "highest_severity": highest}


# ---------------------------------------------------------------------------
# Legacy Node: single-shot explainer (kept as fallback / benchmark)
# ---------------------------------------------------------------------------

async def node_llm_explain(state: RiskState) -> dict:
    hits: list[RuleHit] = state.get("rule_hits", [])
    snap = state.get("snapshot")
    if snap is None or not isinstance(snap, FeatureSnapshot):
        return {}

    prompt = f"""你是一个加密货币市场风控分析助手。请根据以下规则触发信息，生成一段简洁的中文风险告警说明（100字以内），以及给人工审核人员的简短指引（50字以内）。

资产: {snap.asset.value}
当前价格: ${snap.price:,.2f}
1分钟涨跌: {snap.ret_1m:.2%}
5分钟涨跌: {snap.ret_5m:.2%}
5分钟爆仓金额: ${snap.liq_5m_usd:,.0f}
15分钟OI变动: {snap.oi_delta_15m_pct:.1%}

触发规则:
{_format_rule_hits(hits)}

请直接输出 JSON：
{{"summary_zh":"...","review_guidance":"..."}}"""

    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, lambda: _call_llm_sync(prompt, 300))
        data = _extract_json_block(text)
        return {
            "summary_zh": data.get("summary_zh", ""),
            "review_guidance": data.get("review_guidance", ""),
        }
    except Exception as exc:
        logger.error("LLM explain failed: %s", exc)
        return _summary_fallback(snap, hits, [])


# ---------------------------------------------------------------------------
# Node: expert analysts
# ---------------------------------------------------------------------------

async def node_technical_analyst(state: RiskState) -> dict:
    snap = state.get("snapshot")
    hits: list[RuleHit] = state.get("rule_hits", [])
    if state.get("highest_severity") != Severity.P1 or snap is None or not isinstance(snap, FeatureSnapshot):
        return {"technical_analysis": None, "technical_analysis_zh": ""}

    messages = [
        {
            "role": "system",
            "content": (
                "你是技术分析专家。关注价格动作、爆仓、持仓与跨资产共振。"
                "只能直接输出 JSON，不要输出额外解释。"
            ),
        },
        {
            "role": "user",
            "content": f"""分析以下 P1 风险快照，并在需要时调用工具补充上下文。

资产: {snap.asset.value}
价格: {snap.price}
ret_1m: {snap.ret_1m}
ret_5m: {snap.ret_5m}
liq_5m_usd: {snap.liq_5m_usd}
oi_delta_15m_pct: {snap.oi_delta_15m_pct}
funding_z: {snap.funding_z}

触发规则:
{_format_rule_hits(hits)}

输出 JSON：
{{
  "key_metrics": {{
    "ret_1m": number,
    "ret_5m": number,
    "liq_5m_usd": number,
    "oi_delta_15m_pct": number,
    "funding_z": number
  }},
  "confidence": number,
  "narrative_zh": "..."
}}""",
        },
    ]

    try:
        text = await _call_llm_with_tools(messages, tools=GRAPH_TOOLS, max_tokens=400, max_rounds=2)
        data = _extract_json_block(text)
        return {
            "technical_analysis": data,
            "technical_analysis_zh": data.get("narrative_zh", ""),
        }
    except Exception as exc:
        logger.warning("Technical analyst failed: %s", exc)
        fallback = _technical_fallback(snap)
        return {
            "technical_analysis": fallback,
            "technical_analysis_zh": fallback["narrative_zh"],
        }


async def node_macro_context(state: RiskState) -> dict:
    snap = state.get("snapshot")
    hits: list[RuleHit] = state.get("rule_hits", [])
    if state.get("highest_severity") != Severity.P1 or snap is None or not isinstance(snap, FeatureSnapshot):
        return {"macro_context": None, "macro_context_zh": ""}

    messages = [
        {
            "role": "system",
            "content": (
                "你是市场上下文分析专家。关注告警历史、规则版本、持仓和资金费率变化。"
                "只能直接输出 JSON，不要输出额外解释。"
            ),
        },
        {
            "role": "user",
            "content": f"""分析以下 P1 风险快照，并在需要时调用工具补充上下文。

资产: {snap.asset.value}
价格: {snap.price}
oi_delta_15m_pct: {snap.oi_delta_15m_pct}
funding_z: {snap.funding_z}

触发规则:
{_format_rule_hits(hits)}

输出 JSON：
{{
  "key_metrics": {{
    "oi_delta_15m_pct": number,
    "funding_z": number,
    "rule_hit_count": number
  }},
  "confidence": number,
  "narrative_zh": "..."
}}""",
        },
    ]

    try:
        text = await _call_llm_with_tools(messages, tools=GRAPH_TOOLS, max_tokens=300, max_rounds=1)
        data = _extract_json_block(text)
        return {
            "macro_context": data,
            "macro_context_zh": data.get("narrative_zh", ""),
        }
    except Exception as exc:
        logger.warning("Macro context analyst failed: %s", exc)
        fallback = _macro_fallback(snap, hits)
        return {
            "macro_context": fallback,
            "macro_context_zh": fallback["narrative_zh"],
        }


async def node_expert_parallel(state: RiskState) -> dict:
    if state.get("highest_severity") != Severity.P1:
        return {
            "technical_analysis": None,
            "macro_context": None,
            "technical_analysis_zh": "",
            "macro_context_zh": "",
        }

    technical, macro = await asyncio.gather(
        node_technical_analyst(state),
        node_macro_context(state),
    )
    return {**technical, **macro}


async def node_summarizer(state: RiskState) -> dict:
    hits: list[RuleHit] = state.get("rule_hits", [])
    snap = state.get("snapshot")
    if not hits or snap is None or not isinstance(snap, FeatureSnapshot):
        return {"summary_zh": state.get("summary_zh", ""), "review_guidance": state.get("review_guidance", "")}
    if state.get("highest_severity") == Severity.P3:
        return _summary_fallback(snap, hits, [])

    technical = state.get("technical_analysis")
    macro = state.get("macro_context")
    mismatch_flags = _collect_metric_mismatches(snap, [technical, macro])

    messages = [
        {
            "role": "system",
            "content": (
                "你是风险摘要汇总助手。请优先使用原始 snapshot 数据，"
                "若专家输出与 snapshot 不一致，必须以 snapshot 为准并在 review_guidance 中提醒。"
                "只能直接输出 JSON，不要输出额外解释。"
            ),
        },
        {
            "role": "user",
            "content": f"""根据以下信息输出最终中文摘要。

snapshot:
{json.dumps(snap.model_dump(mode="json"), ensure_ascii=False)}

rule_hits:
{json.dumps([hit.model_dump(mode="json") for hit in hits], ensure_ascii=False)}

technical_analysis:
{json.dumps(technical, ensure_ascii=False)}

macro_context:
{json.dumps(macro, ensure_ascii=False)}

mismatch_flags:
{json.dumps(mismatch_flags, ensure_ascii=False)}

输出 JSON：
{{"summary_zh":"...","review_guidance":"..."}}""",
        },
    ]

    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            lambda: _call_chat_completion_sync(messages, max_tokens=250).content.strip(),
        )
        data = _extract_json_block(text)
        review_guidance = data.get("review_guidance", "")
        if mismatch_flags:
            review_guidance = (
                f"{review_guidance} 专家指标与原始快照存在不一致：{'; '.join(mismatch_flags)}。"
            ).strip()
        return {
            "summary_zh": data.get("summary_zh", ""),
            "review_guidance": review_guidance,
        }
    except Exception as exc:
        logger.warning("Summarizer failed: %s", exc)
        return _summary_fallback(snap, hits, mismatch_flags)


# ---------------------------------------------------------------------------
# Node: decision
# ---------------------------------------------------------------------------

def node_decide(state: RiskState) -> dict:
    hits = state.get("rule_hits", [])
    if not hits:
        return {"decision": Decision.SUPPRESS, "fatigue_suppressed": False}

    severity = state.get("highest_severity")
    if severity == Severity.P2 and _fatigue_window_hit(state.get("recent_alert_history", [])):
        return {"decision": Decision.SUPPRESS, "fatigue_suppressed": True}
    if severity == Severity.P1:
        return {"decision": Decision.EMIT, "fatigue_suppressed": False}
    if severity == Severity.P2:
        return {"decision": Decision.MANUAL_REVIEW, "fatigue_suppressed": False}
    return {"decision": Decision.SUPPRESS, "fatigue_suppressed": False}


# ---------------------------------------------------------------------------
# Node: build case + persist
# ---------------------------------------------------------------------------

async def node_build_case(state: RiskState) -> dict:
    from src.domain.models import CaseStatus as CS
    from src.observability.metrics import case_created_total, pending_review_gauge
    from src.persistence.repositories import find_active_case_by_dedupe_key, list_risk_cases
    from src.graph.review_assistants import build_historical_context, quantify_risk

    decision = state.get("decision")
    severity = state.get("highest_severity")
    hits = state.get("rule_hits", [])
    dedupe_key = _select_case_dedupe_key(hits, severity)

    if (
        decision == Decision.EMIT
        and severity == Severity.P1
        and not state.get("is_coordinator_case", False)
        and dedupe_key
    ):
        existing = await find_active_case_by_dedupe_key(state["asset"], dedupe_key, within_seconds=300)
        if existing is not None:
            pending = await list_risk_cases(limit=10000)
            pending_count = sum(1 for item in pending if item.status == CS.PENDING_REVIEW)
            pending_review_gauge.set(pending_count)
            return {
                "case": existing,
                "historical_context_zh": existing.historical_context_zh,
                "risk_quantification_zh": existing.risk_quantification_zh,
            }

    status = CS.OPEN
    suppression_reason = None
    if decision == Decision.MANUAL_REVIEW:
        status = CS.PENDING_REVIEW
    elif decision == Decision.SUPPRESS and state.get("fatigue_suppressed"):
        status = CS.SUPPRESSED
        suppression_reason = "fatigue"

    case = RiskCase(
        case_id=state["thread_id"],
        asset=state["asset"],
        rule_hits=hits,
        decision=decision,
        summary_zh=state.get("summary_zh", ""),
        severity=severity,
        status=status,
        is_coordinator_case=state.get("is_coordinator_case", False),
        historical_context_zh=state.get("historical_context_zh", ""),
        risk_quantification_zh=state.get("risk_quantification_zh", ""),
        suppression_reason=suppression_reason,
    )
    await save_risk_case(case)

    historical_context_zh = state.get("historical_context_zh", "")
    risk_quantification_zh = state.get("risk_quantification_zh", "")
    if (
        decision == Decision.MANUAL_REVIEW
        and severity == Severity.P2
        and not case.is_coordinator_case
        and isinstance(state.get("snapshot"), FeatureSnapshot)
    ):
        historical_context_zh, risk_quantification_zh = await asyncio.gather(
            build_historical_context(case),
            quantify_risk(case, state["snapshot"]),
        )
        case.historical_context_zh = historical_context_zh
        case.risk_quantification_zh = risk_quantification_zh
        await save_risk_case(case)

    case_created_total.labels(
        asset=case.asset.value,
        severity=severity.value if severity else "none",
        decision=decision.value if decision else "none",
    ).inc()

    pending = await list_risk_cases(limit=10000)
    pending_count = sum(1 for item in pending if item.status == CS.PENDING_REVIEW)
    pending_review_gauge.set(pending_count)

    return {
        "case": case,
        "historical_context_zh": historical_context_zh,
        "risk_quantification_zh": risk_quantification_zh,
    }


# ---------------------------------------------------------------------------
# Node: human review interrupt
# ---------------------------------------------------------------------------

def node_human_review(state: RiskState) -> dict:
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

    severity = state.get("highest_severity") or case.severity or Severity.P3
    revision = 1
    channel = "webhook"
    idempotency_key = f"{case.case_id}:{revision}:{channel}"
    title = (
        f"[{severity.value}] 多资产联动风险预警"
        if case.is_coordinator_case
        else f"[{severity.value}] {case.asset.value} 风险预警"
    )

    alert = RiskAlert(
        case_id=case.case_id,
        revision=revision,
        severity=severity,
        title=title,
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
