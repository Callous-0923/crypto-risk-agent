"""LangGraph node implementations."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger
from src.core.proxy import build_openai_client
from src.domain.models import CaseStatus, Decision, FeatureSnapshot, RiskAlert, RiskCase, RuleHit, Severity
from src.graph.agent_tools import GRAPH_TOOLS, call_tool
from src.graph.state import RiskState
from src.notification.dispatcher import dispatch_alert
from src.observability.llm_trace import record_llm_call, run_in_executor_with_context
from src.persistence.repositories import save_risk_alert, save_risk_case

logger = get_logger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = build_openai_client(service="graph LLM")
    return _llm


def _call_llm_sync(prompt: str, max_tokens: int = 300, *, operation: str = "llm_sync") -> str:
    return _call_chat_completion_sync(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        operation=operation,
    ).content.strip()


def _call_chat_completion_sync(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    operation: str = "chat_completion",
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
        response = _get_llm().chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0
        llm_call_duration_seconds.observe(elapsed)
        llm_call_total.labels(status="success").inc()
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        record_llm_call(
            operation=operation,
            status="success",
            duration_ms=elapsed * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=settings.llm_model,
        )
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
        record_llm_call(operation=operation, status=err_type, duration_ms=elapsed * 1000, model=settings.llm_model)
        raise


def _extract_json_block(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("LLM did not return JSON")
    return json.loads(text[start:end])


def _short_text(text: str | None, limit: int) -> str:
    normalized = " ".join((text or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}…"


async def _call_llm_with_tools(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    max_tokens: int,
    max_rounds: int,
) -> str:
    working_messages = list(messages)

    for _ in range(max_rounds + 1):
        message = await run_in_executor_with_context(
            lambda: _call_chat_completion_sync(
                working_messages,
                max_tokens=max_tokens,
                tools=tools,
                operation="tool_augmented_chat",
            )
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


def _merge_rule_hits(existing_hits: list[RuleHit], new_hits: list[RuleHit]) -> list[RuleHit]:
    seen = {(hit.rule_id, hit.dedupe_key) for hit in existing_hits}
    merged = list(existing_hits)
    for hit in new_hits:
        key = (hit.rule_id, hit.dedupe_key)
        if key in seen:
            continue
        merged.append(hit)
        seen.add(key)
    return merged


def _has_early_warning_hit(hits: list[RuleHit]) -> bool:
    return any(hit.rule_id.startswith("EW_") for hit in hits)


def _early_warning_recent_candidate_count(memory_context: dict | None) -> int:
    if not memory_context:
        return 0
    try:
        return int(memory_context.get("early_warning_recent_candidates", 0) or 0)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Node: load memory
# ---------------------------------------------------------------------------

def _count_alerts_in_window(recent_alert_history: list[dict], minutes: int = 5) -> tuple[int, str | None]:
    """返回过去 N 分钟内的告警数量，以及最近一条告警的 severity。"""
    from datetime import datetime, timedelta

    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    count = 0
    last_severity: str | None = None
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
            count += 1
            if last_severity is None:
                last_severity = item.get("severity")
    return count, last_severity


async def node_load_memory(state: RiskState) -> dict:
    from datetime import datetime
    from src.persistence.repositories import (
        get_recent_alerts,
        get_recent_review_actions,
        get_recent_snapshots,
        list_risk_cases,
    )

    asset = state.get("asset")
    if asset is None:
        return {
            "recent_alert_history": [],
            "fatigue_suppressed": False,
            "memory_context": None,
        }

    alerts = await get_recent_alerts(asset, limit=5)
    recent_alert_history = [
        {
            "alert_id": alert.alert_id,
            "case_id": alert.case_id,
            "severity": alert.severity.value,
            "title": alert.title,
            "created_at": alert.created_at.isoformat(),
        }
        for alert in alerts
    ]

    # similar_cases：排除当前 thread_id，保留最多 3 条
    current_thread_id = state.get("thread_id")
    cases = await list_risk_cases(asset=asset, limit=5)
    similar_cases: list[dict] = []
    now = datetime.utcnow()
    for c in cases:
        if c.case_id == current_thread_id:
            continue
        created_at = c.created_at
        if created_at.tzinfo is not None:
            created_at = created_at.replace(tzinfo=None)
        age_minutes = max(0, int((now - created_at).total_seconds() // 60))
        similar_cases.append({
            "case_id": c.case_id,
            "severity": c.severity.value if c.severity else None,
            "decision": c.decision.value if c.decision else None,
            "summary_zh": _short_text(c.summary_zh, 60),
            "age_minutes": age_minutes,
        })
        if len(similar_cases) >= 3:
            break

    # review_actions：最多 3 条
    review_rows = await get_recent_review_actions(asset, limit=5)
    review_actions: list[dict] = []
    for r in review_rows:
        created_at = r.created_at
        if created_at.tzinfo is not None:
            created_at = created_at.replace(tzinfo=None)
        age_minutes = max(0, int((now - created_at).total_seconds() // 60))
        review_actions.append({
            "action": r.action.value,
            "comment": _short_text(r.comment, 30),
            "age_minutes": age_minutes,
        })
        if len(review_actions) >= 3:
            break

    # alert_fatigue
    fatigue_count, last_severity = _count_alerts_in_window(recent_alert_history, minutes=5)
    early_warning_recent_candidates = 0
    early_warning_recent_profiles: list[dict] = []
    snap = state.get("snapshot")
    try:
        from src.rules.config import registry
        from src.rules.engine import early_warning_profile_from_hits, evaluate_with_thresholds

        window = max(1, registry.thresholds.early_warning_persistence_window)
        recent_snapshots = await get_recent_snapshots(asset, n=window + 2)
        for recent in recent_snapshots:
            if isinstance(snap, FeatureSnapshot):
                if recent.snapshot_id == snap.snapshot_id:
                    continue
                recent_time = recent.window_end.replace(tzinfo=None) if recent.window_end.tzinfo else recent.window_end
                snap_time = snap.window_end.replace(tzinfo=None) if snap.window_end.tzinfo else snap.window_end
                if recent_time >= snap_time:
                    continue
            profile = early_warning_profile_from_hits(
                evaluate_with_thresholds(recent, registry.thresholds)
            )
            if profile["candidate"]:
                early_warning_recent_candidates += 1
            early_warning_recent_profiles.append({
                "snapshot_id": recent.snapshot_id,
                "score": profile["score"],
                "signal_count": profile["signal_count"],
                "candidate": profile["candidate"],
            })
            if len(early_warning_recent_profiles) >= max(0, window - 1):
                break
    except Exception as exc:
        logger.debug("Failed to load early-warning persistence context: %s", exc)

    memory_context = {
        "similar_cases": similar_cases,
        "review_actions": review_actions,
        "alert_fatigue": {
            "count_5min": fatigue_count,
            "last_severity": last_severity,
        },
        "early_warning_recent_candidates": early_warning_recent_candidates,
        "early_warning_recent_profiles": early_warning_recent_profiles,
    }

    # Build enriched long-term memory context
    enriched_memory = None
    try:
        from src.memory.manager import get_memory_manager

        mm = get_memory_manager()
        enriched_memory = await mm.build_enriched_context(
            asset=asset,
            snapshot=snap if isinstance(snap, FeatureSnapshot) else None,
            rule_hits=state.get("rule_hits", []),
            short_term_context=memory_context,
        )
    except Exception as exc:
        logger.debug("Failed to build enriched memory: %s", exc)

    return {
        "recent_alert_history": recent_alert_history,
        "fatigue_suppressed": False,
        "memory_context": memory_context,
        "enriched_memory": enriched_memory,
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
    ml_hit = None
    try:
        from src.ml.risk_model import prediction_to_rule_hit

        ml_hit = prediction_to_rule_hit(snap, state.get("ml_prediction"))
    except Exception as exc:
        logger.debug("Failed to convert ML prediction to rule hit: %s", exc)
    if ml_hit is not None:
        hits.append(ml_hit)
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
        text = await run_in_executor_with_context(
            lambda: _call_llm_sync(prompt, 300, operation="legacy_explainer")
        )
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

    severity = state.get("highest_severity")
    if severity == Severity.P3:
        return _summary_fallback(snap, hits, [])

    memory_context = state.get("memory_context") or {}

    if severity == Severity.P1:
        # P1：极简记忆 + 专家流水线
        technical = state.get("technical_analysis")
        macro = state.get("macro_context")
        mismatch_flags = _collect_metric_mismatches(snap, [technical, macro])

        similar_cases = memory_context.get("similar_cases", [])[:1]
        minimal_memory = {"similar_cases": similar_cases} if similar_cases else {}

        messages = [
            {
                "role": "system",
                "content": (
                    "你是风险摘要汇总助手。请优先使用原始 snapshot 数据，"
                    "若专家输出与 snapshot 不一致，必须以 snapshot 为准并在 review_guidance 中提醒。"
                    "只能输出单行 JSON，不得输出 markdown、标题、解释或代码块。"
                ),
            },
            {
                "role": "user",
                "content": f"""根据以下信息输出最终中文摘要。

约束：
1. summary_zh 最多 120 个汉字。
2. review_guidance 最多 60 个汉字。
3. 只保留审核决策最关键的信息，不要重复原始 JSON 字段。

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

memory（仅参考最近一条同资产 case）:
{json.dumps(minimal_memory, ensure_ascii=False)}

输出 JSON：
{{"summary_zh":"...","review_guidance":"..."}}""",
            },
        ]
        max_tokens = 160
        operation = "summarizer_p1"
    else:
        # P2：完整记忆（替代专家流水线）+ 长期记忆增强
        mismatch_flags = []

        # Build long-term memory section for prompt
        enriched = state.get("enriched_memory") or {}
        long_term_section = ""
        semantic_matches = enriched.get("semantic_matches", [])
        if semantic_matches:
            match_lines = []
            for m in semantic_matches[:3]:
                match_lines.append(
                    f"  - [{m.get('severity','?')}] {m.get('summary_zh','')[:60]} "
                    f"(相似度={m.get('similarity',0):.2f}, 决策={m.get('decision','?')})"
                )
            long_term_section += f"\n语义相似历史案例:\n" + "\n".join(match_lines)

        insights = enriched.get("experience_insights", [])
        if insights:
            insight_lines = [f"  - {ins.get('insight_zh','')}" for ins in insights[:3]]
            long_term_section += f"\n\n经验规律:\n" + "\n".join(insight_lines)

        reviewer_pref = enriched.get("reviewer_preference")
        pref_section = ""
        if reviewer_pref and reviewer_pref.get("sample_count", 0) >= 3:
            pref_section = (
                f"\n\n审核者偏好（基于{reviewer_pref['sample_count']}个样本）: "
                f"{reviewer_pref.get('pattern_description_zh', '')}"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是 P2 风险摘要汇总助手。当前路径不经专家流水线，你是唯一的 LLM 分析节点。"
                    "请结合短期记忆（相似 case、审核动作、告警疲劳）和长期记忆（语义相似历史案例、经验规律、审核偏好）综合判断并给出审核指引。"
                    "只能输出单行 JSON，不得输出 markdown、标题、解释或代码块。"
                ),
            },
            {
                "role": "user",
                "content": f"""根据以下信息输出最终中文摘要与审核指引。

约束：
1. summary_zh 最多 120 个汉字；review_guidance 最多 60 个汉字。
2. 若 review_actions 中出现 REJECT 且特征相似，必须在 review_guidance 中提示"历史已拒绝，参考原拒绝理由"。
3. 若 alert_fatigue.count_5min >= 3，review_guidance 需提示"疑似告警疲劳/风暴"。
4. similar_cases 中 age_minutes 越小越相关。
5. 若有语义相似历史案例且其决策为 suppress/reject，需在 review_guidance 中提示。
6. 若审核者偏好显示高拒绝率，需在 review_guidance 中提醒。
7. 只保留审核决策最关键的信息，不要重复原始 JSON 字段。

snapshot:
{json.dumps(snap.model_dump(mode="json"), ensure_ascii=False)}

rule_hits:
{json.dumps([hit.model_dump(mode="json") for hit in hits], ensure_ascii=False)}

短期记忆:
{json.dumps(memory_context, ensure_ascii=False)}
{long_term_section}{pref_section}

输出 JSON：
{{"summary_zh":"...","review_guidance":"..."}}""",
            },
        ]
        max_tokens = 240
        operation = "summarizer_p2"

    try:
        text = await run_in_executor_with_context(
            lambda: _call_chat_completion_sync(
                messages,
                max_tokens=max_tokens,
                operation=operation,
            ).content.strip(),
        )
        data = _extract_json_block(text)
        summary_zh = _short_text(data.get("summary_zh", ""), 120)
        review_guidance = _short_text(data.get("review_guidance", ""), 60)
        if mismatch_flags:
            review_guidance = (
                f"{review_guidance} 专家指标与原始快照存在不一致：{'; '.join(mismatch_flags)}。"
            ).strip()
        return {
            "summary_zh": summary_zh,
            "review_guidance": _short_text(review_guidance, 120),
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
        # Check enriched memory for reviewer preference signals
        enriched = state.get("enriched_memory") or {}
        reviewer_pref = enriched.get("reviewer_preference")

        # If reviewer historically rejects this pattern >80% of the time with enough
        # samples, suppress it to reduce noise (still creates a case for audit)
        if (
            reviewer_pref
            and reviewer_pref.get("sample_count", 0) >= 5
            and reviewer_pref.get("reject_rate", 0) >= 0.80
        ):
            return {"decision": Decision.SUPPRESS, "fatigue_suppressed": False}

        # If semantic matches show similar cases were all suppressed, consider suppression
        semantic_matches = enriched.get("semantic_matches", [])
        if semantic_matches:
            high_sim_matches = [m for m in semantic_matches if m.get("similarity", 0) >= 0.85]
            if (
                len(high_sim_matches) >= 3
                and all(m.get("decision") == "suppress" for m in high_sim_matches)
            ):
                return {"decision": Decision.SUPPRESS, "fatigue_suppressed": False}

        return {"decision": Decision.MANUAL_REVIEW, "fatigue_suppressed": False}
    if severity == Severity.P3:
        return {"decision": Decision.SUPPRESS, "fatigue_suppressed": False}
    return {"decision": Decision.SUPPRESS, "fatigue_suppressed": False}


# ---------------------------------------------------------------------------
# Node: build case + persist
# ---------------------------------------------------------------------------

async def node_build_case(state: RiskState) -> dict:
    from datetime import datetime

    from src.domain.models import CaseStatus as CS
    from src.observability.metrics import case_created_total, pending_review_gauge
    from src.persistence.repositories import find_active_case_by_dedupe_key, list_risk_cases
    from src.graph.review_assistants import build_review_assistance

    decision = state.get("decision")
    severity = state.get("highest_severity")
    hits = state.get("rule_hits", [])
    dedupe_key = _select_case_dedupe_key(hits, severity)

    if severity == Severity.P3:
        return {"case": None, "case_reused": False}

    if (
        decision == Decision.EMIT
        and severity == Severity.P1
        and not state.get("is_coordinator_case", False)
        and dedupe_key
    ):
        existing = await find_active_case_by_dedupe_key(state["asset"], dedupe_key, within_seconds=300)
        if existing is not None:
            from src.persistence.repositories import save_quality_metric_event_sync

            save_quality_metric_event_sync(
                "alert_duplicate_suppressed",
                asset=state["asset"],
                severity=severity,
                case_id=existing.case_id,
                dedupe_key=dedupe_key,
                details={"thread_id": state["thread_id"]},
            )
            pending = await list_risk_cases(limit=10000)
            pending_count = sum(1 for item in pending if item.status == CS.PENDING_REVIEW)
            pending_review_gauge.set(pending_count)
            return {
                "case": existing,
                "case_reused": True,
                "historical_context_zh": existing.historical_context_zh,
                "risk_quantification_zh": existing.risk_quantification_zh,
            }

    if (
        decision == Decision.MANUAL_REVIEW
        and severity == Severity.P2
        and not state.get("is_coordinator_case", False)
        and dedupe_key
    ):
        existing = await find_active_case_by_dedupe_key(state["asset"], dedupe_key, within_seconds=3600)
        if existing is not None and existing.status in {CS.OPEN, CS.PENDING_REVIEW}:
            from src.persistence.repositories import save_quality_metric_event

            existing.rule_hits = _merge_rule_hits(existing.rule_hits, hits)
            existing.updated_at = datetime.utcnow()
            latest_summary = state.get("summary_zh", "")
            if latest_summary:
                existing.summary_zh = latest_summary
            existing.severity = existing.severity or severity
            existing.decision = existing.decision or decision
            if not existing.historical_context_zh:
                existing.historical_context_zh = state.get("historical_context_zh", "")
            if not existing.risk_quantification_zh:
                existing.risk_quantification_zh = state.get("risk_quantification_zh", "")
            await save_risk_case(existing)
            await save_quality_metric_event(
                "p2_case_aggregated",
                asset=state["asset"],
                severity=severity,
                case_id=existing.case_id,
                dedupe_key=dedupe_key,
                details={"thread_id": state["thread_id"]},
            )
            pending = await list_risk_cases(limit=10000)
            pending_count = sum(1 for item in pending if item.status == CS.PENDING_REVIEW)
            pending_review_gauge.set(pending_count)
            return {
                "case": existing,
                "case_reused": True,
                "historical_context_zh": existing.historical_context_zh,
                "risk_quantification_zh": existing.risk_quantification_zh,
            }

    if (
        decision == Decision.MANUAL_REVIEW
        and severity == Severity.P3
        and _has_early_warning_hit(hits)
        and not state.get("is_coordinator_case", False)
        and dedupe_key
    ):
        existing = await find_active_case_by_dedupe_key(state["asset"], dedupe_key, within_seconds=3600)
        if existing is not None and existing.status in {CS.OPEN, CS.PENDING_REVIEW}:
            from src.persistence.repositories import save_quality_metric_event

            existing.rule_hits = _merge_rule_hits(existing.rule_hits, hits)
            existing.updated_at = datetime.utcnow()
            latest_summary = state.get("summary_zh", "")
            if latest_summary:
                existing.summary_zh = latest_summary
            existing.severity = existing.severity or severity
            existing.decision = existing.decision or decision
            await save_risk_case(existing)
            await save_quality_metric_event(
                "early_warning_aggregated",
                asset=state["asset"],
                severity=severity,
                case_id=existing.case_id,
                dedupe_key=dedupe_key,
                details={"thread_id": state["thread_id"]},
            )
            pending = await list_risk_cases(limit=10000)
            pending_count = sum(1 for item in pending if item.status == CS.PENDING_REVIEW)
            pending_review_gauge.set(pending_count)
            return {
                "case": existing,
                "case_reused": True,
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
    if decision == Decision.MANUAL_REVIEW and severity == Severity.P3 and _has_early_warning_hit(hits):
        from src.persistence.repositories import save_quality_metric_event

        await save_quality_metric_event(
            "early_warning_created",
            asset=case.asset,
            severity=severity,
            case_id=case.case_id,
            dedupe_key=dedupe_key,
            details={"rule_ids": [hit.rule_id for hit in hits if hit.rule_id.startswith("EW_")]},
        )

    historical_context_zh = state.get("historical_context_zh", "")
    risk_quantification_zh = state.get("risk_quantification_zh", "")
    if (
        decision == Decision.MANUAL_REVIEW
        and severity == Severity.P2
        and not case.is_coordinator_case
        and isinstance(state.get("snapshot"), FeatureSnapshot)
    ):
        historical_context_zh, risk_quantification_zh = await build_review_assistance(case, state["snapshot"])
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

    # Store case embedding for future semantic retrieval
    try:
        from src.memory.manager import get_memory_manager
        await get_memory_manager().store_case_embedding(case)
    except Exception as exc:
        logger.debug("Failed to store case embedding: %s", exc)

    return {
        "case": case,
        "case_reused": False,
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
    if severity == Severity.P3:
        logger.info("Skip alert emission for P3 case %s", case.case_id)
        case.status = CaseStatus.CLOSED
        await save_risk_case(case)
        return {}
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
