from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger
from src.core.proxy import build_openai_client
from src.domain.models import Asset, FeatureSnapshot
from src.observability.llm_trace import record_llm_call
from src.persistence.repositories import get_model_label, save_model_label

logger = get_logger(__name__)


def _norm_dt(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo else value


def future_summary(
    anchor: FeatureSnapshot,
    future: list[FeatureSnapshot],
    *,
    horizon_seconds: int,
) -> dict[str, Any]:
    """计算锚点快照在未来 horizon 窗口内的统计摘要。

    利用 future 列表已按时间排序的特性，遇到第一个超出 horizon
    的快照即停止扫描，将 O(n²) 降为 O(n)。
    """
    horizon_end = _norm_dt(anchor.window_end) + timedelta(seconds=horizon_seconds)
    anchor_dt = _norm_dt(anchor.window_end)
    future_count = 0
    max_abs_return = 0.0
    max_up_return = 0.0
    max_down_return = 0.0
    max_liq_5m = 0.0
    max_abs_oi = 0.0
    max_abs_fz = 0.0
    max_abs_vz = 0.0

    if anchor.price <= 0:
        return {
            "future_count": 0, "max_abs_return": 0.0, "max_up_return": 0.0,
            "max_down_return": 0.0, "max_liq_5m_usd": 0.0,
            "max_abs_oi_delta_15m_pct": 0.0, "max_abs_funding_z": 0.0,
            "max_abs_vol_z_1m": 0.0,
        }

    for item in future:
        item_dt = _norm_dt(item.window_end)
        if item_dt <= anchor_dt:
            continue
        if item_dt > horizon_end:
            break  # 已排序，后续不可能在窗口内
        if item.price <= 0:
            continue
        ret = (item.price - anchor.price) / anchor.price
        future_count += 1
        if abs(ret) > max_abs_return:
            max_abs_return = abs(ret)
        if ret > max_up_return:
            max_up_return = ret
        if ret < max_down_return:
            max_down_return = ret
        if item.liq_5m_usd > max_liq_5m:
            max_liq_5m = item.liq_5m_usd
        if abs(item.oi_delta_15m_pct) > max_abs_oi:
            max_abs_oi = abs(item.oi_delta_15m_pct)
        if abs(item.funding_z) > max_abs_fz:
            max_abs_fz = abs(item.funding_z)
        if abs(item.vol_z_1m) > max_abs_vz:
            max_abs_vz = abs(item.vol_z_1m)

    return {
        "future_count": future_count,
        "max_abs_return": max_abs_return,
        "max_up_return": max_up_return,
        "max_down_return": max_down_return,
        "max_liq_5m_usd": max_liq_5m,
        "max_abs_oi_delta_15m_pct": max_abs_oi,
        "max_abs_funding_z": max_abs_fz,
        "max_abs_vol_z_1m": max_abs_vz,
    }


def deterministic_label(summary: dict[str, Any]) -> dict[str, Any]:
    max_abs_return = float(summary.get("max_abs_return", 0.0) or 0.0)
    if max_abs_return >= 0.03:
        label = "p1"
        probability = min(1.0, max_abs_return / 0.06)
    elif max_abs_return >= 0.01:
        label = "p2"
        probability = min(0.85, max_abs_return / 0.03)
    else:
        label = "none"
        probability = min(0.25, max_abs_return / 0.01)
    return {
        "label": label,
        "risk_probability": probability,
        "confidence": 0.65,
        "rationale": "deterministic future-window weak label",
    }


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("LLM judge did not return JSON")
    return json.loads(text[start:end])


def _call_llm_judge(anchor: FeatureSnapshot, summary: dict[str, Any]) -> dict[str, Any]:
    client = build_openai_client(service="llm-as-judge labeler")
    prompt = f"""你是加密货币风控训练数据标注员。请判断这个 feature_snapshot 在未来窗口内是否应标为风险。

只能输出 JSON，不要 markdown。字段：
{{"label":"p1|p2|none","risk_probability":0到1,"confidence":0到1,"rationale":"不超过40字"}}

标注规则：
- p1：未来窗口内出现严重风险，例如价格绝对波动 >= 3%，或爆仓/杠杆/资金费率风险非常极端。
- p2：未来窗口内出现中等风险，例如价格绝对波动 >= 1%，或多项风险信号明显恶化。
- none：未来没有足够风险证据。

当前快照：
{json.dumps(anchor.model_dump(mode="json"), ensure_ascii=False)}

未来窗口摘要：
{json.dumps(summary, ensure_ascii=False)}
"""
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.llm_judge_max_tokens,
            timeout=45,
        )
        elapsed = time.perf_counter() - t0
        usage = getattr(response, "usage", None)
        record_llm_call(
            operation="llm_as_judge_feature_snapshot_label",
            status="success",
            duration_ms=elapsed * 1000,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            model=settings.llm_model,
        )
        data = _extract_json(response.choices[0].message.content or "{}")
        label = str(data.get("label", "none")).lower()
        if label not in {"p1", "p2", "none"}:
            label = "none"
        return {
            "label": label,
            "risk_probability": max(0.0, min(1.0, float(data.get("risk_probability", 0.0) or 0.0))),
            "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0))),
            "rationale": str(data.get("rationale", ""))[:200],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        record_llm_call(
            operation="llm_as_judge_feature_snapshot_label",
            status="error",
            duration_ms=elapsed * 1000,
            model=settings.llm_model,
        )
        raise exc


async def label_snapshot(
    anchor: FeatureSnapshot,
    future: list[FeatureSnapshot],
    *,
    horizon_seconds: int,
    use_llm: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    if not force:
        existing = await get_model_label(anchor.snapshot_id)
        if existing is not None:
            return {
                "snapshot_id": existing.snapshot_id,
                "label": existing.label,
                "risk_probability": existing.risk_probability,
                "confidence": existing.confidence,
                "labeling_method": existing.labeling_method,
                "rationale": existing.rationale,
                "cached": True,
            }

    summary = future_summary(anchor, future, horizon_seconds=horizon_seconds)
    method = "llm_as_judge"
    try:
        if use_llm and settings.llm_judge_labeler_enabled and settings.ark_api_key:
            label = _call_llm_judge(anchor, summary)
        else:
            method = "deterministic_fallback"
            label = deterministic_label(summary)
    except Exception as exc:
        logger.warning("LLM judge failed for snapshot %s: %s", anchor.snapshot_id, exc)
        method = "deterministic_fallback_after_llm_error"
        label = deterministic_label(summary)

    await save_model_label(
        snapshot_id=anchor.snapshot_id,
        asset=anchor.asset,
        window_end=anchor.window_end,
        horizon_seconds=horizon_seconds,
        label=label["label"],
        risk_probability=label["risk_probability"],
        confidence=label["confidence"],
        labeling_method=method,
        rationale=label["rationale"],
        judge_payload={"future_summary": summary},
    )
    return {
        "snapshot_id": anchor.snapshot_id,
        "label": label["label"],
        "risk_probability": label["risk_probability"],
        "confidence": label["confidence"],
        "labeling_method": method,
        "rationale": label["rationale"],
        "cached": False,
    }


async def label_snapshot_series(
    series_by_asset: dict[Asset, list[FeatureSnapshot]],
    *,
    horizon_seconds: int,
    max_items: int | None = None,
    use_llm: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    labeled = 0
    cached = 0
    breakdown: dict[str, int] = {}
    method_breakdown: dict[str, int] = {}
    samples = []
    for asset, series in series_by_asset.items():
        ordered = sorted(series, key=lambda item: _norm_dt(item.window_end))
        for index, anchor in enumerate(ordered):
            if max_items is not None and labeled >= max_items:
                return {
                    "labeled": labeled,
                    "cached": cached,
                    "label_breakdown": breakdown,
                    "method_breakdown": method_breakdown,
                    "samples": samples[:5],
                }
            future = ordered[index + 1:]
            if not future:
                continue
            result = await label_snapshot(
                anchor,
                future,
                horizon_seconds=horizon_seconds,
                use_llm=use_llm,
                force=force,
            )
            labeled += 1
            cached += 1 if result.get("cached") else 0
            breakdown[result["label"]] = breakdown.get(result["label"], 0) + 1
            method = result["labeling_method"]
            method_breakdown[method] = method_breakdown.get(method, 0) + 1
            if len(samples) < 5:
                samples.append({"asset": asset.value, **result})
    return {
        "labeled": labeled,
        "cached": cached,
        "label_breakdown": breakdown,
        "method_breakdown": method_breakdown,
        "samples": samples[:5],
    }

