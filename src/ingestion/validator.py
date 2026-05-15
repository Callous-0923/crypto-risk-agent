"""Cross-source price validation between Binance and OKX."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger
from src.core.proxy import build_openai_client
from src.domain.models import Asset
from src.observability.llm_trace import record_llm_call, run_in_executor_with_context
from src.observability.metrics import llm_call_duration_seconds, llm_call_total, llm_error_total
from src.rules.config import registry

logger = get_logger(__name__)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = build_openai_client(service="price validation LLM")
    return _llm


@dataclass
class ValidationResult:
    trusted_price: float
    conflict_detected: bool
    resolution_reason: str


def _call_llm_json_sync(prompt: str) -> dict[str, Any]:
    import time

    t0 = time.perf_counter()
    try:
        response = _get_llm().chat.completions.create(
            model=settings.llm_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        elapsed = time.perf_counter() - t0
        llm_call_duration_seconds.observe(elapsed)
        llm_call_total.labels(status="success").inc()
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        record_llm_call(
            operation="price_validation",
            status="success",
            duration_ms=elapsed * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=settings.llm_model,
        )
        text = (response.choices[0].message.content or "").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
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
        record_llm_call(operation="price_validation", status=err_type, duration_ms=elapsed * 1000, model=settings.llm_model)
        raise


async def _arbitrate_conflict(asset: Asset, binance_price: float, okx_price: float) -> dict[str, Any]:
    prompt = f"""你是行情源冲突仲裁助手。请在 Binance 与 OKX 价格冲突时选择更可信的数据源。

资产: {asset.value}
Binance 价格: {binance_price}
OKX 价格: {okx_price}
价差比例: {abs(binance_price - okx_price) / min(binance_price, okx_price):.4%}

请直接输出 JSON：
{{"trust":"binance|okx|both","confidence":0-1,"reason":"..."}}"""
    return await run_in_executor_with_context(lambda: _call_llm_json_sync(prompt))


async def validate_sources(asset: Asset, binance_price: float, okx_price: float) -> ValidationResult:
    if binance_price <= 0 and okx_price <= 0:
        return ValidationResult(0.0, False, "no source price available")
    if binance_price <= 0:
        return ValidationResult(okx_price, False, "binance price missing")
    if okx_price <= 0:
        return ValidationResult(binance_price, False, "okx price missing")

    diff_pct = abs(binance_price - okx_price) / min(binance_price, okx_price)
    threshold = registry.thresholds.cross_source_conflict_pct
    if diff_pct <= threshold:
        return ValidationResult(
            trusted_price=(binance_price + okx_price) / 2,
            conflict_detected=False,
            resolution_reason="prices within threshold; using midpoint",
        )

    try:
        decision = await _arbitrate_conflict(asset, binance_price, okx_price)
        trust = decision.get("trust", "both")
        reason = decision.get("reason", "llm arbitration result")
        if trust == "binance":
            trusted_price = binance_price
        elif trust == "okx":
            trusted_price = okx_price
        else:
            trusted_price = (binance_price + okx_price) / 2
        return ValidationResult(
            trusted_price=trusted_price,
            conflict_detected=True,
            resolution_reason=reason,
        )
    except Exception as exc:
        logger.warning("Price arbitration failed for %s: %s", asset.value, exc)
        return ValidationResult(
            trusted_price=(binance_price + okx_price) / 2,
            conflict_detected=True,
            resolution_reason="llm arbitration failed; fallback to midpoint",
        )
