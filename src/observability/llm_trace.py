"""Helpers for collecting per-run LLM usage during simulations."""
from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass, field
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class LLMCallRecord:
    operation: str
    status: str
    duration_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    estimated_cost_usd: float = 0.0


@dataclass
class LLMUsageCollector:
    records: list[LLMCallRecord] = field(default_factory=list)

    def record(self, entry: LLMCallRecord) -> None:
        self.records.append(entry)

    def slice_from(self, start_index: int) -> list[LLMCallRecord]:
        return self.records[start_index:]

    def __len__(self) -> int:
        return len(self.records)


_current_collector: contextvars.ContextVar[LLMUsageCollector | None] = contextvars.ContextVar(
    "current_llm_usage_collector",
    default=None,
)


def bind_llm_usage_collector(collector: LLMUsageCollector):
    return _current_collector.set(collector)


def reset_llm_usage_collector(token) -> None:
    _current_collector.reset(token)


def record_llm_call(
    *,
    operation: str,
    status: str,
    duration_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    model: str = "",
    estimated_cost_usd: float | None = None,
) -> None:
    from src.core.config import settings

    model_name = model or settings.llm_model
    cost = (
        estimated_cost_usd
        if estimated_cost_usd is not None
        else (
            (prompt_tokens / 1_000_000) * settings.llm_input_cost_per_million_tokens
            + (completion_tokens / 1_000_000) * settings.llm_output_cost_per_million_tokens
        )
    )
    collector = _current_collector.get()
    if collector is None:
        from src.persistence.repositories import save_llm_call_record_sync

        save_llm_call_record_sync(
            model=model_name,
            operation=operation,
            status=status,
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost,
        )
    else:
        collector.record(
            LLMCallRecord(
                operation=operation,
                status=status,
                duration_ms=duration_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model_name,
                estimated_cost_usd=cost,
            )
        )


async def run_in_executor_with_context(func: Callable[[], T]) -> T:
    """Run a blocking callable in the executor while preserving contextvars."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(None, lambda: ctx.run(func))
