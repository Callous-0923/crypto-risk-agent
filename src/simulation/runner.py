from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone

from src.domain.models import Asset, CaseStatus, Decision, RawEvent, RiskCase, Severity
from src.features.builder import FeatureBuilder
from src.graph.nodes import node_decide, node_expert_parallel, node_run_rules, node_summarizer
from src.graph.review_assistants import build_review_assistance
from src.observability.llm_trace import (
    LLMUsageCollector,
    bind_llm_usage_collector,
    reset_llm_usage_collector,
)
from src.simulation.models import (
    ScenarioCheckpoint,
    ScenarioCheckpointExpectation,
    SimulationLLMMetrics,
    SimulationRun,
    SimulationRunSummary,
    SimulationStageMetrics,
    SimulationStepResult,
)
from src.simulation.scenarios import get_scenario, list_scenarios

_recent_runs: list[SimulationRun] = []


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _build_raw_event(base_time: datetime, spec) -> RawEvent:
    event_ts = base_time + timedelta(seconds=spec.offset_seconds)
    return RawEvent(
        asset=spec.asset,
        source=spec.source,
        event_type=spec.event_type,
        event_ts=event_ts,
        ingest_ts=event_ts,
        payload=spec.payload,
    )


def _initial_state(asset: Asset, snapshot, checkpoint_id: str) -> dict:
    return {
        "thread_id": f"simulation:{checkpoint_id}:{uuid.uuid4()}",
        "asset": asset,
        "snapshot": snapshot,
        "is_coordinator_case": False,
        "recent_alert_history": [],
        "fatigue_suppressed": False,
        "memory_context": None,
        "ml_prediction": None,
        "rule_hits": [],
        "highest_severity": None,
        "technical_analysis": None,
        "macro_context": None,
        "technical_analysis_zh": "",
        "macro_context_zh": "",
        "summary_zh": "",
        "review_guidance": "",
        "historical_context_zh": "",
        "risk_quantification_zh": "",
        "decision": None,
        "case": None,
        "case_reused": False,
        "alert": None,
        "human_approved": None,
        "human_comment": "",
    }


def _records_to_metrics(records) -> SimulationLLMMetrics:
    if not records:
        return SimulationLLMMetrics()
    success_calls = sum(1 for record in records if record.status == "success")
    error_calls = len(records) - success_calls
    prompt_tokens = sum(record.prompt_tokens for record in records)
    completion_tokens = sum(record.completion_tokens for record in records)
    total_tokens = sum(record.total_tokens for record in records)
    usage_available = any(record.total_tokens > 0 for record in records)
    return SimulationLLMMetrics(
        calls=len(records),
        success_calls=success_calls,
        error_calls=error_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        duration_ms=sum(record.duration_ms for record in records),
        usage_available=usage_available,
        operations=[record.operation for record in records],
    )


def _evaluate_correctness(
    expected: ScenarioCheckpointExpectation,
    *,
    severity: Severity | None,
    decision: Decision | None,
    rule_ids: list[str],
    alert_emitted: bool,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if expected.severity != severity:
        notes.append(f"expected severity {expected.severity}, got {severity}")
    if expected.decision != decision:
        notes.append(f"expected decision {expected.decision}, got {decision}")
    missing_rules = [rule_id for rule_id in expected.rule_ids if rule_id not in rule_ids]
    if missing_rules:
        notes.append(f"missing rule ids: {', '.join(missing_rules)}")
    if expected.alert_emitted != alert_emitted:
        notes.append(f"expected alert_emitted={expected.alert_emitted}, got {alert_emitted}")
    return len(notes) == 0, notes


async def _run_checkpoint(
    checkpoint: ScenarioCheckpoint,
    base_time: datetime,
    collector: LLMUsageCollector,
    *,
    injected_events: int,
    snapshot,
    snapshot_ms: float,
) -> SimulationStepResult:
    stage_metrics = SimulationStageMetrics(injected_events=injected_events)

    checkpoint_time = base_time + timedelta(seconds=checkpoint.offset_seconds)
    llm_start = len(collector)
    t_total = time.perf_counter()
    stage_metrics.snapshot_ms = snapshot_ms

    state = _initial_state(checkpoint.asset, snapshot, checkpoint.checkpoint_id)

    stage_metrics.load_memory_ms = 0.0

    t_stage = time.perf_counter()
    state.update(node_run_rules(state))
    stage_metrics.rule_eval_ms = (time.perf_counter() - t_stage) * 1000

    t_stage = time.perf_counter()
    state.update(await node_expert_parallel(state))
    stage_metrics.expert_ms = (time.perf_counter() - t_stage) * 1000

    t_stage = time.perf_counter()
    state.update(await node_summarizer(state))
    stage_metrics.summarizer_ms = (time.perf_counter() - t_stage) * 1000

    state.update(node_decide(state))

    historical_context_zh = ""
    risk_quantification_zh = ""
    if state.get("decision") == Decision.MANUAL_REVIEW and state.get("highest_severity") == Severity.P2:
        synthetic_case = RiskCase(
            asset=checkpoint.asset,
            status=CaseStatus.PENDING_REVIEW,
            rule_hits=state.get("rule_hits", []),
            decision=Decision.MANUAL_REVIEW,
            summary_zh=state.get("summary_zh", ""),
            severity=Severity.P2,
        )
        t_stage = time.perf_counter()
        historical_context_zh, risk_quantification_zh = await build_review_assistance(synthetic_case, snapshot)
        stage_metrics.review_helpers_ms = (time.perf_counter() - t_stage) * 1000

    alert_emitted = state.get("decision") == Decision.EMIT
    actual_rule_ids = [hit.rule_id for hit in state.get("rule_hits", [])]
    correctness_passed, correctness_notes = _evaluate_correctness(
        checkpoint.expectation,
        severity=state.get("highest_severity"),
        decision=state.get("decision"),
        rule_ids=actual_rule_ids,
        alert_emitted=alert_emitted,
    )

    stage_metrics.total_ms = (time.perf_counter() - t_total) * 1000
    llm_metrics = _records_to_metrics(collector.slice_from(llm_start))

    return SimulationStepResult(
        checkpoint_id=checkpoint.checkpoint_id,
        asset=checkpoint.asset,
        checkpoint_time=checkpoint_time,
        description=checkpoint.description,
        snapshot=snapshot,
        expected=checkpoint.expectation,
        actual_severity=state.get("highest_severity"),
        actual_decision=state.get("decision"),
        actual_rule_ids=actual_rule_ids,
        alert_emitted=alert_emitted,
        summary_zh=state.get("summary_zh", ""),
        review_guidance=state.get("review_guidance", ""),
        historical_context_zh=historical_context_zh,
        risk_quantification_zh=risk_quantification_zh,
        correctness_passed=correctness_passed,
        correctness_notes=correctness_notes,
        stage_metrics=stage_metrics,
        llm_metrics=llm_metrics,
    )


def _build_summary(results: list[SimulationStepResult]) -> SimulationRunSummary:
    total_steps = len(results)
    passed_steps = sum(1 for item in results if item.correctness_passed)
    emitted_alerts = sum(1 for item in results if item.alert_emitted)
    manual_reviews = sum(1 for item in results if item.actual_decision == Decision.MANUAL_REVIEW)
    p1_steps = sum(1 for item in results if item.actual_severity == Severity.P1)
    p2_steps = sum(1 for item in results if item.actual_severity == Severity.P2)
    total_latencies = [item.stage_metrics.total_ms for item in results]
    return SimulationRunSummary(
        total_steps=total_steps,
        passed_steps=passed_steps,
        pass_rate=(passed_steps / total_steps) if total_steps else 0.0,
        emitted_alerts=emitted_alerts,
        manual_reviews=manual_reviews,
        p1_steps=p1_steps,
        p2_steps=p2_steps,
        avg_total_latency_ms=(sum(total_latencies) / total_steps) if total_steps else 0.0,
        max_total_latency_ms=max(total_latencies) if total_latencies else 0.0,
        llm_calls=sum(item.llm_metrics.calls for item in results),
        llm_prompt_tokens=sum(item.llm_metrics.prompt_tokens for item in results),
        llm_completion_tokens=sum(item.llm_metrics.completion_tokens for item in results),
        llm_total_tokens=sum(item.llm_metrics.total_tokens for item in results),
        llm_usage_available=any(item.llm_metrics.usage_available for item in results),
    )


async def run_simulation_scenario(scenario_id: str) -> SimulationRun:
    scenario = get_scenario(scenario_id)
    if scenario is None:
        raise ValueError(f"Unknown simulation scenario: {scenario_id}")

    builder = FeatureBuilder()
    base_time = _utc_now()
    started_at = _utc_now()
    run_id = str(uuid.uuid4())
    events = sorted(
        (_build_raw_event(base_time, spec) for spec in scenario.events),
        key=lambda item: item.event_ts,
    )
    checkpoints = sorted(scenario.checkpoints, key=lambda item: item.offset_seconds)
    checkpoints_by_offset: dict[int, list[ScenarioCheckpoint]] = {}
    for checkpoint in checkpoints:
        checkpoints_by_offset.setdefault(checkpoint.offset_seconds, []).append(checkpoint)
    injected_index = 0
    results: list[SimulationStepResult] = []
    collector = LLMUsageCollector()
    token = bind_llm_usage_collector(collector)
    max_offset = max((checkpoint.offset_seconds for checkpoint in checkpoints), default=0)

    try:
        for offset_seconds in range(0, max_offset + scenario.cycle_seconds, scenario.cycle_seconds):
            cutoff = base_time + timedelta(seconds=offset_seconds)
            due_events: list[RawEvent] = []
            while injected_index < len(events) and events[injected_index].event_ts <= cutoff:
                due_events.append(events[injected_index])
                injected_index += 1

            t_ingest = time.perf_counter()
            for event in due_events:
                builder.ingest(event)
            ingest_ms = (time.perf_counter() - t_ingest) * 1000

            t_snapshot = time.perf_counter()
            snapshot = await builder.build_snapshot_validated(scenario.asset, as_of=cutoff)
            snapshot_ms = (time.perf_counter() - t_snapshot) * 1000

            for checkpoint in checkpoints_by_offset.get(offset_seconds, []):
                result = await _run_checkpoint(
                    checkpoint,
                    base_time,
                    collector,
                    injected_events=len(due_events),
                    snapshot=snapshot,
                    snapshot_ms=snapshot_ms,
                )
                result.stage_metrics.ingest_ms = ingest_ms
                results.append(result)
    finally:
        reset_llm_usage_collector(token)

    run = SimulationRun(
        run_id=run_id,
        scenario_id=scenario.scenario_id,
        title=scenario.title,
        title_zh=scenario.title_zh,
        started_at=started_at,
        completed_at=_utc_now(),
        events_injected=len(events),
        summary=_build_summary(results),
        steps=results,
    )
    _recent_runs.insert(0, run)
    del _recent_runs[10:]
    return run


def list_simulation_scenarios():
    return list_scenarios()


def get_recent_simulation_runs() -> list[SimulationRun]:
    return list(_recent_runs)


def get_latest_simulation_run() -> SimulationRun | None:
    return _recent_runs[0] if _recent_runs else None
