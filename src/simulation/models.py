from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.domain.models import Asset, Decision, FeatureSnapshot, Severity


class ScenarioEventSpec(BaseModel):
    offset_seconds: int
    asset: Asset
    source: str
    event_type: str
    payload: dict


class ScenarioCheckpointExpectation(BaseModel):
    severity: Severity | None = None
    decision: Decision | None = None
    rule_ids: list[str] = Field(default_factory=list)
    alert_emitted: bool = False


class ScenarioCheckpoint(BaseModel):
    checkpoint_id: str
    offset_seconds: int
    asset: Asset
    expectation: ScenarioCheckpointExpectation
    description: str = ""


class SimulationScenario(BaseModel):
    scenario_id: str
    title: str
    title_zh: str
    description: str
    description_zh: str
    asset: Asset
    cycle_seconds: int = 30
    events: list[ScenarioEventSpec]
    checkpoints: list[ScenarioCheckpoint]


class SimulationStageMetrics(BaseModel):
    injected_events: int = 0
    ingest_ms: float = 0.0
    snapshot_ms: float = 0.0
    load_memory_ms: float = 0.0
    rule_eval_ms: float = 0.0
    expert_ms: float = 0.0
    summarizer_ms: float = 0.0
    review_helpers_ms: float = 0.0
    total_ms: float = 0.0


class SimulationLLMMetrics(BaseModel):
    calls: int = 0
    success_calls: int = 0
    error_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    usage_available: bool = False
    operations: list[str] = Field(default_factory=list)


class SimulationStepResult(BaseModel):
    checkpoint_id: str
    asset: Asset
    checkpoint_time: datetime
    description: str = ""
    snapshot: FeatureSnapshot
    expected: ScenarioCheckpointExpectation
    actual_severity: Severity | None = None
    actual_decision: Decision | None = None
    actual_rule_ids: list[str] = Field(default_factory=list)
    alert_emitted: bool = False
    summary_zh: str = ""
    review_guidance: str = ""
    historical_context_zh: str = ""
    risk_quantification_zh: str = ""
    correctness_passed: bool = False
    correctness_notes: list[str] = Field(default_factory=list)
    stage_metrics: SimulationStageMetrics = Field(default_factory=SimulationStageMetrics)
    llm_metrics: SimulationLLMMetrics = Field(default_factory=SimulationLLMMetrics)


class SimulationRunSummary(BaseModel):
    total_steps: int = 0
    passed_steps: int = 0
    pass_rate: float = 0.0
    emitted_alerts: int = 0
    manual_reviews: int = 0
    p1_steps: int = 0
    p2_steps: int = 0
    avg_total_latency_ms: float = 0.0
    max_total_latency_ms: float = 0.0
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_usage_available: bool = False


class SimulationRun(BaseModel):
    run_id: str
    scenario_id: str
    title: str
    title_zh: str
    started_at: datetime
    completed_at: datetime
    status: str = "completed"
    events_injected: int = 0
    summary: SimulationRunSummary
    steps: list[SimulationStepResult]
