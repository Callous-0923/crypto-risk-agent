from __future__ import annotations

from src.domain.models import Asset, Decision, Severity
from src.simulation.models import (
    ScenarioCheckpoint,
    ScenarioCheckpointExpectation,
    ScenarioEventSpec,
    SimulationScenario,
)


SCENARIOS: list[SimulationScenario] = [
    # =====================================================================
    # BTC 场景
    # =====================================================================
    SimulationScenario(
        scenario_id="btc_flash_crash_p1",
        title="BTC Flash Crash Escalation",
        title_zh="BTC 闪崩升级场景",
        description="A 30s-snapshot replay that moves from moderate stress into a P1 liquidation cascade.",
        description_zh="基于 30 秒快照回放的 BTC 闪崩场景，先进入中等级风险，再升级为 P1 清算级风险。",
        asset=Asset.BTC,
        events=[
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 100.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 100.2}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 100_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 96.8}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 97.0}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 108_000_000}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 94.0}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 94.4}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="binance_futures", event_type="liquidation", payload={"usd_value": 60_000_000}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 116_000_000}),
        ],
        checkpoints=[
            ScenarioCheckpoint(
                checkpoint_id="t30",
                offset_seconds=30,
                asset=Asset.BTC,
                description="Moderate 1m move should trigger a P2 review path.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P2,
                    decision=Decision.MANUAL_REVIEW,
                    rule_ids=["MKT_EXTREME_VOL_P2"],
                    alert_emitted=False,
                ),
            ),
            ScenarioCheckpoint(
                checkpoint_id="t60",
                offset_seconds=60,
                asset=Asset.BTC,
                description="The move and liquidation burst should escalate to P1 and emit an alert.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P1,
                    decision=Decision.EMIT,
                    rule_ids=["MKT_EXTREME_VOL_P1", "DERIV_CASCADE_LIQ_P1"],
                    alert_emitted=True,
                ),
            ),
        ],
    ),
    SimulationScenario(
        scenario_id="btc_leverage_buildup_p2",
        title="BTC Leverage Buildup",
        title_zh="BTC 杠杆堆积场景",
        description="A slower accumulation that should stay in P2 and route into manual review without emitting an alert.",
        description_zh="一个更平缓的杠杆堆积过程，应停留在 P2 并进入人工审核，不直接发送告警。",
        asset=Asset.BTC,
        events=[
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 100.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 100.1}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 100_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 101.6}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 101.5}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 108_000_000}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 103.2}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 103.1}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 114_000_000}),
        ],
        checkpoints=[
            ScenarioCheckpoint(
                checkpoint_id="t60",
                offset_seconds=60,
                asset=Asset.BTC,
                description="Price and OI should jointly produce a P2 manual-review case.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P2,
                    decision=Decision.MANUAL_REVIEW,
                    rule_ids=["MKT_EXTREME_VOL_P2", "DERIV_OI_BUILDUP"],
                    alert_emitted=False,
                ),
            ),
        ],
    ),
    SimulationScenario(
        scenario_id="btc_early_warning_to_p2",
        title="BTC Early Warning → P2 Escalation",
        title_zh="BTC 提前预警渐进升级场景",
        description="Weak signals first appear as P3 early-warning, then persist and escalate to P2 manual review.",
        description_zh="先触发 P3 级提前预警信号，信号持续积累后升级为 P2 人工审核，完整展示 EW→P2 路径。",
        asset=Asset.BTC,
        events=[
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 100.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 100.1}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 100_000_000}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_swap", event_type="funding_rate", payload={"funding_rate": 0.0001}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 99.6}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 99.7}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 104_500_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_swap", event_type="funding_rate", payload={"funding_rate": -0.0003}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 99.1}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 99.2}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 108_000_000}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.BTC, source="okx_swap", event_type="funding_rate", payload={"funding_rate": -0.0005}),
            ScenarioEventSpec(offset_seconds=90, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 97.0}),
            ScenarioEventSpec(offset_seconds=90, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 97.1}),
            ScenarioEventSpec(offset_seconds=90, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 110_000_000}),
            ScenarioEventSpec(offset_seconds=90, asset=Asset.BTC, source="okx_swap", event_type="funding_rate", payload={"funding_rate": -0.0006}),
        ],
        checkpoints=[
            ScenarioCheckpoint(
                checkpoint_id="t60",
                offset_seconds=60,
                asset=Asset.BTC,
                description="Weak drift + OI buildup + funding bias produces early-warning signals.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P3,
                    decision=Decision.SUPPRESS,
                    rule_ids=["EW_PRICE_DRIFT_5M", "EW_OI_BUILDUP", "EW_FUNDING_BIAS"],
                    alert_emitted=False,
                ),
            ),
            ScenarioCheckpoint(
                checkpoint_id="t90",
                offset_seconds=90,
                asset=Asset.BTC,
                description="Persistent signals escalate to P2 with MKT_EXTREME_VOL_P2.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P2,
                    decision=Decision.MANUAL_REVIEW,
                    rule_ids=["MKT_EXTREME_VOL_P2"],
                    alert_emitted=False,
                ),
            ),
        ],
    ),

    # =====================================================================
    # ETH 场景
    # =====================================================================
    SimulationScenario(
        scenario_id="eth_funding_squeeze_p2",
        title="ETH Funding Rate Squeeze",
        title_zh="ETH 资金费率极端拥挤场景",
        description="ETH funding rate reaches extreme negative territory with OI buildup, triggering P2 review.",
        description_zh="ETH 资金费率极端负值叠加持仓积累，触发 DERIV_FUNDING_SQUEEZE + DERIV_OI_BUILDUP，进入 P2 审核。",
        asset=Asset.ETH,
        events=[
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="binance_spot", event_type="price", payload={"price": 3000.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="okx_spot", event_type="price", payload={"price": 3000.5}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="okx_swap", event_type="open_interest", payload={"oi_usd": 50_000_000}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="okx_swap", event_type="funding_rate", payload={"funding_rate": 0.0001}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="binance_spot", event_type="price", payload={"price": 2985.0}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="okx_spot", event_type="price", payload={"price": 2985.3}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="okx_swap", event_type="open_interest", payload={"oi_usd": 53_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="okx_swap", event_type="funding_rate", payload={"funding_rate": -0.0005}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.ETH, source="binance_spot", event_type="price", payload={"price": 2960.0}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.ETH, source="okx_spot", event_type="price", payload={"price": 2960.2}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.ETH, source="okx_swap", event_type="open_interest", payload={"oi_usd": 56_000_000}),
            ScenarioEventSpec(offset_seconds=60, asset=Asset.ETH, source="okx_swap", event_type="funding_rate", payload={"funding_rate": -0.0008}),
        ],
        checkpoints=[
            ScenarioCheckpoint(
                checkpoint_id="t60",
                offset_seconds=60,
                asset=Asset.ETH,
                description="Funding squeeze + OI buildup triggers P2 review.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P2,
                    decision=Decision.MANUAL_REVIEW,
                    rule_ids=["DERIV_FUNDING_SQUEEZE", "DERIV_OI_BUILDUP"],
                    alert_emitted=False,
                ),
            ),
        ],
    ),
    SimulationScenario(
        scenario_id="eth_volatile_liquidation_p1",
        title="ETH Volatile Liquidation Cascade",
        title_zh="ETH 波动性清算级联场景",
        description="ETH experiences sharp volatility combined with large liquidations, triggering P1 alert.",
        description_zh="ETH 出现大幅波动同时伴随大额爆仓，触发 P1 级别告警。",
        asset=Asset.ETH,
        events=[
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="binance_spot", event_type="price", payload={"price": 3000.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="okx_spot", event_type="price", payload={"price": 3000.3}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="okx_swap", event_type="open_interest", payload={"oi_usd": 50_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="binance_spot", event_type="price", payload={"price": 2840.0}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="okx_spot", event_type="price", payload={"price": 2840.5}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="binance_futures", event_type="liquidation", payload={"usd_value": 55_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="okx_swap", event_type="open_interest", payload={"oi_usd": 55_000_000}),
        ],
        checkpoints=[
            ScenarioCheckpoint(
                checkpoint_id="t30",
                offset_seconds=30,
                asset=Asset.ETH,
                description="Sharp drop + large liq burst should trigger P1 alert.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P1,
                    decision=Decision.EMIT,
                    rule_ids=["MKT_EXTREME_VOL_P1", "DERIV_CASCADE_LIQ_P1"],
                    alert_emitted=True,
                ),
            ),
        ],
    ),

    # =====================================================================
    # SOL 场景
    # =====================================================================
    SimulationScenario(
        scenario_id="sol_vol_spike_p2",
        title="SOL Volatility Spike",
        title_zh="SOL 波动率异常飙升场景",
        description="SOL shows a volatility z-score spike combined with price movement, triggering P2.",
        description_zh="SOL 波动率 z-score 急剧飙升，进入 P2 审核。",
        asset=Asset.SOL,
        events=[
            ScenarioEventSpec(offset_seconds=0, asset=Asset.SOL, source="binance_spot", event_type="price", payload={"price": 150.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.SOL, source="okx_spot", event_type="price", payload={"price": 150.2}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.SOL, source="okx_swap", event_type="open_interest", payload={"oi_usd": 10_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.SOL, source="binance_spot", event_type="price", payload={"price": 145.0}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.SOL, source="okx_spot", event_type="price", payload={"price": 145.3}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.SOL, source="okx_swap", event_type="open_interest", payload={"oi_usd": 11_000_000}),
        ],
        checkpoints=[
            ScenarioCheckpoint(
                checkpoint_id="t30",
                offset_seconds=30,
                asset=Asset.SOL,
                description="SOL rapid 3.3% drop triggers P2 vol warning.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P2,
                    decision=Decision.MANUAL_REVIEW,
                    rule_ids=["MKT_EXTREME_VOL_P2"],
                    alert_emitted=False,
                ),
            ),
        ],
    ),
    SimulationScenario(
        scenario_id="sol_data_conflict_qa",
        title="SOL Cross-Source Data Conflict",
        title_zh="SOL 跨源数据冲突场景",
        description="Binance and OKX prices diverge significantly, triggering data quality alert.",
        description_zh="Binance 和 OKX 价格出现显著偏离，触发 QA_CROSS_SOURCE_CONFLICT 数据质量告警。",
        asset=Asset.SOL,
        events=[
            ScenarioEventSpec(offset_seconds=0, asset=Asset.SOL, source="binance_spot", event_type="price", payload={"price": 150.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.SOL, source="okx_spot", event_type="price", payload={"price": 152.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.SOL, source="okx_swap", event_type="open_interest", payload={"oi_usd": 10_000_000}),
        ],
        checkpoints=[
            ScenarioCheckpoint(
                checkpoint_id="t0",
                offset_seconds=0,
                asset=Asset.SOL,
                description="Cross-source price deviation >0.5% triggers QA conflict warning.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P2,
                    decision=Decision.SUPPRESS,
                    rule_ids=["QA_CROSS_SOURCE_CONFLICT"],
                    alert_emitted=False,
                ),
            ),
        ],
    ),

    # =====================================================================
    # 跨资产联动场景
    # =====================================================================
    SimulationScenario(
        scenario_id="multi_asset_systemic_risk",
        title="Multi-Asset Systemic Risk",
        title_zh="多资产系统性风险联动场景",
        description="BTC and ETH both hit P1 within 60 seconds, triggering cross-asset coordinator intervention.",
        description_zh="BTC 和 ETH 在 60 秒内同时触发 P1，激活跨资产协调器生成联动摘要并进入审核。",
        asset=Asset.BTC,
        events=[
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 100.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 100.1}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 100_000_000}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="binance_spot", event_type="price", payload={"price": 3000.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="okx_spot", event_type="price", payload={"price": 3000.1}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.ETH, source="okx_swap", event_type="open_interest", payload={"oi_usd": 50_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 94.5}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 94.7}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="binance_futures", event_type="liquidation", payload={"usd_value": 55_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 112_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="binance_spot", event_type="price", payload={"price": 2830.0}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="okx_spot", event_type="price", payload={"price": 2830.5}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="binance_futures", event_type="liquidation", payload={"usd_value": 52_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.ETH, source="okx_swap", event_type="open_interest", payload={"oi_usd": 56_000_000}),
        ],
        checkpoints=[
            ScenarioCheckpoint(
                checkpoint_id="t30",
                offset_seconds=30,
                asset=Asset.BTC,
                description="BTC hits P1 with liq cascade; ETH also P1 — coordinator should fire multi-asset case.",
                expectation=ScenarioCheckpointExpectation(
                    severity=Severity.P1,
                    decision=Decision.EMIT,
                    rule_ids=["MKT_EXTREME_VOL_P1", "DERIV_CASCADE_LIQ_P1"],
                    alert_emitted=True,
                ),
            ),
        ],
    ),

    # =====================================================================
    # 无风险 / 正常场景
    # =====================================================================
    SimulationScenario(
        scenario_id="btc_normal_market",
        title="BTC Normal Market Conditions",
        title_zh="BTC 正常市场状态场景",
        description="Normal price movement with no rule triggers — verifies the system does not produce false positives during calm periods.",
        description_zh="平稳市场状态，价格微幅波动，验证系统在平静期不会产生误报。",
        asset=Asset.BTC,
        events=[
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 100.0}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 100.1}),
            ScenarioEventSpec(offset_seconds=0, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 100_000_000}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="binance_spot", event_type="price", payload={"price": 100.3}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_spot", event_type="price", payload={"price": 100.4}),
            ScenarioEventSpec(offset_seconds=30, asset=Asset.BTC, source="okx_swap", event_type="open_interest", payload={"oi_usd": 100_500_000}),
        ],
        checkpoints=[
            ScenarioCheckpoint(
                checkpoint_id="t30",
                offset_seconds=30,
                asset=Asset.BTC,
                description="Normal market with 0.3% move and minimal OI change — no rules should fire.",
                expectation=ScenarioCheckpointExpectation(
                    severity=None,
                    decision=Decision.SUPPRESS,
                    rule_ids=[],
                    alert_emitted=False,
                ),
            ),
        ],
    ),
]


def list_scenarios() -> list[SimulationScenario]:
    return SCENARIOS


def get_scenario(scenario_id: str) -> SimulationScenario | None:
    for scenario in SCENARIOS:
        if scenario.scenario_id == scenario_id:
            return scenario
    return None
