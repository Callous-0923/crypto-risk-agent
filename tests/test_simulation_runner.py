import unittest
from unittest.mock import AsyncMock, patch

from src.domain.models import Asset
from src.observability.llm_trace import record_llm_call
from src.simulation.runner import run_simulation_scenario


def _fake_chat_completion(messages, *, max_tokens, tools=None, operation="chat_completion"):
    record_llm_call(
        operation=operation, status="success", duration_ms=12.0,
        prompt_tokens=120, completion_tokens=40, total_tokens=160,
    )
    if operation == "summarizer":
        return type("Response", (), {"content": '{"summary_zh":"summary","review_guidance":"review"}'})()
    return type("Response", (), {"content": '{"key_metrics":{},"confidence":0.8,"narrative_zh":"analysis"}', "tool_calls": []})()


async def _fake_review_assistance(case, snap):
    record_llm_call(operation="review_assistance", status="success", duration_ms=11.0,
                    prompt_tokens=120, completion_tokens=36, total_tokens=156)
    return "history", "risk"


class SimulationRunnerTests(unittest.IsolatedAsyncioTestCase):
    """模拟场景回放测试。

    注意: 依赖 aiosqlite 连接 init_graph(), 在部分 pytest-asyncio 环境
    下可能因事件循环冲突而挂起。建议在 CI 的独立进程中运行本测试文件。
    """

    async def test_run_flash_crash_scenario_matches_expected_outputs(self):
        import src.ingestion.validator as validator_mod
        import src.graph.nodes as nodes_mod
        import src.simulation.runner as runner_mod

        trusted = (100.0 + 100.2) / 2
        fake_result = validator_mod.ValidationResult(trusted, False, "midpoint")

        with patch.object(validator_mod, "validate_sources", AsyncMock(return_value=fake_result)):
            with patch.object(nodes_mod, "_call_chat_completion_sync", side_effect=_fake_chat_completion):
                with patch.object(runner_mod, "build_review_assistance", _fake_review_assistance):
                    result = await run_simulation_scenario("btc_flash_crash_p1")

        self.assertEqual(result.summary.total_steps, 2)
        self.assertGreater(result.summary.llm_total_tokens, 0)
        self.assertEqual(result.steps[0].checkpoint_id, "t30")
        self.assertEqual(result.steps[1].checkpoint_id, "t60")

    async def test_run_leverage_buildup_keeps_manual_review_path(self):
        import src.ingestion.validator as validator_mod
        import src.graph.nodes as nodes_mod
        import src.simulation.runner as runner_mod

        trusted = (100.0 + 100.1) / 2
        fake_result = validator_mod.ValidationResult(trusted, False, "midpoint")

        with patch.object(validator_mod, "validate_sources", AsyncMock(return_value=fake_result)):
            with patch.object(nodes_mod, "_call_chat_completion_sync", side_effect=_fake_chat_completion):
                with patch.object(runner_mod, "build_review_assistance", _fake_review_assistance):
                    result = await run_simulation_scenario("btc_leverage_buildup_p2")

        self.assertEqual(result.summary.total_steps, 1)
        self.assertEqual(result.steps[0].checkpoint_id, "t60")

    async def test_run_early_warning_to_p2_scenario(self):
        """验证 BTC EW→P2 渐进升级场景可以正常运行并产生 2 个步骤。"""
        import src.ingestion.validator as validator_mod
        import src.graph.nodes as nodes_mod
        import src.simulation.runner as runner_mod

        trusted = 99.65
        fake_result = validator_mod.ValidationResult(trusted, False, "midpoint")

        with patch.object(validator_mod, "validate_sources", AsyncMock(return_value=fake_result)):
            with patch.object(nodes_mod, "_call_chat_completion_sync", side_effect=_fake_chat_completion):
                with patch.object(runner_mod, "build_review_assistance", _fake_review_assistance):
                    result = await run_simulation_scenario("btc_early_warning_to_p2")

        self.assertEqual(result.summary.total_steps, 2)
        self.assertEqual(result.steps[0].checkpoint_id, "t60")
        self.assertEqual(result.steps[1].checkpoint_id, "t90")

    async def test_run_multi_asset_systemic_risk_scenario(self):
        """验证多资产系统性风险场景可以正常运行。"""
        import src.ingestion.validator as validator_mod
        import src.graph.nodes as nodes_mod
        import src.simulation.runner as runner_mod

        trusted = 94.6
        fake_result = validator_mod.ValidationResult(trusted, False, "midpoint")

        with patch.object(validator_mod, "validate_sources", AsyncMock(return_value=fake_result)):
            with patch.object(nodes_mod, "_call_chat_completion_sync", side_effect=_fake_chat_completion):
                with patch.object(runner_mod, "build_review_assistance", _fake_review_assistance):
                    result = await run_simulation_scenario("multi_asset_systemic_risk")

        self.assertEqual(result.summary.total_steps, 1)
        self.assertGreater(result.summary.llm_total_tokens, 0)
        self.assertEqual(result.steps[0].checkpoint_id, "t30")


if __name__ == "__main__":
    unittest.main()
