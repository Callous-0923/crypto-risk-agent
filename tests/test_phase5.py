import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.domain.models import Asset, FeatureSnapshot, RuleHit, Severity
from src.graph.nodes import node_expert_parallel, node_summarizer


def _snap(**kwargs) -> FeatureSnapshot:
    defaults = dict(
        asset=Asset.BTC,
        price=60000.0,
        ret_1m=0.05,
        ret_5m=0.08,
        vol_z_1m=2.0,
        oi_delta_15m_pct=0.12,
        liq_5m_usd=80_000_000.0,
        funding_z=2.8,
        source_stale=False,
        cross_source_conflict=False,
    )
    defaults.update(kwargs)
    return FeatureSnapshot(**defaults)


def _hit() -> RuleHit:
    return RuleHit(
        rule_id="MKT_EXTREME_VOL_P1",
        asset=Asset.BTC,
        severity=Severity.P1,
        description="BTC 触发 P1",
        dedupe_key="p1:btc",
    )


class Phase5Tests(unittest.IsolatedAsyncioTestCase):
    async def test_expert_parallel_runs_two_specialists_concurrently(self):
        import src.graph.nodes as nodes_mod

        async def fake_technical(state):
            await asyncio.sleep(0.1)
            return {"technical_analysis": {"narrative_zh": "技术分析"}, "technical_analysis_zh": "技术分析"}

        async def fake_macro(state):
            await asyncio.sleep(0.1)
            return {"macro_context": {"narrative_zh": "宏观上下文"}, "macro_context_zh": "宏观上下文"}

        with patch.object(nodes_mod, "node_technical_analyst", fake_technical):
            with patch.object(nodes_mod, "node_macro_context", fake_macro):
                started_at = time.perf_counter()
                result = await node_expert_parallel(
                    {
                        "highest_severity": Severity.P1,
                    }
                )
                elapsed = time.perf_counter() - started_at

        self.assertEqual(result["technical_analysis_zh"], "技术分析")
        self.assertEqual(result["macro_context_zh"], "宏观上下文")
        self.assertLess(elapsed, 0.22)

    async def test_summarizer_appends_metric_mismatch_warning(self):
        import src.graph.nodes as nodes_mod

        fake_response = SimpleNamespace(
            content='{"summary_zh":"最终摘要","review_guidance":"先看同步性。"}'
        )
        state = {
            "snapshot": _snap(),
            "rule_hits": [_hit()],
            "highest_severity": Severity.P1,
            "technical_analysis": {
                "key_metrics": {"ret_1m": 0.02, "liq_5m_usd": 80_000_000.0},
                "confidence": 0.9,
                "narrative_zh": "技术分析",
            },
            "macro_context": {
                "key_metrics": {"funding_z": 2.8, "oi_delta_15m_pct": 0.12},
                "confidence": 0.8,
                "narrative_zh": "宏观上下文",
            },
            "summary_zh": "",
            "review_guidance": "",
        }

        with patch.object(nodes_mod, "_call_chat_completion_sync", return_value=fake_response):
            result = await node_summarizer(state)

        self.assertEqual(result["summary_zh"], "最终摘要")
        self.assertIn("ret_1m=0.02 != snapshot(0.05)", result["review_guidance"])

    async def test_summarizer_skips_llm_for_p3_only(self):
        import src.graph.nodes as nodes_mod

        state = {
            "snapshot": _snap(ret_1m=0.0),
            "rule_hits": [
                RuleHit(
                    rule_id="QA_SOURCE_STALE",
                    asset=Asset.BTC,
                    severity=Severity.P3,
                    description="数据陈旧",
                    dedupe_key="p3:btc",
                )
            ],
            "highest_severity": Severity.P3,
            "technical_analysis": None,
            "macro_context": None,
            "summary_zh": "",
            "review_guidance": "",
        }

        with patch.object(nodes_mod, "_call_chat_completion_sync", side_effect=AssertionError("should not call llm")):
            result = await node_summarizer(state)

        self.assertIn("BTC 风险摘要", result["summary_zh"])


if __name__ == "__main__":
    unittest.main()
