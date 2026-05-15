import asyncio
import unittest
from unittest.mock import patch

from src.domain.models import Asset, Decision, FeatureSnapshot, RiskCase, RuleHit, Severity
from src.graph.nodes import node_build_case


def _snap() -> FeatureSnapshot:
    return FeatureSnapshot(
        asset=Asset.BTC,
        price=60000.0,
        ret_1m=0.031,
        ret_5m=0.04,
        oi_delta_15m_pct=0.11,
        liq_5m_usd=10_000_000.0,
        funding_z=2.7,
    )


def _hit(severity: Severity) -> RuleHit:
    return RuleHit(
        rule_id=f"TEST_{severity.value}",
        asset=Asset.BTC,
        severity=severity,
        description=f"{severity.value} hit",
        dedupe_key=f"test:{severity.value}",
    )


class Phase6Tests(unittest.IsolatedAsyncioTestCase):
    async def test_build_case_enriches_manual_review_case_with_review_helpers(self):
        import src.graph.nodes as nodes_mod
        import src.graph.review_assistants as assistants_mod
        import src.persistence.repositories as repo_mod

        saved_cases: list[RiskCase] = []

        async def fake_save(case: RiskCase) -> None:
            saved_cases.append(case.model_copy(deep=True))

        async def fake_list_cases(*args, **kwargs):
            return [saved_cases[-1]] if saved_cases else []

        async def fake_review_assistance(case: RiskCase, snap: FeatureSnapshot) -> tuple[str, str]:
            await asyncio.sleep(0.05)
            return "历史上下文", "风险量化"

        async def fake_find_active(*args, **kwargs):
            return None

        with patch.object(nodes_mod, "save_risk_case", fake_save):
            with patch.object(repo_mod, "list_risk_cases", fake_list_cases):
                with patch.object(repo_mod, "find_active_case_by_dedupe_key", fake_find_active):
                    with patch.object(assistants_mod, "build_review_assistance", fake_review_assistance):
                        result = await node_build_case(
                            {
                                "thread_id": "p2-case",
                                "asset": Asset.BTC,
                                "snapshot": _snap(),
                                "is_coordinator_case": False,
                                "rule_hits": [_hit(Severity.P2)],
                                "highest_severity": Severity.P2,
                                "technical_analysis": None,
                                "macro_context": None,
                                "technical_analysis_zh": "",
                                "macro_context_zh": "",
                                "summary_zh": "摘要",
                                "review_guidance": "指引",
                                "historical_context_zh": "",
                                "risk_quantification_zh": "",
                                "decision": Decision.MANUAL_REVIEW,
                                "case": None,
                                "case_reused": False,
                                "alert": None,
                                "human_approved": None,
                                "human_comment": "",
                            }
                        )

        self.assertEqual(len(saved_cases), 2)
        self.assertEqual(result["historical_context_zh"], "历史上下文")
        self.assertEqual(result["risk_quantification_zh"], "风险量化")
        self.assertEqual(result["case"].historical_context_zh, "历史上下文")
        self.assertEqual(result["case"].risk_quantification_zh, "风险量化")

    async def test_build_case_skips_review_helpers_for_p1(self):
        import src.graph.nodes as nodes_mod
        import src.graph.review_assistants as assistants_mod
        import src.persistence.repositories as repo_mod

        saved_cases: list[RiskCase] = []

        async def fake_save(case: RiskCase) -> None:
            saved_cases.append(case.model_copy(deep=True))

        async def fake_list_cases(*args, **kwargs):
            return [saved_cases[-1]] if saved_cases else []

        async def fake_find_active(*args, **kwargs):
            return None

        with patch.object(nodes_mod, "save_risk_case", fake_save):
            with patch.object(repo_mod, "list_risk_cases", fake_list_cases):
                with patch.object(repo_mod, "find_active_case_by_dedupe_key", fake_find_active):
                    with patch.object(
                        assistants_mod,
                        "build_review_assistance",
                        side_effect=AssertionError("should not call"),
                    ):
                        result = await node_build_case(
                            {
                                "thread_id": "p1-case",
                                "asset": Asset.BTC,
                                "snapshot": _snap(),
                                "is_coordinator_case": False,
                                "rule_hits": [_hit(Severity.P1)],
                                "highest_severity": Severity.P1,
                                "technical_analysis": None,
                                "macro_context": None,
                                "technical_analysis_zh": "",
                                "macro_context_zh": "",
                                "summary_zh": "摘要",
                                "review_guidance": "指引",
                                "historical_context_zh": "",
                                "risk_quantification_zh": "",
                                "decision": Decision.EMIT,
                                "case": None,
                                "alert": None,
                                "human_approved": None,
                                "human_comment": "",
                            }
                        )

        self.assertEqual(len(saved_cases), 1)
        self.assertEqual(result["historical_context_zh"], "")
        self.assertEqual(result["risk_quantification_zh"], "")


if __name__ == "__main__":
    unittest.main()
