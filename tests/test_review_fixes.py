from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

from langgraph.graph import END
from src.domain.models import Asset, CaseStatus, Decision, FeatureSnapshot, RiskCase, RuleHit, Severity
from src.graph.nodes import node_build_case, node_decide
from src.graph.orchestrator import _route_after_build


def _snap() -> FeatureSnapshot:
    return FeatureSnapshot(
        asset=Asset.BTC,
        price=60000.0,
        ret_1m=0.032,
        ret_5m=0.041,
        oi_delta_15m_pct=0.11,
        liq_5m_usd=1_000_000.0,
        funding_z=1.5,
    )


def _hit(severity: Severity, dedupe_key: str = "dedupe:btc:p1") -> RuleHit:
    return RuleHit(
        rule_id=f"TEST_{severity.value}",
        asset=Asset.BTC,
        severity=severity,
        description=f"{severity.value} hit",
        dedupe_key=dedupe_key,
    )


class ReviewFixTests(unittest.IsolatedAsyncioTestCase):
    def test_node_decide_fatigue_suppresses_p2_after_three_recent_alerts(self):
        now = datetime.utcnow()
        state = {
            "rule_hits": [_hit(Severity.P2, dedupe_key="dedupe:btc:p2")],
            "highest_severity": Severity.P2,
            "recent_alert_history": [
                {"created_at": (now - timedelta(minutes=1)).isoformat()},
                {"created_at": (now - timedelta(minutes=2)).isoformat()},
                {"created_at": (now - timedelta(minutes=3)).isoformat()},
            ],
        }

        result = node_decide(state)

        self.assertEqual(result["decision"], Decision.SUPPRESS)
        self.assertTrue(result["fatigue_suppressed"])

    async def test_node_build_case_reuses_existing_p1_case_by_dedupe_key(self):
        import src.graph.nodes as nodes_mod
        import src.persistence.repositories as repo_mod

        existing = RiskCase(
            case_id="existing-case",
            asset=Asset.BTC,
            status=CaseStatus.OPEN,
            decision=Decision.EMIT,
            severity=Severity.P1,
            rule_hits=[_hit(Severity.P1)],
            summary_zh="existing summary",
        )

        async def fake_find(asset, dedupe_key, within_seconds=300):
            self.assertEqual(asset, Asset.BTC)
            self.assertEqual(dedupe_key, "dedupe:btc:p1")
            return existing

        async def fake_list(*args, **kwargs):
            return []

        async def fail_save(case):
            raise AssertionError("deduped path should not save a new case")

        with patch.object(repo_mod, "find_active_case_by_dedupe_key", fake_find):
            with patch.object(repo_mod, "list_risk_cases", fake_list):
                with patch.object(nodes_mod, "save_risk_case", fail_save):
                    result = await node_build_case(
                        {
                            "thread_id": "new-thread",
                            "asset": Asset.BTC,
                            "snapshot": _snap(),
                            "is_coordinator_case": False,
                            "recent_alert_history": [],
                            "fatigue_suppressed": False,
                            "rule_hits": [_hit(Severity.P1)],
                            "highest_severity": Severity.P1,
                            "technical_analysis": None,
                            "macro_context": None,
                            "technical_analysis_zh": "",
                            "macro_context_zh": "",
                            "summary_zh": "new summary",
                            "review_guidance": "",
                            "historical_context_zh": "",
                            "risk_quantification_zh": "",
                            "decision": Decision.EMIT,
                            "case": None,
                            "alert": None,
                            "human_approved": None,
                            "human_comment": "",
                        }
                    )

        self.assertEqual(result["case"].case_id, "existing-case")

    async def test_node_build_case_persists_fatigue_suppressed_case(self):
        import src.graph.nodes as nodes_mod
        import src.persistence.repositories as repo_mod

        saved_cases = []

        async def fake_save(case):
            saved_cases.append(case.model_copy(deep=True))

        async def fake_list(*args, **kwargs):
            return []

        with patch.object(nodes_mod, "save_risk_case", fake_save):
            with patch.object(repo_mod, "list_risk_cases", fake_list):
                result = await node_build_case(
                    {
                        "thread_id": "suppressed-case",
                        "asset": Asset.BTC,
                        "snapshot": _snap(),
                        "is_coordinator_case": False,
                        "recent_alert_history": [],
                        "fatigue_suppressed": True,
                        "rule_hits": [_hit(Severity.P2, dedupe_key="dedupe:btc:p2")],
                        "highest_severity": Severity.P2,
                        "technical_analysis": None,
                        "macro_context": None,
                        "technical_analysis_zh": "",
                        "macro_context_zh": "",
                        "summary_zh": "suppressed summary",
                        "review_guidance": "",
                        "historical_context_zh": "",
                        "risk_quantification_zh": "",
                        "decision": Decision.SUPPRESS,
                        "case": None,
                        "alert": None,
                        "human_approved": None,
                        "human_comment": "",
                    }
                )

        self.assertEqual(len(saved_cases), 1)
        self.assertEqual(result["case"].status, CaseStatus.SUPPRESSED)
        self.assertEqual(result["case"].suppression_reason, "fatigue")

    def test_route_after_build_ends_for_suppressed_cases(self):
        self.assertEqual(_route_after_build({"decision": Decision.SUPPRESS}), END)


if __name__ == "__main__":
    unittest.main()
