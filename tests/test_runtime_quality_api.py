import unittest
from datetime import datetime
from unittest.mock import patch

from src.api import routes as routes_mod
from src.domain.models import Asset, CaseStatus, Decision, RiskCase, Severity


class RuntimeQualityApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_reject_closes_pending_case_even_when_graph_resume_fails(self):
        case = RiskCase(
            case_id="case-1",
            asset=Asset.BTC,
            status=CaseStatus.PENDING_REVIEW,
            decision=Decision.MANUAL_REVIEW,
            severity=Severity.P2,
            summary_zh="pending",
        )
        saved_cases = []
        review_actions = []

        async def fake_get_case(case_id):
            self.assertEqual(case_id, "case-1")
            return case

        async def fake_save_review_action(action):
            review_actions.append(action)

        async def fake_save_risk_case(updated):
            saved_cases.append(updated.model_copy(deep=True))

        async def fake_resume_case(*args, **kwargs):
            raise RuntimeError("checkpoint unavailable")

        async def fake_list_cases(*args, **kwargs):
            return []

        import src.persistence.repositories as repo_mod

        with patch.object(routes_mod, "get_risk_case", fake_get_case):
            with patch.object(routes_mod, "save_review_action", fake_save_review_action):
                with patch.object(routes_mod, "save_risk_case", fake_save_risk_case):
                    with patch.object(routes_mod, "resume_case", fake_resume_case):
                        with patch.object(repo_mod, "list_risk_cases", fake_list_cases):
                            result = await routes_mod.resume_human_review(
                                "case-1",
                                routes_mod.ResumeRequest(
                                    reviewer="operator",
                                    action="reject",
                                    comment="not actionable",
                                ),
                            )

        self.assertFalse(result["approved"])
        self.assertEqual(result["status"], "closed_without_graph_resume")
        self.assertEqual(review_actions[0].action.value, "reject")
        self.assertEqual(saved_cases[-1].status, CaseStatus.CLOSED)
        self.assertIsInstance(saved_cases[-1].updated_at, datetime)

    async def test_cases_endpoint_supports_status_pagination(self):
        case = RiskCase(
            case_id="case-2",
            asset=Asset.ETH,
            status=CaseStatus.PENDING_REVIEW,
            decision=Decision.MANUAL_REVIEW,
            severity=Severity.P2,
        )

        async def fake_list_cases(asset, limit, *, include_suppressed=False, status=None, offset=0):
            self.assertEqual(asset, Asset.ETH)
            self.assertEqual(limit, 10)
            self.assertEqual(offset, 10)
            self.assertEqual(status, CaseStatus.PENDING_REVIEW)
            return [case]

        async def fake_count_cases(asset, *, include_suppressed=False, status=None):
            self.assertEqual(asset, Asset.ETH)
            self.assertEqual(status, CaseStatus.PENDING_REVIEW)
            return 11

        with patch.object(routes_mod, "list_risk_cases", fake_list_cases):
            with patch.object(routes_mod, "count_risk_cases", fake_count_cases):
                result = await routes_mod.get_cases(
                    asset="ETH",
                    limit=10,
                    offset=10,
                    status="pending_review",
                    paginated=True,
                )

        self.assertEqual(result["total"], 11)
        self.assertFalse(result["has_more"])
        self.assertEqual(result["items"][0]["case_id"], "case-2")


if __name__ == "__main__":
    unittest.main()
