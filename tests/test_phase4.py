import asyncio
import time
import unittest
from unittest.mock import patch

from src.domain.models import Asset, Decision, FeatureSnapshot, RiskCase, RuleHit, Severity
from src.features.builder import run_snapshot_cycle
from src.graph.coordinator import coordinate_cross_asset
from src.graph.nodes import node_build_case


def _snap(asset: Asset) -> FeatureSnapshot:
    return FeatureSnapshot(asset=asset, price=100.0)


def _hit(asset: Asset, severity: Severity) -> RuleHit:
    return RuleHit(
        rule_id=f"TEST_{asset.value}_{severity.value}",
        asset=asset,
        severity=severity,
        description=f"{asset.value} {severity.value}",
        dedupe_key=f"test:{asset.value}:{severity.value}",
    )


class Phase4Tests(unittest.IsolatedAsyncioTestCase):
    async def test_run_snapshot_cycle_processes_assets_in_parallel(self):
        import src.features.builder as builder_mod

        saved_assets: list[Asset] = []
        processed_assets: list[Asset] = []
        coordinated_assets: list[Asset] = []

        class DummyBuilder:
            def get_snapshot(self, asset: Asset) -> FeatureSnapshot:
                return _snap(asset)

        async def fake_save(snap: FeatureSnapshot) -> None:
            saved_assets.append(snap.asset)

        async def fake_process(snap: FeatureSnapshot) -> None:
            await asyncio.sleep(0.1)
            processed_assets.append(snap.asset)

        async def fake_coordinate(snaps: dict[Asset, FeatureSnapshot]) -> None:
            coordinated_assets.extend(snaps.keys())

        with patch.object(builder_mod, "save_feature_snapshot", fake_save):
            with patch("src.graph.orchestrator.process_snapshot", fake_process):
                with patch("src.graph.coordinator.coordinate_cross_asset", fake_coordinate):
                    started_at = time.perf_counter()
                    persisted = await run_snapshot_cycle(DummyBuilder())
                    elapsed = time.perf_counter() - started_at

        self.assertEqual(set(saved_assets), {Asset.BTC, Asset.ETH, Asset.SOL})
        self.assertEqual(set(processed_assets), {Asset.BTC, Asset.ETH, Asset.SOL})
        self.assertEqual(set(coordinated_assets), {Asset.BTC, Asset.ETH, Asset.SOL})
        self.assertEqual(set(persisted), {Asset.BTC, Asset.ETH, Asset.SOL})
        self.assertLess(elapsed, 0.22)

    async def test_coordinate_cross_asset_creates_synthetic_case_for_multi_asset_p1(self):
        import src.graph.coordinator as coordinator_mod

        recent_cases = [
            RiskCase(
                case_id="btc-case",
                asset=Asset.BTC,
                severity=Severity.P1,
                decision=Decision.EMIT,
                summary_zh="BTC 触发 P1",
                rule_hits=[_hit(Asset.BTC, Severity.P1)],
            ),
            RiskCase(
                case_id="eth-case",
                asset=Asset.ETH,
                severity=Severity.P1,
                decision=Decision.EMIT,
                summary_zh="ETH 触发 P1",
                rule_hits=[_hit(Asset.ETH, Severity.P1)],
            ),
        ]
        created: dict[str, object] = {}
        scheduled: list[object] = []

        async def fake_recent(*args, **kwargs):
            return recent_cases

        async def fake_find_recent(*args, **kwargs):
            return None

        async def fake_summary(asset_cases):
            return "跨资产摘要", "请审核跨资产联动"

        async def fake_process_coordinator_case(**kwargs):
            created.update(kwargs)
            return "coord-case-id"

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return None

        with patch.object(coordinator_mod, "list_recent_risk_cases", fake_recent):
            with patch.object(coordinator_mod, "find_recent_coordinator_case", fake_find_recent):
                with patch.object(coordinator_mod, "_build_cross_asset_summary", fake_summary):
                    with patch.object(coordinator_mod, "process_coordinator_case", fake_process_coordinator_case):
                        with patch.object(coordinator_mod.asyncio, "create_task", fake_create_task):
                            await coordinate_cross_asset(
                                {
                                    Asset.BTC: _snap(Asset.BTC),
                                    Asset.ETH: _snap(Asset.ETH),
                                    Asset.SOL: _snap(Asset.SOL),
                                }
                            )

        self.assertEqual(created["asset"], Asset.BTC)
        self.assertEqual(created["severity"], Severity.P1)
        self.assertEqual(created["summary_zh"], "跨资产摘要")
        hit = created["rule_hits"][0]
        self.assertEqual(hit.rule_id, "COORD_MULTI_ASSET_EVENT")
        self.assertEqual(hit.evidence["assets"], ["BTC", "ETH"])
        self.assertEqual(len(scheduled), 1)

    async def test_node_build_case_uses_thread_id_and_marks_coordinator(self):
        import src.graph.nodes as nodes_mod
        import src.persistence.repositories as repo_mod

        captured: dict[str, RiskCase] = {}

        async def fake_save(case: RiskCase) -> None:
            captured["case"] = case

        async def fake_list_cases(*args, **kwargs):
            return [captured["case"]]

        with patch.object(nodes_mod, "save_risk_case", fake_save):
            with patch.object(repo_mod, "list_risk_cases", fake_list_cases):
                result = await node_build_case(
                    {
                        "thread_id": "thread-123",
                        "asset": Asset.BTC,
                        "snapshot": _snap(Asset.BTC),
                        "is_coordinator_case": True,
                        "rule_hits": [_hit(Asset.BTC, Severity.P1)],
                        "highest_severity": Severity.P1,
                        "summary_zh": "协调摘要",
                        "review_guidance": "请审核",
                        "decision": Decision.MANUAL_REVIEW,
                        "case": None,
                        "alert": None,
                        "human_approved": None,
                        "human_comment": "",
                    }
                )

        case = result["case"]
        self.assertEqual(case.case_id, "thread-123")
        self.assertTrue(case.is_coordinator_case)
        self.assertEqual(case.decision, Decision.MANUAL_REVIEW)


if __name__ == "__main__":
    unittest.main()
