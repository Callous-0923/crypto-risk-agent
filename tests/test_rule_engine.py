"""规则引擎单元测试 —— 覆盖边界值、组合条件、版本回放。"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.domain.models import Asset, FeatureSnapshot, Severity
from src.rules.config import RuleThresholds
from src.rules.engine import (
    build_early_warning_profile,
    early_warning_profile_from_hits,
    evaluate_with_thresholds,
    is_early_warning_reviewable,
)


def _make_snap(asset: Asset, **kwargs) -> FeatureSnapshot:
    """构建测试用 FeatureSnapshot。"""
    defaults = {
        "asset": asset,
        "window_end": datetime.now(tz=timezone.utc),
        "price": 100.0,
        "ret_1m": 0.0,
        "ret_5m": 0.0,
        "vol_z_1m": 0.0,
        "oi_delta_15m_pct": 0.0,
        "liq_5m_usd": 0.0,
        "funding_z": 0.0,
        "source_stale": False,
        "cross_source_conflict": False,
    }
    defaults.update(kwargs)
    return FeatureSnapshot(**defaults)


class TestRuleThresholds(unittest.TestCase):
    """RuleThresholds 配置模型测试。"""

    def test_default_thresholds(self):
        t = RuleThresholds()
        self.assertAlmostEqual(t.price_change_p1, 0.05)
        self.assertAlmostEqual(t.liq_usd_p1, 50_000_000)

    def test_diff_returns_changed_fields(self):
        t1 = RuleThresholds(price_change_p1=0.05)
        t2 = RuleThresholds(price_change_p1=0.06)
        diff = t1.diff(t2)
        self.assertIn("price_change_p1", diff)
        self.assertEqual(diff["price_change_p1"]["old"], 0.05)
        self.assertEqual(diff["price_change_p1"]["new"], 0.06)

    def test_diff_empty_when_identical(self):
        t1 = RuleThresholds()
        t2 = RuleThresholds()
        diff = t1.diff(t2)
        self.assertEqual(len(diff), 0)


class TestPriceRules(unittest.TestCase):
    """价格规则测试。"""

    def test_p1_triggered_by_large_ret(self):
        snap = _make_snap(Asset.BTC, ret_1m=-0.06, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(price_change_p1=0.05))
        p1_hits = [h for h in hits if h.rule_id == "MKT_EXTREME_VOL_P1"]
        self.assertTrue(p1_hits)
        self.assertEqual(p1_hits[0].severity, Severity.P1)

    def test_p2_triggered_when_below_p1(self):
        snap = _make_snap(Asset.ETH, ret_1m=0.035, price=3000.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(price_change_p1=0.05, price_change_p2=0.03))
        p1_hits = [h for h in hits if h.rule_id == "MKT_EXTREME_VOL_P1"]
        p2_hits = [h for h in hits if h.rule_id == "MKT_EXTREME_VOL_P2"]
        self.assertFalse(p1_hits)
        self.assertTrue(p2_hits)

    def test_no_hit_when_ret_below_threshold(self):
        snap = _make_snap(Asset.SOL, ret_1m=0.01, price=150.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(price_change_p1=0.05, price_change_p2=0.03))
        vol_hits = [h for h in hits if h.rule_id in {"MKT_EXTREME_VOL_P1", "MKT_EXTREME_VOL_P2"}]
        self.assertFalse(vol_hits)

    def test_vol_z_spike_triggered(self):
        snap = _make_snap(Asset.BTC, vol_z_1m=4.0, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(vol_z_spike=3.0))
        self.assertTrue(any(h.rule_id == "MKT_VOL_Z_SPIKE" for h in hits))

    def test_vol_z_no_hit_when_normal(self):
        snap = _make_snap(Asset.BTC, vol_z_1m=1.0, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(vol_z_spike=3.0))
        self.assertFalse(any(h.rule_id == "MKT_VOL_Z_SPIKE" for h in hits))


class TestDerivativesRules(unittest.TestCase):
    """衍生品规则测试。"""

    def test_liquidation_cascade_p1(self):
        snap = _make_snap(Asset.BTC, liq_5m_usd=60_000_000, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(liq_usd_p1=50_000_000))
        self.assertTrue(any(h.rule_id == "DERIV_CASCADE_LIQ_P1" for h in hits))

    def test_liquidation_no_hit_when_below(self):
        snap = _make_snap(Asset.BTC, liq_5m_usd=30_000_000, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(liq_usd_p1=50_000_000))
        self.assertFalse(any(h.rule_id == "DERIV_CASCADE_LIQ_P1" for h in hits))

    def test_oi_buildup_p2(self):
        snap = _make_snap(Asset.ETH, oi_delta_15m_pct=0.12, price=3000.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(oi_delta_p2=0.10))
        self.assertTrue(any(h.rule_id == "DERIV_OI_BUILDUP" for h in hits))

    def test_funding_squeeze(self):
        snap = _make_snap(Asset.BTC, funding_z=3.0, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(funding_z_p2=2.5))
        self.assertTrue(any(h.rule_id == "DERIV_FUNDING_SQUEEZE" for h in hits))


class TestDataQualityRules(unittest.TestCase):
    """数据质量规则测试。"""

    def test_stale_source_triggers_qa(self):
        snap = _make_snap(Asset.BTC, source_stale=True, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds())
        self.assertTrue(any(h.rule_id == "QA_SOURCE_STALE" for h in hits))

    def test_cross_source_conflict(self):
        snap = _make_snap(Asset.SOL, cross_source_conflict=True, price=150.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(cross_source_conflict_pct=0.005))
        self.assertTrue(any(h.rule_id == "QA_CROSS_SOURCE_CONFLICT" for h in hits))

    def test_stale_trumps_other_rules(self):
        """数据过期时其他规则仍应正常评估。"""
        snap = _make_snap(Asset.BTC, source_stale=True, ret_1m=-0.06, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(price_change_p1=0.05))
        self.assertTrue(any(h.rule_id == "QA_SOURCE_STALE" for h in hits))
        self.assertTrue(any(h.rule_id == "MKT_EXTREME_VOL_P1" for h in hits))


class TestEarlyWarningRules(unittest.TestCase):
    """提前预警规则测试。"""

    def test_ew_suppressed_when_p1_p2_hit(self):
        """已有 P1/P2 命中时不触发 EW。"""
        snap = _make_snap(Asset.BTC, ret_1m=-0.06, ret_5m=-0.03, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(
            price_change_p1=0.05, early_warning_ret_5m=0.008,
            early_warning_oi_delta=0.04, early_warning_funding_z=1.5,
        ))
        self.assertTrue(any(h.rule_id == "MKT_EXTREME_VOL_P1" for h in hits))
        self.assertFalse(any(h.rule_id.startswith("EW_") for h in hits))

    def test_ew_profile_with_multi_signal(self):
        """多信号场景应生成 early_warning_profile 且 candidate=True。"""
        snap = _make_snap(
            Asset.BTC,
            ret_5m=-0.015,
            vol_z_1m=1.8,
            oi_delta_15m_pct=0.06,
            price=100.0,
        )
        t = RuleThresholds(
            early_warning_ret_5m=0.008,
            early_warning_vol_z=1.5,
            early_warning_oi_delta=0.04,
            early_warning_funding_z=1.5,
            early_warning_min_score=0.50,
            early_warning_min_signals=2,
            early_warning_single_signal_min_score=0.65,
        )
        profile = build_early_warning_profile(snap, t)
        self.assertGreater(len(profile["signals"]), 1)
        self.assertTrue(profile["candidate"])

    def test_ew_not_reviewable_with_insufficient_persistence(self):
        """持久化次数不足时不应标记为 reviewable。"""
        snap = _make_snap(Asset.BTC, ret_5m=-0.015, price=100.0)
        t = RuleThresholds(
            early_warning_ret_5m=0.008,
            early_warning_persistence_hits=3,
        )
        hits = evaluate_with_thresholds(snap, t)
        result = is_early_warning_reviewable(hits, recent_candidate_count=1, thresholds=t)
        self.assertFalse(result)

    def test_ew_profile_from_hits_empty(self):
        """无 EW_ 前缀命中时返回默认 profile。"""
        profile = early_warning_profile_from_hits([])
        self.assertEqual(profile["score"], 0.0)
        self.assertEqual(profile["signal_count"], 0)
        self.assertFalse(profile["candidate"])


class TestRuleHitConfidence(unittest.TestCase):
    """规则命中置信度计算测试。"""

    def test_confidence_capped_at_1(self):
        snap = _make_snap(Asset.BTC, ret_1m=-0.15, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(price_change_p1=0.05))
        p1 = [h for h in hits if h.rule_id == "MKT_EXTREME_VOL_P1"][0]
        self.assertLessEqual(p1.confidence, 1.0)
        self.assertGreater(p1.confidence, 0.0)

    def test_severity_order(self):
        """P1 规则命中时 P2 同类型规则不应触发。"""
        snap = _make_snap(Asset.BTC, ret_1m=-0.08, price=100.0)
        hits = evaluate_with_thresholds(snap, RuleThresholds(price_change_p1=0.05, price_change_p2=0.03))
        rule_ids = {h.rule_id for h in hits}
        self.assertIn("MKT_EXTREME_VOL_P1", rule_ids)
        self.assertNotIn("MKT_EXTREME_VOL_P2", rule_ids)


if __name__ == "__main__":
    unittest.main()
