import unittest
from datetime import datetime, timedelta

from src.domain.models import Asset, FeatureSnapshot, Severity
from src.evaluation.offline import (
    build_weak_label_episodes,
    evaluate_weak_label_episodes,
)
from src.rules.config import RuleThresholds


def _snap(
    offset_minutes: int,
    *,
    price: float,
    ret_1m: float = 0.0,
    ret_5m: float = 0.0,
    vol_z_1m: float = 0.0,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        asset=Asset.BTC,
        window_end=datetime(2026, 1, 1, 0, 0, 0) + timedelta(minutes=offset_minutes),
        price=price,
        ret_1m=ret_1m,
        ret_5m=ret_5m,
        vol_z_1m=vol_z_1m,
    )


class OfflineEvaluationTests(unittest.TestCase):
    def test_future_move_labels_can_produce_false_negatives_and_false_positives(self):
        thresholds = RuleThresholds(price_change_p1=0.05, price_change_p2=0.03)
        series = {
            Asset.BTC: [
                _snap(0, price=100.0, ret_1m=0.0, ret_5m=0.02, vol_z_1m=2.2),
                _snap(5, price=100.5, ret_1m=0.0, ret_5m=0.02, vol_z_1m=2.2),
                _snap(10, price=106.0, ret_1m=0.0),
                _snap(20, price=100.0, ret_1m=0.04),
                _snap(25, price=100.5, ret_1m=0.0),
                _snap(40, price=101.0, ret_1m=0.0),
            ]
        }

        episodes = build_weak_label_episodes(
            series,
            thresholds=thresholds,
            horizon_seconds=600,
            min_gap_seconds=0,
            max_episodes=10,
        )
        result = evaluate_weak_label_episodes(
            episodes,
            horizon_seconds=600,
            min_gap_seconds=0,
        )

        self.assertGreaterEqual(result["label_breakdown"][Severity.P1.value], 1)
        self.assertGreaterEqual(result["rules_baseline"]["fn"], 1)
        self.assertGreaterEqual(result["rules_baseline"]["fp"], 1)
        self.assertLess(result["rules_baseline"]["recall"], 1.0)
        self.assertGreaterEqual(result["early_warning_policy"]["tp"], 1)
        self.assertGreaterEqual(
            result["risk_detection_policy"]["recall"],
            result["rules_baseline"]["recall"],
        )
        self.assertIn("early_warning_recall", result["core_metrics"])


if __name__ == "__main__":
    unittest.main()
