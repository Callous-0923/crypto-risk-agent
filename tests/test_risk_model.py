import unittest
from datetime import datetime, timedelta

from src.domain.models import Asset, FeatureSnapshot, Severity
from src.graph.nodes import node_run_rules
from src.ml.features import FEATURE_COLUMNS, build_feature_dict, rows_to_matrix
from src.ml.labeling import deterministic_label, future_summary
from src.ml.risk_model import prediction_to_rule_hit, risk_level_from_probability


def _snap(offset_minutes: int = 0, **kwargs) -> FeatureSnapshot:
    defaults = dict(
        asset=Asset.BTC,
        window_end=datetime(2026, 1, 1, 0, 0, 0) + timedelta(minutes=offset_minutes),
        price=100.0,
        ret_1m=0.0,
        ret_5m=0.0,
        vol_z_1m=0.0,
        oi_delta_15m_pct=0.0,
        liq_5m_usd=0.0,
        funding_z=0.0,
    )
    defaults.update(kwargs)
    return FeatureSnapshot(**defaults)


class RiskModelTests(unittest.TestCase):
    def test_feature_matrix_uses_stable_columns(self):
        row = build_feature_dict(
            _snap(ret_1m=0.02, ret_5m=0.03, vol_z_1m=2.0),
            history=[_snap(-1, ret_1m=0.01), _snap(-2, ret_1m=-0.01)],
        )
        matrix = rows_to_matrix([row])

        self.assertEqual(len(matrix[0]), len(FEATURE_COLUMNS))
        self.assertEqual(row["asset_BTC"], 1.0)
        self.assertGreater(row["risk_signal_count"], 0)

    def test_deterministic_label_from_future_move(self):
        anchor = _snap(0, price=100.0)
        summary = future_summary(
            anchor,
            [_snap(10, price=104.0), _snap(20, price=98.0)],
            horizon_seconds=3600,
        )
        label = deterministic_label(summary)

        self.assertEqual(label["label"], "p1")
        self.assertGreater(label["risk_probability"], 0)

    def test_probability_maps_to_risk_level(self):
        self.assertEqual(risk_level_from_probability(0.90), "P1")
        self.assertEqual(risk_level_from_probability(0.60), "P2")
        self.assertEqual(risk_level_from_probability(0.35), "P3")
        self.assertEqual(risk_level_from_probability(0.10), "none")

    def test_prediction_to_rule_hit_and_node_run_rules(self):
        snap = _snap()
        prediction = {
            "model_version": "test-model",
            "raw_probability": 0.7,
            "calibrated_probability": 0.6,
            "risk_level": "P2",
            "top_features": [],
            "horizon_seconds": 3600,
        }
        hit = prediction_to_rule_hit(snap, prediction)
        self.assertEqual(hit.rule_id, "ML_RISK_PROBABILITY")
        self.assertEqual(hit.severity, Severity.P2)

        state = {"snapshot": snap, "ml_prediction": prediction}
        result = node_run_rules(state)
        self.assertTrue(any(item.rule_id == "ML_RISK_PROBABILITY" for item in result["rule_hits"]))
        self.assertEqual(result["highest_severity"], Severity.P2)


if __name__ == "__main__":
    unittest.main()

