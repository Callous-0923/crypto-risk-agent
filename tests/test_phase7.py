from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from src.domain.models import Asset, RawEvent
from src.features.builder import FeatureBuilder
from src.ingestion.validator import ValidationResult, validate_sources


class Phase7Tests(unittest.IsolatedAsyncioTestCase):
    async def test_validate_sources_uses_midpoint_without_llm_when_diff_small(self):
        import src.ingestion.validator as validator_mod

        with patch.object(validator_mod, "_arbitrate_conflict", side_effect=AssertionError("should not call")):
            result = await validate_sources(Asset.BTC, 50000.0, 50100.0)

        self.assertFalse(result.conflict_detected)
        self.assertAlmostEqual(result.trusted_price, 50050.0)
        self.assertIn("midpoint", result.resolution_reason)

    async def test_validate_sources_uses_arbitration_when_diff_large(self):
        import src.ingestion.validator as validator_mod

        async def fake_arbitration(asset, binance_price, okx_price):
            return {"trust": "okx", "confidence": 0.88, "reason": "OKX depth is healthier"}

        with patch.object(validator_mod, "_arbitrate_conflict", fake_arbitration):
            result = await validate_sources(Asset.BTC, 50000.0, 50300.0)

        self.assertTrue(result.conflict_detected)
        self.assertEqual(result.trusted_price, 50300.0)
        self.assertEqual(result.resolution_reason, "OKX depth is healthier")

    async def test_builder_build_snapshot_validated_uses_trusted_price(self):
        import src.ingestion.validator as validator_mod

        builder = FeatureBuilder()
        now = datetime.now(tz=timezone.utc)
        builder.ingest(
            RawEvent(
                asset=Asset.BTC,
                source="binance_spot",
                event_type="price",
                event_ts=now,
                payload={"price": 50000.0},
            )
        )
        builder.ingest(
            RawEvent(
                asset=Asset.BTC,
                source="okx_spot",
                event_type="price",
                event_ts=now,
                payload={"price": 50300.0},
            )
        )

        async def fake_validate(asset, binance_price, okx_price):
            self.assertEqual(binance_price, 50000.0)
            self.assertEqual(okx_price, 50300.0)
            return ValidationResult(50300.0, True, "trusted okx")

        with patch.object(validator_mod, "validate_sources", fake_validate):
            snapshot = await builder.build_snapshot_validated(Asset.BTC)

        self.assertEqual(snapshot.price, 50300.0)
        self.assertTrue(snapshot.cross_source_conflict)

    async def test_snapshot_features_use_trusted_price_series_not_mixed_raw_prices(self):
        builder = FeatureBuilder()
        state = builder._states[Asset.BTC]
        now = datetime.now(tz=timezone.utc)

        trusted_points = [
            (now.replace(microsecond=0) - timedelta(seconds=offset), price)
            for offset, price in [
                (270, 100.0),
                (240, 101.0),
                (210, 102.0),
                (180, 103.0),
                (150, 104.0),
                (120, 105.0),
                (90, 106.0),
                (60, 107.0),
                (30, 108.0),
                (0, 109.0),
            ]
        ]
        for ts, price in trusted_points:
            state.record_trusted_price(ts, price)

        builder.ingest(
            RawEvent(
                asset=Asset.BTC,
                source="binance_spot",
                event_type="price",
                event_ts=now,
                payload={"price": 999999.0},
            )
        )
        builder.ingest(
            RawEvent(
                asset=Asset.BTC,
                source="okx_spot",
                event_type="price",
                event_ts=now,
                payload={"price": 1.0},
            )
        )

        snapshot = state.build_snapshot(Asset.BTC, as_of=trusted_points[-1][0])

        self.assertEqual(snapshot.price, 109.0)
        self.assertAlmostEqual(snapshot.ret_1m, (109.0 - 107.0) / 107.0)
        self.assertAlmostEqual(snapshot.vol_z_1m, state._vol_z(60, 300, now=trusted_points[-1][0]))
        self.assertLess(snapshot.ret_1m, 0.1)


if __name__ == "__main__":
    unittest.main()
