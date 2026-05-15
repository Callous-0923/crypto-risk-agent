from datetime import datetime, timedelta, timezone
import unittest

from src.domain.models import Asset, FeatureSnapshot
from src.market.candles import aggregate_snapshots_to_candles


class MarketCandleTests(unittest.TestCase):
    def test_aggregate_snapshots_to_1m_candles_uses_30s_points(self):
        base = datetime(2026, 4, 22, 11, 0, 0, tzinfo=timezone.utc)
        snapshots = [
            FeatureSnapshot(asset=Asset.BTC, window_end=base + timedelta(seconds=0), price=100.0),
            FeatureSnapshot(asset=Asset.BTC, window_end=base + timedelta(seconds=30), price=104.0),
            FeatureSnapshot(asset=Asset.BTC, window_end=base + timedelta(minutes=1), price=103.0),
            FeatureSnapshot(asset=Asset.BTC, window_end=base + timedelta(minutes=1, seconds=30), price=107.0),
        ]

        candles = aggregate_snapshots_to_candles(
            Asset.BTC,
            snapshots,
            limit=10,
            now=base + timedelta(minutes=2),
        )

        self.assertEqual(len(candles), 2)
        first, second = candles
        self.assertEqual(first.open, 100.0)
        self.assertEqual(first.high, 104.0)
        self.assertEqual(first.low, 100.0)
        self.assertEqual(first.close, 104.0)
        self.assertEqual(first.snapshot_count, 2)
        self.assertTrue(first.is_closed)

        self.assertEqual(second.open, 103.0)
        self.assertEqual(second.high, 107.0)
        self.assertEqual(second.low, 103.0)
        self.assertEqual(second.close, 107.0)
        self.assertEqual(second.snapshot_count, 2)

    def test_aggregate_snapshots_to_1m_candles_marks_current_bucket_open(self):
        base = datetime(2026, 4, 22, 11, 0, 0, tzinfo=timezone.utc)
        snapshots = [
            FeatureSnapshot(asset=Asset.ETH, window_end=base, price=2000.0),
            FeatureSnapshot(asset=Asset.ETH, window_end=base + timedelta(seconds=30), price=2010.0),
        ]

        candles = aggregate_snapshots_to_candles(
            Asset.ETH,
            snapshots,
            now=base + timedelta(seconds=45),
        )

        self.assertEqual(len(candles), 1)
        self.assertFalse(candles[0].is_closed)


if __name__ == "__main__":
    unittest.main()
