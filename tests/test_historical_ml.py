import unittest
from datetime import datetime, timedelta

from src.domain.models import Asset, HistoricalMarketBar
from src.ml.historical_data import binance_monthly_kline_url
from src.ml.historical_features import build_snapshots_from_bars


def _bar(index: int, *, base: float = 100.0, volume: float = 10.0) -> HistoricalMarketBar:
    open_time = datetime(2024, 1, 1) + timedelta(minutes=index)
    close = base + index * 0.1
    return HistoricalMarketBar(
        source="binance_public",
        market_type="spot",
        asset=Asset.BTC,
        symbol="BTCUSDT",
        interval="1m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=close - 0.05,
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=volume + index,
        quote_volume=(volume + index) * close,
        trade_count=100 + index,
        taker_buy_base_volume=(volume + index) * 0.55,
        taker_buy_quote_volume=(volume + index) * close * 0.55,
    )


class HistoricalMlTests(unittest.TestCase):
    def test_binance_public_url_uses_expected_layout(self):
        url = binance_monthly_kline_url(
            asset=Asset.BTC,
            month=datetime(2024, 1, 1),
            interval="1m",
            market_type="spot",
        )

        self.assertEqual(
            url,
            "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip",
        )

    def test_build_snapshots_adds_volume_and_volatility_features(self):
        bars = [_bar(index) for index in range(80)]
        snapshots = build_snapshots_from_bars(
            asset=Asset.BTC,
            spot_bars=bars,
            start=datetime(2024, 1, 1, 1, 0),
            end=datetime(2024, 1, 1, 1, 20),
        )

        self.assertGreater(len(snapshots), 0)
        snap = snapshots[0]
        self.assertGreater(snap.ret_60m, 0)
        self.assertGreater(snap.quote_volume_15m, 0)
        self.assertGreaterEqual(snap.taker_buy_ratio_1m, 0.5)
        self.assertGreaterEqual(snap.realized_vol_15m, 0)


if __name__ == "__main__":
    unittest.main()
