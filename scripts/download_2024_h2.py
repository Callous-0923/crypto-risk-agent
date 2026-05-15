"""下载 2024 年下半年 Binance 历史数据 (7~12月, 已下载上半年可跳过)。"""
import asyncio, sys, time
from datetime import datetime
from src.domain.models import Asset
from src.ml.historical_data import backfill_binance_public_klines


async def main():
    print("=" * 60)
    print("Downloading 2024 H2 data (July ~ December)")
    print("Assets: BTC, ETH, SOL  |  spot + futures_um  |  6 months")
    print("=" * 60)

    t0 = time.perf_counter()
    result = await backfill_binance_public_klines(
        assets=[Asset.BTC, Asset.ETH, Asset.SOL],
        start=datetime(2024, 7, 1),
        end=datetime(2025, 1, 1),
        interval="1m",
        market_types=["spot", "futures_um"],
    )

    elapsed = time.perf_counter() - t0
    print(f"\nDownload complete ({elapsed:.0f}s)")
    print(f"  Total rows saved (this run): {result['total_saved']:,}")
    print(f"  Pre-existing bars in DB are auto-skipped (UNIQUE constraint)")
    print(f"\n  NOTE: Run scripts/check_data_coverage.py to verify 12-month coverage.")


if __name__ == "__main__":
    asyncio.run(main())
