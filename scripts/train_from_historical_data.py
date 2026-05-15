"""下载历史数据 → 训练 LightGBM → 输出全量指标。

运行: python scripts/train_from_historical_data.py

分两步执行:
  Step 1 — 下载 Binance Public Data (2024年 BTC/ETH/SOL)
  Step 2 — 训练 LightGBM 并输出全量指标
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime

from src.domain.models import Asset
from src.ml.historical_data import backfill_binance_public_klines
from src.ml.historical_training import train_historical_risk_model


async def step1_download() -> dict:
    """从 Binance Public Data 下载历史 K 线。"""
    print("=" * 60)
    print("Step 1: Downloading historical market data")
    print("=" * 60)
    print(
        "Assets: BTC, ETH, SOL  |  Markets: spot + futures_um\n"
        "Period: 2024-01 ~ 2024-07 (6 months)\n"
        "Source: data.binance.vision (free, no auth required)\n"
    )
    t0 = time.perf_counter()

    result = await backfill_binance_public_klines(
        assets=[Asset.BTC, Asset.ETH, Asset.SOL],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 7, 1),
        interval="1m",
        market_types=["spot", "futures_um"],
    )

    elapsed = time.perf_counter() - t0
    print(f"\nDownload complete ({elapsed:.0f}s)")
    print(f"  Total rows saved: {result['total_saved']:,}")
    for f in result.get("files", []):
        print(f"  {f['asset']}/{f['market_type']} {f['month']}: "
              f"{f['saved']:,} bars (from {f['rows']} rows)")
    return result


async def step2_train(download_result: dict) -> dict:
    """用历史数据训练 LightGBM。"""
    print("\n" + "=" * 60)
    print("Step 2: Training LightGBM risk model")
    print("=" * 60)
    t0 = time.perf_counter()

    result = await train_historical_risk_model(
        assets=[Asset.BTC, Asset.ETH, Asset.SOL],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 7, 1),
        horizon_seconds=3600,
        max_snapshots_per_asset=80000,
        max_label_items=100000,
        min_samples=1000,
        p2_quantile=0.95,
        p1_quantile=0.995,
    )

    elapsed = time.perf_counter() - t0
    result["_wall_seconds"] = round(elapsed, 1)
    return result


def print_report(download: dict, train: dict) -> None:
    """格式化输出完整训练报告。"""
    print("\n" + "=" * 60)
    print("ETC Risk Agent — LightGBM Training Report")
    print("=" * 60)

    # ---- 模型信息 ----
    print(f"\n{' Model Info ':-^60}")
    print(f"  Version:      {train['model_version']}")
    print(f"  Model size:   {train['model_size_bytes'] / 1024:.1f} KB")
    print(f"  Horizon:      {train.get('horizon_seconds', 3600)}s (1h)")

    # ---- 数据量 ----
    print(f"\n{' Data Pipeline ':-^60}")
    print(f"  Historical bars downloaded: {download['total_saved']:,}")
    print(f"  Labeled candidates:         "
          f"{train.get('historical_labeling_summary', {}).get('candidates', '?'):,}")
    print(f"  Labeled (after sampling):   "
          f"{train.get('historical_labeling_summary', {}).get('labeled', '?'):,}")

    label_dist = train.get("label_distribution", {})
    for split_name in ["train", "val", "test"]:
        dist = label_dist.get(split_name, {})
        print(f"  {split_name:5s}: total={sum(dist.values()):,}  "
              f"p1={dist.get('p1', 0)} p2={dist.get('p2', 0)} none={dist.get('none', 0)}")

    print(f"  Training samples: {train['training_samples']:,}")
    print(f"  Validation:       {train['val_samples']:,}")
    print(f"  Test:             {train['test_samples']:,}")

    # ---- 核心指标 ----
    print(f"\n{' Core Metrics ':-^60}")
    m = train.get("metrics", {})
    print(f"  AUC-ROC (raw):         {train.get('auc_roc_raw', 0):.4f}")
    print(f"  AUC-ROC (calibrated):  {train.get('auc_roc', 0):.4f}")
    print(f"  Precision:             {m.get('precision', 0):.4f}")
    print(f"  Recall:                {m.get('recall', 0):.4f}")
    print(f"  F1 Score:              {m.get('f1', 0):.4f}")
    print(f"  P2 Threshold:          {m.get('threshold', 0):.4f}")
    print(f"  TP={m.get('tp', 0)}  FP={m.get('fp', 0)}  "
          f"TN={m.get('tn', 0)}  FN={m.get('fn', 0)}")

    # ---- 训练效率 ----
    print(f"\n{' Training Efficiency ':-^60}")
    print(f"  Trees used:            {train['n_estimators_used']} / 500 "
          f"({'early_stopped' if train['n_estimators_used'] < 500 else 'used_all'})")
    print(f"  Training duration:     {train['training_duration_seconds']:.1f}s")
    print(f"  Total wall time:       {train.get('_wall_seconds', '?')}s")

    # ---- 特征重要性 Top-10 ----
    print(f"\n{' Top-10 Feature Importances ':-^60}")
    for i, fi in enumerate(train.get("feature_importances", [])[:10]):
        bar = "█" * max(1, int(fi["importance"] / 0.01))
        print(f"  {i + 1:2d}. {fi['feature']:32s} {fi['importance']:.4f} {bar}")

    # ---- 分类报告 ----
    cr = train.get("classification_report", {})
    if isinstance(cr, dict) and "none/P3" in cr:
        print(f"\n{' Per-Class Report ':-^60}")
        for cls_name in ["none/P3", "P1/P2"]:
            if cls_name in cr:
                c = cr[cls_name]
                print(f"  {cls_name}: precision={c.get('precision', 0):.3f}  "
                      f"recall={c.get('recall', 0):.3f}  "
                      f"f1={c.get('f1-score', 0):.3f}  "
                      f"support={c.get('support', 0)}")

    # ---- 标签方法分布 ----
    labeling = train.get("labeling_summary", {})
    method_breakdown = labeling.get("method_breakdown", {})
    if method_breakdown:
        print(f"\n{' Labeling Methods ':-^60}")
        for method, count in sorted(method_breakdown.items()):
            print(f"  {method}: {count:,}")

    print(f"\n{'=' * 60}")
    print("Training complete. Model saved to artifacts/risk_model/latest.joblib")
    print(f"{'=' * 60}")


async def main() -> None:
    print("ETC Agent — Historical Data + LightGBM Training Pipeline")
    print("Assets: BTC, ETH, SOL  |  Period: 2024-01 ~ 2024-07")
    print()

    download_result = await step1_download()
    if download_result["total_saved"] == 0:
        print("\nERROR: No data downloaded. Check network connectivity to data.binance.vision")
        sys.exit(1)

    train_result = await step2_train(download_result)
    print_report(download_result, train_result)


if __name__ == "__main__":
    asyncio.run(main())
