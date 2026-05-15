"""仅训练 LightGBM — 使用 DB 中已有历史数据（跳过下载）。

运行: python scripts/train_only.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime

from src.core.logging import get_logger
from src.domain.models import Asset
from src.ml.historical_training import train_historical_risk_model

logger = get_logger(__name__)


async def main() -> None:
    print("=" * 60)
    print("ETC Agent — LightGBM Training (balanced sampling)")
    print("Period: 2024-01 ~ 2024-07 | per-asset: 80K | labels: 80K")
    print("Improvements: balanced pos/neg ratio + is_unbalance + relaxed quantiles")
    print("=" * 60)

    t0 = time.perf_counter()

    result = await train_historical_risk_model(
        assets=[Asset.BTC, Asset.ETH, Asset.SOL],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 7, 1),
        horizon_seconds=3600,
        max_snapshots_per_asset=80000,
        max_label_items=80000,
        min_samples=1000,
    )

    elapsed = time.perf_counter() - t0

    # ===== 输出完整报告 =====
    print(f"\n{'='*60}")
    print(f"Training complete ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Model:   {result['model_version']}")
    print(f"  Size:    {result['model_size_bytes'] / 1024:.0f} KB")
    print(f"  Train:   {result['training_samples']:,} samples")
    print(f"  Val:     {result['val_samples']:,}")
    print(f"  Test:    {result['test_samples']:,}")
    print(f"  Trees:   {result['n_estimators_used']} (early_stopping)")

    m = result.get("metrics", {})
    print(f"\n  AUC-ROC:       {result.get('auc_roc', 0):.4f}")
    print(f"  Precision:     {m.get('precision', 0):.4f}")
    print(f"  Recall:        {m.get('recall', 0):.4f}")
    print(f"  F1:            {m.get('f1', 0):.4f}")
    print(f"  P2 Threshold:  {m.get('threshold', 0):.4f}")
    print(f"  TP={m.get('tp',0)} FP={m.get('fp',0)} TN={m.get('tn',0)} FN={m.get('fn',0)}")

    dist = result.get("label_distribution", {})
    for split in ["train", "val", "test"]:
        d = dist.get(split, {})
        print(f"  {split}: p1={d.get('p1',0)} p2={d.get('p2',0)} none={d.get('none',0)}")

    labeling = result.get("historical_labeling_summary", {})
    print(f"  Candidates:    {labeling.get('candidates', '?'):,}")
    print(f"  Labeled:       {labeling.get('labeled', '?'):,}")

    fi = result.get("feature_importances", [])[:8]
    if fi:
        print(f"\n  Top features:")
        for f in fi:
            print(f"    {f['feature']:<30s} {f['importance']:.4f}")

    print(f"\n{'='*60}")
    print("Model saved to artifacts/risk_model/latest.joblib")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
