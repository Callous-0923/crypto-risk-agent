"""用 Optuna 找到的最佳参数进行最终训练。

最佳参数来自 30 轮 TPE 搜索:
  learning_rate: 0.0352  num_leaves: 24  max_depth: 10
  min_child_samples: 41  reg_alpha: 0.26  reg_lambda: 0.16
  subsample: 0.84  colsample_bytree: 0.97
"""
import asyncio, time
from datetime import datetime
from src.domain.models import Asset
from src.ml.historical_training import train_historical_risk_model

BEST_PARAMS = {
    "learning_rate": 0.03521358805467869,
    "num_leaves": 24,
    "max_depth": 10,
    "min_child_samples": 41,
    "reg_alpha": 0.2598228519576783,
    "reg_lambda": 0.1602385029019493,
    "subsample": 0.8391599915244341,
    "colsample_bytree": 0.9687496940092467,
}


async def main():
    print("=" * 60)
    print("Optuna Best Params — Final Training (2024 full year)")
    print("=" * 60)

    t0 = time.perf_counter()
    result = await train_historical_risk_model(
        assets=[Asset.BTC, Asset.ETH, Asset.SOL],
        start=datetime(2024, 1, 1),
        end=datetime(2025, 1, 1),
        horizon_seconds=3600,
        max_snapshots_per_asset=150000,
        max_label_items=100000,
        min_samples=1000,
        model_params=BEST_PARAMS,
    )
    elapsed = time.perf_counter() - t0

    m = result.get("metrics", {})
    dist = result.get("label_distribution", {})
    labeling = result.get("historical_labeling_summary", {})

    print(f"\n{' Final Model Report ':=^60}")
    print(f"  Version:        {result['model_version']}")
    print(f"  Train: {result['training_samples']:,}  "
          f"Val: {result['val_samples']:,}  Test: {result['test_samples']:,}")
    print(f"  Trees:          {result['n_estimators_used']}")
    print(f"  Duration:       {result['training_duration_seconds']:.0f}s (wall: {elapsed:.0f}s)")
    print(f"  Size:           {result['model_size_bytes']/1024:.0f} KB")
    print(f"  AUC-ROC:        {result.get('auc_roc', 0):.4f}")
    print(f"  Raw AUC:        {result.get('auc_roc_raw', 0):.4f}")
    print(f"  Precision:      {m.get('precision', 0):.4f}")
    print(f"  Recall:         {m.get('recall', 0):.4f}")
    print(f"  F1 Score:       {m.get('f1', 0):.4f}")
    print(f"  P2 Threshold:   {m.get('threshold', 0):.4f}")
    print(f"  TP={m.get('tp',0)} FP={m.get('fp',0)} "
          f"TN={m.get('tn',0)} FN={m.get('fn',0)}")

    for split in ["train", "val", "test"]:
        d = dist.get(split, {})
        t = sum(d.values())
        if t == 0: continue
        print(f"  {split:5s}: {t:>6,} | p1={d.get('p1',0):>5,} "
              f"({d.get('p1',0)/t*100:.1f}%) "
              f"p2={d.get('p2',0):>5,} ({d.get('p2',0)/t*100:.1f}%)")
    print(f"  Candidates:    {labeling.get('candidates', '?'):,}")
    print(f"  Labeled:       {labeling.get('labeled', '?'):,}")

    fi = result.get("feature_importances", [])[:10]
    if fi:
        print(f"\n  Top features:")
        for f in fi:
            print(f"     {f['feature']:<28s} {f['importance']:>8.1f}")

    print(f"\n{'=' * 60}")
    print("Model: artifacts/risk_model/latest.joblib")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
