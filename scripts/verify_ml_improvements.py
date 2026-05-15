"""LightGBM 改进效果独立验证脚本。

运行: python scripts/verify_ml_improvements.py
"""
import asyncio
import random
import time
import uuid

from src.domain.models import Asset, FeatureSnapshot
from src.ml.risk_model import (
    predict_snapshot,
    train_risk_model,
    model_status,
    load_model_bundle,
)
from src.persistence.database import AsyncSessionLocal, RiskModelLabelRow

random.seed(42)
_BASE_TIME = "2024-06-01T00:00:00"


def _make_snap(asset: Asset, offset: int, *, p2: bool = False) -> FeatureSnapshot:
    """构建合成快照：风险样本 vs 正常样本有明显特征差异。"""
    from datetime import datetime, timedelta, timezone

    bp = {Asset.BTC: 62000.0, Asset.ETH: 3000.0}[asset]
    pid = f"vrfy-{uuid.uuid4().hex[:8]}"
    window_end = datetime.fromisoformat(_BASE_TIME).replace(tzinfo=timezone.utc) + timedelta(minutes=offset)

    if p2:
        return FeatureSnapshot(
            snapshot_id=pid, asset=asset, window_end=window_end,
            price=bp * (1 + random.gauss(-0.04, 0.03)),
            ret_1m=random.gauss(-0.04, 0.02),
            ret_5m=random.gauss(-0.05, 0.03),
            ret_15m=random.gauss(-0.06, 0.03),
            vol_z_1m=random.gauss(2.5, 1.0),
            realized_vol_5m=abs(random.gauss(0.008, 0.004)),
            realized_vol_15m=abs(random.gauss(0.012, 0.005)),
            max_drawdown_15m=abs(random.gauss(0.04, 0.02)),
            max_runup_15m=abs(random.gauss(0.02, 0.01)),
            atr_14=abs(random.gauss(0.008, 0.003)),
            oi_delta_15m_pct=random.gauss(0.10, 0.05),
            oi_delta_5m_pct=random.gauss(0.05, 0.03),
            liq_5m_usd=abs(random.gauss(30_000_000, 15_000_000)),
            funding_z=random.gauss(-2.5, 1.0),
            futures_basis_pct=random.gauss(-0.003, 0.002),
            volume_z_15m=random.gauss(2.0, 1.0),
            trade_count_z_15m=random.gauss(2.0, 1.0),
        )
    return FeatureSnapshot(
        snapshot_id=pid, asset=asset, window_end=window_end,
        price=bp * (1 + random.gauss(0, 0.005)),
        ret_1m=random.gauss(0, 0.005),
        ret_5m=random.gauss(0, 0.008),
        ret_15m=random.gauss(0, 0.01),
        vol_z_1m=random.gauss(0, 0.5),
        realized_vol_5m=abs(random.gauss(0.002, 0.001)),
        realized_vol_15m=abs(random.gauss(0.003, 0.001)),
        max_drawdown_15m=abs(random.gauss(0.005, 0.003)),
        max_runup_15m=abs(random.gauss(0.005, 0.003)),
        atr_14=abs(random.gauss(0.002, 0.001)),
        oi_delta_15m_pct=random.gauss(0, 0.01),
        oi_delta_5m_pct=random.gauss(0, 0.005),
        liq_5m_usd=abs(random.gauss(500_000, 1_000_000)),
        funding_z=random.gauss(0, 0.5),
        futures_basis_pct=random.gauss(0, 0.0005),
        volume_z_15m=random.gauss(0, 0.5),
        trade_count_z_15m=random.gauss(0, 0.5),
    )


async def inject_labels(snaps: list[FeatureSnapshot]) -> None:
    """直接向 DB 注入合成标签（绕过 LLM judge）。"""
    import json
    async with AsyncSessionLocal() as s:
        for snap in snaps:
            is_p2 = (random.random() < 0.33)  # 约 1/3 为正类
            label = "p2" if is_p2 else "none"
            s.add(RiskModelLabelRow(
                label_id=f"vrfy-{uuid.uuid4().hex[:12]}",
                snapshot_id=snap.snapshot_id,
                asset=snap.asset.value,
                window_end=snap.window_end,
                horizon_seconds=3600,
                label=label,
                risk_probability=0.7 if is_p2 else 0.1,
                confidence=0.65,
                labeling_method="verify_script",
                rationale=f"Verify {label}",
                judge_payload_json=json.dumps({"verify": True}),
                created_at=snap.window_end,
            ))
        await s.commit()


async def main() -> None:
    print("=" * 60)
    print("LightGBM Improvements Verification")
    print("=" * 60)

    # 1. 构建数据
    snaps: list[FeatureSnapshot] = []
    for asset in [Asset.BTC, Asset.ETH]:
        for i in range(250):
            snaps.append(_make_snap(asset, i, p2=(random.random() < 0.33)))
    print(f"Built {len(snaps)} synthetic snapshots")

    # 2. 注入标签
    await inject_labels(snaps)
    print(f"Injected labels into DB")

    # 3. 训练
    series = {
        Asset.BTC: [s for s in snaps if s.asset == Asset.BTC],
        Asset.ETH: [s for s in snaps if s.asset == Asset.ETH],
    }
    t0 = time.perf_counter()
    result = await train_risk_model(
        series,
        horizon_seconds=3600,
        max_label_items=0,
        use_llm_judge=False,
        min_samples=30,
        training_metadata={"source": "verify_script"},
    )
    wall_time = time.perf_counter() - t0

    # 4. 打印结果
    print(f"\n{'='*60}")
    print(f"Training Complete (wall: {wall_time:.1f}s)")
    print(f"{'='*60}")
    print(f"Version:       {result['model_version']}")
    print(f"Train samples: {result['training_samples']}")
    print(f"Val samples:   {result['val_samples']}")
    print(f"Test samples:  {result['test_samples']}")
    print(f"Trees used:    {result['n_estimators_used']} / 500")
    print(f"Train duration:{result['training_duration_seconds']:.1f}s")
    print(f"Model size:    {result['model_size_bytes'] / 1024:.1f} KB")

    m = result["metrics"]
    print(f"\n--- Test Metrics ---")
    print(f"AUC-ROC (raw):  {result['auc_roc_raw']:.4f}")
    print(f"AUC-ROC (cal):  {result['auc_roc']:.4f}")
    print(f"Precision:      {m['precision']:.4f}")
    print(f"Recall:         {m['recall']:.4f}")
    print(f"F1:             {m['f1']:.4f}")
    print(f"P2 Threshold:   {m['threshold']:.4f}")
    print(f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")

    print(f"\n--- Label Distribution ---")
    for split, dist in result["label_distribution"].items():
        print(f"  {split}: {dist}")

    print(f"\n--- Top Feature Importances ---")
    for fi in result["feature_importances"][:5]:
        print(f"  {fi['feature']:30s}: {fi['importance']:.4f}")

    # 5. 推理测试
    print(f"\n--- Inference Test ---")
    snap = _make_snap(Asset.BTC, 999, p2=True)
    t1 = time.perf_counter()
    pred = await predict_snapshot(snap, persist=False)
    wall_lat = (time.perf_counter() - t1) * 1000
    if pred:
        print(f"Wall latency:     {wall_lat:.2f}ms")
        print(f"Reported latency: {pred['inference_latency_ms']:.2f}ms")
        print(f"Risk level:       {pred['risk_level']}")
        print(f"Raw prob:         {pred['raw_probability']:.4f}")
        print(f"Calibrated prob:  {pred['calibrated_probability']:.4f}")

    # 6. Model status
    print(f"\n--- Model Status ---")
    st = model_status()
    print(f"Available: {st['available']}")
    print(f"AUC-ROC:   {st['auc_roc']}")
    print(f"Size:      {st['model_size_bytes']} bytes")

    # 7. 验证关键断言
    errors = []
    if result["n_estimators_used"] >= 500:
        errors.append("Early stopping did NOT trigger!")
    if result["auc_roc_raw"] < 0.5:
        errors.append(f"AUC-ROC raw too low: {result['auc_roc_raw']:.4f}")
    if result["training_duration_seconds"] <= 0:
        errors.append("Training duration not recorded!")
    if result["model_size_bytes"] <= 0:
        errors.append("Model size not recorded!")
    if not result["label_distribution"]:
        errors.append("Label distribution not recorded!")
    if not result["feature_importances"]:
        errors.append("Feature importances empty!")
    if pred and pred["inference_latency_ms"] > 100:
        errors.append(f"Inference latency too high: {pred['inference_latency_ms']:.2f}ms")

    if errors:
        print(f"\n{'='*60}")
        print(f"FAILED: {len(errors)} assertion(s)")
        for e in errors:
            print(f"  - {e}")
        exit(1)
    else:
        print(f"\n{'='*60}")
        print(f"ALL CHECKS PASSED")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
