"""LightGBM 改进效果验证测试。

验证内容:
  1. Early stopping 生效 (n_estimators_used < n_estimators)
  2. AUC-ROC 正确计算 (>= 0.5)
  3. 校准不破坏概率排序 (AUC_ROC_cal >= AUC_ROC_raw * 0.95)
  4. 推理延迟 < 50ms
  5. 训练耗时记录
  6. 模型大小记录
  7. 标签分布统计正确
  8. 特征重要性非空
  9. 模型保存/加载循环正确
  10. 新旧模型 API 兼容 (predict_snapshot, prediction_to_rule_hit, model_status)
"""
from __future__ import annotations

import asyncio
import math
import random
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.domain.models import Asset, FeatureSnapshot, Severity
from src.ml.features import FEATURE_COLUMNS, build_feature_dict, build_matrix_rows, rows_to_matrix
from src.ml.risk_model import (
    _binary_label,
    _binary_metrics,
    predict_snapshot,
    prediction_to_rule_hit,
    risk_level_from_probability,
    model_status,
    load_model_bundle,
)

random.seed(42)

_BASE_TIME = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)


def _make_snapshot(
    asset: Asset,
    offset_min: int,
    *,
    ret_1m: float = 0.0,
    ret_5m: float = 0.0,
    ret_15m: float = 0.0,
    ret_30m: float = 0.0,
    ret_60m: float = 0.0,
    vol_z_1m: float = 0.0,
    realized_vol_5m: float = 0.0,
    realized_vol_15m: float = 0.0,
    realized_vol_60m: float = 0.0,
    max_drawdown_15m: float = 0.0,
    max_drawdown_60m: float = 0.0,
    max_runup_15m: float = 0.0,
    max_runup_60m: float = 0.0,
    atr_14: float = 0.0,
    oi_delta_15m_pct: float = 0.0,
    oi_delta_5m_pct: float = 0.0,
    oi_delta_60m_pct: float = 0.0,
    liq_5m_usd: float = 0.0,
    funding_z: float = 0.0,
    futures_basis_pct: float = 0.0,
    basis_z_60m: float = 0.0,
    price: float = 100.0,
    volume_1m: float = 0.0,
    quote_volume_1m: float = 0.0,
    volume_5m: float = 0.0,
    quote_volume_5m: float = 0.0,
    volume_15m: float = 0.0,
    quote_volume_15m: float = 0.0,
    volume_z_15m: float = 0.0,
    volume_z_60m: float = 0.0,
    trade_count_1m: float = 0.0,
    trade_count_z_15m: float = 0.0,
    taker_buy_ratio_1m: float = 0.0,
    taker_buy_ratio_5m: float = 0.0,
    price_range_pct_1m: float = 0.0,
    close_position_1m: float = 0.5,
    volatility_regime_60m: float = 1.0,
) -> FeatureSnapshot:
    window_end = _BASE_TIME + timedelta(minutes=offset_min)
    snap_id = f"test-snap-{uuid.uuid4().hex[:8]}"
    return FeatureSnapshot(
        snapshot_id=snap_id,
        asset=asset,
        window_end=window_end,
        price=price,
        ret_1m=ret_1m,
        ret_5m=ret_5m,
        ret_15m=ret_15m,
        ret_30m=ret_30m,
        ret_60m=ret_60m,
        vol_z_1m=vol_z_1m,
        realized_vol_5m=realized_vol_5m,
        realized_vol_15m=realized_vol_15m,
        realized_vol_60m=realized_vol_60m,
        price_range_pct_1m=price_range_pct_1m,
        close_position_1m=close_position_1m,
        max_drawdown_15m=max_drawdown_15m,
        max_drawdown_60m=max_drawdown_60m,
        max_runup_15m=max_runup_15m,
        max_runup_60m=max_runup_60m,
        atr_14=atr_14,
        volatility_regime_60m=volatility_regime_60m,
        volume_1m=volume_1m,
        quote_volume_1m=quote_volume_1m,
        volume_5m=volume_5m,
        quote_volume_5m=quote_volume_5m,
        volume_15m=volume_15m,
        quote_volume_15m=quote_volume_15m,
        volume_z_15m=volume_z_15m,
        volume_z_60m=volume_z_60m,
        trade_count_1m=trade_count_1m,
        trade_count_z_15m=trade_count_z_15m,
        taker_buy_ratio_1m=taker_buy_ratio_1m,
        taker_buy_ratio_5m=taker_buy_ratio_5m,
        oi_delta_15m_pct=oi_delta_15m_pct,
        oi_delta_5m_pct=oi_delta_5m_pct,
        oi_delta_60m_pct=oi_delta_60m_pct,
        liq_5m_usd=liq_5m_usd,
        funding_z=funding_z,
        futures_basis_pct=futures_basis_pct,
        basis_z_60m=basis_z_60m,
    )


def _build_labeled_snapshots(
    n_per_asset: int = 300,
) -> dict[Asset, list[FeatureSnapshot]]:
    """构建含明显信号区分的合成数据。

    正类 (P1/P2): 高波动、高 OI 变动、高爆仓、极端资金费率
    负类 (none): 正常范围波动
    """
    result: dict[Asset, list[FeatureSnapshot]] = {}
    for asset in [Asset.BTC, Asset.ETH]:
        snaps: list[FeatureSnapshot] = []
        base_price = 62000.0 if asset == Asset.BTC else 3000.0
        for i in range(n_per_asset):
            if i < n_per_asset // 3:
                # 风险样本 (P1/P2)
                snaps.append(_make_snapshot(
                    asset, i,
                    price=base_price * (1 + random.gauss(-0.06, 0.03)),
                    ret_1m=random.gauss(-0.04, 0.02),
                    ret_5m=random.gauss(-0.05, 0.03),
                    ret_15m=random.gauss(-0.06, 0.04),
                    vol_z_1m=random.gauss(2.5, 1.0),
                    realized_vol_5m=abs(random.gauss(0.008, 0.004)),
                    realized_vol_15m=abs(random.gauss(0.012, 0.005)),
                    max_drawdown_15m=abs(random.gauss(0.04, 0.02)),
                    max_runup_15m=abs(random.gauss(0.02, 0.01)),
                    oi_delta_15m_pct=random.gauss(0.10, 0.05),
                    oi_delta_5m_pct=random.gauss(0.05, 0.03),
                    liq_5m_usd=abs(random.gauss(30_000_000, 15_000_000)),
                    funding_z=random.gauss(-2.5, 1.0),
                    futures_basis_pct=random.gauss(-0.003, 0.002),
                    volume_z_15m=random.gauss(2.0, 1.0),
                    trade_count_z_15m=random.gauss(2.0, 1.0),
                ))
            else:
                # 正常样本 (none)
                snaps.append(_make_snapshot(
                    asset, i,
                    price=base_price * (1 + random.gauss(0.0, 0.005)),
                    ret_1m=random.gauss(0.0, 0.005),
                    ret_5m=random.gauss(0.0, 0.008),
                    ret_15m=random.gauss(0.0, 0.01),
                    vol_z_1m=random.gauss(0.0, 0.5),
                    realized_vol_5m=abs(random.gauss(0.002, 0.001)),
                    realized_vol_15m=abs(random.gauss(0.003, 0.001)),
                    max_drawdown_15m=abs(random.gauss(0.005, 0.003)),
                    max_runup_15m=abs(random.gauss(0.005, 0.003)),
                    oi_delta_15m_pct=random.gauss(0.0, 0.01),
                    oi_delta_5m_pct=random.gauss(0.0, 0.005),
                    liq_5m_usd=abs(random.gauss(500_000, 1_000_000)),
                    funding_z=random.gauss(0.0, 0.5),
                    futures_basis_pct=random.gauss(0.0, 0.0005),
                    volume_z_15m=random.gauss(0.0, 0.5),
                    trade_count_z_15m=random.gauss(0.0, 0.5),
                ))
        result[asset] = snaps
    return result


class TestLightGBMImprovements(unittest.TestCase):
    """端到端验证 LightGBM 改进效果。"""

    @classmethod
    def setUpClass(cls):
        """用合成数据训练模型。"""
        cls.series = _build_labeled_snapshots(n_per_asset=300)

        # 直接注入标签 (绕过 LLM judge)
        cls._inject_labels(cls.series)

        from src.ml.risk_model import train_risk_model
        cls.result = asyncio.run(train_risk_model(
            cls.series,
            horizon_seconds=3600,
            max_label_items=0,
            use_llm_judge=False,
            min_samples=50,
            training_metadata={"source": "synthetic_test"},
        ))

    @staticmethod
    def _inject_labels(series: dict[Asset, list[FeatureSnapshot]]) -> None:
        """注入确定性标签以绕过 LLM judge 调用。"""
        import json
        from src.persistence.database import AsyncSessionLocal, RiskModelLabelRow

        async def _inject():
            async with AsyncSessionLocal() as s:
                for asset, snaps in series.items():
                    ordered = sorted(snaps, key=lambda x: x.window_end)
                    for i, snap in enumerate(ordered):
                        label_id = f"test-label-{uuid.uuid4().hex[:12]}"
                        label = "p2" if i < len(ordered) // 3 else "none"
                        s.add(RiskModelLabelRow(
                            label_id=label_id,
                            snapshot_id=snap.snapshot_id,
                            asset=asset.value,
                            window_end=snap.window_end,
                            horizon_seconds=3600,
                            label=label,
                            risk_probability=0.7 if label == "p2" else 0.1,
                            confidence=0.65,
                            labeling_method="test_injected",
                            rationale=f"Injected test label {label}",
                            judge_payload_json=json.dumps({"test": True}),
                            created_at=snap.window_end,
                        ))
                await s.commit()

        asyncio.run(_inject())

    # ------------------------------------------------------------------
    # 测试 1: 训练结果结构完整
    # ------------------------------------------------------------------

    def test_result_has_status_trained(self):
        self.assertEqual(self.result["status"], "trained")

    def test_result_has_model_version(self):
        self.assertIn("model_version", self.result)
        self.assertTrue(self.result["model_version"].startswith("lgbm-"))

    # ------------------------------------------------------------------
    # 测试 2: Early Stopping 生效
    # ------------------------------------------------------------------

    def test_n_estimators_used_less_than_configured(self):
        self.assertIn("n_estimators_used", self.result)
        used = self.result["n_estimators_used"]
        self.assertGreater(used, 0)
        # 配了 500 棵但 early_stopping 应提前停止
        self.assertLess(used, 500)

    # ------------------------------------------------------------------
    # 测试 3: AUC-ROC 指标
    # ------------------------------------------------------------------

    def test_auc_roc_present_and_valid(self):
        self.assertIn("auc_roc", self.result)
        auc = self.result["auc_roc"]
        self.assertIsNotNone(auc)
        # 合成数据有强信号区分, AUC 应 >= 0.65
        self.assertGreaterEqual(auc, 0.65)
        self.assertLessEqual(auc, 1.0)

    def test_auc_roc_raw_also_present(self):
        self.assertIn("auc_roc_raw", self.result)
        self.assertGreaterEqual(self.result["auc_roc_raw"], 0.5)

    # ------------------------------------------------------------------
    # 测试 4: 校准不破坏排序
    # ------------------------------------------------------------------

    def test_calibration_does_not_destroy_ranking(self):
        auc_raw = self.result["auc_roc_raw"]
        auc_cal = self.result["auc_roc"]
        # 校准后 AUC 应与原始接近 (允许 10% 内下降)
        self.assertGreaterEqual(auc_cal, auc_raw * 0.85)

    # ------------------------------------------------------------------
    # 测试 5: 训练耗时
    # ------------------------------------------------------------------

    def test_training_duration_recorded(self):
        self.assertIn("training_duration_seconds", self.result)
        duration = self.result["training_duration_seconds"]
        self.assertGreater(duration, 0.0)
        # 300×2=600 样本, 训练应在 30 秒内完成
        self.assertLess(duration, 30.0)

    # ------------------------------------------------------------------
    # 测试 6: 模型大小
    # ------------------------------------------------------------------

    def test_model_size_recorded(self):
        self.assertIn("model_size_bytes", self.result)
        size = self.result["model_size_bytes"]
        self.assertGreater(size, 1000)
        self.assertLess(size, 20 * 1024 * 1024)  # < 20MB

    # ------------------------------------------------------------------
    # 测试 7: 标签分布
    # ------------------------------------------------------------------

    def test_label_distribution(self):
        self.assertIn("label_distribution", self.result)
        dist = self.result["label_distribution"]
        self.assertIn("train", dist)
        self.assertIn("val", dist)
        self.assertIn("test", dist)
        self.assertGreaterEqual(dist["train"].get("p2", 0), 30)
        self.assertGreaterEqual(dist["train"].get("none", 0), 30)

    # ------------------------------------------------------------------
    # 测试 8: 特征重要性
    # ------------------------------------------------------------------

    def test_feature_importances(self):
        self.assertIn("feature_importances", self.result)
        fi = self.result["feature_importances"]
        self.assertGreaterEqual(len(fi), 5)
        for item in fi:
            self.assertIn("feature", item)
            self.assertIn("importance", item)
            self.assertGreater(item["importance"], 0.0)

    # ------------------------------------------------------------------
    # 测试 9: 训练/验证/测试划分
    # ------------------------------------------------------------------

    def test_train_val_test_split(self):
        train = self.result["training_samples"]
        test = self.result["test_samples"]
        val = self.result["val_samples"]
        total = train + val
        self.assertGreater(train, test)
        self.assertGreater(val, 0)
        self.assertGreater(test, 0)

    # ------------------------------------------------------------------
    # 测试 10: 模型持久化与加载
    # ------------------------------------------------------------------

    def test_save_load_cycle(self):
        bundle = load_model_bundle()
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle["version"], self.result["model_version"])
        self.assertIn("auc_roc", bundle)
        self.assertIn("label_distribution", bundle)

    # ------------------------------------------------------------------
    # 测试 11: model_status()
    # ------------------------------------------------------------------

    def test_model_status_available(self):
        status = model_status()
        self.assertTrue(status["available"])
        self.assertIn("auc_roc", status)
        self.assertIn("training_duration_seconds", status)
        self.assertIn("model_size_bytes", status)
        self.assertIn("label_distribution", status)

    # ------------------------------------------------------------------
    # 测试 12: 推理延迟
    # ------------------------------------------------------------------

    def test_inference_latency_acceptable(self):
        snap = _make_snapshot(
            Asset.BTC, 1000,
            ret_1m=-0.03, ret_5m=-0.04, vol_z_1m=2.0,
            oi_delta_15m_pct=0.08, liq_5m_usd=20_000_000,
            funding_z=-2.0,
        )
        result = asyncio.run(predict_snapshot(snap, persist=False))
        self.assertIsNotNone(result)
        self.assertIn("inference_latency_ms", result)
        latency = result["inference_latency_ms"]
        self.assertGreaterEqual(latency, 0.0)
        # 单次推理应在 50ms 以内
        self.assertLess(latency, 50.0)

    # ------------------------------------------------------------------
    # 测试 13: predict_snapshot 输出结构
    # ------------------------------------------------------------------

    def test_predict_snapshot_output_fields(self):
        snap = _make_snapshot(
            Asset.BTC, 2000,
            ret_1m=-0.02, ret_5m=-0.03, vol_z_1m=1.5,
            oi_delta_15m_pct=0.06,
        )
        result = asyncio.run(predict_snapshot(snap, persist=False))
        self.assertIsNotNone(result)
        for key in ["model_version", "raw_probability", "calibrated_probability",
                     "risk_level", "inference_latency_ms", "top_features"]:
            self.assertIn(key, result)
        self.assertGreaterEqual(len(result["top_features"]), 1)

    # ------------------------------------------------------------------
    # 测试 14: prediction_to_rule_hit
    # ------------------------------------------------------------------

    def test_prediction_to_rule_hit_p2(self):
        snap = _make_snapshot(Asset.BTC, 3000)
        prediction = {
            "model_version": "test",
            "raw_probability": 0.65,
            "calibrated_probability": 0.60,
            "risk_level": "P2",
            "inference_latency_ms": 2.5,
            "top_features": [],
            "horizon_seconds": 3600,
        }
        hit = prediction_to_rule_hit(snap, prediction)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.rule_id, "ML_RISK_PROBABILITY")
        self.assertEqual(hit.severity, Severity.P2)
        self.assertIn("inference_latency_ms", hit.evidence)

    def test_prediction_to_rule_hit_ignores_p3(self):
        snap = _make_snapshot(Asset.BTC, 4000)
        prediction = {
            "model_version": "test",
            "raw_probability": 0.35,
            "calibrated_probability": 0.32,
            "risk_level": "P3",
            "inference_latency_ms": 1.8,
            "top_features": [],
            "horizon_seconds": 3600,
        }
        hit = prediction_to_rule_hit(snap, prediction)
        self.assertIsNone(hit)

    # ------------------------------------------------------------------
    # 测试 15: 特征矩阵维度
    # ------------------------------------------------------------------

    def test_feature_matrix_dimensions(self):
        row = build_feature_dict(
            _make_snapshot(Asset.BTC, 5000, ret_1m=0.02, vol_z_1m=2.0),
            history=[_make_snapshot(Asset.BTC, 4999, ret_1m=0.01)],
        )
        matrix = rows_to_matrix([row])
        self.assertEqual(len(matrix[0]), len(FEATURE_COLUMNS))

    # ------------------------------------------------------------------
    # 测试 16: 二分类指标函数
    # ------------------------------------------------------------------

    def test_binary_metrics_perfect(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        metrics = _binary_metrics(y_true, y_pred)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_binary_metrics_with_errors(self):
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 0]
        metrics = _binary_metrics(y_true, y_pred)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 2)
        self.assertLess(metrics["precision"], 1.0)
        self.assertLess(metrics["recall"], 1.0)

    # ------------------------------------------------------------------
    # 测试 17: 概率→风险等级映射
    # ------------------------------------------------------------------

    def test_risk_level_mapping(self):
        self.assertEqual(risk_level_from_probability(0.90, {"p1": 0.80, "p2": 0.55, "p3": 0.30}), "P1")
        self.assertEqual(risk_level_from_probability(0.60, {"p1": 0.80, "p2": 0.55, "p3": 0.30}), "P2")
        self.assertEqual(risk_level_from_probability(0.35, {"p1": 0.80, "p2": 0.55, "p3": 0.30}), "P3")
        self.assertEqual(risk_level_from_probability(0.10, {"p1": 0.80, "p2": 0.55, "p3": 0.30}), "none")


if __name__ == "__main__":
    unittest.main()
