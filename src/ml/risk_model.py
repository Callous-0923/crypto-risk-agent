"""LightGBM risk model — training, calibration, prediction, and evaluation.

基于 LightGBM 官方调参指南 (Parameters-Tuning.rst) 的最佳实践:
  - leaf-wise 树: num_leaves < 2^max_depth, 配合 min_child_samples 防过拟合
  - 早停: early_stopping(stopping_rounds=30) 避免过度训练
  - 正则化: reg_alpha (L1) + reg_lambda (L2) + min_split_gain
  - 小学习率 + 大迭代数: learning_rate=0.03 + n_estimators=500 + early_stopping
  - 不平衡处理: scale_pos_weight 替代 class_weight='balanced'
  - 评估: AUC-ROC + Precision/Recall/F1 + 标签分布
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.core.config import settings
from src.core.logging import get_logger
from src.domain.models import Asset, FeatureSnapshot, Severity
from src.ml.features import FEATURE_COLUMNS, build_feature_dict, build_matrix_rows, rows_to_matrix
from src.ml.labeling import label_snapshot_series
from src.persistence.repositories import (
    list_model_labels,
    save_model_prediction,
)

logger = get_logger(__name__)

_MODEL_CACHE: dict[str, Any] | None = None


def _sort_dt(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def _artifact_path() -> Path:
    return Path(settings.risk_model_artifact_path)


def _default_thresholds() -> dict[str, float]:
    return {
        "p1": settings.risk_model_p1_threshold,
        "p2": settings.risk_model_p2_threshold,
        "p3": settings.risk_model_p3_threshold,
    }


def risk_level_from_probability(probability: float, thresholds: dict[str, float] | None = None) -> str:
    """将校准后概率映射为 P1/P2/P3/none 风险等级。"""
    active_thresholds = thresholds or _default_thresholds()
    if probability >= active_thresholds["p1"]:
        return Severity.P1.value
    if probability >= active_thresholds["p2"]:
        return Severity.P2.value
    if probability >= active_thresholds["p3"]:
        return Severity.P3.value
    return "none"


def _binary_label(label: str) -> int:
    """将 P1/P2 视为正类，其余为负类。"""
    return 1 if label in {"p1", "p2", Severity.P1.value, Severity.P2.value} else 0


def _safe_metric(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """计算二分类 TP/FP/TN/FN + Precision/Recall/F1。"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = _safe_metric(tp, tp + fp)
    recall = _safe_metric(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "predicted_positive": tp + fp,
        "truth_positive": tp + fn,
    }


def _select_threshold(y_true: list[int], probabilities: list[float]) -> tuple[float, dict[str, Any]]:
    """基于 F1 最大化选择最优 P2 阈值。"""
    candidates = sorted({round(v, 4) for v in probabilities if v > 0.0})
    candidates.extend([settings.risk_model_p3_threshold, settings.risk_model_p2_threshold])
    best_threshold = settings.risk_model_p2_threshold
    best_metrics = _binary_metrics(y_true, [1 if p >= best_threshold else 0 for p in probabilities])
    best_score = -1.0
    for threshold in sorted(set(candidates)):
        if threshold <= 0.0 or threshold > 1.0:
            continue
        y_pred = [1 if p >= threshold else 0 for p in probabilities]
        metrics = _binary_metrics(y_true, y_pred)
        if metrics["predicted_positive"] == 0:
            continue
        score = metrics["f1"] or 0.0
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


def _calibrate_probabilities(raw_probs: list[float], y_true: list[int], assets: list[str]):
    """Isotonic 回归校准：全局 + 按资产（样本 >= 30 且标签不单一）。"""
    from sklearn.isotonic import IsotonicRegression

    global_calibrator = IsotonicRegression(out_of_bounds="clip")
    global_calibrator.fit(raw_probs, y_true)
    calibrators: dict[str, Any] = {}
    for asset in sorted(set(assets)):
        idx = [i for i, v in enumerate(assets) if v == asset]
        y_asset = [y_true[i] for i in idx]
        if len(idx) >= 30 and len(set(y_asset)) > 1:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit([raw_probs[i] for i in idx], y_asset)
            calibrators[asset] = calibrator
    return global_calibrator, calibrators


def _apply_calibrator(bundle: dict[str, Any], asset: str, raw_probability: float) -> float:
    calibrator = bundle.get("asset_calibrators", {}).get(asset) or bundle["global_calibrator"]
    return float(calibrator.predict([raw_probability])[0])


async def ensure_llm_judge_labels(
    series_by_asset: dict[Asset, list[FeatureSnapshot]],
    *,
    horizon_seconds: int,
    max_items: int | None,
    use_llm: bool,
    force: bool = False,
) -> dict[str, Any]:
    return await label_snapshot_series(
        series_by_asset, horizon_seconds=horizon_seconds,
        max_items=max_items, use_llm=use_llm, force=force,
    )


async def train_risk_model(
    series_by_asset: dict[Asset, list[FeatureSnapshot]],
    *,
    horizon_seconds: int | None = None,
    max_label_items: int | None = None,
    use_llm_judge: bool = True,
    min_samples: int = 100,
    training_metadata: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """训练 LightGBM 二分类风险模型。

    基于 LightGBM 官方调参指南:
      - leaf-wise 树: num_leaves=31, max_depth=7, min_child_samples=20
      - early_stopping: stopping_rounds=30, 监控验证集 AUC
      - 正则化: reg_alpha=0.1, reg_lambda=0.1, min_split_gain=0.0
      - 小 lr + 大 n: lr=0.03, n_estimators=500
      - 不平衡: scale_pos_weight (正样本权重自动计算)

    Returns:
        训练结果 dict，包含 model_version / metrics / feature_importances /
        training_duration_seconds / model_size_bytes / label_distribution。
    """
    import joblib
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.model_selection import train_test_split

    t_start = time.perf_counter()
    horizon_seconds = horizon_seconds or settings.risk_model_default_horizon_seconds

    labeling_summary = await ensure_llm_judge_labels(
        series_by_asset,
        horizon_seconds=horizon_seconds,
        max_items=max_label_items,
        use_llm=use_llm_judge,
    )
    labels = await list_model_labels(limit=500000)
    label_by_snapshot = {row.snapshot_id: row for row in labels}

    snapshots: list[FeatureSnapshot] = []
    rows: list[dict[str, float]] = []
    y: list[int] = []
    assets: list[str] = []
    raw_labels: list[str] = []
    for asset, series in series_by_asset.items():
        ordered = sorted(series, key=lambda item: _sort_dt(item.window_end))
        feature_rows = build_matrix_rows(ordered)
        for snap, row in zip(ordered, feature_rows):
            label = label_by_snapshot.get(snap.snapshot_id)
            if label is None:
                continue
            snapshots.append(snap)
            rows.append(row)
            y.append(_binary_label(label.label))
            assets.append(asset.value)
            raw_labels.append(label.label)

    if len(rows) < min_samples:
        raise ValueError(f"not enough labeled samples: {len(rows)} < {min_samples}")
    if len(set(y)) < 2:
        raise ValueError("training labels need both positive and negative samples")

    # 时间序列保持顺序的 train/val/test 划分
    n_total = len(rows)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)
    if train_end < 1:
        train_end = 1
    if val_end <= train_end:
        val_end = min(train_end + 1, n_total - 1)

    x = rows_to_matrix(rows)
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]
    assets_test = assets[val_end:]
    test_labels = raw_labels[val_end:]

    # 标签分布统计
    label_distribution = {
        "train": dict(Counter(raw_labels[:train_end])),
        "val": dict(Counter(raw_labels[train_end:val_end])),
        "test": dict(Counter(test_labels)),
    }

    # 正样本权重 (LightGBM 官方推荐 scale_pos_weight 用于二分类)
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    base = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "min_child_samples": 15,
        "min_split_gain": 0.0,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if model_params:
        base.update(model_params)
    model = LGBMClassifier(**base)
    x_train_np = np.array(x_train, dtype=np.float64)
    x_val_np = np.array(x_val, dtype=np.float64)
    model.fit(
        x_train_np, y_train,
        eval_set=[(x_val_np, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=30, verbose=False),
            log_evaluation(period=0),
        ],
        feature_name=FEATURE_COLUMNS,
    )

    n_estimators_used = model.best_iteration_ or model.n_estimators_

    # 测试集评估
    raw_probs = [float(v) for v in model.predict_proba(x_test)[:, 1]]
    global_calibrator, asset_calibrators = _calibrate_probabilities(raw_probs, y_test, assets_test)
    calibrated_probs = [
        float((asset_calibrators.get(a) or global_calibrator).predict([p])[0])
        for a, p in zip(assets_test, raw_probs)
    ]
    try:
        auc_roc = float(roc_auc_score(y_test, raw_probs))
    except ValueError:
        auc_roc = 0.5
    try:
        auc_roc_cal = float(roc_auc_score(y_test, calibrated_probs))
    except ValueError:
        auc_roc_cal = 0.5
    p2_threshold, metrics = _select_threshold(y_test, calibrated_probs)

    thresholds = {
        "p1": settings.risk_model_p1_threshold,
        "p2": p2_threshold,
        "p3": max(0.01, min(settings.risk_model_p3_threshold, p2_threshold * 0.5)),
    }

    # 按 P1/P2/none 的分类报告
    from sklearn.metrics import classification_report
    try:
        per_class_report = classification_report(
            y_test,
            [1 if p >= p2_threshold else 0 for p in calibrated_probs],
            target_names=["none/P3", "P1/P2"],
            output_dict=True,
            zero_division=0,
        )
    except ValueError:
        per_class_report = {"error": "only one class present in test set"}

    feature_importances = [
        {"feature": f, "importance": float(imp)}
        for f, imp in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_),
            key=lambda item: item[1], reverse=True,
        )[:15]
    ]
    version = f"lgbm-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    training_duration = time.perf_counter() - t_start
    bundle = {
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "horizon_seconds": horizon_seconds,
        "feature_columns": FEATURE_COLUMNS,
        "model": model,
        "global_calibrator": global_calibrator,
        "asset_calibrators": asset_calibrators,
        "metrics": metrics,
        "auc_roc": auc_roc_cal,
        "auc_roc_raw": auc_roc,
        "feature_importances": feature_importances,
        "training_samples": len(rows),
        "test_samples": len(y_test),
        "val_samples": len(y_val),
        "n_estimators_used": n_estimators_used,
        "label_distribution": label_distribution,
        "labeling_summary": labeling_summary,
        "thresholds": thresholds,
        "training_metadata": training_metadata or {"source": "runtime_feature_snapshot"},
        "training_duration_seconds": round(training_duration, 2),
    }

    path = _artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    global _MODEL_CACHE
    _MODEL_CACHE = bundle

    model_size = path.stat().st_size
    bundle["model_size_bytes"] = model_size

    logger.info(
        "Model %s trained in %.1fs: samples=%d trees=%d auc=%.4f f1=%.4f size=%.1fMB",
        version, training_duration, len(rows), n_estimators_used,
        auc_roc_cal, metrics.get("f1") or 0.0, model_size / 1e6,
    )

    return {
        "status": "trained",
        "model_version": version,
        "artifact_path": str(path),
        "training_samples": len(rows),
        "test_samples": len(y_test),
        "val_samples": len(y_val),
        "n_estimators_used": n_estimators_used,
        "training_duration_seconds": round(training_duration, 2),
        "model_size_bytes": model_size,
        "auc_roc": auc_roc_cal,
        "auc_roc_raw": auc_roc,
        "metrics": metrics,
        "classification_report": per_class_report,
        "feature_importances": feature_importances,
        "label_distribution": label_distribution,
        "labeling_summary": labeling_summary,
        "thresholds": thresholds,
        "training_metadata": training_metadata or {"source": "runtime_feature_snapshot"},
    }


def load_model_bundle(*, force_reload: bool = False) -> dict[str, Any] | None:
    global _MODEL_CACHE
    if not force_reload and _MODEL_CACHE is not None:
        return _MODEL_CACHE
    path = _artifact_path()
    if not path.exists():
        _MODEL_CACHE = None
        return None
    import joblib
    _MODEL_CACHE = joblib.load(path)
    return _MODEL_CACHE


def model_status(*, force_reload: bool = False) -> dict[str, Any]:
    bundle = load_model_bundle(force_reload=force_reload)
    if bundle is None:
        return {
            "available": False,
            "artifact_path": str(_artifact_path()),
        }
    return {
        "available": True,
        "artifact_path": str(_artifact_path()),
        "model_version": bundle.get("version"),
        "created_at": bundle.get("created_at"),
        "horizon_seconds": bundle.get("horizon_seconds"),
        "feature_columns": bundle.get("feature_columns", []),
        "training_samples": bundle.get("training_samples"),
        "test_samples": bundle.get("test_samples"),
        "val_samples": bundle.get("val_samples"),
        "n_estimators_used": bundle.get("n_estimators_used"),
        "training_duration_seconds": bundle.get("training_duration_seconds"),
        "model_size_bytes": bundle.get("model_size_bytes"),
        "auc_roc": bundle.get("auc_roc"),
        "auc_roc_raw": bundle.get("auc_roc_raw"),
        "metrics": bundle.get("metrics", {}),
        "feature_importances": bundle.get("feature_importances", []),
        "label_distribution": bundle.get("label_distribution", {}),
        "thresholds": bundle.get("thresholds", _default_thresholds()),
        "training_metadata": bundle.get("training_metadata", {}),
    }


async def predict_snapshot(
    snap: FeatureSnapshot,
    *,
    history: list[FeatureSnapshot] | None = None,
    persist: bool = True,
) -> dict[str, Any] | None:
    """对单个 FeatureSnapshot 进行实时推理。

    Returns:
        dict 含 model_version / raw_probability / calibrated_probability /
        risk_level / inference_latency_ms / top_features，或 None。
    """
    t0 = time.perf_counter()
    if not settings.risk_model_enabled:
        return None
    bundle = load_model_bundle()
    if bundle is None:
        return None
    trained_assets = set(bundle.get("training_metadata", {}).get("assets") or [])
    if trained_assets and snap.asset.value not in trained_assets:
        return None
    row = build_feature_dict(snap, history=history or [])
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    matrix = rows_to_matrix([row], columns=feature_columns)
    raw_probability = float(bundle["model"].predict_proba(matrix)[0][1])
    calibrated_probability = _apply_calibrator(bundle, snap.asset.value, raw_probability)
    risk_level = risk_level_from_probability(
        calibrated_probability,
        thresholds=bundle.get("thresholds", _default_thresholds()),
    )
    feature_importances = {
        item["feature"]: item["importance"]
        for item in bundle.get("feature_importances", [])
    }
    top_features = [
        {"feature": f, "value": row.get(f, 0.0), "importance": feature_importances.get(f, 0.0)}
        for f in sorted(FEATURE_COLUMNS, key=lambda k: feature_importances.get(k, 0.0), reverse=True)[:5]
    ]
    latency_ms = (time.perf_counter() - t0) * 1000

    try:
        from src.observability.metrics import (
            ml_inference_duration_seconds, ml_prediction_total,
        )
        ml_inference_duration_seconds.observe(latency_ms / 1000)
        ml_prediction_total.labels(risk_level=risk_level).inc()
    except Exception:
        pass

    result = {
        "model_version": bundle.get("version", "unknown"),
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "risk_level": risk_level,
        "inference_latency_ms": round(latency_ms, 3),
        "top_features": top_features,
        "horizon_seconds": bundle.get("horizon_seconds"),
    }
    if persist:
        await save_model_prediction(
            snapshot_id=snap.snapshot_id,
            asset=snap.asset,
            model_version=result["model_version"],
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            risk_level=risk_level,
            top_features=top_features,
        )
    return result


def prediction_to_rule_hit(snap: FeatureSnapshot, prediction: dict[str, Any] | None):
    """将 ML 预测结果转换为 RuleHit，供 LangGraph 工作流使用。"""
    if not prediction or prediction.get("risk_level") in {None, "none", Severity.P3.value}:
        return None
    from src.domain.models import RuleHit

    severity = Severity(prediction["risk_level"])
    probability = float(prediction.get("calibrated_probability", 0.0) or 0.0)
    return RuleHit(
        rule_id="ML_RISK_PROBABILITY",
        asset=snap.asset,
        severity=severity,
        confidence=probability,
        description=(
            f"{snap.asset.value} LightGBM calibrated risk probability "
            f"{probability:.1%}; model level {severity.value}"
        ),
        evidence={
            "model_version": prediction.get("model_version"),
            "raw_probability": prediction.get("raw_probability"),
            "calibrated_probability": probability,
            "risk_level": prediction.get("risk_level"),
            "horizon_seconds": prediction.get("horizon_seconds"),
            "inference_latency_ms": prediction.get("inference_latency_ms"),
            "top_features": prediction.get("top_features", []),
        },
        dedupe_key=f"ML_RISK_PROBABILITY:{snap.asset.value}:{severity.value}",
    )
