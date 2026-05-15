"""Optuna 超参数搜索 + LightGBM 最终训练。

搜索空间 (LightGBM 官方调参指南):
  learning_rate:    0.01 ~ 0.10 (log)
  num_leaves:       15 ~ 63
  max_depth:         4 ~ 10
  min_child_samples: 10 ~ 50
  reg_alpha:         1e-5 ~ 0.5 (log)
  reg_lambda:        1e-5 ~ 0.5 (log)
  subsample:         0.60 ~ 1.0
  colsample_bytree:  0.60 ~ 1.0

使用 TPESampler + MedianPruner, n_trials=30。
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import roc_auc_score

from src.core.logging import get_logger
from src.domain.models import Asset

logger = get_logger(__name__)

N_TRIALS = 30

# ---------- 全局变量（协程间共享） ----------

_GLOBAL_SERIES = None
_GLOBAL_LABELS = None
_GLOBAL_HORIZON = 3600


def _sort_dt(v: datetime) -> datetime:
    return v.replace(tzinfo=None) if v.tzinfo else v


def _binary_label(label: str) -> int:
    return 1 if label in {"p1", "p2"} else 0


class _TrialTracker:
    """跨协程的进度计数。"""
    def __init__(self):
        self.total = N_TRIALS
        self.done = 0
        self.best_auc = 0.0
        self.best_params = {}

    def record(self, auc: float, params: dict) -> None:
        self.done += 1
        if auc > self.best_auc:
            self.best_auc = auc
            self.best_params = params
        print(f"  [{self.done}/{self.total}] AUC={auc:.4f}  "
              f"lr={params['learning_rate']:.3f}  "
              f"leaves={params['num_leaves']}  "
              f"depth={params['max_depth']}  "
              f"(best_so_far={self.best_auc:.4f})")


tracker = _TrialTracker()


def _build_objective_fn():
    """返回同步版本的 objective callback (Optuna 需要同步函数)。"""

    def objective(trial: optuna.Trial) -> float:
        series_by_asset = _GLOBAL_SERIES
        labels = _GLOBAL_LABELS
        horizon = _GLOBAL_HORIZON

        label_by_snapshot = {row.snapshot_id: row for row in labels}

        from src.ml.features import FEATURE_COLUMNS, build_matrix_rows, rows_to_matrix

        rows = []
        y_true = []
        assets_list = []
        for asset, series in series_by_asset.items():
            ordered = sorted(series, key=lambda s: _sort_dt(s.window_end))
            feature_rows = build_matrix_rows(ordered)
            for snap, row in zip(ordered, feature_rows):
                label = label_by_snapshot.get(snap.snapshot_id)
                if label is None:
                    continue
                rows.append(row)
                y_true.append(_binary_label(label.label))
                assets_list.append(asset.value)

        if len(rows) < 500 or len(set(y_true)) < 2:
            return 0.0

        n_total = len(rows)
        train_end = int(n_total * 0.70)
        val_end = int(n_total * 0.85)
        if train_end < 1:
            train_end = 1
        if val_end <= train_end:
            val_end = min(train_end + 1, n_total - 1)

        x = rows_to_matrix(rows)
        x_train_np = np.array(x[:train_end], dtype=np.float64)
        y_train = y_true[:train_end]
        x_val_np = np.array(x[train_end:val_end], dtype=np.float64)
        y_val = y_true[train_end:val_end]
        x_test = x[val_end:]
        y_test = y_true[val_end:]

        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 0.5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
        }

        model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=500,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
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

        raw_probs = [float(v) for v in model.predict_proba(x_test)[:, 1]]
        try:
            auc = float(roc_auc_score(y_test, raw_probs))
        except ValueError:
            auc = 0.5

        n_trees = model.best_iteration_ or model.n_estimators_
        trial.set_user_attr("n_trees", n_trees)
        trial.set_user_attr("train_samples", len(rows))

        tracker.record(auc, params)
        return auc

    return objective


async def phase1_search() -> optuna.Study:
    """Phase 1: 加载数据 + 标注 + Optuna 搜索。"""
    global _GLOBAL_SERIES, _GLOBAL_LABELS

    print("=" * 60)
    print("Phase 1: Optuna Hyperparameter Search")
    print(f"Trials: {N_TRIALS} | Pruner: MedianPruner | Sampler: TPESampler")
    print("=" * 60)

    from src.ml.historical_features import load_historical_snapshot_series
    from src.ml.historical_training import label_historical_snapshots

    # Step 1: 加载 2024 全年快照
    t0 = time.perf_counter()
    series_by_asset = await load_historical_snapshot_series(
        assets=[Asset.BTC, Asset.ETH, Asset.SOL],
        start=datetime(2024, 1, 1),
        end=datetime(2025, 1, 1),
        max_snapshots_per_asset=100000,
    )
    total = sum(len(s) for s in series_by_asset.values())
    print(f"Loaded {total:,} snapshots ({time.perf_counter() - t0:.0f}s)")

    # Step 2: 动态弱标注
    t0 = time.perf_counter()
    labeling = await label_historical_snapshots(
        series_by_asset,
        horizon_seconds=3600,
        max_items=100000,
        p2_quantile=0.85,
        p1_quantile=0.95,
    )
    print(f"Labeled {labeling['labeled']:,} samples "
          f"({labeling['label_breakdown']}) "
          f"({time.perf_counter() - t0:.0f}s)")

    from src.persistence.repositories import list_model_labels
    _GLOBAL_SERIES = series_by_asset
    _GLOBAL_LABELS = await list_model_labels(limit=500000)

    # Step 3: Optuna 搜索
    print(f"\n--- Starting {N_TRIALS} trials ---\n")
    study = optuna.create_study(
        study_name="lgbm-risk-2024full",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=3),
    )

    t_search = time.perf_counter()
    objective_fn = _build_objective_fn()
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=False)
    search_time = time.perf_counter() - t_search

    print(f"\nSearch complete ({search_time:.0f}s)\n")
    print(f"  Best trial:       #{study.best_trial.number}")
    print(f"  Best AUC-ROC:     {study.best_value:.4f}")
    print(f"  Best hyperparams:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    top5 = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value, reverse=True,
    )[:5]
    print(f"\n  Top-5 AUCs:")
    for t in top5:
        print(f"    #{t.number:2d}: AUC={t.value:.4f}  "
              f"trees={t.user_attrs.get('n_trees','?')}  "
              f"lr={t.params['learning_rate']:.4f}")

    return study


async def phase2_final_train(study: optuna.Study) -> dict:
    """Phase 2: 用最佳参数完整训练 (含校准/阈值/分类报告)。"""
    print(f"\n{'='*60}")
    print("Phase 2: Final training with best params")
    print(f"{'='*60}")

    from src.ml.risk_model import train_risk_model
    from src.ml.historical_features import load_historical_snapshot_series

    series_by_asset = await load_historical_snapshot_series(
        assets=[Asset.BTC, Asset.ETH, Asset.SOL],
        start=datetime(2024, 1, 1),
        end=datetime(2025, 1, 1),
        max_snapshots_per_asset=100000,
    )

    from src.ml.historical_training import label_historical_snapshots
    _ = await label_historical_snapshots(
        series_by_asset, horizon_seconds=3600,
        max_items=100000, p2_quantile=0.85, p1_quantile=0.95,
    )

    t0 = time.perf_counter()
    result = await train_risk_model(
        series_by_asset,
        horizon_seconds=3600,
        max_label_items=0,
        use_llm_judge=False,
        min_samples=1000,
        model_params=study.best_params,
        training_metadata={"source": "optuna_best_2024"},
    )
    elapsed = time.perf_counter() - t0

    # 报告
    m = result.get("metrics", {})
    dist = result.get("label_distribution", {})

    print(f"\n  Final AUC-ROC:      {result.get('auc_roc', 0):.4f}")
    print(f"  Raw AUC-ROC:        {result.get('auc_roc_raw', 0):.4f}")
    print(f"  Search AUC (Optuna):{study.best_value:.4f}")
    print(f"  Precision:          {m.get('precision', 0):.4f}")
    print(f"  Recall:             {m.get('recall', 0):.4f}")
    print(f"  F1 Score:           {m.get('f1', 0):.4f}")
    print(f"  P2 Threshold:       {m.get('threshold', 0):.4f}")
    print(f"  TP={m.get('tp',0)} FP={m.get('fp',0)} TN={m.get('tn',0)} FN={m.get('fn',0)}")
    print(f"  Trees:              {result['n_estimators_used']}")
    print(f"  Duration:           {result['training_duration_seconds']:.0f}s (wall: {elapsed:.0f}s)")
    print(f"  Size:               {result['model_size_bytes']/1024:.0f} KB")
    print(f"  Train/Val/Test:     {result['training_samples']}/{result['val_samples']}/{result['test_samples']}")

    for split in ["train", "val", "test"]:
        d = dist.get(split, {})
        total = sum(d.values())
        if total == 0:
            continue
        pct_p1 = d.get('p1', 0) / total * 100
        pct_p2 = d.get('p2', 0) / total * 100
        print(f"  {split:5s}: {total:>6,} | p1={d.get('p1',0):>5,} ({pct_p1:.1f}%) p2={d.get('p2',0):>5,} ({pct_p2:.1f}%)")

    fi = result.get("feature_importances", [])[:10]
    if fi:
        print(f"\n  Top-10 Feature Importances:")
        for f in fi:
            print(f"     {f['feature']:<28s} {f['importance']:>8.1f}")

    return result


async def main():
    print(f"Optuna Pipeline — 2024 Full Year Data")
    print(f"Search kwargs: TPE sampler, Median pruner, {N_TRIALS} trials\n")

    study = await phase1_search()
    final = await phase2_final_train(study)

    print(f"\n{'='*60}")
    print(f"Optuna pipeline complete!")
    print(f"  Optuna best AUC: {study.best_value:.4f} (trial #{study.best_trial.number})")
    print(f"  Final  AUC-ROC:  {final.get('auc_roc', 0):.4f}")
    print(f"  F1 Score:        {final.get('metrics', {}).get('f1', 0):.4f}")
    print(f"  Model saved to:  artifacts/risk_model/latest.joblib")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
