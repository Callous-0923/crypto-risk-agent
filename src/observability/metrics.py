"""
Prometheus 指标定义。

面试官要求的四类指标全部覆盖：
- 队列积压（event_bus_queue_size）
- WS 断连率（ws_reconnect_total）
- LLM 超时率（llm_call_duration_seconds / llm_error_total）
- 规则命中分布（rule_hit_total / rule_evaluate_duration_seconds）

额外补充：
- 数据采集健康（ingest_event_total / ingest_lag_seconds）
- Case/Alert 生命周期（case_created_total / alert_sent_total）
- 人工审核积压（pending_review_gauge）
"""
from __future__ import annotations

from prometheus_client import (
    Counter, Gauge, Histogram, CollectorRegistry, REGISTRY,
)

# ---------------------------------------------------------------------------
# 数据采集层
# ---------------------------------------------------------------------------

ingest_event_total = Counter(
    "ingest_event_total",
    "归一化后写入事件总数",
    ["asset", "source", "event_type"],
)

ingest_lag_seconds = Histogram(
    "ingest_lag_seconds",
    "事件从 event_ts 到 ingest_ts 的延迟（秒）",
    ["source"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)

ws_reconnect_total = Counter(
    "ws_reconnect_total",
    "WebSocket 断连重连次数",
    ["source"],  # okx_spot / okx_swap
)

event_bus_queue_size = Gauge(
    "event_bus_queue_size",
    "事件总线队列当前积压条数",
)

# ---------------------------------------------------------------------------
# 特征计算层
# ---------------------------------------------------------------------------

feature_snapshot_total = Counter(
    "feature_snapshot_total",
    "特征快照生成总数",
    ["asset"],
)

# ---------------------------------------------------------------------------
# 规则引擎层
# ---------------------------------------------------------------------------

rule_evaluate_total = Counter(
    "rule_evaluate_total",
    "规则评估总次数",
    ["asset", "rule_version"],
)

rule_hit_total = Counter(
    "rule_hit_total",
    "规则命中总次数",
    ["asset", "rule_id", "severity", "rule_version"],
)

rule_evaluate_duration_seconds = Histogram(
    "rule_evaluate_duration_seconds",
    "单次规则评估耗时（秒）",
    ["asset"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
)

# ---------------------------------------------------------------------------
# LLM 调用层
# ---------------------------------------------------------------------------

llm_call_total = Counter(
    "llm_call_total",
    "LLM 调用总次数",
    ["status"],  # success / timeout / error
)

llm_call_duration_seconds = Histogram(
    "llm_call_duration_seconds",
    "LLM 调用耗时（秒）",
    buckets=[1, 2, 5, 10, 15, 20, 30],
)

llm_error_total = Counter(
    "llm_error_total",
    "LLM 调用失败总次数",
    ["error_type"],  # timeout / auth / network / other
)

# ---------------------------------------------------------------------------
# Case / Alert 生命周期
# ---------------------------------------------------------------------------

case_created_total = Counter(
    "case_created_total",
    "Case 创建总数",
    ["asset", "severity", "decision"],
)

alert_sent_total = Counter(
    "alert_sent_total",
    "告警发送总数",
    ["asset", "severity", "channel"],
)

pending_review_gauge = Gauge(
    "pending_review_gauge",
    "当前 pending_review 状态的 Case 数量",
)

human_review_total = Counter(
    "human_review_total",
    "人工审核操作总数",
    ["action"],  # approve / reject / escalate
)

# ---------------------------------------------------------------------------
# 数据质量
# ---------------------------------------------------------------------------

data_quality_event_total = Counter(
    "data_quality_event_total",
    "数据质量问题事件数",
    ["asset", "issue_type"],  # stale / conflict
)

# ---------------------------------------------------------------------------
# ML 模型 (LightGBM)
# ---------------------------------------------------------------------------

ml_inference_duration_seconds = Histogram(
    "ml_inference_duration_seconds",
    "LightGBM 模型单次推理耗时（秒）",
    buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5],
)

ml_prediction_total = Counter(
    "ml_prediction_total",
    "ML 模型预测总次数",
    ["risk_level"],  # P1 / P2 / P3 / none
)

ml_training_duration_seconds = Gauge(
    "ml_training_duration_seconds",
    "最近一次 LightGBM 训练耗时（秒）",
)

ml_training_samples = Gauge(
    "ml_training_samples",
    "最近一次训练的样本数",
)

ml_model_auc_roc = Gauge(
    "ml_model_auc_roc",
    "最近一次训练的 AUC-ROC（校准后）",
)
