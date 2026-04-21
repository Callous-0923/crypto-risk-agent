# 事件驱动型加密货币市场风控 Agent 代码骨架

## 1. 定位与主链路

本项目不是自动交易系统，也不是纯 LLM 判定器，而是一个面向 BTC / ETH / SOL 的公共市场风控 Agent。

核心设计原则：

- 规则负责异常判定，LLM 负责解释与说明。
- Case 是核心线程单位，`thread_id = case_id`。
- 市场、衍生品、新闻三个检测子图采用隔离执行。
- 告警、人工复核、外部推送等副作用必须幂等。
- 所有原始事件、特征快照、规则命中、Case、Alert 和人审动作必须可审计、可回放。

主执行链路：

```text
多源数据接入
  -> 事件标准化
  -> 特征构建
  -> 数据质量检查
  -> 规则判定
  -> 风险评分与去重冷却
  -> LangGraph Case 编排
  -> LLM 解释
  -> 自动告警或人工复核
  -> 持久化、推送、回放与评测
```

## 2. 推荐目录结构

```text
etc-agent/
  pyproject.toml
  README.md
  .env.example
  docker-compose.yml

  app/
    main.py

    core/
      config.py
      logging.py
      time.py
      idempotency.py

    domain/
      models.py
      enums.py
      state.py
      errors.py

    ingestion/
      bus.py
      normalizer.py
      sources/
        binance_spot_ws.py
        binance_futures.py
        coinglass.py
        coingecko.py
        newsapi.py
      workers/
        ingest_market.py
        ingest_derivatives.py
        ingest_news.py

    features/
      builder.py
      market_features.py
      derivatives_features.py
      news_features.py
      data_quality.py

    rules/
      engine.py
      scoring.py
      market_rules.py
      derivatives_rules.py
      news_rules.py
      quality_rules.py
      registry.py

    graph/
      builder.py
      nodes/
        route.py
        market_subgraph.py
        derivatives_subgraph.py
        news_subgraph.py
        evidence_merge.py
        decision.py
        llm_explainer.py
        human_review.py
        alert_sender.py
        case_monitor.py

    persistence/
      db.py
      repositories/
        raw_event_repo.py
        feature_snapshot_repo.py
        risk_case_repo.py
        risk_alert_repo.py
        human_review_repo.py
      migrations/
        versions/

    api/
      routes/
        health.py
        events.py
        cases.py
        alerts.py
        reviews.py
        replay.py
        admin.py
      ws.py
      schemas.py

    notification/
      dispatcher.py
      websocket.py
      webhook.py
      slack.py
      feishu.py

    replay/
      archive.py
      player.py
      backtest.py
      shadow_ab.py

    evaluation/
      metrics.py
      matcher.py
      reports.py

    observability/
      metrics.py
      tracing.py
      audit.py

  tests/
    unit/
      test_normalizer.py
      test_feature_builder.py
      test_rules.py
      test_scoring.py
      test_data_quality.py
    integration/
      test_graph_resume.py
      test_alert_idempotency.py
      test_api_cases.py
      test_replay.py
```

## 3. 模块职责

### 3.1 `core/`

放置全局基础设施能力：

- `config.py`：读取环境变量、资产范围、阈值、供应商配置。
- `logging.py`：统一日志格式，绑定 `trace_id`、`case_id`、`event_id`。
- `time.py`：处理 `event_ts`、`ingest_ts`、窗口边界、时间桶。
- `idempotency.py`：生成 `alert_id`、`case_id:revision` 等幂等键。

### 3.2 `domain/`

定义业务核心对象和状态枚举。

建议包含：

- `RawEvent`
- `FeatureSnapshot`
- `RuleHit`
- `RiskCase`
- `RiskAlert`
- `HumanReviewAction`
- `RiskState`

这一层不应依赖 FastAPI、数据库、外部 API 或 LangGraph 具体实现，尽量保持纯业务模型。

### 3.3 `ingestion/`

负责多源数据接入与事件总线写入。

数据源包括：

- Binance Spot WebSocket：`aggTrade`、`bookTicker`
- Binance Futures WS / REST：`markPrice`、`forceOrder`、`openInterest`
- CoinGlass：OI、爆仓、资金费率、多空比
- CoinGecko：价格校验、市场快照、陈旧性检查
- News API：中文 / 英文新闻检索

`normalizer.py` 是关键模块，负责把异构源统一成 `RawEvent`：

- 统一 `asset`
- 统一 `source`
- 统一 `event_type`
- 生成 `trace_id`
- 生成 `payload_hash`
- 生成 `dedupe_key`
- 同时保留 `event_ts` 与 `ingest_ts`

### 3.4 `features/`

负责窗口特征构建。

市场特征：

- `ret_1m`
- `ret_5m`
- `vol_z_1m`
- `spread_bps`
- `imbalance_1`

衍生品特征：

- `oi_delta_15m`
- `oi_z`
- `liq_5m_usd`
- `liq_share_one_side_5m`
- `funding_z`
- `long_short_ratio`
- `taker_buy_sell_ratio`

新闻与情绪特征：

- `news_cluster_id`
- `distinct_source_count`
- `sentiment_score`
- `negative_news_volume_z`
- `source_reliability`

数据质量特征：

- `source_stale`
- `ingest_lag_ms`
- `cross_source_conflict`
- `missing_source_count`
- `timestamp_reversed`

### 3.5 `rules/`

规则层是主判定层，不应交给 LLM。

建议拆分为四类：

- `market_rules.py`：极速波动、流动性抽离。
- `derivatives_rules.py`：杠杆静默堆积、爆仓级联、资金费率挤压、大户拥挤反转。
- `news_rules.py`：新闻冲击、情绪持续恶化。
- `quality_rules.py`：喂价分歧、聚合价陈旧、源缺失、流断开。

`scoring.py` 负责统一计算：

- `severity`
- `confidence`
- `dedupe_key`
- `cooldown_until`
- `manual_review_if`
- `decision`

### 3.6 `graph/`

这是 LangGraph 编排层。

设计约束：

- `thread_id = case_id`
- 市场、衍生品、新闻子图采用 `per-invocation` 隔离。
- Case 汇总、人审和后续跟踪采用 `per-thread` 累积状态。
- 人工复核使用 `interrupt()` / `resume`。
- 发送告警、Webhook、Slack、飞书等外部副作用必须使用幂等键。

核心节点：

- `route.py`：按事件类型路由到对应子图。
- `market_subgraph.py`：执行市场异常规则。
- `derivatives_subgraph.py`：执行衍生品风险规则。
- `news_subgraph.py`：执行新闻与情绪规则。
- `evidence_merge.py`：合并证据链、规则命中与数据质量状态。
- `decision.py`：输出 `emit`、`suppress` 或 `manual_review`。
- `llm_explainer.py`：生成中文摘要、告警文案和转人工说明。
- `human_review.py`：暂停等待人工审批、驳回或改写优先级。
- `alert_sender.py`：写入 alert 并推送，必须幂等。
- `case_monitor.py`：更新已有 Case、处理 TTL、判断关闭。

### 3.7 `persistence/`

负责数据库连接、Repository 和迁移。

至少需要五张业务表：

- `raw_event`
- `feature_snapshot`
- `risk_case`
- `risk_alert`
- `human_review_action`

建议额外准备：

- `replay_archive`
- `dead_letter_event`
- `rule_version`
- `source_health`

### 3.8 `api/`

FastAPI 作为查询、订阅、恢复执行和管理入口。

建议路由：

- `GET /health`
- `GET /events`
- `GET /cases`
- `GET /cases/{case_id}`
- `POST /cases/{case_id}/resume`
- `GET /alerts`
- `POST /alerts/{alert_id}/ack`
- `GET /reviews`
- `POST /reviews/{case_id}/approve`
- `POST /reviews/{case_id}/reject`
- `POST /replay/jobs`
- `GET /admin/source-health`
- `GET /admin/rules`

WebSocket：

- `GET /ws/alerts`
- `GET /ws/cases`

### 3.9 `notification/`

统一处理告警推送。

推荐先抽象 `NotificationDispatcher`，再接具体渠道：

- WebSocket
- Webhook
- Slack
- Feishu

所有通知必须带：

- `alert_id`
- `case_id`
- `revision`
- `idempotency_key`

避免 LangGraph durable replay 后重复发送告警。

### 3.10 `replay/`

负责回放、复盘和 Shadow A/B。

核心能力：

- 按 `event_ts` 顺序回放历史事件。
- 重建特征窗口。
- 重跑规则引擎。
- 对比不同规则版本。
- 支持 baseline / treatment 双路评测。

### 3.11 `evaluation/`

负责离线评测指标。

主要指标：

- Precision
- Recall
- F1
- 告警延迟 p95
- 重复率
- 人工通过率
- 人工负载率
- 恢复成功率

匹配逻辑需要按资产、风险域、方向和时间窗口进行约束。

### 3.12 `observability/`

负责工程可观测性。

必须覆盖：

- ingest lag
- source stale
- reconnect count
- rule hit count
- alert emit count
- dedupe hit count
- manual review count
- graph retry count
- DLQ count
- resume success count

所有日志与指标都应绑定：

- `trace_id`
- `event_id`
- `case_id`
- `alert_id`
- `rule_id`
- `rule_version`

## 4. 核心数据对象

### 4.1 RawEvent

```python
class RawEvent:
    event_id: str
    source: str
    asset: str
    event_type: str
    event_ts: datetime
    ingest_ts: datetime
    trace_id: str
    dedupe_key: str
    payload_hash: str
    payload: dict
```

### 4.2 FeatureSnapshot

```python
class FeatureSnapshot:
    snapshot_id: str
    asset: str
    window: str
    event_ids: list[str]
    features: dict
    quality_flags: dict
```

### 4.3 RuleHit

```python
class RuleHit:
    rule_id: str
    rule_version: str
    asset: str
    severity: str
    confidence: float
    evidence: list[dict]
    manual_review_if: dict
    dedupe_key: str
    cooldown_until: datetime | None
```

### 4.4 RiskCase

```python
class RiskCase:
    case_id: str
    thread_id: str
    asset: str
    state: str
    severity: str | None
    confidence: float
    rule_hits: list[RuleHit]
    evidence: list[dict]
    decision: str
```

### 4.5 RiskAlert

```python
class RiskAlert:
    alert_id: str
    case_id: str
    revision: int
    severity: str
    title: str
    body_zh: str
    evidence: list[dict]
    idempotency_key: str
```

### 4.6 LangGraph RiskState

```python
from typing import Literal, TypedDict


class RiskState(TypedDict):
    case_id: str
    asset: Literal["BTC", "ETH", "SOL"]
    event_id: str
    trigger: Literal["market", "derivatives", "news", "manual_resume"]
    feature_snapshot: dict
    evidence: list[dict]
    rule_hits: list[str]
    severity: str | None
    confidence: float
    decision: Literal["emit", "suppress", "manual_review"] | None
    summary_zh: str | None
    retry_count: int
```

## 5. 建议数据库实体

### 5.1 `raw_event`

职责：

- 保存所有标准化后的原始事件。
- 支持审计、回放、去重和问题追踪。

关键字段：

- `event_id`
- `source`
- `asset`
- `event_type`
- `event_ts`
- `ingest_ts`
- `trace_id`
- `dedupe_key`
- `payload`
- `payload_hash`
- `feature_partial`

### 5.2 `feature_snapshot`

职责：

- 保存窗口特征快照。
- 将规则判定与原始事件解耦。

关键字段：

- `snapshot_id`
- `asset`
- `window`
- `event_ids`
- `features`
- `quality_flags`
- `created_at`

### 5.3 `risk_case`

职责：

- 作为系统核心业务对象。
- 对应 LangGraph 的长期线程。

关键字段：

- `case_id`
- `thread_id`
- `asset`
- `state`
- `severity`
- `confidence`
- `first_event_ts`
- `last_event_ts`
- `cooldown_until`
- `rule_hits`
- `evidence`
- `rule_version`
- `model_version`

### 5.4 `risk_alert`

职责：

- 保存已经发出或准备发出的告警。
- 支持多渠道推送和幂等重试。

关键字段：

- `alert_id`
- `case_id`
- `revision`
- `channel`
- `severity`
- `title`
- `body_zh`
- `evidence`
- `emitted_at`
- `idempotency_key`
- `acknowledged`

### 5.5 `human_review_action`

职责：

- 保存人工复核动作。
- 支持审计和后续评估人工通过率。

关键字段：

- `review_id`
- `case_id`
- `reviewer`
- `action`
- `comment`
- `action_ts`

## 6. MVP 落地范围

第一版不建议一次性实现全部能力。

建议 MVP 范围：

- Binance Spot / Futures 接入。
- CoinGecko 价格校验。
- CoinGlass 分钟级衍生品指标。
- `raw_event`、`feature_snapshot`、`risk_case`、`risk_alert`、`human_review_action`。
- 市场规则、衍生品规则、数据质量规则。
- LangGraph Case 编排。
- FastAPI 查询与 `resume` 接口。
- WebSocket 或 Webhook 告警。
- 基础回放能力。

可暂缓：

- News API 实时新闻层。
- 复杂情绪模型。
- 多区域部署。
- 复杂权限系统。
- 机构级审批流。
- 全量前端控制台。

## 7. 推荐实现顺序

### 阶段一：事件与存储打底

目标是把事件稳定落库。

要做：

- 定义 `domain/` 模型。
- 建立数据库连接和 Repository。
- 实现 `RawEvent` 标准化。
- 接入 Redis Streams 或等价事件总线。
- 写入 `raw_event`。

验收标准：

- 任意来源事件都有 `trace_id`、`event_ts`、`ingest_ts`、`payload_hash`。
- 重复事件可以通过 `dedupe_key` 识别。
- 原始事件可查询、可审计。

### 阶段二：特征与规则

目标是跑通基础风控判定。

要做：

- 实现窗口聚合。
- 实现市场规则。
- 实现衍生品规则。
- 实现数据质量规则。
- 实现 `RiskScoring`。

验收标准：

- 能对 BTC / ETH / SOL 生成稳定的 `FeatureSnapshot`。
- 能输出 `RuleHit`。
- 能区分 `emit`、`suppress`、`manual_review`。

### 阶段三：LangGraph Case 编排

目标是把规则命中转为可恢复状态机。

要做：

- 建立 `RiskState`。
- 实现 market / derivatives / news 子图接口。
- 实现 evidence merge。
- 实现 decision node。
- 实现 human review node。
- 实现 alert sender node。

验收标准：

- `thread_id = case_id`。
- 人审可以 interrupt 并 resume。
- 重试不会重复发送告警。
- Case 状态可以恢复。

### 阶段四：服务接口与推送

目标是让系统可操作。

要做：

- FastAPI 查询接口。
- Case 详情接口。
- Alert 确认接口。
- 人审恢复接口。
- WebSocket 或 Webhook 推送。

验收标准：

- 可以查询 Case 证据链。
- 可以人工批准或驳回。
- 告警推送带幂等键。

### 阶段五：回放与评测

目标是让规则可校准、可复盘。

要做：

- 原始事件归档。
- 按 `event_ts` 顺序回放。
- 规则版本对比。
- Precision / Recall / F1 计算。
- Shadow A/B 框架。

验收标准：

- 可以重放历史事件。
- 可以复现某次告警。
- 可以比较两个规则版本的误报和漏报。

## 8. 关键工程约束

### 8.1 LLM 不做主判定

LLM 只负责：

- 新闻聚类辅助。
- 证据摘要。
- 中文告警文案。
- 转人工说明。

LLM 不负责：

- 判断是否发生风险。
- 决定 P0 / P1 / P2。
- 绕过规则直接发告警。

### 8.2 Case 是核心对象

系统不应围绕 Alert 或 Prompt 组织，而应围绕 Case 组织。

原因：

- 去重需要 Case。
- 冷却需要 Case。
- 后续事件合并需要 Case。
- 人审恢复需要 Case。
- 审计和回放需要 Case。

### 8.3 数据质量本身是风险域

数据质量风险不只是日志问题，而应该进入规则系统。

典型数据质量风险：

- BN 与 CGK 价格分歧。
- 聚合价陈旧。
- 源连续缺失。
- WebSocket 断流。
- 时间戳倒挂。
- 本地盘口序列断点。

数据质量风险可以：

- 生成内部 P2 告警。
- 降低其他风险置信度。
- 阻断高优先级自动告警。
- 转人工复核。

### 8.4 所有副作用必须幂等

以下动作必须带幂等键：

- 写 `risk_alert`
- 推送 WebSocket
- 调用 Webhook
- 发送 Slack
- 发送 Feishu
- 邮件或短信通知

推荐幂等键：

```text
alert:{alert_id}
case:{case_id}:revision:{revision}
channel:{channel}:case:{case_id}:revision:{revision}
```

### 8.5 从第一天开始归档

如果不归档原始事件和规则中间结果，后续无法做可靠回放。

必须归档：

- raw event
- feature snapshot
- rule hit
- risk case
- risk alert
- human review action

## 9. 最小可运行闭环

最小可运行闭环可以压缩为：

```text
BN / CGK / CGL
  -> RawEvent
  -> FeatureSnapshot
  -> RuleHit
  -> RiskCase
  -> LangGraph decision
  -> RiskAlert 或 HumanReview
  -> FastAPI 查询 / resume
  -> WebSocket 或 Webhook 推送
```

只要这个闭环跑通，后续再接 News API、前端控制台、Shadow A/B 和完整评测体系。

## 10. 不建议的实现方式

不建议：

- 一上来做纯 LLM 风险判断。
- 把多个 Agent 做成角色扮演式聊天机器人。
- 没有 Case，只生成一次性 Alert。
- 没有 `event_ts` / `ingest_ts` 双时间戳。
- 不保存 `payload_hash` 和 `trace_id`。
- 不做幂等直接发送 Webhook。
- 没有回放归档就调规则阈值。
- 新闻源还没付费实时档时，把新闻冲击作为强依赖。

## 11. 总结

这个 Agent 的代码骨架应围绕七层架构展开：

```text
数据接入层
事件与特征层
规则判定层
Agent 编排层
Case / Alert 持久化层
服务接口与推送层
运维观测与评测层
```

工程实现的核心不是堆模型，而是把以下能力做扎实：

- 多源事件标准化。
- 窗口特征稳定生成。
- 规则可版本化。
- Case 可恢复。
- 告警可解释。
- 人审可闭环。
- 副作用可幂等。
- 历史可回放。
- 规则可评测。
