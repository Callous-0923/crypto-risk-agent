# ETC Risk Agent

加密货币实时风控与提前预警系统。项目以 BTC、ETH、SOL 为监控对象，接入交易所实时行情，构建滚动特征快照，通过规则引擎、LangGraph 工作流、人工审核、评测总结和 LightGBM 风险预测模型完成风险发现、解释、审核和告警闭环。

当前版本已经包含：

- 实时行情接入：Binance / OKX 行情源，支持 WebSocket 与 REST 轮询。
- 风控规则引擎：P1/P2/P3 分级，提前预警，多信号确认，持续性确认，去重与疲劳抑制。
- Agent 工作流：LangGraph 编排规则、专家分析、总结、人工审核、告警发送。
- 人工审核台：待审核案例分页、approve/reject、审核备注、恢复执行。
- 评测总结：真实运行指标、离线弱标签评测、提前预警八指标、吞吐、延迟、去重、LLM 成本、baseline 对比。
- LightGBM 风险模型：实时预测、LLM-as-judge 弱标注、历史行情训练、模型状态展示。
- 历史行情训练链路：Binance public data 2024/2025 1m K 线下载、落库、特征工程、动态弱标签、hard negative 采样。

> 当前默认实现是研发/演示版本，不是可直接用于资金安全决策的生产系统。告警结果需要人工审核或外部风控系统兜底。

---

## 快速启动

### 1. 环境要求

- Docker Desktop
- Python 3.10+
- Node 20，使用 Docker 运行时不需要本机 Node
- 可访问 Binance / OKX / 火山 Ark API 的网络环境

### 2. 配置环境变量

复制 `.env.example`：

```powershell
Copy-Item .env.example .env
```

至少配置：

```env
ARK_API_KEY=your_volcano_engine_api_key_here
LLM_MODEL=doubao-seed-2-0-mini-260215
DATABASE_URL=sqlite+aiosqlite:///./etc_agent.db
WEBHOOK_URL=
LOG_LEVEL=INFO
```

如果容器内需要代理访问交易所或 LLM 服务，配置：

```env
CONTAINER_HTTP_PROXY=http://host.docker.internal:7890
CONTAINER_HTTPS_PROXY=http://host.docker.internal:7890
CONTAINER_ALL_PROXY=
CONTAINER_NO_PROXY=localhost,127.0.0.1,backend
WS_PROXY=socks5://host.docker.internal:7890
```

### 3. 启动前后端

```powershell
docker compose up -d --build
```

访问：

- 前端控制台：http://localhost:5173
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/api/v1/health

停止：

```powershell
docker compose down
```

### 4. 本地开发运行

```powershell
pip install -e .
python main.py
```

前端本地开发：

```powershell
cd frontend
npm install
npm run dev
```

---

## 项目结构

```text
etc-agent/
├── main.py                         # 后端入口
├── docker-compose.yml              # 前后端开发编排
├── Dockerfile.backend.dev          # 后端镜像，包含 libgomp1 以支持 LightGBM
├── pyproject.toml                  # Python 依赖
├── .env.example                    # 环境变量样例
├── artifacts/
│   └── risk_model/
│       └── latest.joblib           # 当前 LightGBM 模型产物
├── frontend/
│   ├── Dockerfile.dev
│   ├── package.json
│   └── src/
│       ├── App.jsx                 # 实时、测试、评测三种工作区
│       ├── main.jsx
│       └── styles.css
├── grafana/
│   └── dashboard.json
├── src/
│   ├── api/
│   │   ├── app.py                  # FastAPI app、启动/停止 ingestion
│   │   └── routes.py               # REST API + WebSocket
│   ├── core/
│   │   ├── config.py               # pydantic-settings 配置
│   │   ├── logging.py
│   │   └── proxy.py                # OpenAI-compatible client/proxy
│   ├── domain/
│   │   └── models.py               # Pydantic 领域模型
│   ├── ingestion/
│   │   ├── normalizer.py
│   │   ├── validator.py
│   │   └── sources/
│   │       ├── binance_ws.py
│   │       ├── coinglass_poll.py
│   │       ├── okx_rest.py
│   │       └── okx_ws.py
│   ├── features/
│   │   └── builder.py              # 实时特征快照构建 + LightGBM 实时预测入口
│   ├── rules/
│   │   ├── config.py               # 规则阈值、版本化发布
│   │   └── engine.py               # P1/P2/P3 与提前预警规则
│   ├── graph/
│   │   ├── state.py
│   │   ├── orchestrator.py         # LangGraph 拓扑与 checkpoint
│   │   ├── nodes.py                # 规则、专家、总结、审核、告警节点
│   │   ├── coordinator.py          # 跨资产协调器
│   │   └── review_assistants.py
│   ├── ml/
│   │   ├── features.py             # LightGBM 特征列与矩阵转换
│   │   ├── risk_model.py           # 训练、校准、预测、模型状态
│   │   ├── labeling.py             # LLM-as-judge / deterministic 弱标注
│   │   ├── historical_data.py      # Binance public data 下载与解析
│   │   ├── historical_features.py  # 历史 K 线转训练 FeatureSnapshot
│   │   └── historical_training.py  # 历史训练、动态标签、hard negative
│   ├── evaluation/
│   │   └── offline.py              # 离线弱标签评测与提前预警调参
│   ├── market/
│   │   └── candles.py              # 1m K 线聚合
│   ├── simulation/
│   │   ├── scenarios.py
│   │   └── runner.py               # 模拟场景测试
│   ├── observability/
│   │   ├── llm_trace.py
│   │   └── metrics.py
│   ├── notification/
│   │   └── dispatcher.py           # Webhook + WebSocket 推送
│   └── persistence/
│       ├── database.py             # SQLAlchemy ORM 表
│       └── repositories.py         # 数据访问层
└── tests/
    ├── test_historical_ml.py
    ├── test_risk_model.py
    ├── test_runtime_quality_api.py
    ├── test_offline_evaluation.py
    ├── test_simulation_runner.py
    └── ...
```

---

## 总体架构

```text
实时交易所数据
  ├─ Binance WS / OKX WS / REST Poll
  └─ Normalizer
        ↓ RawEvent
EventBus
        ↓
FeatureBuilder
  ├─ 价格、波动、OI、funding、爆仓、数据质量
  ├─ FeatureSnapshot 落库
  └─ LightGBM predict_snapshot
        ↓
RuleEngine + ML RuleHit
  ├─ P1：高危，直接告警
  ├─ P2：进入人工审核
  ├─ P3：只记录或抑制，不正式告警
  └─ Early Warning：提前预警候选
        ↓
LangGraph Workflow
  ├─ load_memory
  ├─ run_rules
  ├─ expert_parallel
  ├─ summarizer
  ├─ decide
  ├─ build_case
  ├─ await_review
  └─ send_alert
        ↓
前端控制台 / WebSocket / Webhook / 评测总结
```

历史训练链路：

```text
Binance public data
  └─ monthly 1m klines zip
        ↓
historical_market_bar
        ↓
historical_features
  ├─ ret_1m/5m/15m/30m/60m
  ├─ realized_vol_5m/15m/60m
  ├─ volume_z / trade_count_z
  ├─ taker_buy_ratio
  ├─ futures_basis / basis_z
  └─ drawdown / runup / ATR
        ↓
dynamic weak label
  ├─ p1/p2 by future-return quantile
  ├─ none samples
  └─ hard negatives
        ↓
LightGBM + isotonic calibration + threshold tuning
        ↓
artifacts/risk_model/latest.joblib
```

---

## 前端功能

前端入口：http://localhost:5173

### 实时模式

- 查看 BTC / ETH / SOL 实时状态。
- 查看 1m K 线。
- 查看实时告警流。
- 查看待审核案例。
- 支持待审核分页。
- 支持审核通过和拒绝。
- 支持规则版本发布。

### 测试模式

- 运行内置模拟场景。
- 查看场景执行结果。
- 查看 LLM token、延迟、规则命中、告警触发等测试指标。

### 评测总结

- 真实运行样本量。
- 质量指标：误报代理、漏报代理、precision/recall proxy。
- 离线指标：offline precision、offline recall、offline miss rate。
- 提前预警指标：召回率、准确率、平均提前量、转化率。
- 延迟指标：ingest lag、case-to-alert、review turnaround。
- 吞吐指标：events/sec、snapshots/sec、cases/hour、alerts/hour。
- 去重指标：dedupe rate、duplicate suppressed、P2 聚合。
- LLM 成本：调用次数、token、估算成本、单告警成本。
- baseline 对比：相比纯规则基线的降噪效果。
- LightGBM 模型状态：模型版本、样本量、precision、recall、F1、重要特征。

---

## 核心 API

所有接口前缀为 `/api/v1`。

### 运行状态

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查、资产新鲜度、队列积压 |
| GET | `/metrics` | Prometheus 指标 |
| GET | `/agent/status` | ingestion 状态 |
| POST | `/agent/start` | 启动实时数据接入 |
| POST | `/agent/stop` | 停止实时数据接入 |

### 市场数据

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/market/candles?asset=BTC&interval=1m&limit=60` | 读取聚合后的 1m K 线 |

### 案例和告警

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/cases?status=pending_review&limit=10&offset=0&paginated=true` | 分页读取 case |
| GET | `/cases/{case_id}` | 读取单个 case |
| POST | `/cases/{case_id}/resume` | 人工审核 approve/reject |
| GET | `/alerts?limit=50` | 读取告警 |
| WebSocket | `/ws/alerts` | 实时告警流 |

审核请求：

```json
{
  "reviewer": "operator",
  "action": "approve",
  "comment": "确认风险，发送告警"
}
```

拒绝：

```json
{
  "reviewer": "operator",
  "action": "reject",
  "comment": "误报，关闭案例"
}
```

### 规则管理

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/rules/active` | 当前生效规则 |
| GET | `/rules/versions` | 规则版本列表 |
| GET | `/rules/changelog` | 规则变更日志 |
| POST | `/rules/publish` | 发布新规则版本 |

### 模拟测试

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/simulation/scenarios` | 场景库 |
| GET | `/simulation/runs/latest` | 最近一次运行 |
| GET | `/simulation/runs` | 历史运行 |
| POST | `/simulation/runs` | 执行场景 |

### 评测

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/evaluation/summary` | 真实运行评测总结 |
| GET | `/evaluation/offline` | 离线弱标签评测 |
| GET | `/evaluation/early-warning/tune` | 提前预警阈值搜索 |

### LightGBM 和历史训练

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/ml/risk-model/status` | 当前模型状态 |
| GET | `/ml/risk-model/predictions?limit=20` | 最近模型预测 |
| POST | `/ml/risk-model/label` | 对现有 feature_snapshot 做弱标注 |
| POST | `/ml/risk-model/train` | 用现有 feature_snapshot 训练 |
| POST | `/ml/historical-data/backfill` | 下载 Binance public data 历史 K 线 |
| GET | `/ml/historical-data/status` | 查看历史数据落库量 |
| POST | `/ml/risk-model/train-historical` | 用历史行情训练 LightGBM |

---

## LightGBM 使用说明

### 1. 查看模型状态

```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/api/v1/ml/risk-model/status |
  ConvertTo-Json -Depth 6
```

状态字段包括：

- `available`
- `model_version`
- `training_samples`
- `test_samples`
- `metrics.precision`
- `metrics.recall`
- `metrics.f1`
- `feature_importances`
- `thresholds`
- `training_metadata`
- `label_count`

### 2. 基于实时快照弱标注

```powershell
$body = @{
  days = $null
  horizon_minutes = 60
  max_snapshots_per_asset = 10000
  max_items = 500
  use_llm_judge = $true
  force = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://localhost:8000/api/v1/ml/risk-model/label `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

说明：

- `use_llm_judge=true` 会调用 LLM-as-judge，成本较高。
- 推荐先小批量跑，确认标注质量后再扩大。
- 标注结果写入 `risk_model_label` 表。

### 3. 基于实时快照训练

```powershell
$body = @{
  days = $null
  horizon_minutes = 60
  max_snapshots_per_asset = 10000
  max_label_items = 1000
  use_llm_judge = $false
  min_samples = 100
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://localhost:8000/api/v1/ml/risk-model/train `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### 4. 下载 2024/2025 历史真实行情

数据源使用 Binance public data：

- Spot：`data/spot/monthly/klines`
- USD-M Futures：`data/futures/um/monthly/klines`

建议按月或季度分批下载，不建议一次性直接拉全量 2024/2025。

示例：下载 BTC/ETH/SOL 2024 第一季度 1m K 线：

```powershell
$body = @{
  start = "2024-01-01T00:00:00"
  end = "2024-04-01T00:00:00"
  assets = @("BTC", "ETH", "SOL")
  interval = "1m"
  market_types = @("spot", "futures_um")
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://localhost:8000/api/v1/ml/historical-data/backfill `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

查看落库量：

```powershell
Invoke-RestMethod `
  -Uri http://localhost:8000/api/v1/ml/historical-data/status |
  ConvertTo-Json -Depth 5
```

### 5. 用历史行情训练模型

```powershell
$body = @{
  start = "2024-01-01T00:00:00"
  end = "2026-01-01T00:00:00"
  assets = @("BTC", "ETH", "SOL")
  horizon_minutes = 60
  max_snapshots_per_asset = $null
  max_label_items = 100000
  min_samples = 10000
  p2_quantile = 0.95
  p1_quantile = 0.995
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://localhost:8000/api/v1/ml/risk-model/train-historical `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

训练完成后模型写入：

```text
artifacts/risk_model/latest.joblib
```

线上预测会加载这个模型。模型包里保存了：

- LightGBM classifier
- feature columns
- global calibrator
- per-asset calibrator
- validation metrics
- threshold tuning result
- feature importances
- training metadata

### 6. 实时预测行为

实时快照落库后会执行：

```text
FeatureSnapshot -> predict_snapshot -> risk_model_prediction
```

然后：

- `P1` / `P2`：转成 `ML_RISK_PROBABILITY` 规则命中，进入 LangGraph case/alert 链路。
- `P3`：只记录预测，不触发正式告警。
- `none`：只记录预测。

如果模型元数据里有 `training_metadata.assets`，实时预测只会作用于这些资产，避免用 BTC-only 模型预测 ETH/SOL。

---

## 特征工程

当前 `FeatureSnapshot` 覆盖以下特征族。

### 价格波动

- `ret_1m`
- `ret_5m`
- `ret_15m`
- `ret_30m`
- `ret_60m`
- `price_range_pct_1m`
- `close_position_1m`
- `max_drawdown_15m`
- `max_drawdown_60m`
- `max_runup_15m`
- `max_runup_60m`

### 波动率

- `vol_z_1m`
- `realized_vol_5m`
- `realized_vol_15m`
- `realized_vol_60m`
- `atr_14`
- `volatility_regime_60m`

### 成交量与主动买入

- `volume_1m`
- `quote_volume_1m`
- `volume_5m`
- `quote_volume_5m`
- `volume_15m`
- `quote_volume_15m`
- `volume_z_15m`
- `volume_z_60m`
- `trade_count_1m`
- `trade_count_z_15m`
- `taker_buy_ratio_1m`
- `taker_buy_ratio_5m`

### 衍生品与风险压力

- `oi_delta_5m_pct`
- `oi_delta_15m_pct`
- `oi_delta_60m_pct`
- `liq_5m_usd`
- `funding_z`
- `futures_basis_pct`
- `basis_z_60m`

### 数据质量

- `source_stale`
- `cross_source_conflict`
- `ingest_lag_ms`

> 注意：历史训练链路已经完整构造 OHLCV、成交量、taker buy ratio、basis 等增强字段；实时接入链路目前仍主要填充原有价格、OI、funding、爆仓和数据质量字段，成交量/basis 的实时补齐仍是后续优化项。

---

## 弱标注策略

项目目前有三类标注方式。

### deterministic future-window label

基于未来窗口最大绝对收益率：

- 未来大波动：`p1` / `p2`
- 未出现明显风险：`none`

优点是稳定、低成本。缺点是标签较粗。

### LLM-as-judge

对 `feature_snapshot + future_summary` 调用 LLM，让 LLM 判断：

```json
{
  "label": "p1|p2|none",
  "risk_probability": 0.0,
  "confidence": 0.0,
  "rationale": "..."
}
```

建议用法：

- 不建议全量调用。
- 适合标注边界样本、多信号冲突样本、规则误报样本、规则漏报样本。

### historical dynamic weak label

历史行情训练使用动态分位数：

- `p2_quantile`：默认 `0.95`
- `p1_quantile`：默认 `0.995`

每个资产独立计算未来窗口波动分布，避免 BTC、ETH、SOL 波动水平差异导致标签失真。

同时引入 hard negative：

- 当前信号看起来危险。
- 但未来窗口没有真正风险。
- 这类样本专门用于降低误报、提升 precision 和 F1。

---

## 评测指标

评测页和 API 汇总以下指标。

### 告警质量

- precision proxy
- recall proxy
- false positive proxy
- missed alert proxy
- offline precision
- offline recall
- offline miss rate

### 提前预警八指标

- 提前预警召回率
- 提前预警准确率
- 平均提前量
- 提前预警转化率
- 正式告警召回率
- 正式告警误报率
- P95 告警延迟
- 待审核积压

### 性能与成本

- raw events
- feature snapshots
- events/sec
- snapshots/sec
- cases/hour
- alerts/hour
- avg ingest lag
- p95 ingest lag
- avg case-to-alert
- LLM calls
- LLM tokens
- estimated LLM cost
- cost per alert

### 去重与 baseline

- dedupe rate
- alert dedupe rate
- duplicate suppressed
- P2 aggregated
- alert reduction vs rule baseline
- prevented false positive proxy

### LightGBM 指标

- training samples
- test samples
- precision
- recall
- F1
- selected threshold
- top feature importances
- training metadata

---

## 数据库表

SQLite 默认本地文件为：

```text
etc_agent.db
```

Docker 中使用 volume：

```text
/data/etc_agent.db
```

核心表：

- `raw_event`
- `feature_snapshot`
- `risk_case`
- `risk_alert`
- `human_review_action`
- `rule_version`
- `rule_change_log`
- `llm_call`
- `quality_metric_event`
- `risk_model_label`
- `risk_model_prediction`
- `historical_market_bar`

---

## 运行测试

推荐先跑重点回归：

```powershell
python -m compileall -q src
python -m unittest tests.test_historical_ml tests.test_risk_model -v
```

较完整的核心测试：

```powershell
python -m unittest `
  tests.test_market_candles `
  tests.test_offline_evaluation `
  tests.test_runtime_quality_api `
  tests.test_rules `
  tests.test_historical_ml `
  tests.test_risk_model `
  tests.test_simulation_runner `
  -v
```

前端构建：

```powershell
docker compose exec -T frontend npm run build
```

容器内确认 LightGBM：

```powershell
docker compose exec -T backend python -c "import lightgbm, sklearn; print(lightgbm.__version__, sklearn.__version__)"
```

---

## 常见问题

### 1. `http://localhost:5173` 打不开

检查容器：

```powershell
docker compose ps
docker compose logs frontend
```

### 2. 前端能开，但接口失败

检查后端：

```powershell
docker compose logs backend
Invoke-RestMethod http://localhost:8000/api/v1/health
```

### 3. Binance / OKX 连接失败

优先检查：

- 宿主机能否访问交易所。
- 容器代理是否配置。
- 代理软件是否允许局域网连接。
- `.env` 中 `WS_PROXY` 是否使用 `host.docker.internal`。

### 4. LightGBM 报 `libgomp.so.1` 缺失

后端 Dockerfile 已安装：

```text
libgomp1
```

如果本地非 Docker 环境报错，需要安装系统 OpenMP runtime。

### 5. 历史数据下载慢

Binance public monthly zip 单文件较大。建议按资产、月份或季度分批跑。

### 6. 评测总结接口较慢

`/evaluation/summary` 会聚合大量真实运行数据、离线弱标签和质量指标。数据量大时可能需要几十秒，前端会等待返回。

### 7. 当前历史模型只对部分资产有效

如果只用 BTC 历史数据训练，模型元数据会记录 `assets=["BTC"]`，实时预测只作用于 BTC。补齐 ETH/SOL 历史数据并重新训练后，模型才会覆盖全部资产。

---

## 当前限制

- 实时链路的增强成交量/basis 特征还没有完全补齐，历史训练与实时预测特征存在部分口径差异。
- 2024/2025 全量历史数据需要按批次下载，仓库不会内置完整行情数据。
- LLM-as-judge 不适合全量无脑标注，应优先用于边界样本和争议样本。
- SQLite 适合开发和演示，生产建议迁移到 PostgreSQL。
- 当前模型指标依赖弱标签，不能等同于人工标注真值。
- 告警策略仍应由规则、模型、人工审核共同约束，不能只靠模型概率直接自动交易。

---

## 技术栈

| 模块 | 技术 |
|---|---|
| 后端 API | FastAPI, uvicorn |
| Agent 编排 | LangGraph, langgraph-checkpoint-sqlite |
| LLM | OpenAI-compatible SDK, 火山 Ark |
| 数据接入 | websockets, httpx, aiohttp |
| 数据库 | SQLite, SQLAlchemy async, aiosqlite |
| 规则引擎 | Python deterministic rules |
| 模型 | LightGBM, scikit-learn, joblib, numpy |
| 前端 | React, Vite |
| 可观测性 | prometheus-client, Grafana |
| 测试 | unittest |

---

## 推荐开发顺序

如果继续提升模型效果，建议按这个顺序做：

1. 补齐 BTC/ETH/SOL 2024/2025 spot + futures 1m 历史数据。
2. 重新训练历史版 LightGBM。
3. 对 false positive 样本做 hard negative 加强。
4. 对边界样本使用 LLM-as-judge 复核。
5. 补齐实时成交量、taker buy ratio、basis 特征。
6. 用 walk-forward 验证替代单次时间切分。
7. 在评测页区分 P1/P2 分层 precision、recall、F1。
