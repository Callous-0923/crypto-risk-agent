# 加密货币市场实时风控 Agent

基于 **FastAPI + LangGraph** 构建的事件驱动风控 Agent，接入 Binance / OKX 双源行情，对 BTC / ETH / SOL 进行实时特征提取与规则评估，通过专家 Agent 并行流水线（fan-out/fan-in）生成中文风险告警，支持人工审核介入与进程崩溃可恢复的 interrupt/resume 机制。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据接入层                               │
│  Binance Spot WS ──┐                                            │
│  Binance Futures WS─┤  Normalizer → EventBus (asyncio.Queue)    │
│  OKX WS ───────────┤                                            │
│  OKX REST Poll ────┘                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │ RawEvent
┌────────────────────────────▼────────────────────────────────────┐
│                       特征构建层                                  │
│  FeatureBuilder (_AssetState per asset)                         │
│  • 双源价格追踪：binance_price / okx_price                       │
│  • validate_sources → trusted_price（价差超阈值调 LLM 仲裁）     │
│  • 30s 滚动窗口：ret_1m/5m, vol_z_1m, oi_delta_15m,             │
│                  liq_5m_usd, funding_z                          │
│  • 数据质量标志：source_stale, cross_source_conflict             │
└────────────────────────────┬────────────────────────────────────┘
                             │ FeatureSnapshot (每 30s, 每资产)
┌────────────────────────────▼────────────────────────────────────┐
│                     规则引擎层                                    │
│  RuleEngine (deterministic, 无 LLM)                             │
│  • P1：极端波动 ≥5% / 爆仓 ≥5000万美元                          │
│  • P2：波动 ≥3% / vol_z ≥3σ / OI变动 ≥10% / funding_z ≥2.5σ   │
│  • P3：数据质量降级 (source_stale / cross_source_conflict)       │
│  RuleRegistry：版本化阈值 + 热更新 + diff 审计日志               │
└────────────────────────────┬────────────────────────────────────┘
                             │ list[RuleHit] + Severity
┌────────────────────────────▼────────────────────────────────────┐
│                   LangGraph Agentic Workflow                     │
│                                                                  │
│  load_memory                                                     │
│      │  (近5条告警历史 → state.recent_alert_history)            │
│      ▼                                                           │
│  run_rules                                                       │
│      │                                                           │
│      ▼                                                           │
│  expert_parallel  ←── asyncio.gather ──►  [P1 only]             │
│  ├── technical_analyst  (价格/爆仓/持仓, Tool Use ≤2轮)          │
│  │   输出: {key_metrics, confidence, narrative_zh}               │
│  └── macro_context      (OI/funding/历史告警, Tool Use ≤1轮)     │
│      输出: {key_metrics, confidence, narrative_zh}               │
│      │                                                           │
│      ▼                                                           │
│  summarizer                                                      │
│      │  • 接收原始 snapshot + 两专家输出                         │
│      │  • _collect_metric_mismatches: key_metrics 数值           │
│      │    与 snapshot tolerance 比对，不一致时代码层             │
│      │    强制追加至 review_guidance（非 prompt 层控制）          │
│      │  • P3 fast path: 跳过 LLM，0 tokens                      │
│      ▼                                                           │
│  decide                                                          │
│      │  • P1 → EMIT (dedupe: find_active_case_by_dedupe_key)    │
│      │  • P2 → MANUAL_REVIEW (疲劳抑制: 5min内3+条→SUPPRESSED)  │
│      │  • P3 / 疲劳命中 → SUPPRESS (写库含 suppression_reason)  │
│      ▼                                                           │
│  build_case                                                      │
│      │  • P1: 查 dedupe_key 5min 内 open case，存在则复用        │
│      │  • P2: 并发调 build_historical_context + quantify_risk   │
│      │  • threshold_advisor: 生成阈值调整建议存 case             │
│      ▼                                                           │
│  await_review  ←── interrupt_before (P2/协调器 case 暂停)       │
│      │  resume: aupdate_state → ainvoke                         │
│      │  coordinator case: 300s 超时自动批准兜底                  │
│      ▼                                                           │
│  send_alert                                                      │
│      │  idempotency_key = {case_id}:{revision}:{channel}        │
│      │  Webhook POST + WebSocket broadcast                       │
│      ▼                                                           │
│     END                                                          │
└─────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    跨资产协调器                                   │
│  60s 内 2+ 资产同时 P1 → 规则触发（非 LLM 决策）                │
│  LLM 生成跨资产摘要 → process_coordinator_case                  │
│  → interrupt_before 人审 + 300s 超时自动批准                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键设计决策

### 为什么选 LangGraph 而不是 Celery + Redis？

| 需求 | Celery + Redis | LangGraph |
|------|---------------|-----------|
| 人审中断后跨进程 resume | 需自行实现 task state 持久化 | `AsyncSqliteSaver` + `interrupt_before` 开箱即用 |
| 完整 state 跨节点共享 | 需序列化进外部存储 | TypedDict state 图内一等公民 |
| 多轮 Tool Use agentic loop | chain + retry 需自行拼装 | 节点天然支持多轮 |
| 决策追溯（checkpoint 历史） | 额外开发 | 同一 thread 历史 checkpoint 保留 |

### LLM 决策边界

> LLM 只输出解释性文字和建议，不参与任何路由判断。

- **P1/P2/SUPPRESS 的判断**：规则引擎（deterministic）
- **疲劳抑制的判断**：`_fatigue_window_hit`（代码逻辑）
- **协调器"发不发"的判断**：规则（2+ 资产 P1）
- **LLM 做的事**：生成 `summary_zh`、`review_guidance`、`threshold_suggestion`、跨资产摘要

### 幻觉传播控制

单靠 prompt 指令无法保证幻觉不传播到下游决策。当前实现的两层控制：

1. **结构化输出**：专家节点输出 `{key_metrics: {ret_1m, liq_5m_usd, ...}, confidence, narrative_zh}`，数字和叙述分离
2. **机器校验**：`_collect_metric_mismatches` 对 `key_metrics` 数值字段与原始 snapshot 做 tolerance 比对，不一致时**在代码层**（非 prompt）强制追加至 `review_guidance`

能机器校验的部分做硬断言，不能校验的语义描述通过 `review_guidance` 透传给人工审核。

### P1 告警去重

BTC 闪崩 5 分钟内可能触发 10 个 30s 快照，双层去重避免告警风暴：

```
快照 1 → node_build_case → find_active_case_by_dedupe_key(5min) → 无 → 新建 case_1
快照 2 → node_build_case → find_active_case_by_dedupe_key(5min) → 命中 case_1 → 复用
...
快照 N → 复用 case_1 → node_send_alert → idempotency_key 已存在 → 不重发
```

结果：1 个 case，1 条告警。

### 疲劳抑制审计

被抑制的 P2 case 以 `status=SUPPRESSED` + `suppression_reason="fatigue"` 写库，支持 `GET /api/v1/cases?include_suppressed=true` 事后复盘，区分"疲劳抑制压掉的"和"规则未触发的"。

---

## 项目结构

```
etc-agent/
├── main.py                        # 入口，uvicorn 启动
├── src/
│   ├── api/
│   │   ├── app.py                 # FastAPI 工厂，Agent 启停管理
│   │   └── routes.py              # REST + WebSocket 接口
│   ├── core/
│   │   ├── config.py              # 环境变量与配置（pydantic-settings）
│   │   └── logging.py
│   ├── domain/
│   │   └── models.py              # Pydantic 领域模型（RiskCase, RuleHit, RiskAlert...）
│   ├── features/
│   │   └── builder.py             # 滚动窗口特征构建，build_snapshot_validated
│   ├── graph/
│   │   ├── state.py               # RiskState TypedDict
│   │   ├── orchestrator.py        # 图拓扑，AsyncSqliteSaver，process_snapshot/resume_case
│   │   ├── nodes.py               # 所有节点实现（含专家节点、幻觉校验、dedupe）
│   │   ├── agent_tools.py         # 4 个 OpenAI function-call 工具
│   │   ├── coordinator.py         # 跨资产协调器
│   │   └── review_assistants.py   # 人审助手（历史上下文 + 风险量化）
│   ├── ingestion/
│   │   ├── normalizer.py          # RawEvent 归一化 + EventBus
│   │   ├── validator.py           # 双源价格验证（validate_sources）
│   │   └── sources/
│   │       ├── binance_ws.py      # Binance Spot + Futures WebSocket
│   │       ├── okx_ws.py          # OKX WebSocket
│   │       └── okx_rest.py        # OKX REST 轮询
│   ├── notification/
│   │   └── dispatcher.py          # Webhook + WebSocket 广播
│   ├── observability/
│   │   └── metrics.py             # 12 项 Prometheus 指标定义
│   ├── persistence/
│   │   ├── database.py            # SQLAlchemy ORM 表定义
│   │   └── repositories.py        # 异步 CRUD（含 find_active_case_by_dedupe_key）
│   └── rules/
│       ├── engine.py              # 规则评估（deterministic，无 LLM）
│       └── config.py              # RuleThresholds + RuleRegistry 热更新
├── tests/
│   ├── test_rules.py
│   ├── test_phase4.py             # 并行快照 + 协调器 + case_id 一致性
│   ├── test_phase5.py             # 专家并行时序 + 幻觉校验 + P3 fast path
│   ├── test_phase6.py             # 审核助手并发 + P1 跳过
│   ├── test_phase7.py             # 双源验证（小价差/大价差/builder集成）
│   └── test_phase8.py             # Agent 启停幂等
├── grafana/
│   └── dashboard.json             # Grafana 看板配置
├── .env.example
└── pyproject.toml
```

---

## 快速开始

### 环境要求

- Python ≥ 3.10
- 火山引擎 Ark API Key（或兼容 OpenAI 格式的其他 LLM API）

### 安装

```bash
git clone https://github.com/your-username/etc-agent.git
cd etc-agent
pip install -e .
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`：

```env
# 火山引擎 Ark（豆包/DeepSeek，兼容 OpenAI SDK）
ARK_API_KEY=your_key_here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
LLM_MODEL=doubao-seed-1-6-251015

# 数据库（SQLite，无需额外安装）
DATABASE_URL=sqlite+aiosqlite:///./etc_agent.db

# 可选：告警 Webhook（如飞书、企微机器人）
WEBHOOK_URL=
```

> 也可以替换为 OpenAI / DeepSeek 官方 API，修改 `ARK_BASE_URL` 和 `LLM_MODEL` 即可，代码使用 `openai` SDK 无需改动。

### 启动

```bash
python main.py
```

服务启动后自动：
1. 初始化 SQLite 数据库和 LangGraph checkpoint 表
2. 加载 active 规则版本
3. 启动 Binance / OKX 数据接入（需要网络可访问交易所）

### 运行测试

```bash
python -m unittest tests.test_rules tests.test_phase4 tests.test_phase5 tests.test_phase6 tests.test_phase7 tests.test_phase8 -v
```

所有测试均为纯 mock，不依赖网络和 LLM。

---

## API 接口

服务地址：`http://localhost:8000`，文档：`http://localhost:8000/docs`

### Agent 控制

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/agent/start` | 启动数据接入（幂等，重复调用返回 409） |
| `POST` | `/api/v1/agent/stop` | 优雅停止 |
| `GET` | `/api/v1/agent/status` | 查询运行状态 |
| `GET` | `/api/v1/health` | 数据新鲜度 + 队列积压 |

### Case 管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/cases` | 查询 case 列表（支持 `?asset=BTC&include_suppressed=true`） |
| `GET` | `/api/v1/cases/{case_id}` | 查询单个 case（含 `historical_context_zh`、`threshold_suggestion` 等） |
| `POST` | `/api/v1/cases/{case_id}/resume` | 人工审核（approve / reject） |

### 规则版本管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/rules/active` | 当前生效版本和阈值 |
| `POST` | `/api/v1/rules/publish` | 发布新版本（热更新，无需重启，记录 diff 审计日志） |
| `GET` | `/api/v1/rules/versions` | 所有历史版本 |
| `GET` | `/api/v1/rules/changelog` | 变更审计日志 |

### 实时推送

```
WebSocket: ws://localhost:8000/api/v1/ws/alerts
```

每 30 秒心跳 `{"type":"ping"}`，告警到达时推送完整 alert payload。

### 示例：人工审核 P2 case

```bash
# 查询待审核 case
curl http://localhost:8000/api/v1/cases?asset=BTC

# 批准（触发 LangGraph resume，继续执行 send_alert 节点）
curl -X POST http://localhost:8000/api/v1/cases/{case_id}/resume \
  -H "Content-Type: application/json" \
  -d '{"reviewer": "alice", "action": "approve", "comment": "确认风险，发送告警"}'

# 拒绝
curl -X POST http://localhost:8000/api/v1/cases/{case_id}/resume \
  -H "Content-Type: application/json" \
  -d '{"reviewer": "alice", "action": "reject", "comment": "价格已回归，无需告警"}'
```

### 示例：热更新规则阈值

```bash
# 查看当前阈值
curl http://localhost:8000/api/v1/rules/active

# 发布新版本（只传要修改的字段）
curl -X POST http://localhost:8000/api/v1/rules/publish \
  -H "Content-Type: application/json" \
  -d '{
    "version_tag": "v2",
    "operator": "alice",
    "reason": "SOL 波动率偏高，适当放宽 P2 阈值",
    "price_change_p2": 0.04
  }'
```

---

## 可观测性

### Prometheus 指标

`GET /api/v1/metrics` 暴露以下指标（Prometheus 格式）：

| 指标名 | 类型 | Labels | 说明 |
|--------|------|--------|------|
| `ingest_event_total` | Counter | asset, source, event_type | 归一化事件总数 |
| `ingest_lag_seconds` | Histogram | source | 事件从 event_ts 到入库的延迟 |
| `ws_reconnect_total` | Counter | source | WS 断连重连次数 |
| `event_bus_queue_size` | Gauge | — | 事件总线积压 |
| `rule_evaluate_total` | Counter | asset, rule_version | 规则评估总次数 |
| `rule_hit_total` | Counter | asset, rule_id, severity, rule_version | 规则命中分布 |
| `llm_call_total` | Counter | status | LLM 调用成功/失败 |
| `llm_call_duration_seconds` | Histogram | — | LLM 调用耗时 |
| `llm_error_total` | Counter | error_type | LLM 错误分类（timeout/auth/network） |
| `case_created_total` | Counter | asset, severity, decision | Case 创建数 |
| `alert_sent_total` | Counter | asset, severity, channel | 告警发送数 |
| `pending_review_gauge` | Gauge | — | 当前待人审 case 积压数 |

### Grafana

`grafana/dashboard.json` 可直接导入，覆盖数据接入健康、规则命中热图、LLM 调用成功率、人审积压趋势四个看板。

---

## LangGraph 图状态

`RiskState` TypedDict 全字段：

```python
thread_id: str                    # = case_id，LangGraph thread 标识
asset: Asset                      # BTC / ETH / SOL
snapshot: FeatureSnapshot | None  # resume 路径下可能为 None
is_coordinator_case: bool         # 跨资产协调器产生的 case

rule_hits: list[RuleHit]
highest_severity: Severity | None

# 专家节点输出（结构化 JSON）
technical_analysis: dict | None   # {key_metrics, confidence, narrative_zh}
macro_context: dict | None
technical_analysis_zh: str
macro_context_zh: str

# 最终输出
summary_zh: str
review_guidance: str              # 含幻觉校验 flag（如有不一致）
historical_context_zh: str        # 审核助手：历史 case 摘要（仅 P2）
risk_quantification_zh: str       # 审核助手：风险量化（仅 P2）
threshold_suggestion: dict | None # 阈值调整建议（P1/P2 触发时）

decision: Decision | None         # EMIT / MANUAL_REVIEW / SUPPRESS
case: RiskCase | None
alert: RiskAlert | None
human_approved: bool | None
human_comment: str

# 历史记忆
recent_alert_history: list[dict]
fatigue_suppressed: bool
```

---

## 已知局限

以下是当前版本的明确 gap，不粉饰：

| 局限 | 说明 |
|------|------|
| **无账户数据** | 告警不包含持仓敞口信息，运营仍需手动查交易所 |
| **阈值无回测支撑** | P1/P2 阈值为经验值，未用历史数据验证 |
| **无 LLM eval pipeline** | 专家节点输出质量无定量基准，仅有机器校验（数值字段）和人工审核兜底 |
| **SQLite 单点** | 无 HA，不支持水平扩展，生产建议替换为 PostgreSQL + `AsyncPostgresSaver` |
| **疲劳抑制无豁免通道** | 暂无 API 允许运营在极端行情时临时解除疲劳抑制 |
| **前端未联调** | React 控制台代码已写，受本地 Node.js 环境限制未完成浏览器端联调 |
| **SLO 未校准** | P1 召回率 >99% / P2 精准率 >60% 为推导值，需历史数据验证 |

---

## 技术栈

| 层 | 技术 |
|----|------|
| Web 框架 | FastAPI + uvicorn |
| Agent 编排 | LangGraph 0.2+ (AsyncSqliteSaver) |
| LLM | OpenAI-compatible API（字节 Ark / DeepSeek，可替换） |
| 数据库 | SQLite + SQLAlchemy 2.0 async |
| 数据接入 | websockets, httpx |
| 可观测性 | prometheus-client, Grafana |
| 测试 | unittest.IsolatedAsyncioTestCase |
| 配置 | pydantic-settings, python-dotenv |
