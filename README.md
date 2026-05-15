# ETC Risk Agent — 加密货币实时风控智能体

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Orchestration-FF6F00?style=for-the-badge)](https://langchain.com/langgraph)
[![LightGBM](https://img.shields.io/badge/LightGBM-%20ML%20Model-5CBB5C?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docs.docker.com)

**一个从数据采集 → 特征工程 → ML 预测 → 规则引擎 → Agent 工作流 → 人工审核 → 告警分发 → 离线评测的完整加密货币风控闭环系统**

</div>

---

<!-- 示例截图占位 -->
<p align="center">
  <em>📸 示例截图（请替换为你的实际截图）</em>
  <br>
  <img src="docs/screenshots/placeholder-dashboard.png" width="80%" alt="控制台截图">
  <br>
  <em>🔽 实时风控控制台 — 资产状态、K 线、告警流、待审核案例</em>
</p>

---

## 👁️ 项目概览

这是一个面向**求职展示**的全栈 Agent 项目，涵盖现代 AI 应用开发中的核心技术栈。系统以 **BTC / ETH / SOL** 为监控对象，接入 Binance / OKX 实时行情，通过 **[61 维滚动特征](#特征工程) → [LightGBM 风险排序](#模型指标) → [规则引擎](#规则引擎) → [LangGraph 多 Agent 工作流](#agent-工作流) → [人工审核](#人工审核台) → [告警分发](#告警分发)** 的完整链路，实现对加密货币市场风险的**提前发现、自动分析、人机协同决策和可控告警**。

> **🎯 为什么值得放在简历上**：这是一个从 0 到 1 的完整 AI 系统——不是 DEMO 玩具，不是调包脚本，而是涵盖数据工程、特征工程、模型训练与优化（Optuna + 不平衡处理）、Agent 编排（LangGraph）、前后端分离、Docker 部署、Prometheus 运维监控的真实工程实践。

---

## 📊 核心性能指标

<!-- 示例截图占位 -->
<p align="center">
  <em>📸 模型评测报告截图</em>
  <br>
  <img src="docs/screenshots/placeholder-model-report.png" width="70%" alt="模型评测">
</p>

| 指标 | 数值 | 说明 |
|------|------|------|
| **AUC-ROC** | **0.858** | 从基线 0.656 提升 30.8%，风险排序能力达到实用级别 |
| **Recall** | **85.3%** | 每 10 次真实大幅波动，系统提前捕获 8.5 次 |
| **Precision** | **77.2%** | 每 5 条告警中 4 条命中真风险 |
| **F1 Score** | **0.811** | 漏报与误报之间达到高品质平衡 |
| **数据规模** | **316 万条** | 2024 全年 Binance 1 分钟 K 线（BTC/ETH/SOL × 现货+合约） |
| **特征维度** | **61 维** | 覆盖价格、波动、成交、OI、基差、数据质量 |
| **Optuna 搜索** | **30 轮 TPE** | 8 维超参空间自动搜索 |

### 三代演进

```
v1 随手基线 (F1=0.29, AUC=0.66, 6棵树早停)
  → v2 平衡采样 (F1=0.57, AUC=0.77, 159棵树)
    → v3 Optuna+全年数据 (F1=0.81, AUC=0.86, 275棵树充分收敛)
```

> 从一条"根本没学到东西"的基线到 0.81 F1 的可上线模型——**三版迭代，量化可追溯**。

---

## 🧭 系统总览

### 数据流架构

```
┌─────────────────┐
│  Binance / OKX   │  WebSocket + REST 轮询
│  实时行情接入     │  SOCKS 代理支持
└────────┬────────┘
         │ RawEvent
         ▼
┌─────────────────┐
│   Normalizer     │  标准化 + 校验
│   + Validator    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FeatureBuilder  │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
│  30s 周期性快照   │  61 维特征 │ 滚动窗口计算
│  + LightGBM 推理  │  predict_snapshot() → P1/P2/P3
└────────┬────────┘
         │ FeatureSnapshot + ML RuleHit
         ▼
┌─────────────────┐
│   Rule Engine    │  P1: 高危直接告警
│   多层规则命中    │  P2: 进入审核
│   EW: 提前预警    │  P3: 仅记录
└────────┬────────┘
         │ RuleHit[]
         ▼
┌─────────────────┐
│  LangGraph       │  load_memory → run_rules
│  Agent 工作流     │  → expert_parallel (多分析师)
│                  │  → summarizer → decide
│                  │  → build_case → await_review → send_alert
└────────┬────────┘
         │ RiskCase / RiskAlert
         ▼
┌─────────────────┐
│   前端控制台      │  实时看板 │ 审核台 │ 评测总结
│   WebSocket 推送  │  Webhook 通知
└─────────────────┘
```

### 离线训练链路

```
Binance Public Data (免费公开)
  ↓ 按月 zip 下载 (data.binance.vision)
historical_market_bar 表
  ↓ build_snapshots_from_bars()
FeatureSnapshot × 30 万
  ↓ future_summary()  [O(n) 优化]
动态分位数弱标注 (p2=0.85, p1=0.95)
  ↓ 平衡采样 (正负比 ~1:1.2)
LightGBM + Optuna TPE 搜索
  ↓ Isotonic 校准 + F1 最佳阈值
artifacts/risk_model/latest.joblib
```

---

## 🧱 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **语言** | Python 3.10+ / JavaScript (React) | 后端 + 前端 |
| **Web 框架** | FastAPI + Uvicorn | REST API + WebSocket |
| **前端** | React 18 + Vite | 实时控制台 SPA |
| **数据库** | SQLAlchemy 2.0 + SQLite/aiosqlite | 所有业务数据持久化 |
| **Agent 编排** | LangGraph | 规则→分析→总结→审核→告警 状态图 |
| **ML 模型** | LightGBM + scikit-learn | 二分类风险排序 + Isotonic 校准 |
| **超参优化** | Optuna (TPE Sampler + Median Pruner) | 30 轮自动搜索 |
| **LLM** | OpenAI-compatible API (豆包/DeepSeek) | 多 Agent 分析、LLM-as-judge 标注 |
| **数据源** | Binance WS / OKX WS+REST / Binance Public Data | 实时 + 历史行情 |
| **运维** | Prometheus metrics / structlog | 14 个 Counter/Gauge/Histogram |
| **部署** | Docker Compose | 一键启动前后端 |

---

## 🚀 快速启动

### 1. 环境要求

- Python 3.10+
- Node 20+（Docker 模式不需要）
- Docker Desktop（推荐）
- 可访问 Binance / OKX 的网络（国内需代理）

### 2. 一分钟启动

```powershell
# 克隆仓库
git clone https://github.com/Callous-0923/crypto-risk-agent.git
cd crypto-risk-agent

# 配置 LLM API Key
Copy-Item .env.example .env
# 编辑 .env，填入 ARK_API_KEY

# Docker 启动
docker compose up -d --build
```

访问：

| 服务 | 地址 |
|------|------|
| 前端控制台 | http://localhost:8000 |
| API 文档 (Swagger) | http://localhost:8000/docs |
| 健康检查 | http://localhost:8000/api/v1/health |

### 3. 本地开发

```powershell
pip install -e .
python main.py                    # 后端 http://localhost:8000

cd frontend
npm install && npm run dev        # 前端 http://localhost:5173
```

### 4. 快速体验

```powershell
# 灌入种子数据
python scripts/seed_demo_data.py

# 启动 Agent 开始处理
Invoke-RestMethod -Method POST http://localhost:8000/api/v1/agent/start

# 运行模拟场景
Invoke-RestMethod `
  -Uri http://localhost:8000/api/v1/simulation/runs `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"scenario_id":"btc_flash_crash_p1"}'
```

---

## 🎯 Agent 工作流

系统用 **LangGraph** 实现了一个带人工审核的多 Agent 协作流程：

```
                    ┌──────────────┐
                    │  load_memory  │  加载关联历史案例和偏好
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  run_rules    │  规则引擎 + ML 预测
                    └──────┬───────┘
                           │ 如果有命中
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼───┐ ┌──────▼──────┐ ┌──▼──────────┐
     │ 分析师-A    │ │  分析师-B   │ │  分析师-C    │  parallel execution
     │ 技术面分析   │ │  衍生品分析  │ │  情绪面分析   │  各自给出独立判断
     └────────┬───┘ └──────┬──────┘ └──┬──────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │  summarizer   │  综合多分析师结论
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   decide      │  决策: P1 告警 / P2 审核 / 抑制
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  build_case   │  构建风控案例
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ await_review  │  暂停等待人工审核 (checkpoint)
                    └──────┬───────┘
                           │ approve/reject
                    ┌──────▼───────┐
                    │  send_alert   │  WebSocket + Webhook 推送
                    └──────────────┘
```

> **亮点**：这是 LangGraph 在金融风控场景下的完整落地——checkpoint 持久化支持审核中断恢复、多分析师并行推理、规则+ML 信号融合进入工作流。

---

## 🧠 模型优化历程

<!-- 示例截图占位 -->
<p align="center">
  <em>📸 Optuna 超参搜索可视化</em>
  <br>
  <img src="docs/screenshots/placeholder-optuna.png" width="60%" alt="Optuna 搜索">
</p>

### 从 0.29 到 0.81 F1 的三次迭代

| 阶段 | 关键改进 | AUC | F1 | 正样本比 | 树数 |
|------|----------|-----|-----|---------|------|
| **v1 原始** | 默认参数，正负比 1:8 | 0.656 | 0.287 | 11.4% | 6 (早停!) |
| **v2 平衡** | 正类全保留 + 1:1.2 负类配比 + `is_unbalance` | 0.769 | 0.565 | 49.3% | 159 |
| **v3 Optuna** | 全年数据 + 30轮 TPE 搜索 + 放宽分位数 | **0.858** | **0.811** | **56.6%** | **275** |

### 具体实施的不平衡修复手段

1. **放宽弱标注分位数**（p2: 0.95→0.85, p1: 0.995→0.95）——正样本从 5% 提升到 20%+
2. **正类全保留 + 负类 1:1.2 配比**——改写 `_select_training_records`，不再截断珍贵的正样本
3. **`is_unbalance=True`**——LightGBM 内建的不平衡优化
4. **`future_summary` O(n²)→O(n)**——利用排序提前 break，标注速度提升 ~1000×
5. **Optuna TPE 搜索**——8 维超参空间 30 轮自动探索，最佳参数使 AUC 再提升 11.6%

### 搜索到的最优参数

| 参数 | 默认 | 最优 | 含义 |
|------|------|------|------|
| learning_rate | 0.030 | **0.035** | 略激进的学习速度 |
| num_leaves | 31 | **24** | 更少叶节点(正则更强) |
| max_depth | 7 | **10** | 更深交互 |
| min_child_samples | 15 | **41** | 强防过拟合 |
| reg_alpha | 0.05 | **0.260** | 强 L1 正则 |
| reg_lambda | 0.05 | **0.160** | 强 L2 正则 |
| subsample | 0.80 | **0.84** | 每轮采样比例 |
| colsample_bytree | 0.80 | **0.97** | 几乎用全部特征 |

---

## 📐 特征工程

61 维特征覆盖五大维度——**价格、波动、量能、衍生品、数据质量**。

### 价格动量 (11 维)

`ret_1m` `ret_5m` `ret_15m` `ret_30m` `ret_60m` `price_range_pct_1m` `close_position_1m` `max_drawdown_15m` `max_drawdown_60m` `max_runup_15m` `max_runup_60m`

### 波动率 (7 维)

`vol_z_1m` `realized_vol_5m` `realized_vol_15m` `realized_vol_60m` `atr_14` `volatility_regime_60m`

### 成交量与主动性 (12 维)

`volume_1m` `quote_volume_1m` `volume_5m` `quote_volume_5m` `volume_15m` `quote_volume_15m` `volume_z_15m` `volume_z_60m` `trade_count_1m` `trade_count_z_15m` `taker_buy_ratio_1m` `taker_buy_ratio_5m`

### 衍生品压力 (10 维)

`oi_delta_5m_pct` `oi_delta_15m_pct` `oi_delta_60m_pct` `liq_5m_usd` `funding_z` `futures_basis_pct` `basis_z_60m`

### 数据质量 (3 维)

`source_stale` `cross_source_conflict` `ingest_lag_ms`

### 模型输入矩阵

特征经过 `build_matrix_rows()` 转换为 LightGBM 输入矩阵，包含 log 变换、缺失值处理和时间序列特性的保留。

---

## 📋 规则引擎

系统实现了一个**可版本化、可回放**的规则引擎。

| 规则层 | 触发条件 | 行为 |
|--------|---------|------|
| **P1** | 极端波动 + 大规模爆仓 + OI 异动组合 | 直接告警 |
| **P2** | 中等异动、OI 累积、资金费率偏移 | 进入人工审核 |
| **P3** | 轻微异动、单一信号 | 仅记录 (suppressed) |
| **Early Warning** | 微信号持续累积 + 趋势确认 | 跟踪候选，不直接告警 |

### ML 信号融入规则引擎

LightGBM 实时预测结果通过 `prediction_to_rule_hit()` 转换为 `ML_RISK_PROBABILITY` 规则命中，与纯规则信号平行进入 LangGraph 工作流——实现**规则 + 模型双信号源**融合。

---

## 🔬 评测体系

系统内置两套评测机制和完整的 Prometheus 运维指标覆盖。

### 离线弱标签评测 (`/evaluation/offline`)

基于真实历史快照 + 未来窗口价格变动，构建弱标签数据集，评测：

| 策略 | 监控指标 |
|------|---------|
| rules_baseline | 纯规则全量告警的 precision / recall / F1 |
| early_warning | 提前预警的召回率 + 平均提前量 |
| agent_alert | Agent 审核后正式告警 vs P1/P2 ground truth |

### 运行时代理指标 (`/evaluation/summary`)

| 指标 | 含义 |
|------|------|
| false_positive_proxy_rate | 审核驳回 Case 占比 → Precision 代理 |
| missed_alert_proxy_rate | 高信号未告警 Case 占比 → Recall 代理 |
| dedupe_rate | 重复告警抑制率 |
| approval_rate | 人工审核批准比例 |

### Prometheus 运维指标 (14 项)

```
feature_snapshot_total   │ rule_hit_total        │ llm_call_total
case_created_total       │ alert_sent_total       │ pending_review_gauge
human_review_total       │ data_quality_event_total│ ml_prediction_total
ml_inference_duration_*  │ ml_training_duration   │ ml_model_auc_roc
```

---

## 🎮 模拟测试

内置 **9 个风险场景**用于系统行为验证：

| 场景 | 资产 | 类型 |
|------|------|------|
| btc_flash_crash_p1 | BTC | P1 闪崩 |
| btc_leverage_buildup_p2 | BTC | P2 杠杆堆积 |
| btc_early_warning_to_p2 | BTC | 早期预警→P2 渐进 |
| eth_funding_squeeze_p2 | ETH | 资金费率极端 |
| eth_volatile_liquidation_p1 | ETH | 波动性清算级联 |
| sol_vol_spike_p2 | SOL | 波动率异常飙升 |
| sol_data_conflict_qa | SOL | 数据源冲突告警 |
| multi_asset_systemic_risk | BTC | 多资产联动风险 |
| btc_normal_market | BTC | 正常市场(验证不误报) |

<!-- 示例截图占位 -->
<p align="center">
  <em>📸 模拟测试结果截图</em>
  <br>
  <img src="docs/screenshots/placeholder-simulation.png" width="65%" alt="模拟测试">
</p>

---

## 🗄️ 数据库设计

SQLAlchemy ORM，SQLite 持久化，Docker 环境挂载 volume 防数据丢失。

| 表 | 用途 | 数据量 (Demo) |
|---|------|--------------|
| `raw_event` | 原始行情事件 | — |
| `feature_snapshot` | 30s 滚动特征快照 | 360 |
| `risk_case` | 风控案例 | 60 |
| `risk_alert` | 发出的告警 | 16 |
| `human_review_action` | 审核操作记录 | 30 |
| `llm_call` | LLM 调用追踪 | 120 |
| `quality_metric_event` | 数据质量事件 | 60 |
| `risk_model_label` | 弱标注 | 80 |
| `risk_model_prediction` | ML 预测记录 | 80 |
| `historical_market_bar` | 历史行情 K 线 | 3,162,240 |
| `rule_version` / `rule_change_log` | 规则版本管理 | — |

---

## 🧪 测试覆盖

```powershell
python -m unittest `
  tests/test_risk_model.py `
  tests/test_rule_engine.py `
  tests/test_offline_evaluation.py `
  tests/test_historical_ml.py `
  tests/test_simulation_runner.py `
  tests/test_runtime_quality_api.py `
  tests/test_ml_improvements.py `
  tests/test_market_candles.py `
  -v
```

| 测试文件 | 覆盖范围 |
|---------|---------|
| `test_risk_model.py` | LightGBM 训练、校准、预测、模型状态 |
| `test_rule_engine.py` | P1/P2/P3/EW 规则，边界值、组合条件 |
| `test_offline_evaluation.py` | 弱标签评测全流程 |
| `test_historical_ml.py` | 历史数据下载、特征工程、训练链路 |
| `test_simulation_runner.py` | 9 个场景执行与结果验证 |
| `test_ml_improvements.py` | v1→v2→v3 回归验证 |

---

## 🐳 Docker 部署

```yaml
# docker-compose.yml 核心结构
services:
  backend:   # Python FastAPI + uvicorn + LightGBM
    build: Dockerfile.backend.dev
    ports: [8000:8000]
    volumes: [.:/app, backend_db:/data]

  frontend:  # React SPA (Vite dev server 或 build 静态文件)
    build: frontend/Dockerfile.dev
    ports: [5173:5173]
    depends_on: [backend]
```

---

## 📂 项目结构

```text
crypto-risk-agent/
├── main.py                         # 入口: uvicorn.run
├── docker-compose.yml              # 一键部署
├── pyproject.toml                  # Python 依赖
├── .env.example                    # 环境变量模板
├── artifacts/
│   └── risk_model/latest.joblib    # 当前最优模型 (794 KB)
├── frontend/
│   ├── src/App.jsx                 # 实时/测试/评测三模式
│   └── vite.config.js
├── src/
│   ├── api/app.py                  # FastAPI lifespan + ingestion 启停
│   ├── api/routes.py               # 50+ REST/WS 端点
│   ├── core/config.py              # pydantic-settings
│   ├── core/proxy.py               # OpenAI 兼容客户端
│   ├── domain/models.py            # Pydantic 领域模型
│   ├── features/builder.py         # 特征快照 + ML 推理
│   ├── rules/engine.py             # 规则引擎
│   ├── graph/                      # LangGraph 工作流
│   │   ├── orchestrator.py         # 状态图拓扑
│   │   ├── nodes.py                # 5 个 Agent 节点
│   │   └── coordinator.py          # 跨资产协调
│   ├── ml/                         # ML 全链路
│   │   ├── risk_model.py           # 训练/校准/预测
│   │   ├── labeling.py             # 弱标注 + future_summary
│   │   ├── historical_training.py  # 历史训练 + 采样
│   │   └── historical_data.py      # Binance 数据下载
│   ├── evaluation/offline.py       # 离线评测 + 调参
│   ├── simulation/                 # 场景模拟
│   ├── memory/                     # 向量记忆 + 偏好学习
│   └── observability/metrics.py    # Prometheus 指标
├── scripts/
│   ├── train_optuna.py             # Optuna 搜索脚本
│   ├── train_best_params.py        # 最优参数训练
│   ├── download_2024_h2.py         # 数据下载
│   └── seed_demo_data.py           # 种子数据
└── tests/                          # 单元测试
```

---

## 🛠️ 常用命令

```powershell
# 查看模型状态
Invoke-RestMethod http://localhost:8000/api/v1/ml/risk-model/status

# 启动 Agent
Invoke-RestMethod -Method POST http://localhost:8000/api/v1/agent/start

# 下载历史数据 (按月)
python scripts/download_2024_h2.py

# Optuna 超参搜索
python scripts/train_optuna.py

# 最优参数完整训练
python scripts/train_best_params.py

# 评测总结
Invoke-RestMethod http://localhost:8000/api/v1/evaluation/summary
```

---

## ⚠️ 声明

本项目为**研发演示系统**，模型无法保证在极端市场条件下 100% 捕获所有风险。不构成投资建议或风控保障，不可直接用于资金安全决策。告警结果需人工审核或专业风控系统兜底。

---

## 📄 License

MIT

---

<p align="center">
  <b>🚀 如果你觉得这个项目有帮助，请给一个 Star ⭐</b>
  <br><br>
  <em>Built with Python, React, LangGraph, LightGBM & Optuna</em>
</p>
