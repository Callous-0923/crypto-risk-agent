import { useEffect, useMemo, useState } from "react";

const API_PREFIX = "/api/v1";
const REFRESH_INTERVAL_MS = 30000;
const CANDLE_LIMIT = 24;
const CASE_PAGE_SIZE = 10;
const SUPPORTED_ASSETS = ["BTC", "ETH", "SOL"];
const LANGUAGE_STORAGE_KEY = "etc-agent-language";
const DISPLAY_TIME_ZONE = "Asia/Shanghai";
const DISPLAY_TIME_SUFFIX = "UTC+8";
const EVALUATION_WINDOWS = [
  { key: "all", days: null },
  { key: "7d", days: 7 },
  { key: "30d", days: 30 },
];

const initialRuleForm = {
  version_tag: "",
  operator: "",
  reason: "",
  price_change_p1: "",
  price_change_p2: "",
  oi_delta_p2: "",
  liq_usd_p1: "",
  funding_z_p2: "",
  early_warning_ret_5m: "",
  early_warning_oi_delta: "",
  early_warning_funding_z: "",
  early_warning_vol_z: "",
  early_warning_min_score: "",
  early_warning_min_signals: "",
  early_warning_single_signal_min_score: "",
  early_warning_persistence_window: "",
  early_warning_persistence_hits: "",
  early_warning_dynamic_baseline: "",
  early_warning_dynamic_history: "",
  early_warning_dynamic_quantile: "",
  early_warning_trend_window: "",
  early_warning_min_trend_hits: "",
  vol_z_spike: "",
  cross_source_conflict_pct: "",
};

const TRANSLATIONS = {
  zh: {
    title: "交易风控控制台",
    heroCopy: "实时模式负责监控线上状态，测试模式专门跑模拟场景和评估指标，互不干扰。",
    running: "运行中",
    stopped: "已停止",
    start: "启动",
    stop: "停止",
    health: "健康状态",
    refreshCadence: "刷新节奏：30 秒",
    priceCandles: "价格 K 线",
    chartDerivedFromSnapshots: "基于持久化 30 秒可信快照聚合。",
    dataFreshness: "数据新鲜度",
    fresh: "新鲜",
    stale: "过期",
    price: "价格",
    age: "延迟",
    liveAlerts: "实时告警",
    waitingAlerts: "等待告警流量...",
    alertsDisconnected: "实时告警连接暂时不可用，正在自动重连。",
    alertsConnecting: "正在连接实时告警流...",
    pendingReviews: "待审核案例",
    noPendingCases: "当前没有待审核案例。",
    crossAssetCase: "跨资产案例",
    caseSuffix: "案例",
    show: "展开",
    hide: "收起",
    reviewer: "审核人",
    reviewNotes: "审核备注",
    approve: "批准",
    reject: "拒绝",
    previousPage: "上一页",
    nextPage: "下一页",
    casePageStatus: "{start}-{end} / {total}",
    noCasePage: "0 / 0",
    rulePublishing: "规则发布",
    activeVersion: "当前版本",
    publishVersion: "发布版本",
    marketCandles: "市场 K 线",
    candleTitleSuffix: "1 分钟 K 线",
    sync30s: "30 秒同步",
    closedCandle: "已收盘",
    liveCandle: "进行中",
    waitingSnapshots: "等待 30 秒快照积累...",
    lastClose: "最新收盘",
    window: "窗口",
    minutes: "分钟",
    updated: "更新时间",
    createdAt: "创建时间",
    open: "开盘",
    high: "最高",
    low: "最低",
    close: "收盘",
    snapshots: "快照数",
    candleRange: "时间区间",
    selectedCandle: "选中 K 线",
    clickCandleHint: "点击任意一根 K 线查看该时间节点的详细数据。",
    chartPriceAxis: "价格轴",
    noSummary: "暂无摘要。",
    language: "语言",
    chinese: "中文",
    english: "English",
    agentStarted: "Agent 已启动。",
    agentStopped: "Agent 已停止。",
    caseApproved: "案例 {caseId} 已批准。",
    caseRejected: "案例 {caseId} 已拒绝。",
    rulePublished: "规则版本 {versionTag} 已发布。",
    realtimeMode: "实时模式",
    realtimeModeCopy: "线上监控、审核和规则发布。",
    testMode: "测试模式",
    testModeCopy: "模拟数据流、响应耗时和 token 指标。",
    testModeTitle: "模拟测试工作区",
    testModeCopyFull: "这里的测试运行不会写入正式风险 case 或 alert 表，用于场景验证和性能观察。",
    scenarioLibrary: "场景库",
    runScenario: "运行场景",
    runningScenario: "场景运行中...",
    noScenarioResult: "选择一个场景后即可运行并查看结果。",
    latestRun: "最近一次运行",
    passRate: "通过率",
    totalLatency: "平均总耗时",
    maxLatency: "最大总耗时",
    llmTokens: "LLM 总 token",
    llmCalls: "LLM 调用次数",
    alertsEmitted: "触发告警数",
    manualReviews: "人工审核数",
    p1Steps: "P1 步骤数",
    p2Steps: "P2 步骤数",
    promptTokens: "输入 token",
    completionTokens: "输出 token",
    usageUnavailable: "当前模型未返回 usage，token 指标可能为 0。",
    scenarioExpectation: "预期输出",
    scenarioActual: "实际输出",
    correctnessPassed: "命中预期",
    correctnessFailed: "偏离预期",
    decision: "决策",
    severity: "级别",
    rules: "规则",
    stageMetrics: "阶段耗时",
    summary: "摘要",
    reviewGuidance: "审核指引",
    historicalContext: "历史上下文",
    riskQuantification: "风险量化",
    checkpointTime: "检查点时间",
    injectedEvents: "注入事件数",
    snapshotLatency: "快照耗时",
    ruleLatency: "规则耗时",
    expertLatency: "专家耗时",
    summarizerLatency: "汇总耗时",
    reviewLatency: "审核助手耗时",
    totalLatencyLabel: "总耗时",
    loadMemoryLatency: "记忆加载耗时",
    actualRules: "实际规则",
    expectedRules: "预期规则",
    noScenarios: "当前没有可用测试场景。",
    selectedScenario: "当前场景",
    viewReport: "测试报告",
    outputOperations: "涉及操作",
    status: "状态",
    idle: "空闲",
    completed: "已完成",
    viewMode: "视图模式",
    evaluationMode: "评测总结",
    evaluationModeCopy: "真实运行数据汇总与简历结果。",
    evaluationTitle: "真实运行评测汇总",
    evaluationCopyFull: "基于数据库中沉淀的真实 raw event、snapshot、case、alert 和人工审核记录自动聚合。",
    evaluationWindow: "统计窗口",
    evaluationRefresh: "刷新汇总",
    evaluationGeneratedAt: "生成时间",
    evaluationCoverageHours: "覆盖时长",
    evaluationAllTime: "全量",
    evaluation7d: "近 7 天",
    evaluation30d: "近 30 天",
    runtimeSample: "真实运行样本",
    runtimeCoverage: "覆盖资产",
    runtimeAssets: "资产覆盖",
    runtimeResume: "简历结果",
    runtimeHighlights: "简历亮点",
    runtimeTopRules: "高频规则",
    runtimeAssetBreakdown: "资产拆解",
    runtimeCaseMetrics: "案例指标",
    runtimeLatency: "时延指标",
    runtimeQuality: "质量指标",
    coreEightMetrics: "核心八指标",
    earlyWarningRecall: "提前预警召回率",
    earlyWarningPrecision: "提前预警准确率",
    earlyWarningLeadTime: "平均提前量",
    earlyWarningConversion: "提前预警转化率",
    formalAlertRecall: "正式告警召回率",
    formalAlertFalsePositive: "正式告警误报率",
    p95AlertLatency: "P95 告警延迟",
    earlyWarningCases: "提前预警案例",
    earlyWarningPending: "提前预警待审",
    earlyWarningPolicy: "提前预警策略",
    earlyWarningBestScore: "推荐调参分",
    earlyWarningBestF1: "推荐 F1",
    earlyWarningBestConfig: "推荐参数",
    runtimeReview: "人工审核",
    rawEvents: "原始事件",
    featureSnapshots: "特征快照",
    riskCases: "风险案例",
    riskAlerts: "风险告警",
    humanReviews: "审核操作",
    alertEmissionRate: "告警转化率",
    manualReviewRate: "人工审核率",
    approveRate: "批准率",
    rejectRate: "拒绝率",
    memoryEnrichmentRate: "记忆增强覆盖率",
    avgCaseToAlert: "平均 case→alert",
    avgReviewTurnaround: "平均审核响应",
    avgIngestLag: "平均 ingest lag",
    p95IngestLag: "P95 ingest lag",
    conflictRate: "跨源冲突率",
    staleRate: "数据过期率",
    falsePositiveProxy: "误报率代理",
    missedAlertProxy: "漏报率代理",
    precisionProxy: "精准率代理",
    recallProxy: "召回率代理",
    offlineRecall: "离线召回率",
    offlinePrecision: "离线精准率",
    offlineMissRate: "离线漏报率",
    offlineEpisodes: "离线样本",
    dedupeRate: "去重率",
    alertDedupeRate: "告警去重率",
    duplicateSuppressed: "去重拦截",
    p2Aggregated: "P2 聚合",
    eventsPerSecond: "事件吞吐",
    snapshotsPerSecond: "快照吞吐",
    llmCost: "LLM 成本",
    llmCostPerAlert: "单告警成本",
    llmCostConfigMissing: "成本单价未配置",
    baselineAlertReduction: "较规则基线降噪",
    preventedFalsePositiveProxy: "拦截误报代理",
    pendingReviewCases: "待审核积压",
    coordinatorCases: "协调案例",
    topRuleCount: "命中次数",
    noEvaluationData: "当前数据库里还没有足够的真实运行数据。",
    riskModel: "LightGBM 风险预测模型",
    riskModelStatus: "模型状态",
    riskModelAvailable: "已启用",
    riskModelUnavailable: "模型尚未训练",
    riskModelVersion: "模型版本",
    riskModelLabels: "弱标注样本",
    riskModelTrainSamples: "训练样本",
    riskModelTestSamples: "验证样本",
    riskModelPrecision: "模型准确率",
    riskModelRecall: "模型召回率",
    riskModelF1: "模型 F1",
    riskModelHorizon: "预测窗口",
    riskModelThresholds: "分级阈值",
    riskModelTopFeatures: "重要特征",
  },
  en: {
    title: "Trading Risk Console",
    heroCopy: "Realtime mode watches the live system. Test mode runs replay scenarios and evaluation metrics without touching the live workspace.",
    running: "Running",
    stopped: "Stopped",
    start: "Start",
    stop: "Stop",
    health: "Health",
    refreshCadence: "Refresh cadence: 30 seconds",
    priceCandles: "Price Candles",
    chartDerivedFromSnapshots: "Derived from persisted 30-second trusted snapshots.",
    dataFreshness: "Data Freshness",
    fresh: "Fresh",
    stale: "Stale",
    price: "Price",
    age: "Age",
    liveAlerts: "Live Alerts",
    waitingAlerts: "Waiting for alert traffic...",
    alertsDisconnected: "Live alert streaming is temporarily unavailable. Reconnecting automatically.",
    alertsConnecting: "Connecting to live alert stream...",
    pendingReviews: "Pending Reviews",
    noPendingCases: "No pending review cases.",
    crossAssetCase: "Cross-asset case",
    caseSuffix: "case",
    show: "Show",
    hide: "Hide",
    reviewer: "Reviewer",
    reviewNotes: "Review notes",
    approve: "Approve",
    reject: "Reject",
    previousPage: "Previous",
    nextPage: "Next",
    casePageStatus: "{start}-{end} / {total}",
    noCasePage: "0 / 0",
    rulePublishing: "Rule Publishing",
    activeVersion: "Active version",
    publishVersion: "Publish version",
    marketCandles: "Market Candles",
    candleTitleSuffix: "1m candles",
    sync30s: "30s sync",
    closedCandle: "Closed candle",
    liveCandle: "Live candle",
    waitingSnapshots: "Waiting for synced 30-second snapshots...",
    lastClose: "Last close",
    window: "Window",
    minutes: "minutes",
    updated: "Updated",
    createdAt: "Created",
    open: "Open",
    high: "High",
    low: "Low",
    close: "Close",
    snapshots: "Snapshots",
    candleRange: "Range",
    selectedCandle: "Selected Candle",
    clickCandleHint: "Click any candle to inspect that minute in detail.",
    chartPriceAxis: "Price Axis",
    noSummary: "No summary available.",
    language: "Language",
    chinese: "中文",
    english: "English",
    agentStarted: "Agent started.",
    agentStopped: "Agent stopped.",
    caseApproved: "Case {caseId} approved.",
    caseRejected: "Case {caseId} rejected.",
    rulePublished: "Rule version {versionTag} published.",
    realtimeMode: "Realtime Mode",
    realtimeModeCopy: "Live monitoring, reviews, and rule publishing.",
    testMode: "Test Mode",
    testModeCopy: "Simulated streams, latency, and token metrics.",
    testModeTitle: "Simulation Workspace",
    testModeCopyFull: "Simulation runs do not write into the production risk case or alert tables. Use this space for scenario validation and performance checks.",
    scenarioLibrary: "Scenario Library",
    runScenario: "Run Scenario",
    runningScenario: "Running scenario...",
    noScenarioResult: "Select a scenario to run it and inspect the output.",
    latestRun: "Latest Run",
    passRate: "Pass Rate",
    totalLatency: "Average Total Latency",
    maxLatency: "Max Total Latency",
    llmTokens: "LLM Total Tokens",
    llmCalls: "LLM Calls",
    alertsEmitted: "Alerts Emitted",
    manualReviews: "Manual Reviews",
    p1Steps: "P1 Steps",
    p2Steps: "P2 Steps",
    promptTokens: "Prompt Tokens",
    completionTokens: "Completion Tokens",
    usageUnavailable: "The provider did not return usage, so token metrics may stay at 0.",
    scenarioExpectation: "Expected Output",
    scenarioActual: "Actual Output",
    correctnessPassed: "Matched expectation",
    correctnessFailed: "Deviation detected",
    decision: "Decision",
    severity: "Severity",
    rules: "Rules",
    stageMetrics: "Stage Latencies",
    summary: "Summary",
    reviewGuidance: "Review Guidance",
    historicalContext: "Historical Context",
    riskQuantification: "Risk Quantification",
    checkpointTime: "Checkpoint Time",
    injectedEvents: "Injected Events",
    snapshotLatency: "Snapshot",
    ruleLatency: "Rules",
    expertLatency: "Experts",
    summarizerLatency: "Summarizer",
    reviewLatency: "Review Helpers",
    totalLatencyLabel: "Total",
    loadMemoryLatency: "Memory Load",
    actualRules: "Actual Rules",
    expectedRules: "Expected Rules",
    noScenarios: "No simulation scenarios are available.",
    selectedScenario: "Selected Scenario",
    viewReport: "Test Report",
    outputOperations: "Operations",
    status: "Status",
    idle: "Idle",
    completed: "Completed",
    viewMode: "View Mode",
    evaluationMode: "Evaluation",
    evaluationModeCopy: "Runtime rollup and resume-ready results.",
    evaluationTitle: "Runtime Evaluation Summary",
    evaluationCopyFull: "Aggregated directly from persisted raw events, snapshots, cases, alerts, and human review actions.",
    evaluationWindow: "Window",
    evaluationRefresh: "Refresh Summary",
    evaluationGeneratedAt: "Generated At",
    evaluationCoverageHours: "Coverage Hours",
    evaluationAllTime: "All Time",
    evaluation7d: "Last 7 Days",
    evaluation30d: "Last 30 Days",
    runtimeSample: "Runtime Samples",
    runtimeCoverage: "Coverage",
    runtimeAssets: "Asset Coverage",
    runtimeResume: "Resume Summary",
    runtimeHighlights: "Resume Highlights",
    runtimeTopRules: "Top Rules",
    runtimeAssetBreakdown: "Asset Breakdown",
    runtimeCaseMetrics: "Case Metrics",
    runtimeLatency: "Latency Metrics",
    runtimeQuality: "Quality Metrics",
    coreEightMetrics: "Eight Core Metrics",
    earlyWarningRecall: "Early Warning Recall",
    earlyWarningPrecision: "Early Warning Precision",
    earlyWarningLeadTime: "Average Lead Time",
    earlyWarningConversion: "Early Warning Conversion",
    formalAlertRecall: "Formal Alert Recall",
    formalAlertFalsePositive: "Formal Alert False Positive",
    p95AlertLatency: "P95 Alert Latency",
    earlyWarningCases: "Early Warning Cases",
    earlyWarningPending: "Pending Early Warnings",
    earlyWarningPolicy: "Early Warning Policy",
    earlyWarningBestScore: "Recommended Tune Score",
    earlyWarningBestF1: "Recommended F1",
    earlyWarningBestConfig: "Recommended Config",
    runtimeReview: "Human Review",
    rawEvents: "Raw Events",
    featureSnapshots: "Feature Snapshots",
    riskCases: "Risk Cases",
    riskAlerts: "Risk Alerts",
    humanReviews: "Review Actions",
    alertEmissionRate: "Alert Conversion",
    manualReviewRate: "Manual Review Rate",
    approveRate: "Approval Rate",
    rejectRate: "Reject Rate",
    memoryEnrichmentRate: "Memory Enrichment",
    avgCaseToAlert: "Avg case-to-alert",
    avgReviewTurnaround: "Avg review turnaround",
    avgIngestLag: "Avg ingest lag",
    p95IngestLag: "P95 ingest lag",
    conflictRate: "Conflict Rate",
    staleRate: "Stale Rate",
    falsePositiveProxy: "False Positive Proxy",
    missedAlertProxy: "Missed Alert Proxy",
    precisionProxy: "Precision Proxy",
    recallProxy: "Recall Proxy",
    offlineRecall: "Offline Recall",
    offlinePrecision: "Offline Precision",
    offlineMissRate: "Offline Miss Rate",
    offlineEpisodes: "Offline Episodes",
    dedupeRate: "Dedupe Rate",
    alertDedupeRate: "Alert Dedupe Rate",
    duplicateSuppressed: "Duplicates Blocked",
    p2Aggregated: "P2 Aggregated",
    eventsPerSecond: "Events/sec",
    snapshotsPerSecond: "Snapshots/sec",
    llmCost: "LLM Cost",
    llmCostPerAlert: "Cost/Alert",
    llmCostConfigMissing: "Cost rates not configured",
    baselineAlertReduction: "Noise Reduction vs Rules",
    preventedFalsePositiveProxy: "Prevented FP Proxy",
    pendingReviewCases: "Pending Reviews",
    coordinatorCases: "Coordinator Cases",
    topRuleCount: "Hits",
    noEvaluationData: "There is not enough persisted runtime data yet.",
    riskModel: "LightGBM Risk Model",
    riskModelStatus: "Model Status",
    riskModelAvailable: "Available",
    riskModelUnavailable: "Model has not been trained yet",
    riskModelVersion: "Model Version",
    riskModelLabels: "Weak Labels",
    riskModelTrainSamples: "Training Samples",
    riskModelTestSamples: "Validation Samples",
    riskModelPrecision: "Model Precision",
    riskModelRecall: "Model Recall",
    riskModelF1: "Model F1",
    riskModelHorizon: "Prediction Horizon",
    riskModelThresholds: "Level Thresholds",
    riskModelTopFeatures: "Top Features",
  },
};

function getInitialLanguage() {
  const saved = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
  if (saved === "zh" || saved === "en") {
    return saved;
  }
  return navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
}

function severityClass(severity) {
  return severity ? `severity-${severity.toLowerCase()}` : "severity-p3";
}

function interpolate(template, values = {}) {
  return Object.entries(values).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, String(value)),
    template,
  );
}

function formatNumber(value, digits = 2, locale = "en-US") {
  return Number(value || 0).toLocaleString(locale, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatPercent(value, locale, digits = 1) {
  return Number(value || 0).toLocaleString(locale, {
    style: "percent",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function parseBackendDate(isoValue) {
  if (!isoValue) {
    return null;
  }
  if (isoValue instanceof Date) {
    return isoValue;
  }
  const value = String(isoValue);
  const hasExplicitTimezone = /(?:z|[+-]\d{2}:?\d{2})$/i.test(value);
  return new Date(value.includes("T") && !hasExplicitTimezone ? `${value}Z` : value);
}

function formatTimeLabel(isoValue, locale) {
  const date = parseBackendDate(isoValue);
  if (!date || Number.isNaN(date.getTime())) {
    return "-";
  }
  return date.toLocaleTimeString(locale, {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: DISPLAY_TIME_ZONE,
  });
}

function formatDateTime(isoValue, locale) {
  const date = parseBackendDate(isoValue);
  if (!date || Number.isNaN(date.getTime())) {
    return "-";
  }
  return `${date.toLocaleString(locale, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: DISPLAY_TIME_ZONE,
  })} ${DISPLAY_TIME_SUFFIX}`;
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_PREFIX}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || response.statusText);
  }
  return response.json();
}

function DetailRow({ label, value }) {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const rendered = typeof value === "object" ? JSON.stringify(value, null, 2) : value;
  return (
    <div className="detail-row">
      <div className="detail-label">{label}</div>
      <pre className="detail-value">{rendered}</pre>
    </div>
  );
}

function MetricCard({ label, value, muted }) {
  return (
    <div className={`chart-stat metric-card ${muted ? "metric-muted" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function CandleChart({ asset, candles, locale, t }) {
  const [selectedOpenTime, setSelectedOpenTime] = useState(null);

  useEffect(() => {
    if (!candles.length) {
      setSelectedOpenTime(null);
      return;
    }
    setSelectedOpenTime((current) => current || candles[candles.length - 1].open_time);
  }, [candles]);

  if (!candles.length) {
    return <p className="muted">{t("waitingSnapshots")}</p>;
  }

  const maxPrice = Math.max(...candles.map((item) => item.high));
  const minPrice = Math.min(...candles.map((item) => item.low));
  const range = Math.max(maxPrice - minPrice, maxPrice * 0.004, 1);
  const topPad = 14;
  const bottomPad = 26;
  const leftPad = 64;
  const axisPad = 12;
  const slot = 24;
  const gap = 10;
  const bodyWidth = 12;
  const width = leftPad * 2 + candles.length * (slot + gap);
  const height = 260;
  const plotHeight = height - topPad - bottomPad;
  const labelEvery = Math.max(1, Math.floor(candles.length / 6));
  const latest = candles[candles.length - 1];
  const selectedCandle = candles.find((item) => item.open_time === selectedOpenTime) || latest;

  const priceToY = (price) => {
    const normalized = (price - (minPrice - range * 0.08)) / (range * 1.16);
    return height - bottomPad - normalized * plotHeight;
  };

  const axisLevels = [maxPrice, maxPrice - range / 3, minPrice + range / 3, minPrice];

  return (
    <div className="chart-wrap">
      <div className="chart-toolbar">
        <div>
          <p className="eyebrow">{t("marketCandles")}</p>
          <h2>{asset} {t("candleTitleSuffix")}</h2>
        </div>
        <div className="chart-badges">
          <span className="chart-badge">{t("sync30s")}</span>
          <span className={`chart-badge ${latest.is_closed ? "closed" : "live"}`}>
            {latest.is_closed ? t("closedCandle") : t("liveCandle")}
          </span>
        </div>
      </div>

      <div className="chart-surface">
        <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-label={`${asset} candlestick chart`}>
          <text x={10} y={18} className="chart-axis-title">{t("chartPriceAxis")}</text>
          {[0.2, 0.4, 0.6, 0.8].map((ratio) => {
            const y = topPad + plotHeight * ratio;
            return <line key={ratio} x1={leftPad - axisPad} x2={width - leftPad / 2} y1={y} y2={y} className="chart-grid-line" />;
          })}
          {axisLevels.map((priceValue, index) => {
            const y = priceToY(priceValue);
            return (
              <g key={`${priceValue}-${index}`}>
                <text x={10} y={y + 4} className="chart-axis-price">
                  {formatNumber(priceValue, 3, locale)}
                </text>
                <line x1={leftPad - axisPad} x2={leftPad - 2} y1={y} y2={y} className="chart-axis-tick" />
              </g>
            );
          })}
          {candles.map((candle, index) => {
            const rising = candle.close >= candle.open;
            const x = leftPad + index * (slot + gap) + gap / 2;
            const centerX = x + slot / 2;
            const wickTop = priceToY(candle.high);
            const wickBottom = priceToY(candle.low);
            const openY = priceToY(candle.open);
            const closeY = priceToY(candle.close);
            const bodyY = Math.min(openY, closeY);
            const bodyHeight = Math.max(2, Math.abs(closeY - openY));
            const showLabel = index % labelEvery === 0 || index === candles.length - 1;
            return (
              <g
                key={candle.open_time}
                className={`candle-group ${selectedCandle.open_time === candle.open_time ? "selected" : ""}`}
                onClick={() => setSelectedOpenTime(candle.open_time)}
                role="button"
                tabIndex={0}
                aria-label={`${asset} candle ${formatDateTime(candle.open_time, locale)}`}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setSelectedOpenTime(candle.open_time);
                  }
                }}
              >
                <rect x={x} y={topPad} width={slot} height={plotHeight} rx="8" className="candle-hitbox" />
                <line x1={centerX} x2={centerX} y1={wickTop} y2={wickBottom} className={rising ? "candle-wick up" : "candle-wick down"} />
                <rect
                  x={centerX - bodyWidth / 2}
                  y={bodyY}
                  width={bodyWidth}
                  height={bodyHeight}
                  rx="3"
                  className={rising ? "candle-body up" : "candle-body down"}
                />
                {showLabel ? (
                  <text x={centerX} y={height - 8} textAnchor="middle" className="chart-axis-label">
                    {formatTimeLabel(candle.open_time, locale)}
                  </text>
                ) : null}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="inline-note">{t("clickCandleHint")}</div>

      <div className="chart-summary">
        <MetricCard label={t("lastClose")} value={formatNumber(latest.close, 3, locale)} />
        <MetricCard label={t("window")} value={`${candles.length} ${t("minutes")}`} />
        <MetricCard label={t("updated")} value={formatTimeLabel(latest.close_time, locale)} />
      </div>

      <div className="candle-detail-card">
        <div className="panel-header compact-header">
          <div>
            <h3>{t("selectedCandle")}</h3>
            <p className="muted">{asset} {formatDateTime(selectedCandle.open_time, locale)}</p>
          </div>
          <span className={`chart-badge ${selectedCandle.is_closed ? "closed" : "live"}`}>
            {selectedCandle.is_closed ? t("closedCandle") : t("liveCandle")}
          </span>
        </div>
        <div className="candle-detail-grid">
          <MetricCard label={t("open")} value={formatNumber(selectedCandle.open, 3, locale)} />
          <MetricCard label={t("high")} value={formatNumber(selectedCandle.high, 3, locale)} />
          <MetricCard label={t("low")} value={formatNumber(selectedCandle.low, 3, locale)} />
          <MetricCard label={t("close")} value={formatNumber(selectedCandle.close, 3, locale)} />
          <MetricCard
            label={t("candleRange")}
            value={`${formatTimeLabel(selectedCandle.open_time, locale)} - ${formatTimeLabel(selectedCandle.close_time, locale)}`}
          />
          <MetricCard label={t("snapshots")} value={selectedCandle.snapshot_count} />
        </div>
      </div>
    </div>
  );
}

function RealtimeWorkspace({
  activeRules,
  alerts,
  alertSocketState,
  candles,
  cases,
  casePage,
  casePagination,
  expandedCaseId,
  goToCasePage,
  handleAgentAction,
  handlePublish,
  handleResume,
  health,
  locale,
  message,
  reviewForm,
  ruleForm,
  selectedAsset,
  setExpandedCaseId,
  setReviewForm,
  setRuleForm,
  setSelectedAsset,
  t,
}) {
  const caseTotal = casePagination.total || 0;
  const caseStart = caseTotal === 0 ? 0 : (casePagination.offset || 0) + 1;
  const caseEnd = Math.min((casePagination.offset || 0) + cases.length, caseTotal);
  const hasPreviousCasePage = casePage > 0;
  const hasNextCasePage = Boolean(casePagination.has_more);

  return (
    <>
      {message ? <div className="banner">{message}</div> : null}

      <section className="grid top-grid">
        <div className="panel panel-chart">
          <div className="panel-header">
            <div>
              <h2>{t("priceCandles")}</h2>
              <p className="muted">{t("chartDerivedFromSnapshots")}</p>
            </div>
            <div className="asset-switcher compact">
              {SUPPORTED_ASSETS.map((asset) => (
                <button
                  key={asset}
                  type="button"
                  className={`ghost ${selectedAsset === asset ? "active-tab" : ""}`}
                  onClick={() => setSelectedAsset(asset)}
                >
                  {asset}
                </button>
              ))}
            </div>
          </div>
          <CandleChart asset={selectedAsset} candles={candles} locale={locale} t={t} />
        </div>

        <div className="panel stack-panel">
          <div className="status-card embedded-card">
            <div className="status-line">
              <span className={`status-dot ${health.agent?.running ? "running" : "stopped"}`} />
              <strong>{health.agent?.running ? t("running") : t("stopped")}</strong>
            </div>
            <div className="status-actions">
              <button type="button" onClick={() => handleAgentAction("/agent/start")}>{t("start")}</button>
              <button className="ghost" type="button" onClick={() => handleAgentAction("/agent/stop")}>{t("stop")}</button>
            </div>
            <p className="muted">{t("health")}: {health.status}</p>
            <p className="muted">{t("refreshCadence")}</p>
          </div>

          <div className="panel-subsection">
            <div className="panel-header">
              <h2>{t("dataFreshness")}</h2>
            </div>
            <div className="asset-grid">
              {Object.entries(health.assets || {}).map(([asset, info]) => (
                <button
                  key={asset}
                  type="button"
                  className={`asset-card asset-button ${selectedAsset === asset ? "asset-selected" : ""}`}
                  onClick={() => setSelectedAsset(asset)}
                >
                  <div className="asset-head">
                    <strong>{asset}</strong>
                    <span className={info.ok ? "asset-ok" : "asset-stale"}>{info.ok ? t("fresh") : t("stale")}</span>
                  </div>
                  <div className="asset-metric">{t("price")}: {formatNumber(info.price, 3, locale)}</div>
                  <div className="asset-metric">{t("age")}: {info.age_seconds}s</div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="grid bottom-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t("liveAlerts")}</h2>
          </div>
          {alertSocketState === "connecting" ? <div className="inline-note">{t("alertsConnecting")}</div> : null}
          {alertSocketState === "error" ? <div className="inline-note warning">{t("alertsDisconnected")}</div> : null}
          <div className="alert-list">
            {alerts.length === 0 ? <p className="muted">{t("waitingAlerts")}</p> : null}
            {alerts.map((alert) => (
              <article key={alert.alert_id} className="alert-card">
                <div className="alert-head">
                  <span className={`severity-pill ${severityClass(alert.severity)}`}>{alert.severity}</span>
                  <span className="muted">{formatDateTime(alert.created_at, locale)}</span>
                </div>
                <h3>{alert.title}</h3>
                <p>{alert.body_zh}</p>
              </article>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <h2>{t("pendingReviews")}</h2>
              <p className="muted">
                {caseTotal === 0
                  ? t("noCasePage")
                  : t("casePageStatus", { start: caseStart, end: caseEnd, total: caseTotal })}
              </p>
            </div>
            <div className="pagination-actions">
              <button
                className="ghost"
                type="button"
                disabled={!hasPreviousCasePage}
                onClick={() => goToCasePage(casePage - 1)}
              >
                {t("previousPage")}
              </button>
              <button
                className="ghost"
                type="button"
                disabled={!hasNextCasePage}
                onClick={() => goToCasePage(casePage + 1)}
              >
                {t("nextPage")}
              </button>
            </div>
          </div>
          <div className="case-list">
            {cases.length === 0 ? <p className="muted">{t("noPendingCases")}</p> : null}
            {cases.map((item) => {
              const formState = reviewForm[item.case_id] || { reviewer: "operator", comment: "" };
              const expanded = expandedCaseId === item.case_id;
              return (
                <article key={item.case_id} className="case-card">
                  <div className="case-summary">
                    <div>
                      <span className={`severity-pill ${severityClass(item.severity)}`}>{item.severity}</span>
                      <strong>{item.is_coordinator_case ? t("crossAssetCase") : `${item.asset} ${t("caseSuffix")}`}</strong>
                      <span className="muted case-time">{t("createdAt")}: {formatDateTime(item.created_at, locale)}</span>
                    </div>
                    <button className="ghost" type="button" onClick={() => setExpandedCaseId(expanded ? "" : item.case_id)}>
                      {expanded ? t("hide") : t("show")}
                    </button>
                  </div>
                  <p>{item.summary_zh || t("noSummary")}</p>
                  {expanded ? (
                    <div className="case-details">
                      <DetailRow label="review_guidance" value={item.review_guidance} />
                      <DetailRow label="historical_context_zh" value={item.historical_context_zh} />
                      <DetailRow label="risk_quantification_zh" value={item.risk_quantification_zh} />
                      <DetailRow label="rule_hits" value={item.rule_hits} />
                      <div className="review-form">
                        <input
                          value={formState.reviewer}
                          onChange={(event) =>
                            setReviewForm((current) => ({
                              ...current,
                              [item.case_id]: { ...formState, reviewer: event.target.value },
                            }))
                          }
                          placeholder={t("reviewer")}
                        />
                        <textarea
                          value={formState.comment}
                          onChange={(event) =>
                            setReviewForm((current) => ({
                              ...current,
                              [item.case_id]: { ...formState, comment: event.target.value },
                            }))
                          }
                          placeholder={t("reviewNotes")}
                        />
                        <div className="status-actions">
                          <button type="button" onClick={() => handleResume(item.case_id, "approve")}>{t("approve")}</button>
                          <button className="ghost" type="button" onClick={() => handleResume(item.case_id, "reject")}>{t("reject")}</button>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </article>
              );
            })}
          </div>
        </div>
      </section>

      <section className="grid single-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>{t("rulePublishing")}</h2>
          </div>
          <div className="rule-summary">
            <div>
              <p className="muted">{t("activeVersion")}</p>
              <strong>{activeRules.version_tag || "-"}</strong>
            </div>
            <pre>{JSON.stringify(activeRules.thresholds || {}, null, 2)}</pre>
          </div>

          <form className="rule-form" onSubmit={handlePublish}>
            <input
              placeholder="version_tag"
              value={ruleForm.version_tag}
              onChange={(event) => setRuleForm((current) => ({ ...current, version_tag: event.target.value }))}
              required
            />
            <input
              placeholder="operator"
              value={ruleForm.operator}
              onChange={(event) => setRuleForm((current) => ({ ...current, operator: event.target.value }))}
              required
            />
            <textarea
              placeholder="reason"
              value={ruleForm.reason}
              onChange={(event) => setRuleForm((current) => ({ ...current, reason: event.target.value }))}
            />
            {[
              "price_change_p1",
              "price_change_p2",
              "oi_delta_p2",
              "liq_usd_p1",
              "funding_z_p2",
              "early_warning_ret_5m",
              "early_warning_oi_delta",
              "early_warning_funding_z",
              "early_warning_vol_z",
              "early_warning_min_score",
              "early_warning_min_signals",
              "early_warning_single_signal_min_score",
              "early_warning_persistence_window",
              "early_warning_persistence_hits",
              "early_warning_dynamic_baseline",
              "early_warning_dynamic_history",
              "early_warning_dynamic_quantile",
              "early_warning_trend_window",
              "early_warning_min_trend_hits",
              "vol_z_spike",
              "cross_source_conflict_pct",
            ].map((field) => (
              <input
                key={field}
                placeholder={field}
                value={ruleForm[field]}
                onChange={(event) => setRuleForm((current) => ({ ...current, [field]: event.target.value }))}
              />
            ))}
            <button type="submit">{t("publishVersion")}</button>
          </form>
        </div>
      </section>
    </>
  );
}

function SimulationStepCard({ locale, step, t }) {
  const passed = step.correctness_passed;
  return (
    <article className="panel simulation-step-card">
      <div className="panel-header">
        <div>
          <p className="eyebrow">{step.checkpoint_id}</p>
          <h3>{step.asset} · {formatDateTime(step.checkpoint_time, locale)}</h3>
          <p className="muted">{step.description}</p>
        </div>
        <span className={`severity-pill ${passed ? "severity-ok" : "severity-p2"}`}>
          {passed ? t("correctnessPassed") : t("correctnessFailed")}
        </span>
      </div>

      <div className="simulation-grid">
        <div className="simulation-column">
          <h4>{t("scenarioExpectation")}</h4>
          <DetailRow label={t("severity")} value={step.expected.severity} />
          <DetailRow label={t("decision")} value={step.expected.decision} />
          <DetailRow label={t("expectedRules")} value={step.expected.rule_ids} />
        </div>
        <div className="simulation-column">
          <h4>{t("scenarioActual")}</h4>
          <DetailRow label={t("severity")} value={step.actual_severity} />
          <DetailRow label={t("decision")} value={step.actual_decision} />
          <DetailRow label={t("actualRules")} value={step.actual_rule_ids} />
          <DetailRow label={t("summary")} value={step.summary_zh} />
          <DetailRow label={t("reviewGuidance")} value={step.review_guidance} />
          <DetailRow label={t("historicalContext")} value={step.historical_context_zh} />
          <DetailRow label={t("riskQuantification")} value={step.risk_quantification_zh} />
          <DetailRow label="correctness_notes" value={step.correctness_notes} />
        </div>
      </div>

      <div className="simulation-metrics-grid">
        <MetricCard label={t("checkpointTime")} value={formatDateTime(step.checkpoint_time, locale)} />
        <MetricCard label={t("injectedEvents")} value={step.stage_metrics.injected_events} />
        <MetricCard label={t("loadMemoryLatency")} value={`${formatNumber(step.stage_metrics.load_memory_ms, 1, locale)} ms`} />
        <MetricCard label={t("snapshotLatency")} value={`${formatNumber(step.stage_metrics.snapshot_ms, 1, locale)} ms`} />
        <MetricCard label={t("ruleLatency")} value={`${formatNumber(step.stage_metrics.rule_eval_ms, 1, locale)} ms`} />
        <MetricCard label={t("expertLatency")} value={`${formatNumber(step.stage_metrics.expert_ms, 1, locale)} ms`} />
        <MetricCard label={t("summarizerLatency")} value={`${formatNumber(step.stage_metrics.summarizer_ms, 1, locale)} ms`} />
        <MetricCard label={t("reviewLatency")} value={`${formatNumber(step.stage_metrics.review_helpers_ms, 1, locale)} ms`} />
        <MetricCard label={t("totalLatencyLabel")} value={`${formatNumber(step.stage_metrics.total_ms, 1, locale)} ms`} />
        <MetricCard label={t("llmCalls")} value={step.llm_metrics.calls} />
        <MetricCard label={t("promptTokens")} value={step.llm_metrics.prompt_tokens} />
        <MetricCard label={t("completionTokens")} value={step.llm_metrics.completion_tokens} />
      </div>

      <DetailRow label={t("outputOperations")} value={step.llm_metrics.operations} />
    </article>
  );
}

function TestWorkspace({
  latestSimulation,
  locale,
  selectedScenario,
  selectedScenarioId,
  setSelectedScenarioId,
  simulationRunning,
  simulationScenarios,
  t,
  runSimulation,
}) {
  return (
    <div className="test-workspace">
      <div className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">{t("testMode")}</p>
            <h2>{t("testModeTitle")}</h2>
          </div>
        </div>
        <p className="muted">{t("testModeCopyFull")}</p>
      </div>

      <div className="grid test-grid">
        <aside className="panel test-sidebar">
          <div className="panel-header">
            <h2>{t("scenarioLibrary")}</h2>
          </div>
          <div className="scenario-list">
            {simulationScenarios.length === 0 ? <p className="muted">{t("noScenarios")}</p> : null}
            {simulationScenarios.map((scenario) => (
              <button
                key={scenario.scenario_id}
                type="button"
                className={`scenario-button ${selectedScenarioId === scenario.scenario_id ? "active-tab" : "ghost"}`}
                onClick={() => setSelectedScenarioId(scenario.scenario_id)}
              >
                <strong>{locale === "zh-CN" ? scenario.title_zh : scenario.title}</strong>
                <span>{locale === "zh-CN" ? scenario.description_zh : scenario.description}</span>
              </button>
            ))}
          </div>
          <div className="scenario-runner">
            <p className="muted">
              {t("selectedScenario")}: {selectedScenario ? (locale === "zh-CN" ? selectedScenario.title_zh : selectedScenario.title) : "-"}
            </p>
            <button type="button" onClick={runSimulation} disabled={!selectedScenarioId || simulationRunning}>
              {simulationRunning ? t("runningScenario") : t("runScenario")}
            </button>
          </div>
        </aside>

        <div className="test-main">
          {!latestSimulation ? (
            <div className="panel">
              <p className="muted">{t("noScenarioResult")}</p>
            </div>
          ) : (
            <>
              <div className="panel">
                <div className="panel-header">
                  <div>
                    <p className="eyebrow">{t("latestRun")}</p>
                    <h2>{locale === "zh-CN" ? latestSimulation.title_zh : latestSimulation.title}</h2>
                  </div>
                  <span className="chart-badge closed">{t(latestSimulation.status === "completed" ? "completed" : "idle")}</span>
                </div>
                <div className="simulation-metrics-grid">
                  <MetricCard label={t("passRate")} value={`${formatNumber(latestSimulation.summary.pass_rate * 100, 1, locale)}%`} />
                  <MetricCard label={t("totalLatency")} value={`${formatNumber(latestSimulation.summary.avg_total_latency_ms, 1, locale)} ms`} />
                  <MetricCard label={t("maxLatency")} value={`${formatNumber(latestSimulation.summary.max_total_latency_ms, 1, locale)} ms`} />
                  <MetricCard label={t("llmTokens")} value={latestSimulation.summary.llm_total_tokens} />
                  <MetricCard label={t("llmCalls")} value={latestSimulation.summary.llm_calls} />
                  <MetricCard label={t("alertsEmitted")} value={latestSimulation.summary.emitted_alerts} />
                  <MetricCard label={t("manualReviews")} value={latestSimulation.summary.manual_reviews} />
                  <MetricCard label={t("p1Steps")} value={latestSimulation.summary.p1_steps} />
                  <MetricCard label={t("p2Steps")} value={latestSimulation.summary.p2_steps} />
                  <MetricCard label={t("promptTokens")} value={latestSimulation.summary.llm_prompt_tokens} />
                  <MetricCard label={t("completionTokens")} value={latestSimulation.summary.llm_completion_tokens} />
                  <MetricCard label={t("status")} value={latestSimulation.status} />
                </div>
                {!latestSimulation.summary.llm_usage_available ? (
                  <div className="inline-note warning">{t("usageUnavailable")}</div>
                ) : null}
              </div>

              <div className="simulation-steps">
                {latestSimulation.steps.map((step) => (
                  <SimulationStepCard key={`${latestSimulation.run_id}-${step.checkpoint_id}`} locale={locale} step={step} t={t} />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function EvaluationWorkspace({
  evaluationSummary,
  evaluationWindowKey,
  locale,
  refreshEvaluationSummary,
  riskModelStatus,
  setEvaluationWindowKey,
  t,
}) {
  const qualityMetrics = evaluationSummary?.quality_metrics || {};
  const throughputMetrics = evaluationSummary?.throughput_metrics || {};
  const llmCostMetrics = evaluationSummary?.llm_cost_metrics || {};
  const baselineComparison = evaluationSummary?.baseline_comparison || {};
  const offlineEvaluation = evaluationSummary?.offline_evaluation || {};
  const offlineRiskDetection = offlineEvaluation?.risk_detection_policy || {};
  const coreMetrics = evaluationSummary?.core_quality_metrics || {};
  const earlyWarningMetrics = evaluationSummary?.early_warning_metrics || {};
  const earlyWarningBestConfig = earlyWarningMetrics?.best_offline_config || null;
  const riskModelMetrics = riskModelStatus?.metrics || {};
  const optionalPercent = (value) => (value == null ? "N/A" : formatPercent(value, locale));
  const optionalUsd = (value) => (value == null ? "N/A" : `$${formatNumber(value, 4, locale)}`);
  const optionalSeconds = (value) => (value == null ? "N/A" : `${formatNumber(value, 1, locale)} s`);

  return (
    <div className="test-workspace">
      <div className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">{t("evaluationMode")}</p>
            <h2>{t("evaluationTitle")}</h2>
          </div>
          <button type="button" className="ghost" onClick={() => refreshEvaluationSummary()}>
            {t("evaluationRefresh")}
          </button>
        </div>
        <p className="muted">{t("evaluationCopyFull")}</p>
        <div className="asset-switcher compact">
          {EVALUATION_WINDOWS.map((windowOption) => (
            <button
              key={windowOption.key}
              type="button"
              className={`ghost ${evaluationWindowKey === windowOption.key ? "active-tab" : ""}`}
              onClick={() => setEvaluationWindowKey(windowOption.key)}
            >
              {t(`evaluation${windowOption.key === "all" ? "AllTime" : windowOption.key}`)}
            </button>
          ))}
        </div>
      </div>

      {!evaluationSummary || evaluationSummary.sample_sizes.raw_events === 0 ? (
        <div className="panel">
          <p className="muted">{t("noEvaluationData")}</p>
        </div>
      ) : (
        <>
          <section className="grid single-grid">
            <div className="panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">{t("runtimeSample")}</p>
                  <h2>{t("runtimeResume")}</h2>
                </div>
              </div>
              <div className="simulation-metrics-grid">
                <MetricCard label={t("rawEvents")} value={evaluationSummary.sample_sizes.raw_events} />
                <MetricCard label={t("featureSnapshots")} value={evaluationSummary.sample_sizes.feature_snapshots} />
                <MetricCard label={t("riskCases")} value={evaluationSummary.sample_sizes.risk_cases} />
                <MetricCard label={t("riskAlerts")} value={evaluationSummary.sample_sizes.risk_alerts} />
                <MetricCard label={t("humanReviews")} value={evaluationSummary.sample_sizes.human_reviews} />
                <MetricCard label={t("runtimeCoverage")} value={`${formatNumber(evaluationSummary.window.coverage_hours, 1, locale)} h`} />
                <MetricCard label={t("evaluationGeneratedAt")} value={formatDateTime(evaluationSummary.generated_at, locale)} />
                <MetricCard label={t("runtimeAssets")} value={evaluationSummary.coverage.assets.join(", ")} />
              </div>
              <div className="panel-subsection">
                <h3>{t("runtimeHighlights")}</h3>
                <div className="detail-value">
                  <ul className="flat-list">
                    {(locale === "zh-CN"
                      ? evaluationSummary.resume_ready.highlights_zh
                      : evaluationSummary.resume_ready.highlights_en
                    ).map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </section>

          <section className="grid single-grid">
            <div className="panel">
              <div className="panel-header">
                <h2>{t("coreEightMetrics")}</h2>
              </div>
              <div className="simulation-metrics-grid">
                <MetricCard label={t("earlyWarningRecall")} value={optionalPercent(coreMetrics.early_warning_recall)} />
                <MetricCard label={t("earlyWarningPrecision")} value={optionalPercent(coreMetrics.early_warning_precision)} />
                <MetricCard label={t("earlyWarningLeadTime")} value={optionalSeconds(coreMetrics.early_warning_avg_lead_time_seconds)} />
                <MetricCard label={t("earlyWarningConversion")} value={optionalPercent(coreMetrics.early_warning_conversion_rate)} />
                <MetricCard label={t("formalAlertRecall")} value={optionalPercent(coreMetrics.formal_alert_recall)} />
                <MetricCard label={t("formalAlertFalsePositive")} value={optionalPercent(coreMetrics.formal_alert_false_positive_rate)} />
                <MetricCard label={t("p95AlertLatency")} value={optionalSeconds(coreMetrics.p95_alert_latency_seconds)} />
                <MetricCard label={t("pendingReviewCases")} value={coreMetrics.pending_review_cases ?? 0} />
              </div>
            </div>
          </section>

          <section className="grid single-grid">
            <div className="panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">{t("riskModelStatus")}</p>
                  <h2>{t("riskModel")}</h2>
                </div>
              </div>
              <div className="simulation-metrics-grid">
                <MetricCard label={t("riskModelStatus")} value={riskModelStatus?.available ? t("riskModelAvailable") : t("riskModelUnavailable")} />
                <MetricCard label={t("riskModelVersion")} value={riskModelStatus?.model_version || "-"} />
                <MetricCard label={t("riskModelLabels")} value={riskModelStatus?.label_count || 0} />
                <MetricCard label={t("riskModelTrainSamples")} value={riskModelStatus?.training_samples || 0} />
                <MetricCard label={t("riskModelTestSamples")} value={riskModelStatus?.test_samples || 0} />
                <MetricCard label={t("riskModelPrecision")} value={optionalPercent(riskModelMetrics.precision)} />
                <MetricCard label={t("riskModelRecall")} value={optionalPercent(riskModelMetrics.recall)} />
                <MetricCard label={t("riskModelF1")} value={optionalPercent(riskModelMetrics.f1)} />
                <MetricCard label={t("riskModelHorizon")} value={optionalSeconds(riskModelStatus?.horizon_seconds)} />
              </div>
              <DetailRow label={t("riskModelThresholds")} value={riskModelStatus?.thresholds || {}} />
              <DetailRow label={t("riskModelTopFeatures")} value={riskModelStatus?.feature_importances || []} />
            </div>
          </section>

          <section className="grid bottom-grid">
            <div className="panel">
              <div className="panel-header">
                <h2>{t("runtimeCaseMetrics")}</h2>
              </div>
              <div className="simulation-metrics-grid">
                <MetricCard label={t("alertEmissionRate")} value={formatPercent(evaluationSummary.case_metrics.emitted_alert_rate, locale)} />
                <MetricCard label={t("manualReviewRate")} value={formatPercent(evaluationSummary.case_metrics.manual_review_rate, locale)} />
                <MetricCard label={t("memoryEnrichmentRate")} value={formatPercent(evaluationSummary.case_metrics.memory_enrichment_rate, locale)} />
                <MetricCard label={t("pendingReviewCases")} value={evaluationSummary.case_metrics.pending_review_cases} />
                <MetricCard label={t("coordinatorCases")} value={evaluationSummary.case_metrics.coordinator_cases} />
                <MetricCard label={t("earlyWarningCases")} value={earlyWarningMetrics.cases || 0} />
                <MetricCard label={t("earlyWarningPending")} value={earlyWarningMetrics.pending_cases || 0} />
                <MetricCard label={t("earlyWarningPolicy")} value={earlyWarningMetrics.runtime_policy || "-"} />
                <MetricCard label={t("earlyWarningBestScore")} value={earlyWarningBestConfig ? formatNumber(earlyWarningBestConfig.score, 3, locale) : "N/A"} />
                <MetricCard label={t("earlyWarningBestF1")} value={earlyWarningBestConfig?.f1 == null ? "N/A" : formatPercent(earlyWarningBestConfig.f1, locale)} />
                <MetricCard label="P1" value={evaluationSummary.case_metrics.severity_breakdown.P1 || 0} />
                <MetricCard label="P2" value={evaluationSummary.case_metrics.severity_breakdown.P2 || 0} />
                <MetricCard label="P3" value={evaluationSummary.case_metrics.severity_breakdown.P3 || 0} />
              </div>
              <DetailRow label={t("earlyWarningBestConfig")} value={earlyWarningBestConfig?.thresholds || {}} />
              <DetailRow label={t("decision")} value={evaluationSummary.case_metrics.decision_breakdown} />
              <DetailRow label={t("status")} value={evaluationSummary.case_metrics.status_breakdown} />
            </div>

            <div className="panel">
              <div className="panel-header">
                <h2>{t("runtimeReview")}</h2>
              </div>
              <div className="simulation-metrics-grid">
                <MetricCard label={t("approveRate")} value={formatPercent(evaluationSummary.human_review.approve_rate, locale)} />
                <MetricCard label={t("rejectRate")} value={formatPercent(evaluationSummary.human_review.reject_rate, locale)} />
                <MetricCard label={t("avgReviewTurnaround")} value={evaluationSummary.latency_metrics.avg_review_turnaround_seconds == null ? "N/A" : `${formatNumber(evaluationSummary.latency_metrics.avg_review_turnaround_seconds, 1, locale)} s`} />
                <MetricCard label={t("avgCaseToAlert")} value={evaluationSummary.latency_metrics.avg_case_to_alert_seconds == null ? "N/A" : `${formatNumber(evaluationSummary.latency_metrics.avg_case_to_alert_seconds, 1, locale)} s`} />
              </div>
              <DetailRow label={t("runtimeReview")} value={evaluationSummary.human_review.action_breakdown} />
            </div>
          </section>

          <section className="grid bottom-grid">
            <div className="panel">
              <div className="panel-header">
                <h2>{t("runtimeLatency")}</h2>
              </div>
              <div className="simulation-metrics-grid">
                <MetricCard label={t("avgIngestLag")} value={evaluationSummary.latency_metrics.avg_snapshot_ingest_lag_ms == null ? "N/A" : `${formatNumber(evaluationSummary.latency_metrics.avg_snapshot_ingest_lag_ms, 1, locale)} ms`} />
                <MetricCard label={t("p95IngestLag")} value={evaluationSummary.latency_metrics.p95_snapshot_ingest_lag_ms == null ? "N/A" : `${formatNumber(evaluationSummary.latency_metrics.p95_snapshot_ingest_lag_ms, 1, locale)} ms`} />
                <MetricCard label={t("avgCaseToAlert")} value={evaluationSummary.latency_metrics.avg_case_to_alert_seconds == null ? "N/A" : `${formatNumber(evaluationSummary.latency_metrics.avg_case_to_alert_seconds, 1, locale)} s`} />
                <MetricCard label={t("avgReviewTurnaround")} value={evaluationSummary.latency_metrics.avg_review_turnaround_seconds == null ? "N/A" : `${formatNumber(evaluationSummary.latency_metrics.avg_review_turnaround_seconds, 1, locale)} s`} />
              </div>
            </div>

            <div className="panel">
              <div className="panel-header">
                <h2>{t("runtimeQuality")}</h2>
              </div>
              <div className="simulation-metrics-grid">
                <MetricCard label={t("precisionProxy")} value={optionalPercent(qualityMetrics.precision_proxy)} />
                <MetricCard label={t("falsePositiveProxy")} value={optionalPercent(qualityMetrics.false_positive_proxy_rate)} />
                <MetricCard label={t("recallProxy")} value={optionalPercent(qualityMetrics.recall_proxy)} />
                <MetricCard label={t("missedAlertProxy")} value={optionalPercent(qualityMetrics.missed_alert_proxy_rate)} />
                <MetricCard label={t("offlineRecall")} value={optionalPercent(offlineRiskDetection.recall)} />
                <MetricCard label={t("offlinePrecision")} value={optionalPercent(offlineRiskDetection.precision)} />
                <MetricCard label={t("offlineMissRate")} value={optionalPercent(offlineRiskDetection.miss_rate)} />
                <MetricCard label={t("offlineEpisodes")} value={offlineEvaluation.episodes || 0} />
                <MetricCard label={t("dedupeRate")} value={formatPercent(qualityMetrics.dedupe_rate, locale)} />
                <MetricCard label={t("alertDedupeRate")} value={formatPercent(qualityMetrics.alert_dedupe_rate || 0, locale)} />
                <MetricCard label={t("duplicateSuppressed")} value={qualityMetrics.duplicate_suppressed || 0} />
                <MetricCard label={t("p2Aggregated")} value={qualityMetrics.p2_case_aggregated || 0} />
                <MetricCard label={t("baselineAlertReduction")} value={formatPercent(baselineComparison.alert_reduction_rate, locale)} />
                <MetricCard label={t("preventedFalsePositiveProxy")} value={baselineComparison.prevented_false_positive_proxy || 0} />
                <MetricCard label={t("conflictRate")} value={formatPercent(evaluationSummary.data_quality.cross_source_conflict_rate, locale)} />
                <MetricCard label={t("staleRate")} value={formatPercent(evaluationSummary.data_quality.source_stale_rate, locale)} />
              </div>
            </div>
          </section>

          <section className="grid bottom-grid">
            <div className="panel">
              <div className="panel-header">
                <h2>{t("runtimeCoverage")}</h2>
              </div>
              <div className="simulation-metrics-grid">
                <MetricCard label={t("eventsPerSecond")} value={formatNumber(throughputMetrics.events_per_second, 3, locale)} />
                <MetricCard label={t("snapshotsPerSecond")} value={formatNumber(throughputMetrics.snapshots_per_second, 3, locale)} />
                <MetricCard label={t("riskCases")} value={`${formatNumber(throughputMetrics.cases_per_hour, 2, locale)}/h`} />
                <MetricCard label={t("riskAlerts")} value={`${formatNumber(throughputMetrics.alerts_per_hour, 2, locale)}/h`} />
                <MetricCard label={t("runtimeAssets")} value={evaluationSummary.coverage.assets.length} />
                <MetricCard label={t("evaluationCoverageHours")} value={`${formatNumber(evaluationSummary.window.coverage_hours, 1, locale)} h`} />
              </div>
            </div>

            <div className="panel">
              <div className="panel-header">
                <h2>{t("llmCost")}</h2>
              </div>
              <div className="simulation-metrics-grid">
                <MetricCard label={t("llmCalls")} value={llmCostMetrics.calls || 0} />
                <MetricCard label={t("llmTokens")} value={llmCostMetrics.total_tokens || 0} />
                <MetricCard label={t("llmCost")} value={optionalUsd(llmCostMetrics.estimated_cost_usd)} />
                <MetricCard label={t("llmCostPerAlert")} value={optionalUsd(llmCostMetrics.cost_per_alert_usd)} />
              </div>
              {!llmCostMetrics.cost_configured ? (
                <p className="muted">{t("llmCostConfigMissing")}</p>
              ) : null}
              <DetailRow label={t("outputOperations")} value={llmCostMetrics.operation_breakdown} />
            </div>
          </section>

          <section className="grid bottom-grid">
            <div className="panel">
              <div className="panel-header">
                <h2>{t("runtimeAssetBreakdown")}</h2>
              </div>
              <div className="asset-grid">
                {evaluationSummary.coverage.asset_breakdown.map((item) => (
                  <div key={item.asset} className="asset-card">
                    <div className="asset-head">
                      <strong>{item.asset}</strong>
                      <span className="muted">{item.raw_events} {t("rawEvents")}</span>
                    </div>
                    <div className="asset-metric">{t("featureSnapshots")}: {item.snapshots}</div>
                    <div className="asset-metric">{t("riskCases")}: {item.cases}</div>
                    <div className="asset-metric">{t("riskAlerts")}: {item.alerts}</div>
                    <div className="asset-metric">{t("humanReviews")}: {item.reviews}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="panel">
              <div className="panel-header">
                <h2>{t("runtimeTopRules")}</h2>
              </div>
              <div className="alert-list">
                {evaluationSummary.top_rules.map((item) => (
                  <article key={item.rule_id} className="alert-card">
                    <div className="alert-head">
                      <strong>{item.rule_id}</strong>
                      <span className="muted">{t("topRuleCount")}: {item.count}</span>
                    </div>
                    <DetailRow label={t("severity")} value={item.severity_breakdown} />
                  </article>
                ))}
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
}

export default function App() {
  const [language, setLanguage] = useState(getInitialLanguage);
  const [viewMode, setViewMode] = useState("realtime");
  const [selectedAsset, setSelectedAsset] = useState("BTC");
  const [health, setHealth] = useState({ status: "loading", agent: { running: false }, assets: {} });
  const [candles, setCandles] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [cases, setCases] = useState([]);
  const [casePage, setCasePage] = useState(0);
  const [casePagination, setCasePagination] = useState({
    total: 0,
    limit: CASE_PAGE_SIZE,
    offset: 0,
    has_more: false,
  });
  const [expandedCaseId, setExpandedCaseId] = useState("");
  const [activeRules, setActiveRules] = useState({ version_tag: "", thresholds: {} });
  const [ruleForm, setRuleForm] = useState(initialRuleForm);
  const [reviewForm, setReviewForm] = useState({});
  const [message, setMessage] = useState("");
  const [alertSocketState, setAlertSocketState] = useState("connecting");
  const [simulationScenarios, setSimulationScenarios] = useState([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState("");
  const [latestSimulation, setLatestSimulation] = useState(null);
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [evaluationWindowKey, setEvaluationWindowKey] = useState("all");
  const [evaluationSummary, setEvaluationSummary] = useState(null);
  const [riskModelStatus, setRiskModelStatus] = useState(null);

  const locale = language === "zh" ? "zh-CN" : "en-US";
  const t = (key, values) => interpolate(TRANSLATIONS[language][key] || key, values);

  const selectedScenario = useMemo(
    () => simulationScenarios.find((item) => item.scenario_id === selectedScenarioId) || null,
    [selectedScenarioId, simulationScenarios],
  );

  useEffect(() => {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
  }, [language]);

  async function refreshHealth() {
    const data = await requestJson("/health");
    setHealth(data);
  }

  async function refreshCandles(asset = selectedAsset) {
    const data = await requestJson(`/market/candles?asset=${asset}&interval=1m&limit=${CANDLE_LIMIT}`);
    setCandles(data);
  }

  async function refreshCases(page = casePage) {
    const safePage = Math.max(0, page);
    const offset = safePage * CASE_PAGE_SIZE;
    const data = await requestJson(
      `/cases?status=pending_review&limit=${CASE_PAGE_SIZE}&offset=${offset}&paginated=true`,
    );
    if (data.items.length === 0 && data.total > 0 && safePage > 0) {
      setCasePage(safePage - 1);
      return refreshCases(safePage - 1);
    }
    setCases(data.items);
    setCasePagination({
      total: data.total,
      limit: data.limit,
      offset: data.offset,
      has_more: data.has_more,
    });
    return data;
  }

  async function refreshRules() {
    const data = await requestJson("/rules/active");
    setActiveRules(data);
  }

  async function refreshAll(asset = selectedAsset) {
    await Promise.all([refreshHealth(), refreshCandles(asset), refreshCases(), refreshRules()]);
  }

  function goToCasePage(nextPage) {
    setCasePage(Math.max(0, nextPage));
    setExpandedCaseId("");
  }

  async function refreshSimulationWorkspace() {
    const [scenarios, latest] = await Promise.all([
      requestJson("/simulation/scenarios"),
      requestJson("/simulation/runs/latest"),
    ]);
    setSimulationScenarios(scenarios);
    if (!selectedScenarioId && scenarios.length > 0) {
      setSelectedScenarioId(scenarios[0].scenario_id);
    }
    setLatestSimulation(latest);
  }

  async function refreshEvaluationSummary(windowKey = evaluationWindowKey) {
    const selectedWindow = EVALUATION_WINDOWS.find((item) => item.key === windowKey) || EVALUATION_WINDOWS[0];
    const query = selectedWindow.days == null ? "" : `?days=${selectedWindow.days}`;
    const [summary, modelStatus] = await Promise.all([
      requestJson(`/evaluation/summary${query}`),
      requestJson("/ml/risk-model/status"),
    ]);
    setEvaluationSummary(summary);
    setRiskModelStatus(modelStatus);
  }

  useEffect(() => {
    if (viewMode !== "realtime") {
      return undefined;
    }
    refreshAll(selectedAsset).catch((error) => setMessage(error.message));
    const poller = window.setInterval(() => {
      refreshAll(selectedAsset).catch((error) => setMessage(error.message));
    }, REFRESH_INTERVAL_MS);
    return () => window.clearInterval(poller);
  }, [casePage, selectedAsset, viewMode]);

  useEffect(() => {
    if (viewMode !== "test") {
      return undefined;
    }
    refreshSimulationWorkspace().catch((error) => setMessage(error.message));
    return undefined;
  }, [viewMode]);

  useEffect(() => {
    if (viewMode !== "evaluation") {
      return undefined;
    }
    refreshEvaluationSummary(evaluationWindowKey).catch((error) => setMessage(error.message));
    return undefined;
  }, [evaluationWindowKey, viewMode]);

  useEffect(() => {
    if (viewMode !== "realtime") {
      setAlertSocketState("idle");
      return undefined;
    }

    let socket;
    let reconnectTimer;
    let cancelled = false;

    const connect = () => {
      if (cancelled) {
        return;
      }
      setAlertSocketState("connecting");
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${window.location.host}${API_PREFIX}/ws/alerts`);

      socket.onopen = () => {
        setAlertSocketState("open");
      };

      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "ping") {
          return;
        }
        setAlerts((current) => [payload, ...current].slice(0, 30));
      };

      socket.onerror = () => {
        setAlertSocketState("error");
      };

      socket.onclose = () => {
        if (cancelled) {
          return;
        }
        setAlertSocketState("error");
        reconnectTimer = window.setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) {
        window.clearTimeout(reconnectTimer);
      }
      if (socket && socket.readyState < WebSocket.CLOSING) {
        socket.close();
      }
    };
  }, [viewMode]);

  async function handleAgentAction(path) {
    try {
      await requestJson(path, { method: "POST" });
      await refreshHealth();
      setMessage(path.endsWith("start") ? t("agentStarted") : t("agentStopped"));
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function handleResume(caseId, action) {
    const payload = reviewForm[caseId] || { reviewer: "operator", comment: "" };
    try {
      await requestJson(`/cases/${caseId}/resume`, {
        method: "POST",
        body: JSON.stringify({
          reviewer: payload.reviewer || "operator",
          action,
          comment: payload.comment || "",
        }),
      });
      await refreshCases();
      setMessage(action === "approve" ? t("caseApproved", { caseId }) : t("caseRejected", { caseId }));
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function handlePublish(event) {
    event.preventDefault();
    const payload = Object.fromEntries(
      Object.entries(ruleForm).map(([key, value]) => {
        if (value === "") {
          return [key, undefined];
        }
        const numberFields = [
          "price_change_p1",
          "price_change_p2",
          "oi_delta_p2",
          "liq_usd_p1",
          "funding_z_p2",
          "early_warning_ret_5m",
          "early_warning_oi_delta",
          "early_warning_funding_z",
          "early_warning_vol_z",
          "early_warning_min_score",
          "early_warning_min_signals",
          "early_warning_single_signal_min_score",
          "early_warning_persistence_window",
          "early_warning_persistence_hits",
          "early_warning_dynamic_history",
          "early_warning_dynamic_quantile",
          "early_warning_trend_window",
          "early_warning_min_trend_hits",
          "vol_z_spike",
          "cross_source_conflict_pct",
        ];
        const boolFields = ["early_warning_dynamic_baseline"];
        if (boolFields.includes(key)) {
          return [key, ["true", "1", "yes", "on"].includes(String(value).toLowerCase())];
        }
        return [key, numberFields.includes(key) ? Number(value) : value];
      }),
    );
    try {
      await requestJson("/rules/publish", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setRuleForm(initialRuleForm);
      await refreshRules();
      setMessage(t("rulePublished", { versionTag: payload.version_tag }));
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function handleRunSimulation() {
    if (!selectedScenarioId) {
      return;
    }
    setSimulationRunning(true);
    try {
      const result = await requestJson("/simulation/runs", {
        method: "POST",
        body: JSON.stringify({ scenario_id: selectedScenarioId }),
      });
      setLatestSimulation(result);
      setMessage("");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setSimulationRunning(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="backdrop backdrop-a" />
      <div className="backdrop backdrop-b" />

      <div className="workspace-shell">
        <aside className="mode-sidebar">
          <div className="panel side-panel">
            <p className="eyebrow">ETC Risk Agent</p>
            <h2>{t("viewMode")}</h2>
            <div className="mode-list">
              <button
                type="button"
                className={viewMode === "realtime" ? "active-tab" : "ghost"}
                onClick={() => setViewMode("realtime")}
              >
                <strong>{t("realtimeMode")}</strong>
                <span>{t("realtimeModeCopy")}</span>
              </button>
              <button
                type="button"
                className={viewMode === "test" ? "active-tab" : "ghost"}
                onClick={() => setViewMode("test")}
              >
                <strong>{t("testMode")}</strong>
                <span>{t("testModeCopy")}</span>
              </button>
              <button
                type="button"
                className={viewMode === "evaluation" ? "active-tab" : "ghost"}
                onClick={() => setViewMode("evaluation")}
              >
                <strong>{t("evaluationMode")}</strong>
                <span>{t("evaluationModeCopy")}</span>
              </button>
            </div>
            <div className="lang-switcher sidebar-switcher" aria-label={t("language")}>
              <button
                type="button"
                className={`ghost ${language === "zh" ? "active-tab" : ""}`}
                onClick={() => setLanguage("zh")}
              >
                {t("chinese")}
              </button>
              <button
                type="button"
                className={`ghost ${language === "en" ? "active-tab" : ""}`}
                onClick={() => setLanguage("en")}
              >
                {t("english")}
              </button>
            </div>
          </div>
        </aside>

        <main className="workspace-main">
          <header className="hero compact-hero">
            <div>
              <p className="eyebrow">ETC Risk Agent</p>
              <h1>{t("title")}</h1>
              <p className="hero-copy">{t("heroCopy")}</p>
            </div>
          </header>

          {viewMode === "realtime" ? (
            <RealtimeWorkspace
              activeRules={activeRules}
              alerts={alerts}
              alertSocketState={alertSocketState}
              candles={candles}
              cases={cases}
              casePage={casePage}
              casePagination={casePagination}
              expandedCaseId={expandedCaseId}
              goToCasePage={goToCasePage}
              handleAgentAction={handleAgentAction}
              handlePublish={handlePublish}
              handleResume={handleResume}
              health={health}
              locale={locale}
              message={message}
              reviewForm={reviewForm}
              ruleForm={ruleForm}
              selectedAsset={selectedAsset}
              setExpandedCaseId={setExpandedCaseId}
              setReviewForm={setReviewForm}
              setRuleForm={setRuleForm}
              setSelectedAsset={setSelectedAsset}
              t={t}
            />
          ) : viewMode === "test" ? (
            <>
              {message ? <div className="banner">{message}</div> : null}
              <TestWorkspace
                latestSimulation={latestSimulation}
                locale={locale}
                selectedScenario={selectedScenario}
                selectedScenarioId={selectedScenarioId}
                setSelectedScenarioId={setSelectedScenarioId}
                simulationRunning={simulationRunning}
                simulationScenarios={simulationScenarios}
                t={t}
                runSimulation={handleRunSimulation}
              />
            </>
          ) : (
            <>
              {message ? <div className="banner">{message}</div> : null}
              <EvaluationWorkspace
                evaluationSummary={evaluationSummary}
                evaluationWindowKey={evaluationWindowKey}
                locale={locale}
                refreshEvaluationSummary={refreshEvaluationSummary}
                riskModelStatus={riskModelStatus}
                setEvaluationWindowKey={setEvaluationWindowKey}
                t={t}
              />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
