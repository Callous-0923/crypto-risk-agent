import { useEffect, useState } from "react";

const API_PREFIX = "/api/v1";

const initialRuleForm = {
  version_tag: "",
  operator: "",
  reason: "",
  price_change_p1: "",
  price_change_p2: "",
  oi_delta_p2: "",
  liq_usd_p1: "",
  funding_z_p2: "",
  vol_z_spike: "",
  cross_source_conflict_pct: "",
};

function severityClass(severity) {
  return severity ? `severity-${severity.toLowerCase()}` : "severity-p3";
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

export default function App() {
  const [health, setHealth] = useState({ status: "loading", agent: { running: false }, assets: {} });
  const [alerts, setAlerts] = useState([]);
  const [cases, setCases] = useState([]);
  const [expandedCaseId, setExpandedCaseId] = useState("");
  const [activeRules, setActiveRules] = useState({ version_tag: "", thresholds: {} });
  const [ruleForm, setRuleForm] = useState(initialRuleForm);
  const [reviewForm, setReviewForm] = useState({});
  const [message, setMessage] = useState("");

  async function refreshHealth() {
    const data = await requestJson("/health");
    setHealth(data);
  }

  async function refreshCases() {
    const data = await requestJson("/cases?limit=50");
    setCases(data.filter((item) => item.status === "pending_review"));
  }

  async function refreshRules() {
    const data = await requestJson("/rules/active");
    setActiveRules(data);
  }

  async function refreshAll() {
    await Promise.all([refreshHealth(), refreshCases(), refreshRules()]);
  }

  useEffect(() => {
    refreshAll().catch((error) => setMessage(error.message));
    const poller = window.setInterval(() => {
      refreshAll().catch((error) => setMessage(error.message));
    }, 10000);
    return () => window.clearInterval(poller);
  }, []);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(`${protocol}//${window.location.host}${API_PREFIX}/ws/alerts`);
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "ping") {
        return;
      }
      setAlerts((current) => [payload, ...current].slice(0, 30));
    };
    socket.onerror = () => setMessage("WebSocket 连接异常，实时告警可能延迟。");
    return () => socket.close();
  }, []);

  async function handleAgentAction(path) {
    try {
      await requestJson(path, { method: "POST" });
      await refreshHealth();
      setMessage(path.endsWith("start") ? "Agent 已启动" : "Agent 已停止");
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
      setMessage(`案例 ${caseId} 已${action === "approve" ? "批准" : "拒绝"}`);
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
          "vol_z_spike",
          "cross_source_conflict_pct",
        ];
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
      setMessage(`规则版本 ${payload.version_tag} 已发布`);
    } catch (error) {
      setMessage(error.message);
    }
  }

  return (
    <div className="app-shell">
      <div className="backdrop backdrop-a" />
      <div className="backdrop backdrop-b" />
      <header className="hero">
        <div>
          <p className="eyebrow">ETC Risk Agent</p>
          <h1>风控控制台</h1>
          <p className="hero-copy">集中处理 agent 生命周期、实时告警、人审案例和规则版本发布。</p>
        </div>
        <div className="status-card">
          <div className="status-line">
            <span className={`status-dot ${health.agent?.running ? "running" : "stopped"}`} />
            <strong>{health.agent?.running ? "运行中" : "已停止"}</strong>
          </div>
          <div className="status-actions">
            <button onClick={() => handleAgentAction("/agent/start")}>启动</button>
            <button className="ghost" onClick={() => handleAgentAction("/agent/stop")}>停止</button>
          </div>
          <p className="muted">健康状态：{health.status}</p>
        </div>
      </header>

      {message ? <div className="banner">{message}</div> : null}

      <section className="grid top-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>数据新鲜度</h2>
          </div>
          <div className="asset-grid">
            {Object.entries(health.assets || {}).map(([asset, info]) => (
              <div key={asset} className="asset-card">
                <div className="asset-head">
                  <strong>{asset}</strong>
                  <span className={info.ok ? "asset-ok" : "asset-stale"}>{info.ok ? "fresh" : "stale"}</span>
                </div>
                <div className="asset-metric">价格: {Number(info.price || 0).toFixed(2)}</div>
                <div className="asset-metric">延迟: {info.age_seconds}s</div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>实时告警</h2>
          </div>
          <div className="alert-list">
            {alerts.length === 0 ? <p className="muted">等待实时告警流...</p> : null}
            {alerts.map((alert) => (
              <article key={alert.alert_id} className="alert-card">
                <div className="alert-head">
                  <span className={`severity-pill ${severityClass(alert.severity)}`}>{alert.severity}</span>
                  <span className="muted">{new Date(alert.created_at).toLocaleString()}</span>
                </div>
                <h3>{alert.title}</h3>
                <p>{alert.body_zh}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="grid bottom-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>待审核案例</h2>
          </div>
          <div className="case-list">
            {cases.length === 0 ? <p className="muted">当前没有待审核案例。</p> : null}
            {cases.map((item) => {
              const formState = reviewForm[item.case_id] || { reviewer: "operator", comment: "" };
              const expanded = expandedCaseId === item.case_id;
              return (
                <article key={item.case_id} className="case-card">
                  <div className="case-summary">
                    <div>
                      <span className={`severity-pill ${severityClass(item.severity)}`}>{item.severity}</span>
                      <strong>{item.is_coordinator_case ? "多资产协调案例" : `${item.asset} 案例`}</strong>
                    </div>
                    <button className="ghost" onClick={() => setExpandedCaseId(expanded ? "" : item.case_id)}>
                      {expanded ? "收起" : "展开"}
                    </button>
                  </div>
                  <p>{item.summary_zh || "暂无摘要"}</p>
                  {expanded ? (
                    <div className="case-details">
                      <DetailRow label="review_guidance" value={item.review_guidance} />
                      <DetailRow label="historical_context_zh" value={item.historical_context_zh} />
                      <DetailRow label="risk_quantification_zh" value={item.risk_quantification_zh} />
                      <DetailRow label="threshold_suggestion" value={item.threshold_suggestion} />
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
                          placeholder="reviewer"
                        />
                        <textarea
                          value={formState.comment}
                          onChange={(event) =>
                            setReviewForm((current) => ({
                              ...current,
                              [item.case_id]: { ...formState, comment: event.target.value },
                            }))
                          }
                          placeholder="审核备注"
                        />
                        <div className="status-actions">
                          <button onClick={() => handleResume(item.case_id, "approve")}>批准</button>
                          <button className="ghost" onClick={() => handleResume(item.case_id, "reject")}>拒绝</button>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </article>
              );
            })}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>规则版本管理</h2>
          </div>
          <div className="rule-summary">
            <div>
              <p className="muted">当前版本</p>
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
            <button type="submit">发布新版本</button>
          </form>
        </div>
      </section>
    </div>
  );
}
