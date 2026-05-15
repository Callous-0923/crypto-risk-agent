from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 火山引擎
    ark_api_key: str = ""
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    llm_model: str = "doubao-seed-2-0-mini-260215"
    llm_input_cost_per_million_tokens: float = 0.0
    llm_output_cost_per_million_tokens: float = 0.0

    redis_url: str = "redis://localhost:6379"
    database_url: str = "sqlite+aiosqlite:///./etc_agent.db"
    webhook_url: str = ""
    log_level: str = "INFO"
    http_proxy: str = ""
    https_proxy: str = ""
    all_proxy: str = ""

    # Feature windows (seconds)
    vol_window_short: int = 60
    vol_window_long: int = 300

    # Rule thresholds
    price_change_p1: float = 0.05
    price_change_p2: float = 0.03
    oi_delta_p2: float = 0.10
    liq_usd_p1: float = 50_000_000
    funding_z_p2: float = 2.5
    early_warning_ret_5m: float = 0.008
    early_warning_oi_delta: float = 0.04
    early_warning_funding_z: float = 1.5
    early_warning_vol_z: float = 1.5
    early_warning_min_score: float = 0.50
    early_warning_min_signals: int = 2
    early_warning_single_signal_min_score: float = 0.65
    early_warning_persistence_window: int = 3
    early_warning_persistence_hits: int = 2
    early_warning_dynamic_baseline: bool = False
    early_warning_dynamic_history: int = 24
    early_warning_dynamic_quantile: float = 0.80
    early_warning_trend_window: int = 4
    early_warning_min_trend_hits: int = 1
    cross_source_conflict_pct: float = 0.005
    coordinator_auto_approve_seconds: int = 300
    risk_model_enabled: bool = True
    risk_model_artifact_path: str = "artifacts/risk_model/latest.joblib"
    risk_model_p1_threshold: float = 0.80
    risk_model_p2_threshold: float = 0.55
    risk_model_p3_threshold: float = 0.30
    risk_model_default_horizon_seconds: int = 3600
    llm_judge_labeler_enabled: bool = True
    llm_judge_max_tokens: int = 180

    # Memory layer
    memory_enabled: bool = True
    embedding_model: str = "doubao-embedding-large-text-240915"
    memory_similarity_threshold: float = 0.65
    memory_top_k: int = 5
    memory_distill_interval_hours: int = 24
    memory_preference_learning_interval_hours: int = 12

    # 代理设置（Docker 容器内访问 Binance/OKX 时使用）
    # 格式示例：socks5://host.docker.internal:7890
    # 或：http://host.docker.internal:7890
    ws_proxy: str = ""


settings = Settings()
