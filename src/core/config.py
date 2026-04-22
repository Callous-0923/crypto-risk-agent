from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 火山引擎
    ark_api_key: str = ""
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    llm_model: str = "doubao-seed-1-6-251015"

    redis_url: str = "redis://localhost:6379"
    database_url: str = "sqlite+aiosqlite:///./etc_agent.db"
    webhook_url: str = ""
    log_level: str = "INFO"

    # Feature windows (seconds)
    vol_window_short: int = 60
    vol_window_long: int = 300

    # Rule thresholds
    price_change_p1: float = 0.05
    price_change_p2: float = 0.03
    oi_delta_p2: float = 0.10
    liq_usd_p1: float = 50_000_000
    funding_z_p2: float = 2.5
    coordinator_auto_approve_seconds: int = 300


settings = Settings()
