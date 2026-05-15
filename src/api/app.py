"""FastAPI application factory."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.core.logging import setup_logging, get_logger
from src.persistence.database import init_db

logger = get_logger(__name__)

_ingestion_task: asyncio.Task | None = None
_memory_tasks: list[asyncio.Task] = []


def _start_memory_background_tasks(settings) -> None:
    """启动记忆层后台定时任务：经验蒸馏 + 审核偏好学习。"""
    global _memory_tasks

    async def _distill_loop() -> None:
        interval_s = max(3600, settings.memory_distill_interval_hours * 3600)
        logger.info(
            "Memory distill loop started (interval=%s hours)",
            settings.memory_distill_interval_hours,
        )
        await asyncio.sleep(300)
        while True:
            try:
                from src.memory.experience import distill_experiences
                result = await distill_experiences()
                if result:
                    logger.info("Experience distill completed: %d new insights", len(result))
            except Exception as exc:
                logger.warning("Experience distill failed: %s", exc)
            await asyncio.sleep(interval_s)

    async def _preference_learn_loop() -> None:
        interval_s = max(3600, settings.memory_preference_learning_interval_hours * 3600)
        logger.info(
            "Preference learning loop started (interval=%s hours)",
            settings.memory_preference_learning_interval_hours,
        )
        await asyncio.sleep(600)
        while True:
            try:
                from src.memory.reviewer_preference import learn_preferences
                result = await learn_preferences()
                if result:
                    logger.info("Preference learning completed: %d patterns updated", len(result))
            except Exception as exc:
                logger.warning("Preference learning failed: %s", exc)
            await asyncio.sleep(interval_s)

    _memory_tasks.append(asyncio.create_task(_distill_loop()))
    _memory_tasks.append(asyncio.create_task(_preference_learn_loop()))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan 上下文管理器——替代已弃用的 on_event。"""
    await init_db()
    from src.graph.orchestrator import init_graph
    from src.rules.config import registry
    await init_graph()
    await registry.load_active()

    from src.core.config import settings
    if settings.memory_enabled:
        from src.memory.store import init_vector_store
        await init_vector_store()
        _start_memory_background_tasks(settings)

    await start_ingestion()
    yield


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="ETC Risk Agent", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    # 直接注册 health/detailed 以绕过 router 级别注册问题
    @app.get("/api/v1/health/detailed")
    async def health_detailed():
        from datetime import datetime, timezone
        from src.features.builder import get_feature_builder
        from src.domain.models import Asset
        from src.observability.metrics import event_bus_queue_size, pending_review_gauge
        from src.persistence.database import AsyncSessionLocal

        agent_info = get_agent_status()

        builder = get_feature_builder()
        asset_status = {}
        for asset in Asset:
            snap = builder.get_snapshot(asset)
            age = (datetime.now(timezone.utc) - snap.window_end.replace(tzinfo=timezone.utc)).total_seconds() \
                  if snap.window_end and snap.price > 0 else 9999
            asset_status[asset.value] = {
                "price": snap.price, "age_seconds": round(age),
                "fresh": snap.price > 0 and age < 120,
                "oi_delta_15m_pct": snap.oi_delta_15m_pct,
                "funding_z": snap.funding_z,
            }

        llm_status = "unknown"
        try:
            from src.core.proxy import build_openai_client
            from src.core.config import settings as app_settings
            build_openai_client(service="health_check")
            llm_status = "configured" if app_settings.ark_api_key else "missing_api_key"
        except Exception:
            llm_status = "unavailable"

        queue_size = event_bus_queue_size._value.get()
        pending_reviews = pending_review_gauge._value.get()

        db_stats = {}
        try:
            from sqlalchemy import func, select
            from src.persistence.database import (
                FeatureSnapshotRow, RiskCaseRow, RiskAlertRow,
                LLMCallRow, HumanReviewRow, QualityMetricEventRow,
            )
            async with AsyncSessionLocal() as s:
                db_stats["feature_snapshots"] = (await s.execute(
                    select(func.count(FeatureSnapshotRow.snapshot_id))
                )).scalar() or 0
                db_stats["risk_cases"] = (await s.execute(
                    select(func.count(RiskCaseRow.case_id))
                )).scalar() or 0
                db_stats["risk_alerts"] = (await s.execute(
                    select(func.count(RiskAlertRow.alert_id))
                )).scalar() or 0
                db_stats["llm_calls"] = (await s.execute(
                    select(func.count(LLMCallRow.call_id))
                )).scalar() or 0
                db_stats["review_actions"] = (await s.execute(
                    select(func.count(HumanReviewRow.action_id))
                )).scalar() or 0
                db_stats["quality_events"] = (await s.execute(
                    select(func.count(QualityMetricEventRow.event_id))
                )).scalar() or 0
        except Exception as exc:
            db_stats = {"error": str(exc)}

        ml_status = "unavailable"
        try:
            from src.ml.risk_model import model_status
            ml_info = model_status()
            ml_status = "loaded" if ml_info.get("available") else "not_trained"
        except Exception:
            ml_status = "error"

        return {
            "status": "ok", "agent": agent_info,
            "llm": {"status": llm_status}, "ml_model": {"status": ml_status},
            "queue": {"event_bus_queue_size": queue_size, "pending_reviews": pending_reviews},
            "assets": asset_status, "database": db_stats,
        }

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


def get_agent_status() -> dict:
    return {
        "running": _ingestion_task is not None and not _ingestion_task.done(),
    }


async def start_ingestion() -> None:
    global _ingestion_task
    if _ingestion_task is not None and not _ingestion_task.done():
        raise RuntimeError("Agent is already running")
    _ingestion_task = asyncio.create_task(_run_ingestion_supervisor())


async def stop_ingestion() -> None:
    global _ingestion_task
    if _ingestion_task is None or _ingestion_task.done():
        raise RuntimeError("Agent is not running")
    _ingestion_task.cancel()
    await asyncio.gather(_ingestion_task, return_exceptions=True)
    _ingestion_task = None


async def _run_ingestion_supervisor() -> None:
    global _ingestion_task
    from src.features.builder import run_feature_builder
    from src.ingestion.sources.binance_ws import run_binance_futures, run_binance_spot
    from src.ingestion.sources.okx_ws import run_okx_ws
    from src.ingestion.sources.okx_rest import run_okx_rest_poll

    tasks = [
        asyncio.create_task(run_binance_spot()),
        asyncio.create_task(run_binance_futures()),
        asyncio.create_task(run_okx_ws()),
        asyncio.create_task(run_okx_rest_poll()),
        asyncio.create_task(run_feature_builder()),
    ]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        if _ingestion_task is asyncio.current_task():
            _ingestion_task = None


app = create_app()
