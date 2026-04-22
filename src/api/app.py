"""FastAPI application factory."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.core.logging import setup_logging
from src.persistence.database import init_db

_ingestion_task: asyncio.Task | None = None


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="ETC Risk Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    @app.on_event("startup")
    async def startup():
        await init_db()
        from src.graph.orchestrator import init_graph
        from src.rules.config import registry
        await init_graph()
        await registry.load_active()   # 从数据库加载 active 规则版本
        await start_ingestion()

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
