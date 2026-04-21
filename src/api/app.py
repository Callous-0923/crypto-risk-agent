"""FastAPI application factory."""
from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.logging import setup_logging
from src.persistence.database import init_db


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

    @app.on_event("startup")
    async def startup():
        await init_db()
        from src.graph.orchestrator import init_graph
        from src.rules.config import registry
        await init_graph()
        await registry.load_active()   # 从数据库加载 active 规则版本
        asyncio.create_task(_start_ingestion())

    return app


async def _start_ingestion() -> None:
    from src.features.builder import run_feature_builder, get_feature_builder
    from src.ingestion.sources.okx_ws import run_okx_ws
    from src.ingestion.sources.okx_rest import run_okx_rest_poll
    from src.graph.orchestrator import process_snapshot
    from src.domain.models import Asset
    import asyncio

    async def _rule_eval_loop():
        """Every 30s, evaluate latest snapshots and run graph for assets with hits."""
        while True:
            await asyncio.sleep(30)
            builder = get_feature_builder()
            for asset in Asset:
                snap = builder.get_snapshot(asset)
                if snap.price > 0:
                    asyncio.create_task(process_snapshot(snap))

    asyncio.create_task(run_okx_ws())
    asyncio.create_task(run_okx_rest_poll())
    asyncio.create_task(run_feature_builder())
    asyncio.create_task(_rule_eval_loop())


app = create_app()
