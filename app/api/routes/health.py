import asyncio
from typing import Literal

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.container import Container

router = APIRouter(prefix="/health", tags=["health"])
logger = structlog.get_logger(__name__)

Status = Literal["ok", "error"]

PROBE_TIMEOUT_SECONDS = 3.0


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    container: Container = request.app.state.container
    neo4j = await _check_neo4j(container)
    redis = await _check_redis(container)
    healthy = neo4j == "ok" and redis == "ok"
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "error", "neo4j": neo4j, "redis": redis},
    )


async def _check_neo4j(container: Container) -> Status:
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS):
            await container.driver.verify_connectivity()
    except Exception:
        logger.warning("readiness_neo4j_failed", exc_info=True)
        return "error"
    return "ok"


async def _check_redis(container: Container) -> Status:
    try:
        async with asyncio.timeout(PROBE_TIMEOUT_SECONDS):
            await container.redis.ping()
    except Exception:
        logger.warning("readiness_redis_failed", exc_info=True)
        return "error"
    return "ok"
