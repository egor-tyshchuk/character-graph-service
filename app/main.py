from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.middleware import RequestIdMiddleware
from app.api.routes import auth, health
from app.config import Settings
from app.container import build_container, close_container
from app.errors import install_exception_handlers
from app.logging import configure_logging


def create_app(settings: Settings | None = None, *, llm: object | None = None) -> FastAPI:
    resolved = settings or Settings()
    configure_logging(resolved.log_level, resolved.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await build_container(resolved)
        app.state.container = container
        try:
            yield
        finally:
            await close_container(container)

    app = FastAPI(title="Character Graph Service", lifespan=lifespan)
    app.state.llm = llm
    app.add_middleware(RequestIdMiddleware)
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(auth.router)
    return app
