from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import install_exception_handlers
from app.api.middleware import RequestIdMiddleware
from app.api.routes import articles, auth, characters, health, relations
from app.config import Settings
from app.container import build_container, close_container
from app.llm.client import LLMClient
from app.logging import configure_logging
from app.queue import JobQueue


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLMClient | None = None,
    queue: JobQueue | None = None,
) -> FastAPI:
    resolved = settings or Settings()
    configure_logging(resolved.log_level, resolved.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await build_container(resolved, llm=llm, queue=queue)
        app.state.container = container
        try:
            yield
        finally:
            await close_container(container)

    app = FastAPI(title="Character Graph Service", lifespan=lifespan)
    app.add_middleware(RequestIdMiddleware)
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(articles.router)
    app.include_router(characters.router)
    app.include_router(relations.router)
    return app
