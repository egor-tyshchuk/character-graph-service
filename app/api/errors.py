from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.errors import AppError
from app.llm.client import LLMError, LLMNotConfiguredError

logger = structlog.get_logger(__name__)

_HTTP_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
}


def _error_response(
    status_code: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


def internal_error_response() -> JSONResponse:
    return _error_response(500, "internal_error", "Internal server error")


def _summarize_validation_errors(errors: list[Any]) -> str:
    parts = []
    for err in errors:
        location = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{location}: {err.get('msg', 'invalid value')}")
    return "; ".join(parts) or "invalid request"


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message, dict(exc.headers) or None)

    @app.exception_handler(LLMError)
    async def handle_llm_error(_: Request, exc: LLMError) -> JSONResponse:
        if isinstance(exc, LLMNotConfiguredError):
            logger.warning("llm_unavailable", error=str(exc))
            return _error_response(503, "llm_unavailable", "LLM is not configured")
        logger.warning("llm_unavailable", exc_info=exc)
        return _error_response(503, "llm_unavailable", "LLM provider is unavailable")

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            422, "validation_error", _summarize_validation_errors(list(exc.errors()))
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        headers = dict(exc.headers) if exc.headers else None
        return _error_response(exc.status_code, code, str(exc.detail), headers)

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", exc_info=exc)
        return internal_error_response()
