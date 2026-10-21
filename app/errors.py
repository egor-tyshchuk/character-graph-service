from collections.abc import Mapping
from typing import ClassVar


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    headers: ClassVar[Mapping[str, str]] = {}

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class CharactersNotFoundError(NotFoundError):
    pass


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"
    headers: ClassVar[Mapping[str, str]] = {"WWW-Authenticate": "Bearer"}


class ValidationAppError(AppError):
    status_code = 422
    code = "validation_error"


class ResourceBusyError(AppError):
    status_code = 409
    code = "resource_busy"
    headers: ClassVar[Mapping[str, str]] = {"Retry-After": "5"}


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedMediaError(AppError):
    status_code = 415
    code = "unsupported_media_type"
