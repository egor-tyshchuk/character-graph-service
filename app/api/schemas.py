from pydantic import BaseModel

from app.domain.models import Connection


class TokenRequest(BaseModel):
    app_token: str


class TextUploadRequest(BaseModel):
    text: str
    title: str | None = None


class RefreshRequest(BaseModel):
    refresh_token: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Page[T](BaseModel):
    items: list[T]
    total: int
    limit: int
    offset: int


class ConnectedResponse(BaseModel):
    character_id: str
    requested_type: str
    resolved_type: str | None
    items: list[Connection]
