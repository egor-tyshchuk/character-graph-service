from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from app.domain.models import Connection

TOKEN_MAX_CHARS = 4096
TITLE_MAX_CHARS = 300
MERGE_MAX_SOURCES = 50


class TokenRequest(BaseModel):
    app_token: str = Field(max_length=TOKEN_MAX_CHARS)


class TextUploadRequest(BaseModel):
    text: str
    title: str | None = Field(default=None, max_length=TITLE_MAX_CHARS)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(max_length=TOKEN_MAX_CHARS)


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


class MergeCharactersRequest(BaseModel):
    target_id: str
    source_ids: list[str] = Field(min_length=1, max_length=MERGE_MAX_SOURCES)

    @model_validator(mode="after")
    def _distinct_characters(self) -> Self:
        if self.target_id in self.source_ids:
            raise ValueError("target_id must not be among source_ids")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be distinct")
        return self


class QueuedResponse(BaseModel):
    status: Literal["queued", "already_queued"]
