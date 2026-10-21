from typing import Self

from pydantic import BaseModel, Field, model_validator

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


class MergeCharactersRequest(BaseModel):
    target_id: str
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_characters(self) -> Self:
        if self.target_id in self.source_ids:
            raise ValueError("target_id must not be among source_ids")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be distinct")
        return self
