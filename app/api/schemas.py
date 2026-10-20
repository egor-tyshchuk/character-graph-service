from pydantic import BaseModel


class TokenRequest(BaseModel):
    app_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
