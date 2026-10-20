import asyncio
from dataclasses import dataclass
from pathlib import PurePath

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile

from app.api.schemas import TextUploadRequest
from app.domain.models import ArticleFormat
from app.errors import PayloadTooLargeError, UnsupportedMediaError, ValidationAppError
from app.ingestion.fingerprint import fingerprint, normalize_text_for_hash
from app.ingestion.markdown import parse_markdown
from app.ingestion.models import DocumentParseError
from app.ingestion.pdf import read_pdf_title

PDF_MAGIC = b"%PDF-"
DEFAULT_TITLE = "Untitled"
JSON_MEDIA_TYPE = "application/json"
REQUEST_OVERHEAD_BYTES = 64 * 1024
BODY_EXPANSION = {
    JSON_MEDIA_TYPE: 6,
    "multipart/form-data": 1,
    "application/x-www-form-urlencoded": 3,
}


@dataclass(frozen=True)
class UploadedArticle:
    data: bytes
    format: ArticleFormat
    content_hash: str
    title: str | None
    filename: str | None

    async def resolve_title(self) -> str:
        content_title = await self._content_title()
        return self.title or content_title or _filename_stem(self.filename) or DEFAULT_TITLE

    async def _content_title(self) -> str | None:
        if self.format is ArticleFormat.MARKDOWN:
            return parse_markdown(self.data).title
        try:
            return await asyncio.to_thread(read_pdf_title, self.data)
        except DocumentParseError as exc:
            raise ValidationAppError(str(exc)) from exc


async def read_upload(request: Request, max_bytes: int) -> UploadedArticle:
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    expansion = BODY_EXPANSION.get(media_type)
    if expansion is None:
        raise UnsupportedMediaError(f"Content-Type must be one of {', '.join(BODY_EXPANSION)}")
    body_limit = max_bytes * expansion + REQUEST_OVERHEAD_BYTES
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > body_limit:
        raise _too_large(max_bytes)
    if media_type == JSON_MEDIA_TYPE:
        return await _read_json(request, body_limit, max_bytes)
    async with request.form(max_part_size=body_limit) as form:
        return await _form_article(form, max_bytes)


async def _read_json(request: Request, body_limit: int, max_bytes: int) -> UploadedArticle:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > body_limit:
            raise _too_large(max_bytes)
    try:
        payload = TextUploadRequest.model_validate_json(body)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors(include_url=False)) from exc
    return _text_article(payload.text, payload.title, max_bytes)


async def _form_article(form: FormData, max_bytes: int) -> UploadedArticle:
    text, file, title = form.get("text"), form.get("file"), form.get("title")
    if isinstance(file, str) or isinstance(text, UploadFile) or isinstance(title, UploadFile):
        raise ValidationAppError("file must be an uploaded file; text and title must be strings")
    if text is not None and file is None:
        return _text_article(text, title, max_bytes)
    if file is not None and text is None:
        data = await file.read(max_bytes + 1)
        _check_size(data, max_bytes)
        return _file_article(data, file.filename, title)
    raise ValidationAppError("Provide exactly one of text or file")


def _text_article(text: str, title: str | None, max_bytes: int) -> UploadedArticle:
    data = text.encode("utf-8")
    _check_size(data, max_bytes)
    return _markdown_article(data, text, title, filename=None)


def _file_article(data: bytes, filename: str | None, title: str | None) -> UploadedArticle:
    if data.startswith(PDF_MAGIC):
        return UploadedArticle(
            data=data,
            format=ArticleFormat.PDF,
            content_hash=fingerprint(data),
            title=_clean_title(title),
            filename=filename,
        )
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnsupportedMediaError("File must be a PDF or UTF-8 encoded Markdown/text") from exc
    return _markdown_article(data, text, title, filename=filename)


def _markdown_article(
    data: bytes, text: str, title: str | None, *, filename: str | None
) -> UploadedArticle:
    normalized = normalize_text_for_hash(text)
    if not normalized:
        raise ValidationAppError("Article content is empty")
    return UploadedArticle(
        data=data,
        format=ArticleFormat.MARKDOWN,
        content_hash=fingerprint(normalized.encode("utf-8")),
        title=_clean_title(title),
        filename=filename,
    )


def _check_size(data: bytes, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise _too_large(max_bytes)


def _too_large(limit: int) -> PayloadTooLargeError:
    return PayloadTooLargeError(f"Upload exceeds the limit of {limit} bytes")


def _clean_title(title: str | None) -> str | None:
    return (title or "").strip() or None


def _filename_stem(filename: str | None) -> str | None:
    return PurePath(filename or "").stem.strip() or None
