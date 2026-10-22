from pydantic import BaseModel


class ParsedDocument(BaseModel):
    title: str | None
    text: str
    images_described: int
    images_failed: int = 0
    images_skipped: int = 0


class DocumentParseError(Exception):
    pass
