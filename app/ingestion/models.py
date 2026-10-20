from pydantic import BaseModel


class ParsedDocument(BaseModel):
    title: str | None
    text: str
    images_described: int


class DocumentParseError(Exception):
    pass
