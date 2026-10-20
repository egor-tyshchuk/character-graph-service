from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class ArticleStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class ArticleFormat(StrEnum):
    MARKDOWN = "markdown"
    PDF = "pdf"


class ArticleStats(BaseModel):
    characters: int
    relations: int
    images_described: int


class Article(BaseModel):
    id: str
    title: str
    format: ArticleFormat
    content_hash: str
    status: ArticleStatus
    error: str | None = None
    stats: ArticleStats | None = None
    created_at: datetime
    updated_at: datetime


class Character(BaseModel):
    id: str
    name: str
    aliases: list[str]
    description: str
    article_count: int


class RelationTypeInfo(BaseModel):
    name: str
    description: str
    usage_count: int


class Direction(StrEnum):
    OUTGOING = "outgoing"
    INCOMING = "incoming"


class Connection(BaseModel):
    character: Character
    relation_type: str
    direction: Direction
    description: str
    evidence: str
    article_id: str
