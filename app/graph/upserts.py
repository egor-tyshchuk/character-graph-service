from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class MatchKind(StrEnum):
    NEW = "new"
    VECTOR = "vector"
    JUDGE = "judge"
    MANUAL = "manual"
    RECONCILE = "reconcile"


MATCH_STRENGTH = (MatchKind.VECTOR, MatchKind.JUDGE, MatchKind.NEW)


class Mention(BaseModel):
    surface_name: str
    description: str
    match: MatchKind
    similarity: float | None


class CharacterUpsert(BaseModel):
    id: str
    name: str
    aliases: list[str]
    description: str
    embedding: list[float] | None
    mentions: list[Mention]


class RelationUpsert(BaseModel):
    source_id: str
    target_id: str
    type: str
    description: str
    evidence: str


class RelationTypeUpsert(BaseModel):
    name: str
    description: str
    embedding: list[float]


class SimilarCharacter(BaseModel):
    id: str
    name: str
    aliases: list[str]
    description: str
    similarity: float
    created_at: datetime


class NotSameUpsert(BaseModel):
    a_id: str
    b_id: str
    reason: str
    hash_a: str
    hash_b: str


class EmbeddingRefresh(BaseModel):
    id: str
    name: str
    aliases: list[str]
    description: str
    embedding: list[float]
