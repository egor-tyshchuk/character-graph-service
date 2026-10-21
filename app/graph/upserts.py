from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class MatchKind(StrEnum):
    NEW = "new"
    VECTOR = "vector"
    JUDGE = "judge"
    RECONCILE = "reconcile"


MATCH_STRENGTH = (MatchKind.VECTOR, MatchKind.JUDGE, MatchKind.NEW)


def match_strength(kind: str | None, similarity: float | None) -> tuple[int, float]:
    rank = MATCH_STRENGTH.index(kind) if kind in MATCH_STRENGTH else len(MATCH_STRENGTH)
    return rank, -(similarity or 0.0)


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
    needs_reconcile: bool = False


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


class StoredCharacter(BaseModel):
    id: str
    name: str
    aliases: list[str]
    description: str
    embedding: list[float]
    created_at: datetime
    needs_reconcile: bool = False


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
