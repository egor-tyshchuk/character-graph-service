from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedCharacter(_StrictModel):
    name: str
    aliases: list[str]
    description: str


class ExtractedRelation(_StrictModel):
    source: str
    target: str
    type: str
    description: str
    evidence: str


class ExtractionResult(_StrictModel):
    characters: list[ExtractedCharacter]
    relations: list[ExtractedRelation]


class ImageDescription(_StrictModel):
    description: str


class DedupVerdict(_StrictModel):
    same: bool
    reason: str
