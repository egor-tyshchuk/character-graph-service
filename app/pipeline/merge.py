import re
import unicodedata
from collections.abc import Sequence

from pydantic import BaseModel

from app.llm.schemas import ExtractedCharacter, ExtractedRelation, ExtractionResult

NON_ALPHANUMERIC_RUN = re.compile(r"[\W_]+")


class MergedCharacter(BaseModel):
    key: str
    name: str
    aliases: list[str]
    description: str


class MergedRelation(BaseModel):
    source_key: str
    target_key: str
    type: str
    description: str
    evidence: str


class MergedArticle(BaseModel):
    characters: list[MergedCharacter]
    relations: list[MergedRelation]


def normalize_name(name: str) -> str:
    collapsed = " ".join(unicodedata.normalize("NFKC", name).casefold().split())
    return _strip_edge_noise(collapsed)


def normalize_relation_type(raw: str) -> str:
    snake = NON_ALPHANUMERIC_RUN.sub("_", _split_camel_case(raw))
    return snake.upper().strip("_")


def merge_extractions(results: Sequence[ExtractionResult]) -> MergedArticle:
    characters = [
        character
        for result in results
        for character in result.characters
        if normalize_name(character.name)
    ]
    groups = _UnionFind(len(characters))
    form_owner: dict[str, int] = {}
    for index, character in enumerate(characters):
        for form in _forms(character):
            if form in form_owner:
                groups.union(form_owner[form], index)
            else:
                form_owner[form] = index

    members: dict[int, list[ExtractedCharacter]] = {}
    for index, character in enumerate(characters):
        members.setdefault(groups.find(index), []).append(character)
    merged_by_root = {root: _merge_group(group) for root, group in members.items()}
    key_by_form = {
        form: merged_by_root[groups.find(owner)].key for form, owner in form_owner.items()
    }

    relations = [relation for result in results for relation in result.relations]
    return MergedArticle(
        characters=sorted(merged_by_root.values(), key=lambda character: character.key),
        relations=_merge_relations(relations, key_by_form),
    )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        self.parent[self.find(second)] = self.find(first)


def _strip_edge_noise(text: str) -> str:
    start, end = 0, len(text)
    while start < end and _is_edge_noise(text[start]):
        start += 1
    while end > start and _is_edge_noise(text[end - 1]):
        end -= 1
    return text[start:end]


def _is_edge_noise(char: str) -> bool:
    return char.isspace() or unicodedata.category(char).startswith("P")


def _split_camel_case(text: str) -> str:
    pieces: list[str] = []
    for index, current in enumerate(text):
        previous = text[index - 1] if index else ""
        if current.isupper() and (previous.islower() or previous.isdigit()):
            pieces.append("_")
        pieces.append(current)
    return "".join(pieces)


def _forms(character: ExtractedCharacter) -> set[str]:
    return {form for raw in (character.name, *character.aliases) if (form := normalize_name(raw))}


def _merge_group(group: list[ExtractedCharacter]) -> MergedCharacter:
    name = max((character.name.strip() for character in group), key=len)
    key = normalize_name(name)
    surface_by_form: dict[str, str] = {}
    for character in group:
        for raw in (character.name, *character.aliases):
            form = normalize_name(raw)
            if form and form != key:
                surface_by_form.setdefault(form, raw.strip())
    description = max((character.description.strip() for character in group), key=len)
    return MergedCharacter(
        key=key,
        name=name,
        aliases=sorted(surface_by_form.values()),
        description=description,
    )


def _merge_relations(
    relations: list[ExtractedRelation], key_by_form: dict[str, str]
) -> list[MergedRelation]:
    merged: dict[tuple[str, str, str], MergedRelation] = {}
    for relation in relations:
        source_key = key_by_form.get(normalize_name(relation.source))
        target_key = key_by_form.get(normalize_name(relation.target))
        relation_type = normalize_relation_type(relation.type)
        if source_key is None or target_key is None or source_key == target_key:
            continue
        if not relation_type:
            continue
        identity = (source_key, target_key, relation_type)
        existing = merged.get(identity)
        if existing is None or len(relation.evidence) > len(existing.evidence):
            merged[identity] = MergedRelation(
                source_key=source_key,
                target_key=target_key,
                type=relation_type,
                description=relation.description,
                evidence=relation.evidence,
            )
    return list(merged.values())
