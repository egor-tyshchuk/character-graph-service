import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

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
    compatible = unicodedata.normalize("NFKC", raw)
    snake = NON_ALPHANUMERIC_RUN.sub("_", _split_camel_case(compatible))
    return snake.upper().strip("_")


def merge_extractions(results: Sequence[ExtractionResult]) -> MergedArticle:
    buckets = _name_buckets(results)
    groups = _UnionFind([bucket.results for bucket in buckets])
    neighbors = {index: _linked_buckets(index, buckets) for index in range(len(buckets))}
    ambiguous = {index for index, linked in neighbors.items() if _has_conflict(linked, buckets)}
    for index, linked in neighbors.items():
        for other in linked:
            if index not in ambiguous and other not in ambiguous:
                groups.union_if_disjoint(index, other)

    members: dict[int, list[tuple[int, ExtractedCharacter]]] = {}
    for index, bucket in enumerate(buckets):
        members.setdefault(groups.find(index), []).extend(bucket.records)
    characters = [
        _merge_group([record for _, record in sorted(group, key=_position)])
        for group in members.values()
    ]

    relations = [relation for result in results for relation in result.relations]
    return MergedArticle(
        characters=sorted(characters, key=lambda character: character.key),
        relations=_merge_relations(relations, _EndpointResolver(characters)),
    )


@dataclass
class _NameBucket:
    name: str
    records: list[tuple[int, ExtractedCharacter]] = field(default_factory=list)
    results: set[int] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)


class _UnionFind:
    def __init__(self, result_sets: list[set[int]]) -> None:
        self.parent = list(range(len(result_sets)))
        self.results = [set(result_set) for result_set in result_sets]

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            item = self.parent[item]
        return item

    def union_if_disjoint(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root == second_root or self.results[first_root] & self.results[second_root]:
            return
        self.parent[second_root] = first_root
        self.results[first_root] |= self.results[second_root]


class _EndpointResolver:
    def __init__(self, characters: list[MergedCharacter]) -> None:
        self.canonical_keys = {character.key for character in characters}
        self.keys_by_form: dict[str, set[str]] = {}
        for character in characters:
            for form in {character.key, *map(normalize_name, character.aliases)}:
                self.keys_by_form.setdefault(form, set()).add(character.key)

    def resolve(self, endpoint: str) -> str | None:
        form = normalize_name(endpoint)
        if form in self.canonical_keys:
            return form
        owners = self.keys_by_form.get(form, set())
        return next(iter(owners)) if len(owners) == 1 else None


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


def _name_buckets(results: Sequence[ExtractionResult]) -> list[_NameBucket]:
    buckets: dict[str, _NameBucket] = {}
    position = 0
    for result_index, result in enumerate(results):
        for character in result.characters:
            name = normalize_name(character.name)
            if not name:
                continue
            bucket = buckets.setdefault(name, _NameBucket(name))
            bucket.records.append((position, character))
            bucket.results.add(result_index)
            bucket.aliases |= {
                form for alias in character.aliases if (form := normalize_name(alias))
            }
            position += 1
    return list(buckets.values())


def _position(record: tuple[int, ExtractedCharacter]) -> int:
    return record[0]


def _linked_buckets(index: int, buckets: list[_NameBucket]) -> list[int]:
    bucket = buckets[index]
    return [
        other_index
        for other_index, other in enumerate(buckets)
        if other_index != index
        and not bucket.results & other.results
        and (bucket.name in other.aliases or other.name in bucket.aliases)
    ]


def _has_conflict(indices: list[int], buckets: list[_NameBucket]) -> bool:
    return any(
        buckets[first].results & buckets[second].results
        for first, second in combinations(indices, 2)
    )


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
    relations: list[ExtractedRelation], resolver: _EndpointResolver
) -> list[MergedRelation]:
    merged: dict[tuple[str, str, str], MergedRelation] = {}
    for relation in relations:
        source_key = resolver.resolve(relation.source)
        target_key = resolver.resolve(relation.target)
        relation_type = normalize_relation_type(relation.type)
        if source_key is None or target_key is None or source_key == target_key:
            continue
        if not relation_type:
            continue
        evidence = relation.evidence.strip()
        identity = (source_key, target_key, relation_type)
        existing = merged.get(identity)
        if existing is None or len(evidence) > len(existing.evidence):
            merged[identity] = MergedRelation(
                source_key=source_key,
                target_key=target_key,
                type=relation_type,
                description=relation.description.strip(),
                evidence=evidence,
            )
    return list(merged.values())
