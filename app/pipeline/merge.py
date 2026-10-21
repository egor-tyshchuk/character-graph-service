from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

from pydantic import BaseModel

from app.domain.names import normalize_name, normalize_relation_type
from app.llm.schemas import ExtractedCharacter, ExtractedRelation, ExtractionResult


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


def merge_extractions(results: Sequence[ExtractionResult]) -> MergedArticle:
    buckets = _name_buckets(results)
    characters = [
        _merge_group([record for _, record in sorted(group, key=_position)])
        for component in _link_components(buckets)
        for group in _resolve_component(component)
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
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        self.parent[self.find(second)] = self.find(first)


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


def _link_components(buckets: list[_NameBucket]) -> list[list[_NameBucket]]:
    links = _UnionFind(len(buckets))
    for (first_index, first), (second_index, second) in combinations(enumerate(buckets), 2):
        if first.name in second.aliases or second.name in first.aliases:
            links.union(first_index, second_index)
    components: dict[int, list[_NameBucket]] = {}
    for index, bucket in enumerate(buckets):
        components.setdefault(links.find(index), []).append(bucket)
    return list(components.values())


def _resolve_component(
    component: list[_NameBucket],
) -> list[list[tuple[int, ExtractedCharacter]]]:
    shares_chunk = any(
        first.results & second.results for first, second in combinations(component, 2)
    )
    if shares_chunk:
        return [bucket.records for bucket in component]
    return [[record for bucket in component for record in bucket.records]]


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
    return sorted(
        merged.values(),
        key=lambda relation: (relation.source_key, relation.target_key, relation.type),
    )
