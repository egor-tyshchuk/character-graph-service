import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from app.config import Settings
from app.domain.models import CharacterProfile
from app.domain.profiles import (
    character_profile,
    merge_profiles,
    profile_embedding_text,
    profile_hash,
    relation_type_description,
    relation_type_embedding_text,
)
from app.graph.repository import GraphRepository
from app.graph.upserts import (
    CharacterUpsert,
    MatchKind,
    Mention,
    NotSameUpsert,
    RelationTypeUpsert,
    RelationUpsert,
    match_strength,
)
from app.llm.client import LLMClient
from app.pipeline.matching import JudgeBudget, MatchOutcome, Rejection, find_match
from app.pipeline.merge import MergedArticle, MergedRelation


@dataclass(frozen=True)
class ArticleEmbeddings:
    characters: dict[str, list[float]]
    new_relation_types: dict[str, list[float]]


class ResolvedArticle(BaseModel):
    characters: list[CharacterUpsert]
    relations: list[RelationUpsert]
    relation_types: list[RelationTypeUpsert]
    not_same: list[NotSameUpsert]


@dataclass
class _Node:
    profile: CharacterProfile
    embedding: list[float] | None
    mentions: list[Mention]

    def upsert(self, node_id: str) -> CharacterUpsert:
        return CharacterUpsert(
            id=node_id,
            name=self.profile.name,
            aliases=self.profile.aliases,
            description=self.profile.description,
            embedding=self.embedding,
            mentions=sorted(
                self.mentions, key=lambda mention: match_strength(mention.match, mention.similarity)
            ),
        )


async def embed_article(
    merged: MergedArticle, *, repo: GraphRepository, llm: LLMClient
) -> ArticleEmbeddings:
    texts = [profile_embedding_text(character_profile(c)) for c in merged.characters]
    vectors = await llm.embed(texts) if texts else []
    types = sorted({relation.type for relation in merged.relations})
    new_types = sorted(set(types) - await repo.existing_relation_types(types))
    type_texts = [relation_type_embedding_text(name) for name in new_types]
    type_vectors = await llm.embed(type_texts) if type_texts else []
    return ArticleEmbeddings(
        characters={c.key: v for c, v in zip(merged.characters, vectors, strict=True)},
        new_relation_types=dict(zip(new_types, type_vectors, strict=True)),
    )


async def resolve_article(
    merged: MergedArticle,
    embeddings: ArticleEmbeddings,
    *,
    article_title: str,
    repo: GraphRepository,
    llm: LLMClient,
    settings: Settings,
) -> ResolvedArticle:
    budget = JudgeBudget(settings.dedup_max_judge_calls_per_article)
    lines = _relation_lines(merged)
    nodes: dict[str, _Node] = {}
    ids_by_key: dict[str, str] = {}
    rejected: list[tuple[str, Rejection]] = []
    for character in merged.characters:
        profile = character_profile(character)
        embedding = embeddings.characters[character.key]
        judge_profile = character_profile(
            character, relations=lines.get(character.key, []), articles=[article_title]
        )
        outcome = await find_match(
            judge_profile, embedding, repo=repo, llm=llm, settings=settings, budget=budget
        )
        node_id = _place(nodes, profile, embedding, outcome)
        ids_by_key[character.key] = node_id
        rejected.extend((node_id, rejection) for rejection in outcome.rejections)
    relations = _rewrite_relations(merged.relations, ids_by_key)
    used_types = {relation.type for relation in relations}
    return ResolvedArticle(
        characters=[node.upsert(node_id) for node_id, node in nodes.items()],
        relations=relations,
        relation_types=[
            RelationTypeUpsert(
                name=name, description=relation_type_description(name), embedding=vector
            )
            for name, vector in embeddings.new_relation_types.items()
            if name in used_types
        ],
        not_same=_not_same_verdicts(rejected, nodes),
    )


def _relation_lines(merged: MergedArticle) -> dict[str, list[str]]:
    names = {character.key: character.name for character in merged.characters}
    lines: dict[str, set[str]] = {}
    for relation in merged.relations:
        lines.setdefault(relation.source_key, set()).add(
            f"{relation.type} → {names[relation.target_key]}"
        )
        lines.setdefault(relation.target_key, set()).add(
            f"← {relation.type} {names[relation.source_key]}"
        )
    return {key: sorted(key_lines) for key, key_lines in lines.items()}


def _place(
    nodes: dict[str, _Node],
    profile: CharacterProfile,
    embedding: list[float],
    outcome: MatchOutcome,
) -> str:
    if outcome.match is None:
        node_id = str(uuid.uuid4())
        mention = _mention(profile, MatchKind.NEW, outcome.best_similarity)
        nodes[node_id] = _Node(profile=profile, embedding=embedding, mentions=[mention])
        return node_id
    existing, kind = outcome.match
    node = nodes.setdefault(
        existing.id, _Node(profile=character_profile(existing), embedding=None, mentions=[])
    )
    node.profile = merge_profiles(node.profile, profile)
    node.mentions.append(_mention(profile, kind, existing.similarity))
    return existing.id


def _mention(profile: CharacterProfile, kind: MatchKind, similarity: float | None) -> Mention:
    return Mention(
        surface_name=profile.name,
        description=profile.description,
        match=kind,
        similarity=similarity,
    )


def _rewrite_relations(
    relations: Sequence[MergedRelation], ids_by_key: dict[str, str]
) -> list[RelationUpsert]:
    by_identity: dict[tuple[str, str, str], RelationUpsert] = {}
    for relation in relations:
        source_id, target_id = ids_by_key[relation.source_key], ids_by_key[relation.target_key]
        if source_id == target_id:
            continue
        identity = (source_id, target_id, relation.type)
        kept = by_identity.get(identity)
        if kept is None or len(relation.evidence) > len(kept.evidence):
            by_identity[identity] = RelationUpsert(
                source_id=source_id,
                target_id=target_id,
                type=relation.type,
                description=relation.description,
                evidence=relation.evidence,
            )
    return list(by_identity.values())


def _not_same_verdicts(
    rejected: Sequence[tuple[str, Rejection]], nodes: dict[str, _Node]
) -> list[NotSameUpsert]:
    verdicts: dict[frozenset[str], NotSameUpsert] = {}
    for node_id, rejection in rejected:
        judged = rejection.existing
        judged_node = nodes.get(judged.id)
        judged_profile = judged_node.profile if judged_node else character_profile(judged)
        verdicts[frozenset((node_id, judged.id))] = NotSameUpsert(
            a_id=node_id,
            b_id=judged.id,
            reason=rejection.reason,
            hash_a=profile_hash(nodes[node_id].profile),
            hash_b=profile_hash(judged_profile),
        )
    return list(verdicts.values())
