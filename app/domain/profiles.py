import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

from app.domain.models import CharacterProfile
from app.domain.names import normalize_name

JUDGE_MAX_RELATIONS = 20
JUDGE_MAX_ARTICLES = 5


class ProfileSource(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def aliases(self) -> Sequence[str]: ...

    @property
    def description(self) -> str: ...


def character_profile(
    character: ProfileSource,
    *,
    relations: Sequence[str] = (),
    articles: Sequence[str] = (),
) -> CharacterProfile:
    return CharacterProfile(
        name=character.name,
        aliases=list(character.aliases),
        description=character.description,
        relations=list(relations),
        articles=list(articles),
    )


def profile_hash(profile: CharacterProfile) -> str:
    identity = [
        normalize_name(profile.name),
        sorted({normalize_name(alias) for alias in profile.aliases}),
        profile.description,
    ]
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def merge_profiles(existing: CharacterProfile, incoming: CharacterProfile) -> CharacterProfile:
    kept_form = normalize_name(existing.name)
    surface_by_form: dict[str, str] = {}
    for alias in (*existing.aliases, *incoming.aliases, incoming.name):
        form = normalize_name(alias)
        if form and form != kept_form:
            surface_by_form.setdefault(form, alias.strip())
    return CharacterProfile(
        name=existing.name,
        aliases=sorted(surface_by_form.values()),
        description=max(existing.description, incoming.description, key=len),
    )


def profile_embedding_text(profile: CharacterProfile) -> str:
    return f"{profile.name}; aliases: {', '.join(profile.aliases) or 'none'}; {profile.description}"


def relation_type_embedding_text(name: str) -> str:
    return f"{name}: {relation_type_description(name)}"


def relation_type_description(name: str) -> str:
    return name.replace("_", " ").lower()
