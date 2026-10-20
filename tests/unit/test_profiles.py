from app.domain.profiles import (
    merge_profiles,
    profile_embedding_text,
    profile_hash,
    relation_type_embedding_text,
)
from app.llm.schemas import CharacterProfile


def _profile(
    name: str = "Anna Petrova",
    aliases: list[str] | None = None,
    description: str = "A lighthouse keeper.",
) -> CharacterProfile:
    return CharacterProfile(
        name=name, aliases=["Anya", "Аня"] if aliases is None else aliases, description=description
    )


def test_profile_hash_ignores_name_case_whitespace_and_alias_order() -> None:
    variant = _profile(name="  ANNA   petrova ", aliases=["аня", "ANYA"])

    assert profile_hash(variant) == profile_hash(_profile())


def test_profile_hash_ignores_graph_context() -> None:
    with_context = _profile().model_copy(
        update={"relations": ["SIBLING_OF → Mark Petrov"], "articles": ["A tale"]}
    )

    assert profile_hash(with_context) == profile_hash(_profile())


def test_profile_hash_changes_with_name_aliases_or_description() -> None:
    baseline = profile_hash(_profile())

    assert profile_hash(_profile(name="Anna Ivanova")) != baseline
    assert profile_hash(_profile(aliases=["Anya"])) != baseline
    assert profile_hash(_profile(description="A sailor.")) != baseline


def test_profile_hash_does_not_confuse_field_boundaries() -> None:
    first = _profile(name="Anna", aliases=["Petrova"], description="")
    second = _profile(name="Anna Petrova", aliases=[], description="")

    assert profile_hash(first) != profile_hash(second)


def test_profile_hash_ignores_duplicate_aliases() -> None:
    duplicated = _profile(aliases=["Anya", "anya", "Аня", " АНЯ "])

    assert profile_hash(duplicated) == profile_hash(_profile())


def test_merge_profiles_keeps_existing_name_and_unions_aliases() -> None:
    existing = CharacterProfile(
        name="Anna Petrova", aliases=["Anya", "Аня"], description="A keeper."
    )
    incoming = CharacterProfile(
        name="Анна Петрова",
        aliases=["anya", "Анечка", "ANNA PETROVA"],
        description="A lighthouse keeper on the northern coast.",
    )

    merged = merge_profiles(existing, incoming)

    assert merged == CharacterProfile(
        name="Anna Petrova",
        aliases=["Anya", "Анечка", "Анна Петрова", "Аня"],
        description="A lighthouse keeper on the northern coast.",
    )


def test_merge_profiles_keeps_existing_description_when_not_shorter() -> None:
    existing = CharacterProfile(name="Mark", aliases=[], description="A sailor.")
    incoming = CharacterProfile(name="Mark", aliases=[], description="A tailor")

    merged = merge_profiles(existing, incoming)

    assert merged.description == "A sailor."
    assert merged.aliases == []


def test_profile_embedding_text_starts_with_name() -> None:
    assert profile_embedding_text(_profile()) == (
        "Anna Petrova; aliases: Anya, Аня; A lighthouse keeper."
    )
    assert profile_embedding_text(_profile(aliases=[])) == (
        "Anna Petrova; aliases: none; A lighthouse keeper."
    )


def test_relation_type_embedding_text() -> None:
    assert relation_type_embedding_text("FRIEND_OF") == "FRIEND_OF: friend of"
