from app.domain.profiles import profile_hash
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
