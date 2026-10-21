from app.domain.models import CharacterProfile
from app.llm.prompts import render_extraction_user, render_judge, render_vision


def test_render_judge_includes_relations_and_articles() -> None:
    candidate = CharacterProfile(
        name="Anna Petrova",
        aliases=["Anna"],
        description="A doctor.",
        relations=["SIBLING_OF → Mark Petrov", "← EMPLOYS Pyotr Sidorov"],
        articles=["Family chronicle"],
    )
    existing = CharacterProfile(name="Анна Петрова", aliases=[], description="Врач.")

    prompt = render_judge(candidate, existing)

    assert (
        "Name: Anna Petrova\n"
        "Aliases: Anna\n"
        "Description: A doctor.\n"
        "Relations: SIBLING_OF → Mark Petrov; ← EMPLOYS Pyotr Sidorov\n"
        "Articles: Family chronicle"
    ) in prompt
    assert (
        "Name: Анна Петрова\nAliases: none\nDescription: Врач.\nRelations: none\nArticles: none"
    ) in prompt
    assert prompt.index("Anna Petrova") < prompt.index("Анна Петрова")


def test_render_judge_limits_relations_and_articles() -> None:
    profile = CharacterProfile(
        name="Ivan",
        aliases=[],
        description="A pilot.",
        relations=[f"FRIEND_OF → Person {i}" for i in range(25)],
        articles=[f"Article {i}" for i in range(8)],
    )

    prompt = render_judge(profile, profile)

    assert "Person 19" in prompt
    assert "Person 20" not in prompt
    assert "Article 4" in prompt
    assert "Article 5" not in prompt


def test_render_extraction_user_without_known_types_says_none_yet() -> None:
    prompt = render_extraction_user("Some text", [])

    assert "Known relationship types: none yet" in prompt
    assert "<<<\nSome text\n>>>" in prompt


def test_render_extraction_user_joins_known_types() -> None:
    prompt = render_extraction_user("Some text", ["FRIEND_OF", "SIBLING_OF"])

    assert "Known relationship types: FRIEND_OF, SIBLING_OF" in prompt


def test_render_vision_truncates_context_to_1500_chars() -> None:
    prompt = render_vision("a" * 1500 + "b" * 10)

    assert "a" * 1500 in prompt
    assert "b" not in prompt.split("<<<")[-1]
