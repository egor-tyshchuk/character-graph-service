import math

from app.domain.models import CharacterProfile
from app.llm.schemas import ExtractedCharacter, ExtractionResult
from tests.fakes import FakeLLMClient, vec


def test_vec_pads_and_normalizes() -> None:
    vector = vec(3.0, 4.0, dim=4)

    assert vector == [0.6, 0.8, 0.0, 0.0]


async def test_fake_embed_prefix_override_and_determinism() -> None:
    override = vec(1.0)
    fake = FakeLLMClient(dim=8, embeddings={"Anna": override})

    first = await fake.embed(["Anna Petrova, a doctor", "Mark Petrov", "About Anna"])
    second = await fake.embed(["Mark Petrov"])

    assert first[0] == override
    assert first[1] == second[0]
    assert first[2] != override
    assert all(len(vector) == 8 for vector in first)
    assert math.isclose(sum(x * x for x in first[1]), 1.0)
    assert fake.embed_calls == [
        ["Anna Petrova, a doctor", "Mark Petrov", "About Anna"],
        ["Mark Petrov"],
    ]


async def test_fake_extract_returns_first_matching_key() -> None:
    anna = ExtractionResult(
        characters=[ExtractedCharacter(name="Anna", aliases=[], description="A doctor.")],
        relations=[],
    )
    fake = FakeLLMClient(
        extractions={"Anna": anna, "Mark": ExtractionResult(characters=[], relations=[])}
    )

    assert await fake.extract("Anna met Mark", []) == anna
    assert await fake.extract("Nobody here", []) == ExtractionResult(characters=[], relations=[])
    assert fake.extract_calls == ["Anna met Mark", "Nobody here"]


async def test_fake_judge_same_pairs_is_symmetric() -> None:
    fake = FakeLLMClient(same_pairs=[("Ivan Petrov", "Иван Петров")])
    ivan = CharacterProfile(name="Ivan Petrov", aliases=[], description="")
    ivan_ru = CharacterProfile(name="Иван Петров", aliases=[], description="")
    mark = CharacterProfile(name="Mark", aliases=[], description="")

    assert (await fake.judge_same_character(ivan, ivan_ru)).same is True
    assert (await fake.judge_same_character(ivan_ru, ivan)).same is True
    assert (await fake.judge_same_character(ivan, mark)).same is False
    assert fake.judge_calls == [(ivan, ivan_ru), (ivan_ru, ivan), (ivan, mark)]


async def test_fake_describe_image_records_context() -> None:
    fake = FakeLLMClient(image_description="A cat.")

    assert await fake.describe_image(b"png", "page text") == "A cat."
    assert fake.image_calls == ["page text"]
