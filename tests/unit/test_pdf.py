import pytest
from pydantic import SecretStr

from app.config import Settings
from app.ingestion.models import DocumentParseError
from app.ingestion.pdf import parse_pdf
from tests.fakes import FakeLLMClient
from tests.pdf_factory import PageSpec, make_pdf, make_png, set_title

LONG_TEXT = (
    "Anna Petrova is a surgeon in Kazan. Her brother Mark Petrov teaches history. "
    "They meet every summer at their grandmother's dacha."
)


def _settings(*, pdf_max_images: int = 20) -> Settings:
    return Settings(
        jwt_secret=SecretStr("unit-test-secret-unit-test-secret"), pdf_max_images=pdf_max_images
    )


async def test_text_pdf_without_images() -> None:
    llm = FakeLLMClient()
    data = make_pdf(
        [PageSpec(text=LONG_TEXT), PageSpec(text=f"Second page about Mark Petrov. {LONG_TEXT}")]
    )

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert "Anna Petrova is a surgeon" in document.text
    assert "Second page about Mark Petrov." in document.text
    assert document.images_described == 0
    assert document.title is None
    assert llm.image_calls == []


async def test_embedded_image_described_inline() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT, images=[make_png(300, 300, "siblings")])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert "[Image on page 1: A photo of Anna Petrova" in document.text
    assert document.text.index("Anna Petrova is a surgeon") < document.text.index("[Image on")
    assert document.images_described == 1
    assert len(llm.image_calls) == 1
    assert "Anna Petrova is a surgeon" in llm.image_calls[0]


async def test_small_images_skipped() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT, images=[make_png(50, 50)])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert llm.image_calls == []
    assert document.images_described == 0
    assert "[Image on page" not in document.text


async def test_scanned_page_rendered_once() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(text=LONG_TEXT), PageSpec(images=[make_png(300, 300, "scan")])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert "[Image on page 2:" in document.text
    assert "[Image on page 1:" not in document.text
    assert document.images_described == 1


async def test_scanned_first_page_block() -> None:
    llm = FakeLLMClient()
    data = make_pdf([PageSpec(images=[make_png(300, 300, "scan")])])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert document.text.startswith("[Image on page 1:")


async def test_repeated_image_described_once() -> None:
    llm = FakeLLMClient()
    png = make_png(300, 300, "logo")
    data = make_pdf([PageSpec(text=f"{LONG_TEXT} Page {n}.", images=[png]) for n in range(3)])

    document = await parse_pdf(data, llm=llm, settings=_settings())

    assert len(llm.image_calls) == 1
    assert document.images_described == 1
    assert document.text.count("[Image on page") == 1


async def test_image_cap_respected() -> None:
    llm = FakeLLMClient()
    images = [make_png(200 + index * 10, 200, f"image {index}") for index in range(5)]
    data = make_pdf([PageSpec(text=LONG_TEXT, images=images)])

    document = await parse_pdf(data, llm=llm, settings=_settings(pdf_max_images=2))

    assert len(llm.image_calls) == 2
    assert document.images_described == 2


async def test_title_from_metadata() -> None:
    data = set_title(make_pdf([PageSpec(text=LONG_TEXT)]), "The Petrovs")

    document = await parse_pdf(data, llm=FakeLLMClient(), settings=_settings())

    assert document.title == "The Petrovs"


async def test_corrupt_pdf_raises() -> None:
    with pytest.raises(DocumentParseError):
        await parse_pdf(b"%PDF-1.7 garbage", llm=FakeLLMClient(), settings=_settings())


async def test_encrypted_pdf_raises() -> None:
    data = make_pdf([PageSpec(text=LONG_TEXT)], password="x")

    with pytest.raises(DocumentParseError):
        await parse_pdf(data, llm=FakeLLMClient(), settings=_settings())
