import pytest

from app.ingestion.markdown import parse_markdown
from app.ingestion.models import DocumentParseError


def test_title_from_first_heading() -> None:
    document = parse_markdown(b"intro\n# Title\n## Sub")

    assert document.title == "Title"
    assert "## Sub" in document.text
    assert document.images_described == 0


def test_no_heading_title_none() -> None:
    document = parse_markdown(b"Anna met Mark.\n#hashtag is not a heading")

    assert document.title is None
    assert document.text.startswith("Anna met Mark.")


def test_bom_stripped() -> None:
    document = parse_markdown(b"\xef\xbb\xbf# Caf\xc3\xa9 Petrov\nbody")

    assert document.title == "Café Petrov"
    assert not document.text.startswith("\ufeff")


def test_invalid_utf8_raises() -> None:
    with pytest.raises(DocumentParseError):
        parse_markdown(b"# Title\n\xff\xfe\xfa")
