from itertools import pairwise

from app.ingestion.chunking import chunk_text


def _paragraphs(count: int, words: int = 12) -> list[str]:
    return [" ".join(f"p{index}w{word}" for word in range(words)) for index in range(count)]


def _shared_overlap(previous: str, current: str, *, limit: int) -> str:
    for size in range(min(limit, len(previous)), 0, -1):
        if current.startswith(previous[-size:]):
            return previous[-size:]
    return ""


def test_short_text_single_chunk() -> None:
    text = "Anna met Mark.\n\nThey are siblings."

    assert chunk_text(text, max_chars=1000, overlap_chars=100) == [text]


def test_chunks_respect_max_chars() -> None:
    text = "\n\n".join(_paragraphs(40))

    chunks = chunk_text(text, max_chars=300, overlap_chars=60)

    assert len(chunks) > 1
    assert all(len(chunk) <= 300 + 60 for chunk in chunks)


def test_overlap_carries_tail_of_previous_chunk() -> None:
    text = "\n\n".join(_paragraphs(20))

    chunks = chunk_text(text, max_chars=300, overlap_chars=60)

    assert len(chunks) > 1
    for previous, current in pairwise(chunks):
        overlap = _shared_overlap(previous, current, limit=60)
        assert overlap
        assert previous[-len(overlap) - 1].isspace()
        assert current[len(overlap)].isspace()


def test_long_paragraph_hard_split() -> None:
    paragraph = " ".join(f"word{index}" for index in range(200))

    chunks = chunk_text(paragraph, max_chars=100, overlap_chars=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    joined = " ".join(chunks)
    assert all(f"word{index}" in joined.split() for index in range(200))


def test_empty_text_returns_empty_list() -> None:
    assert chunk_text("", max_chars=100, overlap_chars=10) == []
    assert chunk_text(" \n\n \t\n", max_chars=100, overlap_chars=10) == []


def test_no_text_lost() -> None:
    paragraphs = _paragraphs(30, words=8)
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, max_chars=250, overlap_chars=50)

    assert all(any(paragraph in chunk for chunk in chunks) for paragraph in paragraphs)
