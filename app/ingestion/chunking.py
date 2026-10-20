import re

PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
PARAGRAPH_SEPARATOR = "\n\n"


def chunk_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    units = [
        piece for paragraph in _paragraphs(text) for piece in _split_oversized(paragraph, max_chars)
    ]
    chunks: list[str] = []
    for body in _pack(units, max_chars):
        overlap = _tail(chunks[-1], overlap_chars - len(PARAGRAPH_SEPARATOR)) if chunks else ""
        chunks.append(f"{overlap}{PARAGRAPH_SEPARATOR}{body}" if overlap else body)
    return chunks


def _paragraphs(text: str) -> list[str]:
    return [stripped for part in PARAGRAPH_BREAK.split(text) if (stripped := part.strip())]


def _split_oversized(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    words = [
        word[start : start + max_chars]
        for word in paragraph.split()
        for start in range(0, len(word), max_chars)
    ]
    return _pack(words, max_chars, separator=" ")


def _pack(units: list[str], max_chars: int, separator: str = PARAGRAPH_SEPARATOR) -> list[str]:
    packed: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(separator) + len(unit) > max_chars:
            packed.append(current)
            current = unit
        else:
            current = f"{current}{separator}{unit}" if current else unit
    if current:
        packed.append(current)
    return packed


def _tail(chunk: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(chunk) <= limit:
        return chunk.strip()
    window = chunk[-limit:]
    if not chunk[-limit - 1].isspace():
        boundary = re.search(r"\s", window)
        if boundary is None:
            return ""
        window = window[boundary.start() :]
    return window.strip()
