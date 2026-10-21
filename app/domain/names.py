import re
import unicodedata

NON_ALPHANUMERIC_RUN = re.compile(r"[\W_]+")


def normalize_name(name: str) -> str:
    collapsed = " ".join(unicodedata.normalize("NFKC", name).casefold().split())
    return _strip_edge_noise(collapsed)


def normalize_relation_type(raw: str) -> str:
    compatible = unicodedata.normalize("NFKC", raw)
    snake = NON_ALPHANUMERIC_RUN.sub("_", _split_camel_case(compatible))
    return snake.upper().strip("_")


def _strip_edge_noise(text: str) -> str:
    start, end = 0, len(text)
    while start < end and _is_edge_noise(text[start]):
        start += 1
    while end > start and _is_edge_noise(text[end - 1]):
        end -= 1
    return text[start:end]


def _is_edge_noise(char: str) -> bool:
    return char.isspace() or unicodedata.category(char).startswith("P")


def _split_camel_case(text: str) -> str:
    pieces: list[str] = []
    for index, current in enumerate(text):
        previous = text[index - 1] if index else ""
        if current.isupper() and (previous.islower() or previous.isdigit()):
            pieces.append("_")
        pieces.append(current)
    return "".join(pieces)
