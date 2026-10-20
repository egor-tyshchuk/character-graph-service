import hashlib
import re
import unicodedata

BYTE_ORDER_MARK = "﻿"
_INLINE_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")


def normalize_text_for_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text.removeprefix(BYTE_ORDER_MARK))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = (_INLINE_WHITESPACE.sub(" ", line.rstrip()) for line in normalized.split("\n"))
    return _BLANK_LINE_RUN.sub("\n\n", "\n".join(lines)).strip()


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
