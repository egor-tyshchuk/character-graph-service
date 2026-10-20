import re

from app.ingestion.models import DocumentParseError, ParsedDocument

ATX_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(?P<title>.+?)(?:[ \t]+#+)?[ \t]*$", re.MULTILINE)


def parse_markdown(data: bytes) -> ParsedDocument:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError(f"markdown is not valid UTF-8: {exc}") from exc
    heading = ATX_HEADING.search(text)
    title = heading.group("title").strip() if heading else None
    return ParsedDocument(title=title or None, text=text, images_described=0)
