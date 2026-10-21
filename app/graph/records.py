from datetime import datetime
from typing import Any

from neo4j import AsyncDriver
from pydantic import TypeAdapter

from app.graph.upserts import Mention

MENTIONS = TypeAdapter(list[Mention])


class Neo4jStore:
    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver


def to_native(value: Any) -> datetime:
    result: datetime = value.to_native()
    return result


def character_projection(var: str) -> str:
    return (
        f"{var} {{.id, .name, .aliases, .description, "
        f"article_count: COUNT {{ ({var})-[:APPEARS_IN]->(:Article) }}}}"
    )
