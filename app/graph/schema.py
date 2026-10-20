from neo4j import AsyncDriver

CHARACTER_VECTOR_INDEX = "character_embedding"
RELATION_TYPE_VECTOR_INDEX = "relation_type_embedding"
INDEX_AWAIT_SECONDS = 300


class SchemaMismatchError(Exception):
    pass


_CONSTRAINTS = (
    "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT article_content_hash IF NOT EXISTS "
    "FOR (a:Article) REQUIRE a.content_hash IS UNIQUE",
    "CREATE CONSTRAINT character_id IF NOT EXISTS FOR (c:Character) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT relation_type_name IF NOT EXISTS "
    "FOR (t:RelationType) REQUIRE t.name IS UNIQUE",
    "CREATE INDEX related_to_type IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.type)",
)


def _vector_index(name: str, label: str, embedding_dim: int) -> str:
    return (
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dim}, "
        "`vector.similarity_function`: 'cosine'}}"
    )


async def apply_schema(driver: AsyncDriver, embedding_dim: int) -> None:
    statements = (
        *_CONSTRAINTS,
        _vector_index(CHARACTER_VECTOR_INDEX, "Character", embedding_dim),
        _vector_index(RELATION_TYPE_VECTOR_INDEX, "RelationType", embedding_dim),
    )
    for statement in statements:
        await driver.execute_query(statement)
    await driver.execute_query(f"CALL db.awaitIndexes({INDEX_AWAIT_SECONDS})")
    await _check_vector_dimensions(driver, embedding_dim)


async def _check_vector_dimensions(driver: AsyncDriver, embedding_dim: int) -> None:
    records, _, _ = await driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name, options WHERE name IN $names RETURN name, options",
        names=[CHARACTER_VECTOR_INDEX, RELATION_TYPE_VECTOR_INDEX],
    )
    dimensions = {
        str(record["name"]): int(record["options"]["indexConfig"]["vector.dimensions"])
        for record in records
    }
    mismatches = [
        f"vector index {name!r} has dimension {actual}"
        for name, actual in sorted(dimensions.items())
        if actual != embedding_dim
    ]
    if mismatches:
        raise SchemaMismatchError(
            f"{'; '.join(mismatches)}, but embedding_dim is {embedding_dim}. "
            "Drop the vector indexes or use a fresh database."
        )
