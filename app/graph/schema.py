from neo4j import AsyncDriver

CHARACTER_VECTOR_INDEX = "character_embedding"
RELATION_TYPE_VECTOR_INDEX = "relation_type_embedding"

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
