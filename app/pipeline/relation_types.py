from app.domain.names import normalize_relation_type
from app.pipeline.deps import PipelineDeps


async def resolve_relation_type(requested: str, deps: PipelineDeps) -> str | None:
    normalized = normalize_relation_type(requested)
    if normalized and normalized in await deps.repo.existing_relation_types([normalized]):
        return normalized
    [embedding] = await deps.llm.embed([requested])
    for name, similarity in await deps.repo.find_similar_relation_types(embedding, 1):
        if similarity >= deps.settings.relation_type_query_threshold:
            return name
    return None
