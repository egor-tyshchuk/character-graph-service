from app.domain.names import normalize_relation_type
from app.pipeline.deps import PipelineDeps


async def resolve_relation_type(requested: str, deps: PipelineDeps) -> str | None:
    normalized = normalize_relation_type(requested)
    if normalized and normalized in await deps.repo.existing_relation_types([normalized]):
        return normalized
    [embedding] = await deps.llm.embed([requested])
    similar = await deps.repo.find_similar_relation_types(embedding, deps.settings.vector_search_k)
    best = min(similar, key=lambda item: (-item[1], item[0]), default=None)
    if best is not None and best[1] >= deps.settings.relation_type_query_threshold:
        return best[0]
    return None
