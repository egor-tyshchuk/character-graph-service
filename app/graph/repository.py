from app.graph.articles import ArticleRepository
from app.graph.characters import CharacterRepository
from app.graph.merge import MergeRepository
from app.graph.persist import PersistRepository


class GraphRepository(ArticleRepository, CharacterRepository, PersistRepository, MergeRepository):
    pass
