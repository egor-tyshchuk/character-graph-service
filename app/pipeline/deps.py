from dataclasses import dataclass

from redis.asyncio import Redis

from app.config import Settings
from app.graph.repository import GraphRepository
from app.llm.client import LLMClient
from app.storage import BlobStore


@dataclass(frozen=True)
class PipelineDeps:
    repo: GraphRepository
    llm: LLMClient
    blobs: BlobStore
    redis: Redis
    settings: Settings
