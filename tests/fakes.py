import asyncio
import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence

from app.llm.schemas import CharacterProfile, DedupVerdict, ExtractionResult

DEFAULT_IMAGE_DESCRIPTION = "A photo of Anna Petrova and Mark Petrov, captioned 'siblings'."


def vec(*xs: float, dim: int = 8) -> list[float]:
    padded = [*xs, *([0.0] * (dim - len(xs)))]
    norm = math.sqrt(sum(x * x for x in padded))
    return [x / norm for x in padded]


def _hash_vector(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    while len(digest) < dim:
        digest += hashlib.sha256(digest).digest()
    return vec(*(byte - 127.5 for byte in digest[:dim]), dim=dim)


class FakeLLMClient:
    def __init__(
        self,
        *,
        dim: int = 8,
        extractions: Mapping[str, ExtractionResult] | None = None,
        embeddings: Mapping[str, list[float]] | None = None,
        same_pairs: Iterable[tuple[str, str]] = (),
        image_description: str = DEFAULT_IMAGE_DESCRIPTION,
    ) -> None:
        self.dim = dim
        self.extractions = dict(extractions or {})
        self.embeddings = dict(embeddings or {})
        self.same_pairs = [frozenset(pair) for pair in same_pairs]
        self.image_description = image_description
        self.extract_calls: list[str] = []
        self.embed_calls: list[list[str]] = []
        self.judge_calls: list[tuple[CharacterProfile, CharacterProfile]] = []
        self.image_calls: list[str] = []

    async def extract(self, chunk: str, known_relation_types: Sequence[str]) -> ExtractionResult:
        await asyncio.sleep(0)
        self.extract_calls.append(chunk)
        for key, result in self.extractions.items():
            if key in chunk:
                return result
        return ExtractionResult(characters=[], relations=[])

    async def describe_image(self, image_png: bytes, page_context: str) -> str:
        self.image_calls.append(page_context)
        return self.image_description

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self._embed_one(text) for text in texts]

    async def judge_same_character(
        self, candidate: CharacterProfile, existing: CharacterProfile
    ) -> DedupVerdict:
        self.judge_calls.append((candidate, existing))
        same = frozenset({candidate.name, existing.name}) in self.same_pairs
        return DedupVerdict(same=same, reason="configured pair" if same else "not configured")

    def _embed_one(self, text: str) -> list[float]:
        for prefix, vector in self.embeddings.items():
            if text.startswith(prefix):
                return vector
        return _hash_vector(text, self.dim)
