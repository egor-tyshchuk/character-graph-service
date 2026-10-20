import asyncio
from pathlib import Path


class BlobStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def save(self, article_id: str, data: bytes) -> None:
        await asyncio.to_thread(self._write, self._path(article_id), data)

    async def load(self, article_id: str) -> bytes:
        return await asyncio.to_thread(self._path(article_id).read_bytes)

    async def delete(self, article_id: str) -> None:
        await asyncio.to_thread(self._path(article_id).unlink, missing_ok=True)

    def _path(self, article_id: str) -> Path:
        return self._root / article_id

    def _write(self, path: Path, data: bytes) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
