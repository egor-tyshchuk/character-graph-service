from pathlib import Path

import pytest

from app.storage import BlobStore


async def test_save_load_delete_roundtrip(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "nested" / "uploads")

    await store.save("article-1", b"%PDF-1.7 bytes")

    assert await store.load("article-1") == b"%PDF-1.7 bytes"
    await store.delete("article-1")
    with pytest.raises(FileNotFoundError):
        await store.load("article-1")


async def test_save_overwrites_existing_blob(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)

    await store.save("article-1", b"first")
    await store.save("article-1", b"second")

    assert await store.load("article-1") == b"second"


async def test_delete_missing_is_noop(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "never-created")

    await store.delete("missing")
